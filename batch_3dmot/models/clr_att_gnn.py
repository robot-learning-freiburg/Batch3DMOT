import os
import torch
from torch import nn

from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

import torch_geometric.nn
from torch_geometric.typing import Adj, Size
from typing import List, Optional, Set

from batch_3dmot.models import message_passing, attention_message_passing, resnet_fully_conv, pointnet, radarnet


class GNN(torch.nn.Module):
    def __init__(self, img_encoder , lidar_encoder, radar_encoder, use_attention=True, gnn_depth=6, edge_dim=64, node_dim=179):
        super(GNN, self).__init__()
        self.depth = gnn_depth
        self.use_attention = use_attention

        self.resnet = img_encoder
        self.pointnet = lidar_encoder
        self.radarnet = radar_encoder

        for n, param in self.resnet.named_parameters():
            param.requires_grad = False

        for n, param in self.pointnet.named_parameters():
            param.requires_grad = False

        for n, param in self.radarnet.named_parameters():
            param.requires_grad = False

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64)
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(19, 48),
            nn.ReLU(),
            nn.Linear(48, 96)
        )

        self.edge_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.fc_lidar_encoder = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 128),
        )

        self.fc_radar_encoder = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

        self.message_passing = CausalMessagePassing()

        # Node-to-node cross attention
        self.c2c_att = nn.MultiheadAttention(embed_dim=96, num_heads=2, kdim=96, vdim=96, batch_first=True)
        self.l2l_att = nn.MultiheadAttention(embed_dim=128, num_heads=2, kdim=128, vdim=128, batch_first=True)
        self.r2r_att = nn.MultiheadAttention(embed_dim=64, num_heads=2, kdim=64, vdim=64, batch_first=True)

        self.att_edge_encoder = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.knn_conv = torch_geometric.nn.GATConv(96, 96, add_self_loops=False)

    def forward(self, data):

        pose_feats, img_feats, lidar_feats, radar_feats, edge_index, edge_attr, node_timestamps = (
            data.pose_feats,
            data.img_feats,
            data.lidar_feats,
            data.radar_feats,
            data.edge_index,
            data.edge_attr,
            data.node_timestamps,
        )

        pc_nodes = pose_feats.new_zeros((pose_feats.size(0)), dtype=torch.bool)
        pcl_nodes = pose_feats.new_zeros((pose_feats.size(0)), dtype=torch.bool)
        pr_nodes = pose_feats.new_zeros((pose_feats.size(0)), dtype=torch.bool)

        for node_idx, lidar_feat  in enumerate(lidar_feats):
            l = torch.sum(lidar_feat)
            if l:
                pcl_nodes[node_idx] = 1
            else:
                pc_nodes[node_idx] = 1

        for node_idx, radar_feat  in enumerate(radar_feats):
            r = torch.sum(radar_feat)
            if r:
                pr_nodes[node_idx] = 1

        edge_attr = self.edge_encoder(edge_attr.float())

        x_img = self.resnet.encode(img_feats)

        pointnet_out = lidar_feats.new_zeros((pose_feats.size(0), 256))
        if lidar_feats[pcl_nodes].view(-1, 3, 128).size(0) < 2:
            self.pointnet.eval()
            self.fc_lidar_encoder.eval()
        pointnet_out[pcl_nodes] = self.pointnet.forward_feat(lidar_feats[pcl_nodes].view(-1, 3, 128))
        x_lidar = lidar_feats.new_zeros((pose_feats.size(0), 128))
        x_lidar[pcl_nodes] = self.fc_lidar_encoder(pointnet_out[pcl_nodes])

        radarnet_out = radar_feats.new_zeros((pose_feats.size(0), 256))
        if radar_feats[pr_nodes].view(-1, 4, 64).size(0) < 2:
            self.radarnet.eval()
            self.fc_radar_encoder.eval()
        radarnet_out[pr_nodes] = self.radarnet.forward_feat(radar_feats[pr_nodes].view(-1, 4, 64))
        x_radar = radar_feats.new_zeros((pose_feats.size(0), 64))
        x_radar[pr_nodes] = self.fc_radar_encoder(radarnet_out[pr_nodes])

        if self.use_attention:
            x_j_img, x_i_img = x_img[edge_index[0]].view(-1,1,96), x_img[edge_index[1]].view(-1,1,96)
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0]].view(-1,1,128), x_lidar[edge_index[1]].view(-1,1,128)
            x_j_radar, x_i_radar = x_radar[edge_index[0]].view(-1,1,64), x_radar[edge_index[1]].view(-1,1,64)

            x_j_img_att, _ = self.c2c_att(query=x_i_img, key=x_j_img, value=x_j_img, need_weights=False)
            x_i_img_att, _ = self.c2c_att(query=x_j_img, key=x_i_img, value=x_i_img, need_weights=False)

            x_j_lidar_att, _ = self.l2l_att(query=x_i_lidar, key=x_j_lidar, value=x_j_lidar, need_weights=False)
            x_i_lidar_att, _ = self.l2l_att(query=x_j_lidar, key=x_i_lidar, value=x_i_lidar, need_weights=False)

            x_j_radar_att, _ = self.r2r_att(query=x_i_radar, key=x_j_radar, value=x_j_radar, need_weights=False)
            x_i_radar_att, _ = self.r2r_att(query=x_j_radar, key=x_i_radar, value=x_i_radar, need_weights=False)

            x_j_img, x_i_img = x_j_img_att.squeeze(1), x_i_img_att.squeeze(1)
            x_j_lidar, x_i_lidar = x_j_lidar_att.squeeze(1), x_i_lidar_att.squeeze(1)
            x_j_radar, x_i_radar = x_j_radar_att.squeeze(1), x_i_radar_att.squeeze(1)
            
            x_sens_j, x_sens_i = torch.cat([x_j_radar, x_j_lidar, x_j_img], dim=1), torch.cat([x_i_radar, x_i_lidar, x_i_img], dim=1)

            att_edge_feats = torch.cat([x_sens_i, x_sens_j, edge_attr], dim=1)
            att_edge_attr = self.att_edge_encoder(att_edge_feats)

        else:
            x_j_lidar, x_i_lidar = x_lidar[edge_index[0]], x_lidar[edge_index[1]]
            x_j_img, x_i_img = x_img[edge_index[0]], x_img[edge_index[1]]
            att_edge_feats = torch.cat([x_i_img, x_i_lidar, x_j_img, x_j_lidar, edge_attr], dim=1)
            att_edge_attr = self.att_edge_encoder(att_edge_feats)   

        x_sens = torch.cat([x_img, x_lidar, x_radar], dim=1)

        x = torch.cat([pose_feats], dim=1)
        x = self.node_encoder(x)
        initial_x = x

        for i in range(self.depth):
            if i % 2 == 0:
                for t in torch.unique(node_timestamps).tolist():
                    x_t = x[node_timestamps == t]
                    edge_index_t = torch_geometric.nn.knn_graph(x_t, k=20, loop=False)
                    x_t = self.knn_conv.forward(x_t, edge_index_t)
                    x[node_timestamps == t] == x_t
            
            x, edge_attr = self.message_passing.forward(x, edge_index, edge_attr, initial_x, att_edge_attr)

        return self.edge_classifier(edge_attr), x_sens
    

class CausalMessagePassing(torch_geometric.nn.MessagePassing):

    def __init__(self):
        super(CausalMessagePassing, self).__init__(aggr='add')

        self.edge_update = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.create_past_msgs = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 128)
        )

        self.create_future_msgs = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
        )

        self.combine_future_past = nn.Sequential(
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
        )

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def forward(self, x, edge_index, edge_attr, initial_x, att_edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index,
                              size=(x.size(0), x.size(0)),
                              x=x,
                              edge_attr=edge_attr,
                              initial_x=initial_x,
                              att_edge_attr=att_edge_attr)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            past_msgs, future_msgs, update_edges_attr = self.message(**msg_kwargs)

            rows, cols = edge_index

            # Run aggregate for the past and for the future separately
            messages_past = self.aggregate(past_msgs, cols, dim_size=size[1])
            messages_future = self.aggregate(future_msgs, rows, dim_size=size[0])

            messages = torch.cat([messages_past, messages_future], dim=1)

            update_kwargs = self.inspector.distribute('update', coll_dict)

            return self.update(messages, **update_kwargs), update_edges_attr

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr, initial_x_i, initial_x_j, att_edge_attr) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        # Update the edge features based on the adjacent nodes
        edge_update_features = torch.cat([x_i, x_j, edge_attr, att_edge_attr], dim=1)
        updated_edge_attr = self.edge_update(edge_update_features)

        # To construct messages that are in the future we look at
        # the features going into the nodes (edge directed into the future), thus x_i
        future_msg_feats = torch.cat([x_i, updated_edge_attr, initial_x_i], dim=1)
        future_msgs = self.create_future_msgs(future_msg_feats)
        # use x_i to source all information that flows into a node (=into the future)

        # For past messages one takes the feature of the node the edge points to
        # To construct messages from the past we look at all features "leaving"
        # a node, thus x_j (they point to the node in the present)
        past_msg_feats = torch.cat([x_j, updated_edge_attr, initial_x_j], dim=1)
        past_msgs = self.create_past_msgs(past_msg_feats)
        # use x_j to source all neighbor nodes in the past that send features

        return past_msgs, future_msgs, updated_edge_attr

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.
        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        #Combine future and past
        updated_nodes = self.combine_future_past(inputs)

        return updated_nodes
