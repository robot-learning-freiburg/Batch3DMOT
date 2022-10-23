import torch
from torch import nn

import os
import re
import inspect
import os.path as osp
from uuid import uuid1
from itertools import chain
from inspect import Parameter
from typing import List, Optional, Set
from torch_geometric.typing import Adj, Size

from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import gather_csr, scatter, segment_csr

import torch_geometric.nn
from torch_geometric.nn import Sequential

from batch_3dmot.models.heterolinear import HeteroLinear, Linear


class PoseGNN(torch.nn.Module):
    def __init__(self, gnn_depth=6, edge_dim=16, node_dim=19, mp_type: str = "attention"):
        super(PoseGNN, self).__init__()
        self.depth = gnn_depth

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(19, 24),
            nn.ReLU(),
            nn.Linear(24, 36),
            nn.ReLU(),
            nn.Linear(36, 48),
        )
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

        self.knn_conv = torch_geometric.nn.GATConv(48, 48, add_self_loops=False)
        self.message_passing = CausalMessagePassing()

    def forward(self, data):
        pose_feats, edge_index, edge_attr, node_timestamps, batch = (
            data.pose_feats,
            data.edge_index,
            data.edge_attr,
            data.node_timestamps,
            data.batch
        )

        edge_attr = self.edge_encoder(edge_attr.float())
        initial_x = self.node_encoder(pose_feats)
        x = self.node_encoder(pose_feats)
 
        x_enc = x


        for i in range(self.depth):
            if i % 2  == 0:
                for t in torch.unique(node_timestamps).tolist():
                    x_t = x[node_timestamps == t]
                    edge_index_t = torch_geometric.nn.knn_graph(x_t, k=20, loop=False)
                    x_t = self.knn_conv.forward(x_t, edge_index_t)
                    x[node_timestamps == t] == x_t

            # T-A Message Passing
            x, edge_attr = self.message_passing.forward(x, edge_index, edge_attr, initial_x)
        
        # message passing
        return self.edge_classifier(edge_attr), x_enc


class CausalMessagePassing(torch_geometric.nn.MessagePassing):

    def __init__(self):
        super(CausalMessagePassing, self).__init__(aggr='add', node_dim=0)

        self.edge_update = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.create_past_msgs = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )

        self.create_future_msgs = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64)
        )

        self.combine_future_past = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 48)
        )

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def forward(self, x, edge_index, edge_attr, initial_x):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index,
                              size=(x.size(0), x.size(0)),
                              x=x,
                              edge_attr=edge_attr,
                              initial_x=initial_x)

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
                edge_attr, initial_x_i, initial_x_j) -> Tensor:
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
        edge_update_features = torch.cat([x_i, x_j, edge_attr], dim=1)
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
