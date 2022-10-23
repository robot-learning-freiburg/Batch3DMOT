import torch
import torchvision
import torch_geometric.data
import os
import glob
import json
import numpy as np
import networkx as nx

from collections import defaultdict
from collections.abc import Mapping
from batch_3dmot.predict_contrastive import inference

from batch_3dmot.utils.config import ParamLib
import batch_3dmot.utils.dataset

import torch_geometric.utils
import torch_geometric.transforms



class GraphDataset(torch_geometric.data.Dataset):
    """Characterizes a graph dataset to torch geometric."""

    def __init__(self, params: ParamLib, 
                 scenes: list, graph_data_dir: str, batch_size_graph: int, inference: bool):
        super(GraphDataset, self).__init__(params, scenes)
        """
        Initializes a graph dataset.
        Args:
              config: global dict defining paths
              scenes: list with multiple dicts each defining a scene that is loaded to this dataset.
              transform: Some transform to augment/change the graph data.
              pre_transform: Some preprocessing function applied to the sourced graph data.

        """
        self.scenes = scenes
        self.inference = inference
        self.batches = []
        self.list_of_scenes = []
        self.batch_size_graph = batch_size_graph
        self.edge_weighting = True
        self.params = params
        
        graph_files_dir = graph_data_dir
        print(graph_files_dir)
        
        scene_count = 0
        for scene in scenes[0::params.main.slice_factor]:
            scene_count += 1
            num_batches = int(scene['nbr_samples']) - self.batch_size_graph

            for batch_no in range(0, num_batches):
                self.batches.append(graph_files_dir +
                                    str(scene['token']) + "_len" +
                                    str(params.gnn.batch_size_graph) + "_" +
                                    str(batch_no))
          

        # Run scripts/statistics.py with specified dataset to attain relative frequencies.
        self.rel_freq_train = {'bicycle': 0.07455396870915335,
                                'bus': 0.013947840246335299,
                                'car': 0.44736907722651076,
                                'motorcycle': 0.055813302136334404,
                                'pedestrian': 0.1980141158741746,
                                'trailer': 0.06407160593555014,
                                'truck': 0.14623008987194142,
                                }
        # val only
        self.rel_freq_val = {'bicycle': 0.01673684284519299, 
                             'bus': 0.022198634903452107, 
                             'car': 0.5623747899986644, 
                             'motorcycle': 0.017629568188048728, 
                             'pedestrian': 0.24143651457532284, 
                             'trailer': 0.02923499764517331, 
                             'truck': 0.11038865184414562}


        # train and val
        self.rel_freq_trainval = {'bicycle': 0.01366729208060003, 
                         'bus': 0.018809669790663047, 
                         'car': 0.5685450597677517,
                         'motorcycle': 0.014540873950664522,
                         'pedestrian': 0.25376977084034424, 
                         'trailer': 0.028650719379687724, 
                         'truck': 0.10201661419028872}
        
        
        self.abs_freq_train = {'truck': 31410495, 
                                'bus': 2996022, 
                                'car': 96095709, 
                                'trailer': 13762700, 
                                'bicycle': 16014331, 
                                'pedestrian': 42533800, 
                                'motorcycle': 11988801}

        self.abs_freq_mini_train = {'truck': 270799, 
                                    'bus': 50896, 
                                    'car': 928625, 
                                    'trailer': 112093, 
                                    'bicycle': 195073, 
                                    'pedestrian': 777648, 
                                    'motorcycle': 127637}
        
    def get_metadata(self):
        return self.batches, self.scenes
        
    def get_instance_metadata(node_meta):
        meta = {k: node_meta[k] for k in ('sample_token',
                                        'translation',
                                        'size',
                                        'rotation',
                                        'velocity',
                                        'num_lidar_pts',
                                        'category_name',
                                        'score',
                                        'token',
                                        'time')}

        if 'incoming' and 'outgoing' not in meta.keys():
            meta.update({'incoming': dict(), 'outgoing': dict()})

        return meta


    def cb_scaling_factor(self, edge_class):
        """
        num_edges = N
        edge_factor = (1-beta)/(1-beta^(n_i))
        """
        
        num_edges = 5 #(best) trainval: 5, mini: 10000
        beta = (num_edges-1)/num_edges # mini: 0.5
        
        edges_per_cls = {cls: num_edges * cls_freq for cls, cls_freq in self.rel_freq_train.items()}
        edge_factor = (1-beta)/(1-beta**edges_per_cls[edge_class])

        return edge_factor

    def cb_scaling_factor_rev(self, edge_class):
        
        beta = 0.999
        edge_factor = (1-beta)/(1-beta**self.abs_freq_mini_train[edge_class])

        return edge_factor



    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        """
        Source a batch by index
        Args:
            idx: some index used by dataloader to identify a batch

        Returns:
            data: torch_geometric.data.Data (one batch of multiple frames, not a batch of batches in the torch sense)
        """

        pose_features = torch.load(self.batches[idx] + '_pose_features.pth')
        img_features = torch.load(self.batches[idx] + '_img_features.pth')
        lidar_features = torch.load(self.batches[idx] + '_lidar_features.pth')
        radar_features = torch.load(self.batches[idx] + '_radar_features.pth')
        node_timestamps = torch.load(self.batches[idx] + '_node_timestamps.pth')
        edge_features = torch.load(self.batches[idx] + '_edge_features.pth')
        edges = torch.load(self.batches[idx] + '_edges.pth')
        gt = torch.load(self.batches[idx] + '_gt.pth')

        if self.inference:
            boxes = torch.load(self.batches[idx] + '_node_boxes.pth')

        with open(self.batches[idx] + '_node_metadata.json', 'r') as file:
            node_metadata = json.load(file)


        if self.inference:
            global_edge_index = torch.zeros_like(edges)
            global_node_timestamps = torch.zeros((node_timestamps.shape[0],2))
            
            # Attain global edge indices
            for row_idx, edge in enumerate(edges):
                
                global_node_j = node_metadata[str(edge[0].item())]['global_node_id'] 
                global_node_i = node_metadata[str(edge[1].item())]['global_node_id']
                global_edge_index[row_idx] = torch.tensor([global_node_j, global_node_i])

            # Attain global node timestamps
            for node_idx, node_time in enumerate(node_timestamps):
                global_node_timestamps[node_idx] = torch.tensor([node_metadata[str(node_idx)]['global_node_id'], node_time])
                global_node_timestamps[node_idx] = torch.tensor([node_metadata[str(node_idx)]['global_node_id'], node_time])

        if self.edge_weighting:

            class_dict_used = batch_3dmot.utils.dataset.get_class_config(params=self.params, class_dict_name=self.params.main.class_dict)
            weights = torch.zeros(edges.shape[0])

            edge_classes = torch.zeros(edges.shape[0])
            node_classes = torch.zeros(pose_features.shape[0])
            # Loop through edges tensor
            for row_idx, edge in enumerate(edges):
                try:
                    class_a = node_metadata[str(edge[0].item())]['category_name']
                    class_b = node_metadata[str(edge[1].item())]['category_name']

                except TypeError:
                    try:
                        class_a = node_metadata[int(edge[0].item())]['category_name']
                        class_b = node_metadata[int(edge[1].item())]['category_name']
                    except TypeError:
                        print(self.batches[idx])
                
                if class_a == class_b:
                    weights[row_idx] = self.cb_scaling_factor(edge_class=class_a)

                    edge_classes[row_idx] = class_dict_used[class_a]

                    node_classes[edge[0]] = class_dict_used[class_a]
                    node_classes[edge[1]] = class_dict_used[class_a]
                else:
                    # Use weight minimum selection between different node categories
                    if self.rel_freq[class_a] < self.rel_freq[class_b]:
                        weights[row_idx] = self.cb_scaling_factor(edge_class=class_a)
                    elif self.rel_freq[class_b] < self.rel_freq[class_a]:
                        weights[row_idx] = self.cb_scaling_factor(edge_class=class_b)
        else:
            weights = torch.ones(edges.shape[0])

        data = torch_geometric.data.Data(pose_feats=pose_features,
                                         img_feats=img_features,
                                         lidar_feats=lidar_features,
                                         radar_feats=radar_features,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_features,
                                         y=gt.t().contiguous(),
                                         node_timestamps=node_timestamps,
                                         edge_weights=weights,
                                         edge_classes=edge_classes,
                                         node_classes=node_classes,
                                         num_nodes=pose_features.shape[0],
                                         batch_idx=idx,)

        if self.inference:
            data.global_edge_index=global_edge_index.t().contiguous()
            data.global_node_timestamps=global_node_timestamps
            data.boxes=boxes

            global_node_metadata = dict()
            for node_id in range(data.num_nodes):
                global_node_metadata[node_metadata[str(node_id)]['global_node_id']] = node_metadata[str(node_id)]
            #data.boxes = global_node_metadata

            return data, str(global_node_metadata)
        
        return data


class DatasetStatistics(torch_geometric.data.Dataset):
    """Characterizes a graph dataset to torch geometric."""

    def __init__(self, params: ParamLib, scenes: list, batch_size_graph: int):
        super(DatasetStatistics, self).__init__(params, scenes)
        """
        Initializes a graph dataset.
        Args:
              config: global dict defining paths
              scenes: list with multiple dicts each defining a scene that is loaded to this dataset.
              transform: Some transform to augment/change the graph data.
              pre_transform: Some preprocessing function applied to the sourced graph data.

        """

        self.batches = []
        self.list_of_scenes = []
        self.batch_size_graph = batch_size_graph
        self.edge_weighting = True
        
        graph_files_dir = params.paths.graphs_megvii_disj_len5_global_index
        print(graph_files_dir)
        
        for scene in scenes[0::params.main.slice_factor]:
            print(scene)
            num_batches = int(scene['nbr_samples']) - self.batch_size_graph
            scene_files = [f for f in
                           glob.glob(graph_files_dir + "**/" +
                                     str(scene['token']) + "_len" +
                                     str(params.gnn.batch_size_graph) + '*.pth', recursive=False)]

            self.list_of_scenes.append(scene_files)
        
            for batch_no in range(0, num_batches):
                self.batches.append(graph_files_dir +
                                    str(scene['token']) + "_len" +
                                    str(params.gnn.batch_size_graph) + "_" +
                                    str(batch_no))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        """
        Source a batch by index
        Args:
            idx: some index used by dataloader to identify a batch

        Returns:
            data: torch_geometric.data.Data (one batch of multiple frames, not a batch of batches in the torch sense)
        """

        edges = torch.load(self.batches[idx] + '_edges.pth')
        
        with open(self.batches[idx] + '_node_metadata.json', 'r') as file:
            node_metadata = json.load(file)
            
        edges_class_dict = defaultdict(int)
        # Loop through edges tensor
        for row_idx, edge in enumerate(edges):

            edge_class = node_metadata[int(edge[0].cpu())]['category_name']
            edges_class_dict[edge_class] += 1

        return edges_class_dict





