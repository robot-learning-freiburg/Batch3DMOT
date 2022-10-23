import argparse
from statistics import mean
from collections import defaultdict
import os
import json
import numpy as np
import ray
from ray.util.multiprocessing import Pool

import torch
import torch_geometric
from tqdm import tqdm
import torch_geometric.data
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

import nuscenes
import nuscenes.utils.data_classes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.utils import category_to_tracking_name
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.render import TrackingRenderer

import networkx as nx
import matplotlib.pyplot as plt

import batch_3dmot.models.gnn_transfer_cl_med
import batch_3dmot.models.gnn_transfer_cl
import batch_3dmot.models.cl_gnn_trad


import batch_3dmot.models.gnn_baseline
import batch_3dmot.models.clr_att_gnn
import batch_3dmot.models.cl_att_gnn

import batch_3dmot.models.pointnet
import batch_3dmot.models.resnet_fully_conv
import batch_3dmot.models.radarnet
import batch_3dmot.utils.load_scenes
import batch_3dmot.utils.dataset
import batch_3dmot.utils.interpolation

from batch_3dmot.utils.config import ParamLib


# ----------- PARAMETER SOURCING --------------

parser = argparse.ArgumentParser()

# General args (namespace: main)
parser.add_argument('--dataset', type=str, help="dataset path")
parser.add_argument('--version', type=str, help="provide dataset version/split")
parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

# Specific args (namespace: predict)
parser.add_argument('--batch_size_graph', type=int, help='define number of frames contained in one graph')

opt = parser.parse_args()

params = ParamLib(opt.config)
params.main.overwrite(opt)
params.pointnet.overwrite(opt)

# ---------------------------------------------


def load_batch_detections(params: ParamLib, scene: dict, batch_no: int, batch_size_graph: int):

    batch_path = params.paths.graphs_pose_megvii_disj_len5 + str(scene['token']) + "_len" + str(batch_size_graph) + "_" + str(batch_no)

    pose_features = torch.load(batch_path + '_pose_features.pth')
    img_features = torch.load(batch_path + '_img_features.pth')
    lidar_features = torch.load(batch_path + '_lidar_features.pth')
    radar_features = torch.load(batch_path + '_radar_features.pth')
    node_timestamps = torch.load(batch_path + '_node_timestamps.pth')
    edge_features = torch.load(batch_path + '_edge_features.pth')
    edges = torch.load(batch_path + '_edges.pth')
    boxes = torch.load(batch_path + '_node_boxes.pth')

    with open(batch_path + '_node_metadata.json', 'r') as json_file:
        node_metadata = json.load(json_file)

    return pose_features, img_features, lidar_features, radar_features, node_timestamps, edge_features, edges, boxes, node_metadata


def greedy_filter_node_flux(meta):
    # Check all incoming nodes
    if len(meta['incoming']) > 1:
        # Select predecessor with the highest edge score
        pred_idx = max(meta['incoming'], key=meta['incoming'].get)
        pred_edge_score = meta['incoming'][pred_idx]
        predecessor = {pred_idx: pred_edge_score}

    elif len(meta['incoming']) == 1:
        predecessor = meta['incoming']
    else:
        predecessor = {}

    # Check all outgoing nodes
    if len(meta['outgoing']) > 1:
        # Select node successor with the highest edge score
        succ_idx = max(meta['outgoing'], key=meta['outgoing'].get)
        succ_edge_score = meta['outgoing'][succ_idx]
        successor = {succ_idx: succ_edge_score}

    elif len(meta['outgoing']) == 1:
        successor = meta['outgoing']
    else:
        successor = {}

    return predecessor, successor


def aggregate_node_flux(nodes_meta, edge_scores):
    for (out_idx, in_idx), score in edge_scores.items():
        nodes_meta[in_idx]['incoming'].update({out_idx: float(score)})
        nodes_meta[out_idx]['outgoing'].update({in_idx: float(score)})
    return nodes_meta

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

def combine_batches_to_scene(params: ParamLib,
                             processed_scene: dict,
                             gnn: batch_3dmot.models.cl_att_gnn.GNN,
                             batch_size_graph: int,
                             mode: str = 'detections',
                             device: str = 'cuda'):
    """
    Takes overlapping batches and aggregates them to one big consistent graph.
    """

    cls2idx = batch_3dmot.utils.dataset.get_class_config(params=params,
                                                         class_dict_name=params.main.class_dict)
    idx2cls = {v: k for k, v in cls2idx.items()}

    scene_metadata = nusc.get('scene', processed_scene['token'])
    tqdm.write(scene_metadata['name'])
    last_node_timestep = scene_metadata['nbr_samples'] - 1

    save_root = os.path.join(params.paths.eval, 'predict/', 'detections/')

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Iterate over each of the batches
    scene_edges = defaultdict(list)
    meta2gid = defaultdict()

    global_idx = 0

    for batch_no in tqdm(range(0, scene_metadata['nbr_samples'] - batch_size_graph + 1)):
        pose_feats, img_feats, lidar_feats, radar_feats, node_timestamps, edge_features, edges, boxes, node_metadata = load_batch_detections(params=params,
                                                                                                            scene=processed_scene,
                                                                                                            batch_no=batch_no,
                                                                                                            batch_size_graph=batch_size_graph)

        # Case where no detections in frame
        if len(pose_feats) == 0 or len(edge_features) == 0:
            continue

        data = torch_geometric.data.Data(pose_feats=pose_feats,
                                         img_feats=img_feats,
                                         lidar_feats=lidar_feats,
                                         radar_feats=radar_feats,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_features,
                                         node_timestamps=node_timestamps,
                                         boxes=boxes,
                                         num_nodes=pose_feats.shape[0],).to(device)

        with torch.no_grad():
            # Shape (E,) array with classification score for every edge
            pred_scores, x_sens = gnn.forward(data)
            
            pred_scores = pred_scores.squeeze(1).cpu().numpy()
            print(pred_scores)

        # Loop through all nodes and aggregate edges
        for node_idx in range(len(pose_feats)):
            meta = get_instance_metadata(node_metadata[node_idx])

            meta.update({'incoming': dict(), 'outgoing': dict()})

            # Create reverse index to reference node indices via node meta hash globally.
            if str(meta) not in meta2gid.keys():
                meta2gid[str(meta)] = global_idx
                global_idx += 1

        # Loop through the (2,E)-edges-tensor edge-by-edge (row-by-row)
        for edge_idx, (out_idx, in_idx) in tqdm(enumerate(edges)):

            time_out = node_timestamps[out_idx].item()
            meta_out = get_instance_metadata(node_metadata[out_idx])
            scene_idx_out = meta2gid[str(meta_out)]

            time_in = node_timestamps[in_idx].item()
            meta_in = get_instance_metadata(node_metadata[in_idx])
            scene_idx_in = meta2gid[str(meta_in)]

            scene_edges[(scene_idx_out, scene_idx_in)].append(pred_scores[edge_idx].item())

    # Reverse index all nodes
    scene_nodes = {index: eval(meta) for meta, index in meta2gid.items()}

    # Average the prediction scores over all batches
    avg_edge_scores = {edge: np.mean(scores) for edge, scores in scene_edges.items()}
    
    print(avg_edge_scores)

    edge_scores_thresholds = {'bicycle': 0.1, 'bus': 0.005, 'car': 0.02, 'motorcycle': 0.03, 'pedestrian': 0.025, 'trailer': 0.04, 'truck': 0.005}

    avg_edge_scores = {edge: avg_score for edge, avg_score in avg_edge_scores.items() if avg_score > edge_scores_thresholds[scene_nodes[edge[0]]['category_name']]}

    # Add info on connecting edges to each node
    scene_nodes_w_flux = aggregate_node_flux(scene_nodes, avg_edge_scores)

    global_edges_log = {str(key): val for key, val in avg_edge_scores.items()}
    with open(os.path.join(save_root, processed_scene['token'] + '_edge_scores.json'), 'w') as outdict:
        json.dump(global_edges_log, outdict)

    # GREEDY ROUNDING:
    for node_idx, node_meta in scene_nodes_w_flux.items():
        scene_nodes_w_flux[node_idx]['incoming'], \
        scene_nodes_w_flux[node_idx]['outgoing'] = greedy_filter_node_flux(node_meta)

    # Create dict with only the greedy-filtered edges
    greedy_edges = dict()
    for node_idx in scene_nodes_w_flux:

        if len(scene_nodes_w_flux[node_idx]['outgoing']) > 0:
            greedy_edges[(node_idx, list(scene_nodes_w_flux[node_idx]['outgoing'].keys())[0])] = list(scene_nodes_w_flux[node_idx]['outgoing'].values())[0]

        if len(scene_nodes_w_flux[node_idx]['incoming']) > 0:
            greedy_edges[(list(scene_nodes_w_flux[node_idx]['incoming'].keys())[0], node_idx)] = list(scene_nodes_w_flux[node_idx]['incoming'].values())[0]

    pred_edges = [(edge,score) for edge, score in greedy_edges.items()]

    return scene_nodes_w_flux, pred_edges


def create_trajectories(pred_edges, scene_nodes):

    mode = "hier"

    if mode == "recursive_dfs":
        fut_nodes = defaultdict(list)
        for (i, j), score in pred_edges:
            fut_nodes[i].append(j)
            fut_nodes[j] = list()

        def recursive_dfs(neighbors, rem_nodes, track):
            for node in neighbors:
                print("NODE", node, "with neighbors:", fut_nodes[node])
                print(node in rem_nodes)
                if node in rem_nodes:
                    track.append(node)
                    rem_nodes.remove(node)
                    recursive_dfs(fut_nodes[node], rem_nodes, track)
            return track

        all_tracks = []
        remaining_nodes = list(fut_nodes.keys())
        while remaining_nodes:
            new_track = recursive_dfs([remaining_nodes[0]], remaining_nodes, [])
            all_tracks.append(new_track)
        # print(all_tracks)
        print(all_tracks)
    
    elif mode == 'hier':
        pred_edges_dict = {edge_plus_score[0]: edge_plus_score[1] for edge_plus_score in pred_edges}
        pred_edges_vals_desc = sorted(pred_edges_dict.items(), key=lambda item: item[1], reverse=True)
        pred_edges_desc = {k: v for k, v in pred_edges_vals_desc}

        all_tracks = defaultdict(list)
        vis = defaultdict()

        clusters = defaultdict(list)
        clusters_scores = defaultdict(list)

        join_score = {'bicycle': 0.1, 'bus': 0.005, 'car': 0.02, 'motorcycle': 0.03, 'pedestrian': 0.025, 'trailer': 0.04, 'truck': 0.005}

        for edge, score in pred_edges_desc.items():
            j,i = edge

            edge_cat = scene_nodes[i]['category_name'] 

            print("------")
            print("CAND EDGE:", edge, score)
            # Unconstrained edge
            if j not in vis.keys() and i not in vis.keys():
                print("UNCONSTRAINED")
                if len(list(clusters.keys())) == 0:
                    cluster_idx = 0
                else:
                    cluster_idx = max(list(clusters.keys())) + 1
                clusters[cluster_idx].extend([j,i])
                clusters_scores[cluster_idx].append(score)
                vis[i] = cluster_idx
                vis[j] = cluster_idx
                print(cluster_idx)

                print(clusters[cluster_idx])

            # Constrained edge
            else:
                print("CONSTRAINED")
                # Preceding edge
                if j not in vis.keys() and i in vis.keys():
                    print("Potentially preceding")
                    cluster2extend = vis[i]
                    if clusters[cluster2extend][0] == i:
                        clusters[cluster2extend].insert(0,j)
                        clusters_scores[cluster2extend].insert(0,score)
                        vis[j] = cluster2extend
                    else:
                        continue

                # Succeeding edge
                elif j in vis.keys() and i not in vis.keys():
                    print("Potentially succeeding")
                    cluster2extend = vis[j]
                    print(clusters[cluster2extend])
                    print(cluster2extend)
                    if clusters[cluster2extend][-1] == j:
                        clusters[cluster2extend].append(i)
                        clusters_scores[cluster2extend].append(score)
                        vis[i] = cluster2extend
                    else:
                        continue

                # Edge joins two clusters
                elif j in vis.keys() and i in vis.keys():
                    cluster0 = vis[j]
                    cluster1 = vis[i]

                    cluster_score = np.mean(clusters_scores[cluster0] + clusters_scores[cluster1])
                    print("Potentially joining")
                    # Condition for joining two clusters
                    if j == clusters[cluster0][-1] and i == clusters[cluster1][0] and score > join_score[edge_cat]:
                        clusters[cluster0] = clusters[cluster0] + clusters[cluster1]
                        clusters_scores[cluster0] = clusters_scores[cluster0] + clusters_scores[cluster1]

                        for node in clusters[cluster0]:
                            vis[node] = cluster0

                        del clusters[cluster1]
                        del clusters_scores[cluster1]

                    else:
                        continue

        all_tracks = [v for k,v in clusters.items()]
        print(all_tracks)
    return all_tracks


def get_track_dict(params: ParamLib,
                   processed_scene: dict,
                   batch_size_graph: int,
                   device: str = 'cpu',
                   interpolate_tracks: bool = True) -> dict:
    
    class_dict_used = batch_3dmot.utils.dataset.get_class_config(params=params, class_dict_name=params.main.class_dict)

    resnet_ae = batch_3dmot.models.resnet_fully_conv.ResNetAE()
    resnet_ae.load_state_dict(torch.load(os.path.join(params.paths.models,params.resnet.checkpoint), map_location=torch.device('cpu')))

    pointnet_classifier = batch_3dmot.models.pointnet.PointNetClassifier(k=len(class_dict_used), feature_transform=params.pointnet.feature_transform).to()
    pointnet_classifier.load_state_dict(torch.load(os.path.join(params.paths.models, params.pointnet.checkpoint), map_location=torch.device('cpu')))

    radarnet_classifier = batch_3dmot.models.radarnet.RadarNetClassifier(k=len(class_dict_used), feature_transform=params.radarnet.feature_transform)
    radarnet_classifier.load_state_dict(torch.load(os.path.join(params.paths.models, params.radarnet.checkpoint), map_location=torch.device('cpu')))

    edge_classifier = batch_3dmot.models.cl_att_gnn.GNN(gnn_depth=params.gnn.gnn_depth,
                                                                node_dim=params.gnn.node_dim,
                                                                edge_dim=params.gnn.edge_dim,
                                                                img_encoder=resnet_ae,
                                                                lidar_encoder=pointnet_classifier,
                                                                radar_encoder=radarnet_classifier,
                                                                use_attention=params.gnn.attention,
                                                                ).to(device).eval()
                                          
    edge_classifier.load_state_dict(torch.load(os.path.join(params.paths.models, params.predict.checkpoint),
                                               map_location=torch.device('cpu')))

    nodes, pred_edges = combine_batches_to_scene(params=params,
                                                 processed_scene=processed_scene,
                                                 gnn=edge_classifier,
                                                 mode='detections',
                                                 batch_size_graph=batch_size_graph,
                                                 device=device)

    tracks = create_trajectories(pred_edges, nodes)

    track_scores = dict()
    for (i, j), score in pred_edges:
        track_scores[(i,j)] = score

    for track in tracks:
        print(track)
        gt_track_coverage = list()
        print_scores = list()
        for node_idx in track:
            if 'token' in nodes[node_idx].keys():
                gt_track_coverage.append(nodes[node_idx]['token'])
            else:
                gt_track_coverage.append("noise")
        for idx in range(len(track)-1):
            try:
                print_scores.append(track_scores[(track[idx], track[idx+1])])
            except:
                print("")
        print(gt_track_coverage)
        print(print_scores)

    track_dict = defaultdict(dict)
    for track_id, track in enumerate(tracks):
        #print(track)
        for node_id in track:
            print(nodes[node_id])

            detection_time = nodes[node_id]['time']
            detection_box = {field: nodes[node_id][field] for field in list(nodes[node_id].keys()) if field not in ['time']}

            track_dict[track_id][detection_time] = detection_box

    return track_dict

class Batch3DMOTSceneEval:

    def __init__(self, params: ParamLib, processed_scene: dict, batch_size_graph: int):

        self.batch_size_graph = batch_size_graph

        self.scene_token = processed_scene['token']
        self.all_sample_tokens = self.create_sample_token_dict(self.scene_token)

        self.all_tracks = get_track_dict(params=params,
                                         processed_scene=processed_scene,
                                         batch_size_graph=self.batch_size_graph)

        self.omitted_tracks = list()

        self.eval_boxes = EvalBoxes()

    def create_eval_boxes(self):
        for token, list_of_boxes in enumerate(self.all_sample_tokens):
            self.eval_boxes.add_boxes(token, list_of_boxes)

    # This method works correctly
    def create_sample_token_dict(self, scene_token):
        """
        TODO: Needs a docstring
        Args:
            scene_token:

        Returns:

        """
        scene_meta = nusc.get('scene', scene_token)

        sample_token = scene_meta['first_sample_token']
        sample_meta = nusc.get('sample', sample_token)

        all_sample_tokens = dict()

        for _ in range(scene_meta['nbr_samples']):
            sample_meta = nusc.get('sample', sample_token)
            all_sample_tokens[sample_token] = []
            sample_token = sample_meta['next']

        all_sample_tokens[scene_meta['last_sample_token']] = list()

        return all_sample_tokens

    def traverse_generated_tracks(self):
        """
        TODO: Needs a docstring

        """
        for track_id in range(len(self.all_tracks)):
            self.traverse_single_track(track_id=track_id,
                                       single_track_dict=self.all_tracks[track_id],
                                       interpolate_tracks=True)

        # Return scene results structured as a dict over sample tokens (keys)
        # Each sample token holds a list of all tracking boxes contained
        # in that frame (value in dict).
        return self.all_sample_tokens

    def traverse_single_track(self, track_id, single_track_dict: dict, interpolate_tracks: bool):
        """
        TODO: Needs a docstring
        Args:
            track_id:
            single_track_dict:

        Returns:

        """

        # Perform interpolation if stated
        if interpolate_tracks and list(single_track_dict.values())[0] == "trailer":
            interpolated_boxes = batch_3dmot.utils.interpolation.interpolate_linear(track_id=track_id,
                                                                                  track_dict=single_track_dict,
                                                                                  nusc=nusc)
        
            for sample_token, tracking_box in interpolated_boxes.items():
                self.all_sample_tokens[sample_token].append(tracking_box)

        for detection_time, detection_params in single_track_dict.items():

            category_name = detection_params['category_name']

            # per detection on track create a TrackingBox
            tracking_box = TrackingBox(sample_token=detection_params['sample_token'],
                                       translation=detection_params['translation'],
                                       size=detection_params['size'],
                                       rotation=detection_params['rotation'],
                                       tracking_id=str(track_id),  # instance id of this object
                                       tracking_name=category_name,
                                       tracking_score=detection_params['score'])  # class name used in tracking challenge

            # print(tracking_box.translation)
            self.all_sample_tokens[detection_params['sample_token']].append(tracking_box)


def convert_to_submission_dict(results_across_scenes):
    results = dict()

    for sample_token, box_list in results_across_scenes.items():
        results[sample_token] = []
        for box in box_list:
            results[sample_token].append({"sample_token": sample_token,
                                          "translation":  box.translation,
                                          "size": box.size,
                                          "rotation": box.rotation,
                                          "velocity": box.velocity,
                                          "tracking_id":  box.tracking_id,
                                          "tracking_name":  box.tracking_name,
                                          "tracking_score": box.tracking_score
                                          })

    submission = dict()
    submission['meta'] = {"use_camera": True,
                          "use_lidar": True,
                          "use_radar": False,
                          "use_map": False,
                          "use_external": False}
    submission['results'] = results

    return submission
    # Evidence: The submission bzw. result dict contains an empty list for each last_sample per scene.

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_one_scene(data):

    params,scene = data

    eval_inst = Batch3DMOTSceneEval(params=params,
                                    processed_scene=scene,
                                    batch_size_graph=params.predict.batch_size_graph)

    # Just store the results per scene. Since the results
    # are disjunct we can just extend the overall results dict
    results_scene = eval_inst.traverse_generated_tracks()

    return results_scene

def process_chunk(data):
    
    meta_list_chunk, params = data
    
    chunk_result = list()
    
    for scene in meta_list_chunk:
        eval_inst = Batch3DMOTSceneEval(params=params,
                                        processed_scene=scene,
                                        batch_size_graph=params.predict.batch_size_graph)

        # Just store the results per scene. Since the results
        # are disjoint we can just extend the overall results dict
        results_scene = eval_inst.traverse_generated_tracks()
        chunk_result.append(results_scene)
        
    return chunk_result

if __name__ == '__main__':

    # Load data splits and a nuscenes instance
    nusc, meta_lists = batch_3dmot.utils.load_scenes.load_scene_meta_list(data_path=params.paths.data,
                                                                          dataset=params.main.dataset,
                                                                          version=params.main.version)
    # Load metadata of data splits
    if params.main.version == "v1.0-mini":
        train_scene_meta_list, val_scene_meta_list = meta_lists
    elif params.main.version == "v1.0-trainval":
        train_scene_meta_list, val_scene_meta_list = meta_lists
    elif params.main.version == "v1.0-test":
        test_scene_meta_list = meta_lists
        train_scene_meta_list, val_scene_meta_list = [], []
    else:
        train_scene_meta_list, val_scene_meta_list, test_scene_meta_list = [], [], []

    results_over_all_scenes = dict()
    # Loop through the different data splits provided above
    for meta_list in meta_lists:

        # Loop through all scenes within one data split
        if len(meta_list) < 200:
            meta_list_chunks = list(chunks(meta_list, 19))
            ray.init(num_cpus=8,
                     include_dashboard=False,
                     _system_config={"automatic_object_spilling_enabled": True,
                                     "object_spilling_config": json.dumps(
                                         {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

            pool = Pool()
            chunk_results = pool.map(process_chunk, [(chunk,params) for chunk in meta_list_chunks])
            
            for chunk_result in chunk_results:
                for scene_result in chunk_result:
                    results_over_all_scenes.update(scene_result)

            ray.shutdown()


    submission_dict = convert_to_submission_dict(results_over_all_scenes)
    
    filename = '***_submission.json'
    results_path = os.path.join(params.paths.eval, filename)
    print(results_path)

    with open(results_path, 'w') as outfile:
        json.dump(submission_dict, outfile)

    # Specifying the evaluation configuration parameters.
    if params.eval.eval_config == 'tracking_nips_2019':
        eval_cfg = config_factory('tracking_nips_2019')
    else:
        with open(params.eval.eval_config, 'r') as _f:
            eval_cfg = TrackingConfig.deserialize(json.load(_f))
    print(vars(params.eval))
    # Running the evaluation using the specified evaluation config parameters.
    nusc_eval = TrackingEval(config=eval_cfg,
                             result_path=results_path,
                             eval_set=params.eval.eval_set,
                             output_dir=params.paths.eval,
                             nusc_version=params.main.version,
                             nusc_dataroot=params.paths.data,
                             verbose=bool(params.eval.verbose))

    nusc_eval.main(render_curves=bool(params.eval.render_curves))
        
    # CHECKLIST
    # 1. Set submission.json filename
    # 2. Check len(meta_list)
    # 3. Check chunk_size
    # 4. Check num_cpus
    # 5. Check graphfile_dir in load_batch_detections/load_batch_gt
