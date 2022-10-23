from email.policy import default
import pickle
import argparse
from pathlib import Path
import os
import json

import torch
import numpy as np
from tqdm import tqdm
import PIL
from collections import defaultdict

import torchvision

import ray
from ray.util.multiprocessing import Pool

from pyquaternion import Quaternion

import nuscenes
import nuscenes.utils.data_classes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.tracking.utils import category_to_tracking_name

from batch_3dmot.utils.config import ParamLib
import batch_3dmot.utils.dataset
import batch_3dmot.utils.geo_utils as geo_utils
import batch_3dmot.utils.graph_utils
import batch_3dmot.utils.load_scenes
import batch_3dmot.utils.nuscenes
import batch_3dmot.utils.radar

import batch_3dmot.preprocessing.match_detections


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_all_chunks(params: ParamLib, nusc: nuscenes.nuscenes.NuScenes, meta_list: list, chunk_size: int):

    # Divide the set of scenes (per split) into chunks that are each processed by a different worker node.
    scene_chunks = list(chunks(meta_list, chunk_size))

    data = dict()
    data['results'] = dict()
    data['meta'] = dict()

    tqdm.write("------ LOADING GT/DETECTION DATA TO NUSCENES VERSION:")
    gt_anns, pred_boxes = batch_3dmot.preprocessing.match_detections.load_detections(params=params, nusc=nusc)
    # filter detections for chunks

    tqdm.write("------ CHUNKING AND CREATING OF MATCHED/UNMATCHED DETECTIONS")
    chunked_detections = list()
    for chunk in tqdm(scene_chunks):
        chunk_samples = dict()
        for scene_meta in chunk:
            scene = nusc.get('scene', scene_meta['token'])
            next_sample_token = scene['first_sample_token']

            sample_detections = batch_3dmot.preprocessing.match_detections.match_sample(nusc=nusc,
                                                                                         sample_token=next_sample_token,
                                                                                         pred_boxes=pred_boxes,
                                                                                         gt_anns=gt_anns,
                                                                                         classes=class_dict_used,
                                                                                         detector_type=params.main.detections)
            chunk_samples[next_sample_token] = sample_detections

            while nusc.get('sample', next_sample_token)['next'] is not "":
                next_sample_token = nusc.get('sample', next_sample_token)['next']
                sample_detections = batch_3dmot.preprocessing.match_detections.match_sample(nusc=nusc,
                                                                                            sample_token=next_sample_token,
                                                                                            pred_boxes=pred_boxes,
                                                                                            gt_anns=gt_anns,
                                                                                            classes=class_dict_used,
                                                                                            detector_type=params.main.detections)


                chunk_samples[next_sample_token] = sample_detections

        chunked_detections.append(chunk_samples)

    chunk_data = list()
    for chunk_meta, chunk_dets in zip(scene_chunks, chunked_detections):
        chunk_data.append((chunk_meta, chunk_dets))
        
    #process_chunk(chunk_data[0])

    pool = Pool()
    pool.map(process_chunk, [data for data in chunk_data])


def process_chunk(data):

    chunk_scene_meta_list, detections = data

    # Iterate over the scenes in the list of scenes to be used
    batch_len = params.graph_construction.batch_size_graph

    for scene in tqdm(chunk_scene_meta_list):
        tqdm.write('Processing "{}"'.format(nusc.get('scene', scene['token'])))

        next_batch_token = scene['first_sample_token']

        # Iterate over each of the batches
        for i in tqdm(range(scene['nbr_samples'] - batch_len + 1)):
            
            edges = []  # (2, num_edges) with pairs of connected node ids
            edge_features = []  # (num_edges, num_feat_edges) edge_id with features
            gt_edges = []  # (num_edges) with 0/1 depending on edge is gt

            past_nodes = []
            node_id = 0

            pose_features = torch.empty([0,19])

            batch_init_sample = nusc.get('sample', next_batch_token)

            # define next_sample_token for next batch
            next_sample_token = next_batch_token

            graph_filename = os.path.join(params.paths.graphs_disj_pose_centerpoint_1440_flip_40nn_iou_14feb_len7, scene['token']+'_len'+str(batch_len)+'_'+str(i))

            # Iterative over all samples within one batch
            for idx, val in enumerate(range(i, i + batch_len)):

                sample_detections = detections[next_sample_token]
                tqdm.write('Batch #{} - Frame #{} - Dets #{}'.format(i, val, len(sample_detections)))
                cur_nodes = []

                for det_box in sample_detections:

                    # Transform box to global coord. frame
                    sample = nusc.get('sample', next_sample_token)
                    sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                    # This returns a box in ego-vehicle coord. frame (IMU).
                    ego_box = det_box.copy()

                    # Compute the distance between detection box and ego pos while neglecting the LIDAR-z-axis (upwards)
                    ego_box_radius = np.linalg.norm(ego_box.center[0:2])

                    if ego_box.name in class_dict_used.keys() \
                            and params.pointnet.ego_rad_min < ego_box_radius < params.pointnet.ego_rad_max:

                        # From ego to global coordinates for box metadata 
                        global_box = det_box.copy()
                        global_box.rotate(Quaternion(pose_record['rotation']))
                        global_box.translate(np.array(pose_record['translation']))

                        ego_box_yaw = torch.from_numpy(
                            np.array([batch_3dmot.utils.geo_utils.quaternion_yaw(ego_box.orientation)]))
                        
                        feat_3d_pose = torch.cat([torch.from_numpy(ego_box.center).float(),
                                                  torch.from_numpy(ego_box.wlh).float(),
                                                  ego_box_yaw.float(),
                                                  torch.from_numpy(ego_box.velocity).float()],
                                                 dim=0)
                        feat_3d_pose = feat_3d_pose.reshape(-1, 1)
                        score_feat = torch.tensor(ego_box.score).reshape(-1,1)
                        timestamp = val
                        class_label = torch.tensor(int(class_dict_used[ego_box.name]))
                        class_one_hot = torch.nn.functional.one_hot(class_label - 1,
                                                                    num_classes=len(class_dict_used.keys()))
                                
                        class_one_hot = class_one_hot.reshape(-1, 1).float()
                        rel_time_tensor = torch.tensor(int(val-i)).reshape(-1,1).float()

                        box_metadata = {'token': global_box.token,
                                          'sample_token': next_sample_token,
                                          'translation': list(global_box.center),
                                          'size': list(global_box.wlh),
                                          'rotation': list(global_box.orientation),
                                          'category_name': str(global_box.name),
                                          'score': float(global_box.score),
                                          }

                        node_feature = torch.cat([feat_3d_pose,
                                                  class_one_hot,
                                                  score_feat,
                                                  rel_time_tensor], dim=0).reshape(1,-1)

                        pose_features = torch.cat([pose_features, node_feature], dim=0)

                        cur_nodes.append({'box': global_box,
                                          'sample_token': next_sample_token,
                                          'metadata': box_metadata,
                                          'node_id': node_id,
                                          'time': timestamp,
                                          'feat_3d_pose': feat_3d_pose,
                                          'full_feat': node_feature,
                                          'category': global_box.name,
                                          })
                            
                        node_id += 1
                #tqdm.write("FRAMES @ PAST: {}".format(len(past_nodes)))
                #tqdm.write("CURRENT FRAME: {}".format(len(cur_nodes)))

                if len(past_nodes) > 0:

                    for cur in tqdm(cur_nodes):
                        
                        past_categ_nodes = list()
                        for past_node in past_nodes:
                            if past_node['category'] == cur['category']:
                                past_categ_nodes.append(past_node)

                        if len(past_categ_nodes) > vars(params.graph_construction)['top_knn_classes'][cur['category']]:
                            k = vars(params.graph_construction)['top_knn_classes'][cur['category']]
                        else:
                            k = len(past_categ_nodes)
                            
                        if len(past_categ_nodes) > 0:
                            
                            knn_past_nodes = get_knn_nodes_in_graph(cur, past_categ_nodes, k)

                            for ex in knn_past_nodes:
                                ex_id, cur_id = ex['node_id'], cur['node_id']
                                edges.append([ex_id, cur_id])


                                if ex['metadata']['token'] is not None and cur['metadata']['token'] is not None:
                                    if ex['metadata']['token'] == cur['metadata']['token']:

                                        cur_ex_time_diff = abs(cur['time'] - ex['time'])
                                        # Simple case, we just know this is the closest appearance of the same instance
                                        if cur_ex_time_diff == 1:
                                            gt_edges.append(1)

                                        # This is the complex case. We do not know if the ex-cur-time difference either
                                        # represents the closest appearance or a farther one.
                                        elif cur_ex_time_diff > 1:
                                            # Search for other nodes with equal tokens at other times.
                                            # Compute a ranking position of the cur-ex-edge

                                            oth_deltas = list()
                                            for oth_node in knn_past_nodes:
                                                if oth_node['time'] != ex['time'] and oth_node['metadata']['token'] == cur['metadata']['token']:
                                                    time_diff_oth = abs(cur['time'] - oth_node['time'])
                                                    oth_deltas.append(time_diff_oth)

                                            if len(oth_deltas) == 0:
                                                gt_edges.append(1)
                                            else:
                                                if np.min(oth_deltas) > cur_ex_time_diff:
                                                    gt_edges.append(1)

                                                elif np.min(oth_deltas) < cur_ex_time_diff:
                                                    gt_edges.append(0)
                                        else:
                                            gt_edges.append(0)
                                    else:
                                        gt_edges.append(0)
                                else:
                                    gt_edges.append(0)

                                # Compute edge features based on relative differences
                                box_feats = batch_3dmot.utils.graph_utils.compute_motion_edge_feats(ex, cur)

                                # Compute absolute time delta (UNIX timestamp, using microseconds per second)
                                time_delta = abs(cur['time'] - ex['time'])
                                box_feats.append(time_delta)

                                # Add edge feature for that particular ex-cur-node-combination
                                edge_features.append(box_feats)

                past_nodes.extend(cur_nodes)
                next_sample_token = sample['next']

            all_nodes = sorted(past_nodes, key=lambda n: n['node_id'])
            
            edges = torch.tensor(edges)
            gt_edges = torch.tensor(gt_edges)
            edge_features = torch.tensor(edge_features)
            pose_features = pose_features.reshape(len(all_nodes), -1)
            node_timestamps = torch.tensor([node['time'] for node in all_nodes])
            boxes = torch.cat([node['feat_3d_pose'] for node in all_nodes], dim=1).reshape(len(all_nodes), -1)

            # Store graph data for easy read-in via pyg dataset class.
            torch.save(edges, graph_filename + '_edges.pth')
            torch.save(gt_edges, graph_filename + '_gt.pth')
            torch.save(node_timestamps, graph_filename + '_node_timestamps.pth')
            torch.save(edge_features, graph_filename + '_edge_features.pth')
            torch.save(pose_features, graph_filename + '_pose_features.pth')
            torch.save(boxes, graph_filename + '_node_boxes.pth')

            # Node metadata JSON dump
            all_nodes_metadata = [{'sample_token': node['sample_token'],
                                    'translation': list(node['box'].center),
                                    'size': list(node['box'].wlh),
                                    'rotation': list(node['box'].orientation),
                                    'velocity': list(node['box'].velocity),
                                    'category_name': node['category'],
                                    'node_id': node['node_id'],
                                    'time': node['time'],
                                    'score': node['box'].score,
                                    'token': node['box'].token} for node in all_nodes]

            # Per batch: Store a node dict
            with open(graph_filename + '_node_metadata.json', 'w') as outfile:
                json.dump(all_nodes_metadata, outfile)

            next_batch_token = batch_init_sample['next']


if __name__ == '__main__':

    # ----------- PARAMETER SOURCING --------------

    parser = argparse.ArgumentParser()

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Specific arguments (namespace: graph_construction)
    parser.add_argument('--batch_size_graph', type=int, help='define number of frames contained in one graph')
    parser.add_argument('--resnet_checkpoint', type=str, help='model path')
    parser.add_argument('--pointnet_checkpoint', type=str, help='model path')
    parser.add_argument('--radarnet_checkpoint', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.graph_construction.overwrite(opt)

    # Load lists with all scene meta data based on the dataset, the version,
    nusc, meta_lists = batch_3dmot.utils.load_scenes.load_scene_meta_list(data_path=params.paths.data,
                                                                          dataset=params.main.dataset,
                                                                          version=params.main.version)

    class_dict_used = batch_3dmot.utils.dataset.get_class_config(params, class_dict_name=params.main.class_dict)

    ray.init(num_cpus=15,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # Construct graphs using the generated scene metadata & store them to nuscenes/graphs dir
    for meta_list in meta_lists:
        
        if len(meta_list) == 700:
            process_all_chunks(params, nusc, meta_list[350:], 25)
