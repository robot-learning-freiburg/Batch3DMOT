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
import batch_3dmot.utils.geo_utils
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

            pose_features = torch.empty([0, params.graph_construction.feat_3d_pose_dim])


            img_features = torch.empty([0, 3, params.graph_construction.feat_2d_app_dim, params.graph_construction.feat_2d_app_dim])
            lidar_features = torch.empty([0, 3, params.graph_construction.feat_3d_app_dim])
            radar_features = torch.empty([0, 4, params.graph_construction.feat_3d_radar_dim])

            batch_init_sample = nusc.get('sample', next_batch_token)

            # define next_sample_token for next batch
            next_sample_token = next_batch_token

            graph_filename = os.path.join(params.paths.graphs_clr_centerpoint_1440_flip_40nn_iou_14feb_len2,
                                          scene['token']+'_len'+str(batch_len)+'_'+str(i))

            # Iterative over all samples within one batch
            for idx, val in enumerate(range(i, i + batch_len)):

                tqdm.write('Batch #{} - Frame #{}'.format(i, val))

                sample_detections = detections[next_sample_token]

                cur_nodes = []

                for det_box in sample_detections:

                    # Transform box to global coord. frame
                    sample = nusc.get('sample', next_sample_token)
                    sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                    # This returns a box in ego-vehicle coord. frame (IMU).
                    det_box = det_box.copy()

                    # Compute the distance between detection box and ego pos while neglecting the LIDAR-z-axis (upwards)
                    det_box_radius = np.linalg.norm(det_box.center[0:2])

                    if det_box.name in class_dict_used.keys() \
                            and params.pointnet.ego_rad_min < det_box_radius < params.pointnet.ego_rad_max:

                        # Every annotation has both image and LIDAR information
                        # If the number of LIDAR points is zero we just return zeros
                        if params.main.sensors_used["img"]:
                            with torch.no_grad():

                                # Get a detection box in ego-vehicle coord. frame.
                                cam_box = det_box.copy()
                                
                                # Attain a list of cameras available in the respective sample.
                                available_cameras = dict()
                                for sensor_name, sensor_token in sample['data'].items():
                                    if 'CAM' in sensor_name:
                                        available_cameras[sensor_name] = sensor_token

                                visibility = dict()
                                for cam, sample_cam_token in available_cameras.items():
                                    # Transform box to camera-specific coord system
                                    sd_record = nusc.get('sample_data', sample_cam_token)
                                    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                                    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                                    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                                    cam_imsize = (sd_record['width'], sd_record['height'])

                                    # Get box in ego coord frame
                                    box = cam_box.copy()

                                    #  Move box to calibrated sensor coord system.
                                    box.translate(-np.array(cs_record['translation']))
                                    box.rotate(Quaternion(cs_record['rotation']).inverse)

                                    cam_vis = batch_3dmot.utils.nuscenes.count_box_corners_in_image(box=box,
                                                                                                    intrinsic=cam_intrinsic,
                                                                                                    imsize=cam_imsize)

                                    visibility[cam] = cam_vis[0]

                                #print(visibility)

                                highest_vis_camera = max(visibility.keys(), key=(lambda key: visibility[key]))

                                cam_used_sample_token = available_cameras[highest_vis_camera]

                                sd_record = nusc.get('sample_data', cam_used_sample_token)
                                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

                                sample_data_path = os.path.join(params.paths.data, sd_record['filename'])

                                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                                cam_imsize = (sd_record['width'], sd_record['height'])

                                #  Move box from ego to calibrated sensor frame = the selected camera.
                                cam_box.translate(-np.array(cs_record['translation']))
                                cam_box.rotate(Quaternion(cs_record['rotation']).inverse)

                                corners_3d = cam_box.corners()

                                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                                corners_3d = corners_3d[:, in_front]

                                corners_img = batch_3dmot.utils.nuscenes.view_points(corners_3d,
                                                                                        cam_intrinsic,
                                                                                        normalize=True).T[:, :2].tolist()

                                # Keep only corners that fall within the image.
                                #print(corners_img)
                                final_coords = batch_3dmot.utils.nuscenes.post_process_coords(corners_img)

                                # Some noisy detections are not observed in any camera (e.g. located in the sky)
                                if final_coords is None:

                                    tqdm.write("Detection omitted")
                                    print(cam_box)
                                    continue
                                else:
                                    min_x, min_y, max_x, max_y = final_coords

                                    img = PIL.Image.open(sample_data_path)

                                    crop_corners = (round(min_x),
                                                    round(min_y),
                                                    round(max_x),
                                                    round(max_y))

                                    masked_img = img.crop(crop_corners)

                                    transform_img = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((params.preprocessing.res_size,
                                                                        params.preprocessing.res_size)),
                                        torchvision.transforms.ToTensor(),
                                    ])

                                    transformed_img = transform_img(masked_img)

                                    transformed_img = transformed_img.unsqueeze(0)  # add batch dim
                                    # torchvision.utils.save_image(transformed_img, 'detection_graph/' + next_sample_token+str(node_id)+'.png')

                                    img_feat = transformed_img
                                    img_features = torch.cat([img_features, img_feat], dim=0)


                        if params.main.sensors_used["lidar"]:
                            with torch.no_grad():

                                # Load point cloud and feed through PointNet
                                # try:
                                # Copy box in ego-vehicle coord. frame.
                                lidar_box = det_box.copy()

                                sd_data_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                                cs_record = nusc.get('calibrated_sensor', sd_data_record['calibrated_sensor_token'])
                                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                                pose_record = nusc.get('ego_pose', sd_data_record['ego_pose_token'])

                                # lidar_pc = nuscenes.utils.data_classes.LidarPointCloud.from_file(lidar_pc_path)
                                lidar_pc, times = nuscenes.utils.data_classes.LidarPointCloud.from_file_multisweep(
                                    nusc=nusc,
                                    sample_rec=sample,
                                    chan='LIDAR_TOP',
                                    ref_chan='LIDAR_TOP',
                                    nsweeps=params.preprocessing.nsweeps_lidar)

                                # Transform point cloud to the ego vehicle frame.
                                lidar_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                                lidar_pc.translate(np.array(cs_record['translation']))

                                # Mask the pointcloud and extract points within box.
                                mask = nuscenes.utils.geometry_utils.points_in_box(lidar_box, lidar_pc.points[0:3, :])
                                masked_lidar_points = lidar_pc.points[:, mask]

                                no_pc_feat = torch.zeros((1, 3, params.graph_construction.feat_3d_app_dim))

                                if masked_lidar_points.shape[1] < params.pointnet.min_lidar_pts:
                                    lidar_feat = no_pc_feat

                                else:
                                    # Normalize: center & scale
                                    X = masked_lidar_points - np.expand_dims(np.mean(masked_lidar_points, axis=0), 0)
                                    dist = np.max(np.sqrt(np.sum(X ** 2, axis=1)), 0)
                                    X = X / dist

                                    # Augment point cloud with zeros or sample to fixed size
                                    # We simulate a batch with a size of 1.
                                    pc_list_tensor, labels = batch_3dmot.utils.dataset.collate_lidar([(X, 0)])

                                    lidar_feat = pc_list_tensor.float()

                                lidar_features = torch.cat([lidar_features, lidar_feat], dim=0)
                            
                            
                        # Load radar point cloud and feed through trained RadarNet
                        if params.main.sensors_used["radar"]:

                            map_cam2radar = defaultdict(list)
                            map_cam2radar['CAM_FRONT_LEFT'].extend(["RADAR_FRONT_LEFT", "RADAR_BACK_LEFT"])
                            map_cam2radar['CAM_FRONT'].extend(
                                ["RADAR_FRONT_RIGHT", "RADAR_FRONT", "RADAR_FRONT_LEFT"])
                            map_cam2radar['CAM_FRONT_RIGHT'].extend(["RADAR_FRONT_RIGHT", "RADAR_BACK_RIGHT"])
                            map_cam2radar['CAM_BACK_RIGHT'].extend(["RADAR_FRONT_RIGHT", "RADAR_BACK_RIGHT"])
                            map_cam2radar['CAM_BACK'].extend(["RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"])
                            map_cam2radar['CAM_BACK_LEFT'].extend(["RADAR_FRONT_LEFT", "RADAR_BACK_LEFT"])

                            # Get detection box in ego coord. frame
                            radar_box = det_box.copy()

                            # get radar pointclouds
                            all_radar_pcs = batch_3dmot.utils.radar.RadarPointCloudWithVelocity(np.zeros((18, 0)))
                            #tqdm.write(str(map_cam2radar[highest_vis_camera]))
                            for radar_channel in map_cam2radar[highest_vis_camera]:
                                radar_pcs, _ = batch_3dmot.utils.radar.RadarPointCloudWithVelocity.from_file_multisweep(nusc=nusc,
                                                                     sample_rec=sample,
                                                                     chan=radar_channel,
                                                                     ref_chan='LIDAR_TOP',
                                                                     nsweeps=params.preprocessing.nsweeps_radar)

                                all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))

                            # Create slightly enlarged annotation box because sometimes radar points belonging to some object
                            # are located a little outside the actual box.
                            enlarged_box_size = [dim * 1.05 for dim in radar_box.wlh]
                            radar_box = nuscenes.utils.data_classes.Box(center=radar_box.center,
                                                                        size=enlarged_box_size,
                                                                        orientation=Quaternion(
                                                                          radar_box.orientation))

                            # Get reference pose
                            ref_sd_token = sample['data']['LIDAR_TOP']
                            ref_sd_rec = nusc.get('sample_data', ref_sd_token)
                            ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])

                            # lidar_sensor translation & rotation wrt. to ego vehicle frame
                            ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])

                            # Transform both radar points as well as bounding box to ego vehicle frame and check whether
                            # the points are actually falling into the box.
                            radar_in_ego_frame = all_radar_pcs

                            radar_in_ego_frame.rotate(Quaternion(ref_cs_rec['rotation']).rotation_matrix)
                            radar_in_ego_frame.translate(np.array(ref_cs_rec['translation']))

                            # Attain boolean array on
                            box_mask = batch_3dmot.utils.radar.points_in_box(box=radar_box, points=radar_in_ego_frame.points[0:3, :])
                            masked_radar_pc = all_radar_pcs.points[:, box_mask]


                            if masked_radar_pc.shape[1] < params.radarnet.min_radar_pts:
                                radar_feat = torch.zeros((1, 4, params.graph_construction.feat_3d_radar_dim))

                            else:
                                # Normalize: center & scale
                                X = masked_radar_pc - np.expand_dims(np.mean(masked_radar_pc, axis=0), 0)
                                dist = np.max(np.sqrt(np.sum(X ** 2, axis=1)), 0)
                                X = X / dist

                                radar_vector = X[[0, 1, 8, 9], :]

                                # Augment point cloud with zeros or sample to fixed size
                                # We simulate a batch with a size of 1.
                                pc_list_tensor, labels = batch_3dmot.utils.dataset.collate_radar([(radar_vector, 0)])

                                radar_feat = pc_list_tensor.float()

                            radar_features = torch.cat([radar_features, radar_feat], dim=0)
                                
                            

                        # We use global coordinates for 3D pose feature
                        det_box = det_box.copy()
                        ego_box = det_box.copy()

                        sd_data_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                        pose_record = nusc.get('ego_pose', sd_data_record['ego_pose_token'])

                        
                        det_box.rotate(Quaternion(pose_record['rotation']))
                        det_box.translate(np.array(pose_record['translation']))

                        ego_box_yaw = torch.from_numpy(
                            np.array([batch_3dmot.utils.geo_utils.quaternion_yaw(ego_box.orientation)]))

                        feat_3d_pose = torch.cat([torch.from_numpy(ego_box.center).float(),
                                                  torch.from_numpy(ego_box.wlh).float(),
                                                  ego_box_yaw.float(),
                                                  torch.from_numpy(ego_box.velocity).float()],
                                                 dim=0)
                        feat_3d_pose = feat_3d_pose.reshape(1, -1)

                        score_feat = torch.tensor(det_box.score).reshape(-1,1)

                        timestamp = val
                        class_label = torch.tensor(int(class_dict_used[ego_box.name]))
                        class_one_hot = torch.nn.functional.one_hot(class_label - 1,
                                                                    num_classes=len(class_dict_used.keys()))
                                
                        class_one_hot = class_one_hot.reshape(-1, 1).float()


                        rel_time_tensor = torch.tensor(int(val-i)).reshape(-1,1).float()

                        box_metadata = {'token': det_box.token,
                                          'sample_token': next_sample_token,
                                          'translation': list(det_box.center),
                                          'size': list(det_box.wlh),
                                          'rotation': list(det_box.orientation),
                                          'category_name': str(det_box.name),
                                          'score': float(det_box.score),
                                          }
                        
                        #print(feat_3d_pose.shape, class_one_hot.shape, score_feat.shape, rel_time_tensor.shape)

                        pose_feat = torch.cat([feat_3d_pose,
                                                class_one_hot.reshape(1,-1),
                                                score_feat,
                                                rel_time_tensor], dim=1)

                        #print(pose_feat.shape, pose_features.shape)
                        pose_features = torch.cat([pose_features, pose_feat.reshape(1,-1)], dim=0)
                        
                        # img, lidar, radar
                        if params.main.sensors_used['radar'] \
                                and params.main.sensors_used['lidar'] and params.main.sensors_used['img']:

                            cur_nodes.append({'box': det_box,
                                              'sample_token': next_sample_token,
                                              'metadata': box_metadata,
                                              'node_id': node_id,
                                              'time': timestamp,
                                              'feat_2d_app': img_feat,
                                              'feat_3d_pose': pose_feat,
                                              'feat_3d_app': lidar_feat,
                                              'feat_3d_radar': radar_feat, 
                                              'category': det_box.name,
                                              'num_lidar_pts': masked_lidar_points.shape[1],
                                              'num_radar_pts': masked_radar_pc.shape[1],
                                              })
                            
                        elif params.main.sensors_used['lidar'] and params.main.sensors_used['img'] \
                                and params.main.sensors_used['radar'] is False:

                            cur_nodes.append({'box': det_box,
                                              'sample_token': next_sample_token,
                                              'metadata': box_metadata,
                                              'node_id': node_id,
                                              'time': timestamp,
                                              'feat_2d_app': img_feat,
                                              'feat_3d_pose': pose_feat,
                                              'feat_3d_app': lidar_feat,
                                              'category': det_box.name,
                                              'num_lidar_pts': masked_lidar_points.shape[1],
                                              'num_radar_pts': 0,
                                              })
                        # img    
                        elif params.main.sensors_used['img'] and params.main.sensors_used['lidar'] is False\
                                and params.main.sensors_used['radar'] is False:

                            cur_nodes.append({'box': det_box,
                                              'sample_token': next_sample_token,
                                              'metadata': box_metadata,
                                              'node_id': node_id,
                                              'time': timestamp,
                                              'feat_2d_app': img_feat,
                                              'feat_3d_pose': pose_feat,
                                              'category': det_box.name,
                                              'num_lidar_pts': 0,
                                              'num_radar_pts': 0,
                                              })
                        
                        # img, radar
                        elif params.main.sensors_used['img'] and params.main.sensors_used['lidar'] is False\
                                and params.main.sensors_used['radar']:

                            cur_nodes.append({'box': det_box,
                                              'sample_token': next_sample_token,
                                              'metadata': box_metadata,
                                              'node_id': node_id,
                                              'time': timestamp,
                                              'feat_3d_pose': pose_feat,
                                              'feat_2d_app': img_feat,
                                              'feat_3d_radar': radar_feat, 
                                              'category': det_box.name,
                                              'num_lidar_pts': 0,
                                              'num_radar_pts': masked_radar_pc.shape[1],
                                              })
                        # lidar    
                        elif params.main.sensors_used['img'] is False and params.main.sensors_used['lidar'] \
                                and params.main.sensors_used['radar'] is False:
                            
                            cur_nodes.append({'box': det_box,
                                              'sample_token': next_sample_token,
                                              'metadata': box_metadata,
                                              'node_id': node_id,
                                              'time': timestamp,
                                              'feat_3d_app': lidar_feat,
                                              'feat_3d_pose': pose_feat,
                                              'category': det_box.name,
                                              'num_lidar_pts': masked_lidar_points.shape[1],
                                              'num_radar_pts': 0,
                                              })
                            
                        node_id += 1

                
                if len(past_nodes) > 0:
                    
                    #if len(past_nodes) > params.graph_construction.top_knn_nodes:
                    #    k = params.graph_construction.top_knn_nodes
                    #else:
                    #    k = len(past_nodes)

                    for cur in tqdm(cur_nodes):
                        
                        past_categ_nodes = list()
                        for past_node in past_nodes:
                            if past_node['category'] == cur['category']:
                                past_categ_nodes.append(past_node)
                                
                        if len(past_categ_nodes) > params.graph_construction.top_knn_nodes:
                            k = params.graph_construction.top_knn_nodes
                        else:
                            k = len(past_categ_nodes)
                            
                        if len(past_categ_nodes) > 0:

                                knn_past_nodes = batch_3dmot.utils.graph_utils.get_knn_nodes_in_graph(cur_node=cur, 
                                                                                                      other_nodes=past_categ_nodes, 
                                                                                                      k=k, use_img=params.main.sensors_used['img'], use_lidar=params.main.sensors_used['lidar'])

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
                                                oth_candidates = list()
                                                for oth_node in knn_past_nodes:
                                                    if oth_node['metadata']['token'] == cur['metadata']['token']:
                                                        time_diff_oth = abs(cur['time'] - oth_node['time'])
                                                        oth_candidates.append(time_diff_oth)
                                                oth_candidates.sort()  # sort list in ascending order

                                                # Compute the time-difference-wise ranking of the investigated edge
                                                rank_cur_ex_edge = np.argmin(np.abs(np.array(oth_candidates)-cur_ex_time_diff))

                                                # Perform label smoothing based on rank
                                                # However, when cur and ex are closest neighbors, the label stays 1.
                                                if rank_cur_ex_edge == 0:
                                                    gt_edges.append(1)
                                                else:
                                                    gt_edges.append(0)
                                            else:
                                                gt_edges.append(0)
                                        else:
                                            gt_edges.append(0)
                                    else:
                                        gt_edges.append(0)

                                    # Compute edge features based on relative differences
                                    box_feats = batch_3dmot.utils.graph_utils.compute_motion_edge_feats(ex, cur)

                                    # Compute relative appearance difference (2D only)
                                    #appear_2d_difference = np.linalg.norm(cur['feat_2d_app'] - ex['feat_2d_app'], ord=2)
                                    #box_feats.append(appear_2d_difference)

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
            img_features = img_features.reshape(len(all_nodes), -1, params.graph_construction.feat_2d_app_dim,
                                                                    params.graph_construction.feat_2d_app_dim)
            lidar_features = lidar_features.reshape(len(all_nodes), params.graph_construction.feat_3d_app_dim, -1)
            radar_features = radar_features.reshape(len(all_nodes), params.graph_construction.feat_3d_radar_dim, -1)

            print(pose_features.shape, img_features.shape, lidar_features.shape, radar_features.shape)
            node_timestamps = torch.tensor([node['time'] for node in all_nodes])
            boxes = torch.cat([node['feat_3d_pose'] for node in all_nodes], dim=1).reshape(len(all_nodes), -1)

            # Store graph data for easy read-in via pyg dataset class.
            torch.save(edges, graph_filename + '_edges.pth')
            torch.save(gt_edges, graph_filename + '_gt.pth')
            torch.save(node_timestamps, graph_filename + '_node_timestamps.pth')
            torch.save(edge_features, graph_filename + '_edge_features.pth')
            torch.save(pose_features, graph_filename + '_pose_features.pth')
            torch.save(img_features, graph_filename + '_img_features.pth')
            torch.save(lidar_features, graph_filename + '_lidar_features.pth')
            torch.save(radar_features, graph_filename + '_radar_features.pth')
            torch.save(boxes, graph_filename + '_node_boxes.pth')

            # Node metadata JSON dump
            all_nodes_metadata = [{'sample_token': node['sample_token'],
                                    'translation': list(node['box'].center),
                                    'size': list(node['box'].wlh),
                                    'rotation': list(node['box'].orientation),
                                    'velocity': list(node['box'].velocity),
                                    'num_lidar_pts': node['num_lidar_pts'],
                                    'num_radar_pts': node['num_radar_pts'],
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
    #parser.add_argument('--resnet_checkpoint', type=str, help='model path')
    #parser.add_argument('--pointnet_checkpoint', type=str, help='model path')
    #parser.add_argument('--radarnet_checkpoint', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.graph_construction.overwrite(opt)

    # Load lists with all scene meta data based on the dataset, the version,
    nusc, meta_lists = batch_3dmot.utils.load_scenes.load_scene_meta_list(data_path=params.paths.data,
                                                                          dataset=params.main.dataset,
                                                                          version=params.main.version)

    # Define model paths for 3D PointNet classifier and 2D ResNet Autoencoder and RadarNet
    #resnet_checkpoint = os.path.join(params.paths.models, params.graph_construction.resnet_checkpoint)
    #pointnet_checkpoint = os.path.join(params.paths.models, params.graph_construction.pointnet_checkpoint)
    #radarnet_checkpoint = os.path.join(params.paths.models, params.graph_construction.radarnet_checkpoint)

    class_dict_used = batch_3dmot.utils.dataset.get_class_config(params, class_dict_name=params.main.class_dict)

    ray.init(num_cpus=8,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # Construct graphs using the generated scene metadata & store them to nuscenes/graphs dir
    #new_meta_list = list()
    for meta_list in meta_lists:
        #print(len(meta_list))

        if params.main.version == "v1.0-trainval":
            if len(meta_list) == 150:
                #for scene in meta_list:
                #    print(scene['token'])
                #    if scene['token'] == '4bb9626f13184da7a67ba1b241c19558':
                #        new_meta_list.append(scene)
                process_all_chunks(params, nusc, meta_list[0:], 19)
        
        
        elif params.main.version == "v1.0-mini":
            if len(meta_list) == 8:
                #for scene in meta_list:
                #    print(scene['token'])
                #    if scene['token'] == '4bb9626f13184da7a67ba1b241c19558':
                #        new_meta_list.append(scene)
                process_all_chunks(params, nusc, meta_list, 9)    

    #if len(meta_list) == 150:
    #    #new_meta_list = list()
        #for scene in meta_list:
        #    if scene['token'] == 'cddd0d5be10b4313a859524952455f43':
        #        new_meta_list.append(scene)
                
        #print(new_meta_list)
    #    process_all_chunks(params, nusc, meta_list, 1)
