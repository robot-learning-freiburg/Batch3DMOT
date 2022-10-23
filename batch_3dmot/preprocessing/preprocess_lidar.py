import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager
import pickle
import ray
from ray.util.multiprocessing import Pool
from collections import defaultdict
import datetime

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import PIL
from pyquaternion import Quaternion

import nuscenes

import nuscenes.utils.data_classes
import nuscenes.utils.geometry_utils
from nuscenes.eval.tracking.utils import category_to_tracking_name

from batch_3dmot.utils.config import ParamLib
import batch_3dmot.utils.dataset
import batch_3dmot.utils.load_scenes

# ----------- PARAMS (TO BE SOURCED) --------------

parser = argparse.ArgumentParser()

# General args (namespace: main)
parser.add_argument('--dataset', type=str, default='nuscenes', help="dataset path")
parser.add_argument('--version', type=str, help="dataset type nuscenes/gen1/KITTI")
parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)

# Specific args (namespace: preprocessing)
parser.add_argument('--res_size', type=int, help="specify the image resolution used")

opt = parser.parse_args()

params = ParamLib(opt.config)
params.main.overwrite(opt)
params.preprocessing.overwrite(opt)
print(vars(params.preprocessing))
opt = parser.parse_args()


### ------------------------------------------------


all_processed_anns = list()

# Load lists with all scene meta data based on the dataset, the version,
nusc, meta_lists = batch_3dmot.utils.load_scenes.load_scene_meta_list(data_path=params.paths.data,
                                                                      dataset=params.main.dataset,
                                                                      version=params.main.version)

# Check which dataset version is used
if params.main.version == "v1.0-mini":
    mini_train_scene_meta_list, mini_val_scene_meta_list = meta_lists

elif params.main.version == "v1.0-trainval":
    train_scene_meta_list, val_scene_meta_list = meta_lists

elif params.main.version == "v1.0-test":
    test_scene_meta_list = meta_lists

else:
    print("Unable to load meta data to the provided dataset, the version identifier is not known.")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


pc_path = os.path.join(params.paths.preprocessed_data, 'lidar/')

batch_3dmot.utils.dataset.check_mkdir(pc_path)

with open(params.paths.image_anns, 'r') as f:
    anns_metadata = json.load(f)

metadata_chunks = list(chunks(anns_metadata, params.preprocessing.chunk_size_lidar))
print(len(metadata_chunks))

batch_len = 12 # 20 works well on the cluster

iteration_ranges = chunks(list(range(0,len(metadata_chunks))), batch_len)
iteration_idcs = list(iteration_ranges)

global_result = defaultdict()
global_result['train'] = list()
global_result['val'] = list()
global_result['test'] = list()
global_result['mini_train'] = list()
global_result['mini_val'] = list()


def get_masked_pc(ann_metadata: dict,
                  channel: str = 'LIDAR_TOP',
                  min_dist: float = 1.0,
                  max_dist: float = 50.0,
                  nsweeps: int = 5):
    """
    Extracts points that lie inside of box as defined by annotation.
    Transforms both annotation box as well as pointcloud into ego vehicle frame.

    max_dist: 50m corresponds to the nuscenes tracking eval spec
    Args:
        params:
        nusc:
        ann_metadata:
        channel:
        min_dist:
        max_dist:
        nsweeps:
    Returns:
        masked_lidar_pc: LidarPointCloud only holding the masked points
    """
    ann_token = ann_metadata['sample_annotation_token']

    # Source metadata belonging to the annotation.
    ann_meta = nusc.get('sample_annotation', ann_token)

    # Source pointcloud belonging to that particular annotation.
    sample = nusc.get('sample', ann_meta['sample_token'])

    # sd = sample data (here: lidar data record)
    sd_data_record = nusc.get('sample_data', sample['data'][channel])
    cs_record = nusc.get('calibrated_sensor', sd_data_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_data_record['ego_pose_token'])

    lidar_pc_path = os.path.join(nusc.dataroot, sd_data_record['filename'])

    lidar_pc = nuscenes.utils.data_classes.LidarPointCloud.from_file(lidar_pc_path)
    lidar_pc, times = nuscenes.utils.data_classes.LidarPointCloud.from_file_multisweep(nusc=nusc,
                                                                                       sample_rec=sample,
                                                                                       chan=channel,
                                                                                       ref_chan='LIDAR_TOP',
                                                                                       nsweeps=nsweeps)

    num_points = lidar_pc.nbr_points()
    # print("1 sweep: " + str(num_points))

    num_points_alt = lidar_pc.nbr_points()
    # print("5 sweep: " + str(num_points_alt))

    # Create box corresponding to annotation
    ann_box = nuscenes.utils.data_classes.Box(center=ann_meta['translation'], size=ann_meta['size'],
                                              orientation=Quaternion(ann_meta['rotation'], token=ann_token))

    # Transform point cloud to the ego vehicle frame.
    lidar_pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    lidar_pc.translate(np.array(cs_record['translation']))

    # copy global params
    global_box = ann_box.copy()

    # move to origin and rotate randomly, then move back to original pos in global frame.
    ann_box.translate(-ann_box.center)
    rand_rot_quat = Quaternion(axis=[0, 0, 1],
                               angle=np.random.uniform(-(3.14159265 / 10), +(3.14159265 / 10), 1)[0])
    ann_box.rotate(rand_rot_quat)
    ann_box.translate(global_box.center)

    # perform random resizing
    rand_resize = np.random.uniform(0.85, 1.15, 3)
    ann_box.wlh = np.array([ann_box.wlh[0] * rand_resize[0],
                            ann_box.wlh[1] * rand_resize[1],
                            ann_box.wlh[2] * rand_resize[2]])

    # Move box to ego vehicle coord frame.
    ann_box.translate(-np.array(pose_record['translation']))
    ann_box.rotate(Quaternion(pose_record['rotation']).inverse)

    # Mask the pointcloud and extract points within box.
    mask = nuscenes.utils.geometry_utils.points_in_box(ann_box, lidar_pc.points[0:3, :])
    masked_lidar_points = lidar_pc.points[:, mask]

    masked_lidar_pc = nuscenes.utils.data_classes.LidarPointCloud(masked_lidar_points)

    # Compute the distance between annotation box and ego vehicle while neglecting the LIDAR-z-axis (upwards)
    ann_ego_radius = np.linalg.norm(np.array(ann_meta['translation'][0:2]) - np.array(pose_record['translation'][0:2]))

    if min_dist < ann_ego_radius < max_dist:
        use = True
        tqdm.write('MASKED LIDAR: ' + str(masked_lidar_pc.nbr_points()))
        # print('DISTANCE TO EGO VEHICLE VALID: ' + str(ann_ego_radius))
    else:
        use = False
        # print('DISTANCE TO EGO VEHICLE NOT VALID' + str(ann_ego_radius))

    return masked_lidar_pc, ann_ego_radius, ann_box, sample


def process_chunk_ext(chunk):

    print(len(chunk))
    processed_anns = dict()
    processed_anns['train'] = list()
    processed_anns['val'] = list()
    processed_anns['test'] = list()
    processed_anns['mini_train'] = list()
    processed_anns['mini_val'] = list()

    for ann_metadata in tqdm(chunk):

        if category_to_tracking_name(ann_metadata['category_name']) is not None \
                and ann_metadata['num_lidar_pts'] >= params.pointnet.min_lidar_pts:
            # Extract both masked image as well as pointcloud.
            masked_pc, dist, box_ego, sample = get_masked_pc(ann_metadata=ann_metadata,
                                                             channel='LIDAR_TOP',
                                                             min_dist=params.pointnet.ego_rad_min,
                                                             max_dist=params.pointnet.ego_rad_max,
                                                             nsweeps=params.preprocessing.nsweeps_lidar)

            if params.pointnet.ego_rad_min < dist < params.pointnet.ego_rad_max:

                scene_meta = nusc.get('scene', sample['scene_token'])
                ann_metadata['scene_name'] = scene_meta['name']

                # Save masked pointcloud as numpy bin
                if masked_pc is not None:
                    filename = str(ann_metadata['sample_annotation_token'])
                    np.save(os.path.join(pc_path, filename), masked_pc.points)

                # Write distance between ego vehicle and annotation into JSON.
                ann_metadata['ann_ego_radius'] = dist

                box_ann = dict()

                box_ann['center'] = {'x': box_ego.center[0],
                                     'y': box_ego.center[1],
                                     'z': box_ego.center[2]}

                box_ann['wlh'] = {'w': box_ego.wlh[0],
                                  'l': box_ego.wlh[1],
                                  'h': box_ego.wlh[2]}

                box_ann['orientation'] = {'w': box_ego.orientation.w,
                                          'x': box_ego.orientation.x,
                                          'y': box_ego.orientation.y,
                                          'z': box_ego.orientation.z}

                ann_metadata['3d_box'] = box_ann

                if scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['train']:
                    processed_anns['train'].append(ann_metadata)

                elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['val']:
                    processed_anns['val'].append(ann_metadata)

                elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['test']:
                    processed_anns['test'].append(ann_metadata)

                # Since the mini split is a subset of the full dataset we have to check separately.
                if scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['mini_train']:
                    processed_anns['mini_train'].append(ann_metadata)

                elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['mini_val']:
                    processed_anns['mini_val'].append(ann_metadata)

    # Write-Overwrites
    export_log_txt = open(os.path.join(params.paths.preprocessed_data,
                                       "lidar_export_chunk_"+str(datetime.datetime.now()).split('.')[0]+".txt"),
                          "w")  # write mode
    export_log_txt.write(str(params.preprocessing.chunk_size_lidar))
    export_log_txt.close()

    return processed_anns

index = params.preprocessing.lidar_batch_idx
print(iteration_idcs[index])
chunks_in_batch = [metadata_chunks[single_idx] for single_idx in iteration_idcs[index]]
print(len(chunks_in_batch))


ray.init(num_cpus=12,
         #log_to_driver=False,
         include_dashboard=False,
         #_lru_evict=True,
         _system_config={"automatic_object_spilling_enabled": True,
                         "object_spilling_config": json.dumps({"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},)},)
pool = Pool()
batch_result = pool.map(process_chunk_ext, [chunk for chunk in chunks_in_batch]) #metadata_chunks])

for chunk_result in batch_result:
    for split, list in chunk_result.items():
        global_result[split].extend(list)

if __name__ == '__main__':

    #all_processed_anns = process_chunks()
    # Export the img and pc data

    batch_path = "".join(params.paths.processed_lidar_anns.split('.')[:-1])+ str(params.preprocessing.lidar_batch_idx)+".json"
    with open(batch_path, 'w') as outfile:
        json.dump(global_result,  outfile)
