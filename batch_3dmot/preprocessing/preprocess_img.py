import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Process, Manager
import pickle
import ray
from ray.util.multiprocessing import Pool

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

class PreprocessedDataset:

    def __init__(self, params: ParamLib, nusc: nuscenes.nuscenes.NuScenes):

        """
        Initialize preprocessed dataset class that shall contains all
        annotations
        Args:
            params: dict holding dataset parameters such as paths
            nusc:
        """
        self.nusc = nusc
        self.params = params

        self.img_path = os.path.join(params.paths.preprocessed_data, 'img/')

        batch_3dmot.utils.dataset.check_mkdir(self.img_path)

        with open(params.paths.image_anns, 'r') as f:
            self.anns_metadata = json.load(f)

    def process_all_boxes(self):

        processed_anns = dict()
        processed_anns['train'] = list()
        processed_anns['val'] = list()
        processed_anns['test'] = list()
        processed_anns['mini_train'] = list()
        processed_anns['mini_val'] = list()

        for ann_metadata in tqdm(self.anns_metadata):

            if category_to_tracking_name(ann_metadata['category_name']) is not None:

                ann_token = ann_metadata['sample_annotation_token']

                # Source metadata belonging to the annotation.
                ann_meta = nusc.get('sample_annotation', ann_token)

                # Source pointcloud belonging to that particular annotation.
                sample = nusc.get('sample', ann_meta['sample_token'])

                sd_data_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                pose_record = nusc.get('ego_pose', sd_data_record['ego_pose_token'])

                # Create box corresponding to annotation
                ann_box = nuscenes.utils.data_classes.Box(center=ann_meta['translation'], size=ann_meta['size'],
                                                          orientation=Quaternion(ann_meta['rotation'], token=ann_token))

                # Move box to ego vehicle coord frame.
                ann_box.translate(-np.array(pose_record['translation']))
                ann_box.rotate(Quaternion(pose_record['rotation']).inverse)

                # Compute the distance between box and ego vehicle while neglecting the LIDAR-z-axis (upwards)
                ann_ego_radius = np.linalg.norm(
                    np.array(ann_meta['translation'][0:2]) - np.array(pose_record['translation'][0:2]))

                if self.params.resnet.ego_rad_min < ann_ego_radius < self.params.resnet.ego_rad_max:
                    # Write distance between ego vehicle and annotation into JSON.
                    ann_metadata['ann_ego_radius'] = ann_ego_radius

                    scene_meta = nusc.get('scene', sample['scene_token'])
                    ann_metadata['scene_name'] = scene_meta['name']

                    box_ann = dict()

                    box_ann['center'] = {'x': ann_box.center[0],
                                         'y': ann_box.center[1],
                                         'z': ann_box.center[2]}

                    box_ann['wlh'] = {'w': ann_box.wlh[0],
                                      'l': ann_box.wlh[1],
                                      'h': ann_box.wlh[2]}

                    box_ann['orientation'] = {'w': ann_box.orientation.w,
                                              'x': ann_box.orientation.x,
                                              'y': ann_box.orientation.y,
                                              'z': ann_box.orientation.z}

                    ann_metadata['3d_box'] = box_ann

                    if scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['train']:
                        processed_anns['train'].append(ann_metadata)

                    elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['val']:
                        processed_anns['val'].append(ann_metadata)

                    elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['test']:
                        processed_anns['test'].append(ann_metadata)

                    # Check for mini-split separately since its a subset of the full dataset.
                    if scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['mini_train']:
                        processed_anns['mini_train'].append(ann_metadata)

                    elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['mini_val']:
                        processed_anns['mini_val'].append(ann_metadata)

        return processed_anns

if __name__ == '__main__':
    # Evaluate parser args

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

    preprocessed_dataset = PreprocessedDataset(params, nusc)

    all_processed_anns = preprocessed_dataset.process_all_boxes()

    # Export the img and pc data
    with open(params.paths.processed_img_anns, 'w') as outfile:
        json.dump(all_processed_anns, outfile)

