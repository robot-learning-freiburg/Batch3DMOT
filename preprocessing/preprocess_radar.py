import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import ray
from ray.util.multiprocessing import Pool
from functools import reduce
import datetime

import torch
from torchvision import transforms

from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, transform_matrix

import nuscenes
import nuscenes.utils.data_classes

from batch_3dmot.utils.config import ParamLib
import batch_3dmot.utils.dataset
import batch_3dmot.utils.load_scenes
import batch_3dmot.utils.radar
from nuscenes.eval.tracking.utils import category_to_tracking_name

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class PreprocessedDataset:

    def __init__(self, params: ParamLib, nusc: nuscenes.nuscenes.NuScenes):

        """
        Initialize preprocessed dataset class that shall contains all
        annotations
        Args:
            params: dict holding all parameters to be used.
            nusc: a NuScenes instance
        """
        self.nusc = nusc
        self.nsweeps = params.preprocessing.nsweeps_radar

        self.RADAR_PATH = os.path.join(params.paths.preprocessed_data + 'radar/')
        batch_3dmot.utils.dataset.check_mkdir(self.RADAR_PATH)

        with open(params.paths.image_anns, 'r') as json_file:
            self.anns_metadata = json.load(json_file)
            
        print(len(self.anns_metadata))

        self.map_cam2radar = defaultdict(list)
        self.map_cam2radar['CAM_FRONT_LEFT'].extend(["RADAR_FRONT_LEFT", "RADAR_BACK_LEFT"])
        self.map_cam2radar['CAM_FRONT'].extend(["RADAR_FRONT_RIGHT", "RADAR_FRONT", "RADAR_FRONT_LEFT"])
        self.map_cam2radar['CAM_FRONT_RIGHT'].extend(["RADAR_FRONT_RIGHT", "RADAR_BACK_RIGHT"])
        self.map_cam2radar['CAM_BACK_RIGHT'].extend(["RADAR_FRONT_RIGHT", "RADAR_BACK_RIGHT"])
        self.map_cam2radar['CAM_BACK'].extend(["RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"])
        self.map_cam2radar['CAM_BACK_LEFT'].extend(["RADAR_FRONT_LEFT", "RADAR_BACK_LEFT"])

        self.metadata_chunks = list(chunks(self.anns_metadata, params.preprocessing.chunk_size_radar))
        print(len(self.metadata_chunks))

        self.global_result = defaultdict()
        self.global_result['train'] = list()
        self.global_result['val'] = list()
        self.global_result['test'] = list()
        self.global_result['mini_train'] = list()
        self.global_result['mini_val'] = list()



    def process_chunk_ext(self, chunk):
        
        print(len(chunk))

        processed_anns = dict()
        processed_anns['train'] = list()
        processed_anns['val'] = list()
        processed_anns['test'] = list()
        processed_anns['mini_train'] = list()
        processed_anns['mini_val'] = list()

        for ann_metadata in tqdm(chunk):

            if category_to_tracking_name(ann_metadata['category_name']) is not None \
                    and ann_metadata['num_radar_pts'] >= params.radarnet.min_radar_pts:

                # Select the camera capturing the annotation
                camera_channel = ann_metadata['filename'].split('/')[1]

                ann_token = ann_metadata['sample_annotation_token']
                tqdm.write(ann_token)

                # Source metadata belonging to the annotation.
                ann_meta = nusc.get('sample_annotation', ann_token)
                # Source sample belonging to that particular annotation.
                sample = nusc.get('sample', ann_meta['sample_token'])

                # get radar pointclouds
                all_radar_pcs = batch_3dmot.utils.radar.RadarPointCloudWithVelocity(np.zeros((18, 0)))
                tqdm.write(str(self.map_cam2radar[camera_channel]))
                for radar_channel in self.map_cam2radar[camera_channel]:
                    radar_pcs, _ = batch_3dmot.utils.radar.RadarPointCloudWithVelocity.from_file_multisweep(nusc=nusc,
                                                                                                            sample_rec=sample,
                                                                                                            chan=radar_channel,
                                                                                                            ref_chan='LIDAR_TOP',
                                                                                                            nsweeps=self.nsweeps)


                    all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))

                # Create slightly enlarged annotation box because sometimes radar points belonging to some object
                # are located a little outside the actual box.
                enlarged_box_size = [dim * 1.05 for dim in ann_meta['size']]
                ann_box = nuscenes.utils.data_classes.Box(center=ann_meta['translation'], size=enlarged_box_size,
                                                          orientation=Quaternion(ann_meta['rotation'], token=ann_token))

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
                ann_box.translate(-np.array(ref_pose_rec['translation']))
                ann_box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

                ann_ego_radius = np.linalg.norm(
                    np.array(ann_meta['translation'][0:2]) - np.array(ref_pose_rec['translation'][0:2]))

                # Attain boolean array on
                box_mask = batch_3dmot.utils.radar.points_in_box(box=ann_box, points=radar_in_ego_frame.points[0:3, :])

                masked_radar_pc = all_radar_pcs.points[:, box_mask]

                tqdm.write(str(masked_radar_pc.shape))

                if params.pointnet.ego_rad_min < ann_ego_radius < params.pointnet.ego_rad_max:

                    scene_meta = nusc.get('scene', sample['scene_token'])
                    ann_metadata['scene_name'] = scene_meta['name']

                    # Save masked pointcloud as numpy bin
                    if masked_radar_pc is not None:
                        filename = str(ann_metadata['sample_annotation_token'])
                        np.save(os.path.join(self.RADAR_PATH, filename), masked_radar_pc)

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
                    # Extract the 3D box parameters and add to the JSON.
                    ann_metadata['3d_box'] = box_ann

                    # Compute the distance between annotation box and ego vehicle while neglecting the LIDAR-z-axis (upwards)

                    ann_metadata['ann_ego_radius'] = ann_ego_radius

                    if scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['train']:
                        processed_anns['train'].append(ann_metadata)

                    elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['val']:
                        processed_anns['val'].append(ann_metadata)

                    elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['test']:
                        processed_anns['test'].append(ann_metadata)

                    if scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['mini_train']:
                        processed_anns['mini_train'].append(ann_metadata)

                    elif scene_meta['name'] in nuscenes.utils.splits.create_splits_scenes(verbose=False)['mini_val']:
                        processed_anns['mini_val'].append(ann_metadata)

        # Write-Overwrites
        export_log_txt = open(os.path.join(params.paths.preprocessed_data,
                                           "radar_export_chunk_" + str(datetime.datetime.now()).split('.')[0] + ".txt"),
                              "w")  # write mode
        export_log_txt.write(str(params.preprocessing.chunk_size_radar))
        export_log_txt.close()

        return processed_anns


    def process_all_boxes_parallel(self):
        ray.init(num_cpus=12,
                 # log_to_driver=False,
                 include_dashboard=False,
                 # _lru_evict=True,
                 _system_config={"automatic_object_spilling_enabled": True,
                                 "object_spilling_config": json.dumps(
                                     {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )
        pool = Pool()
        batch_result = pool.map(self.process_chunk_ext, [chunk for chunk in self.metadata_chunks])  # metadata_chunks])

        for chunk_result in batch_result:
            for split, list in chunk_result.items():
                self.global_result[split].extend(list)

        return self.global_result



if __name__ == '__main__':

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

    all_processed_anns = preprocessed_dataset.process_all_boxes_parallel()

    # Export the img and pc data
    with open(params.paths.processed_radar_anns, 'w') as outfile:
        json.dump(all_processed_anns, outfile)




