import os
import glob
import torch
from PIL import Image
import PIL
from collections import defaultdict
import numpy as np
import random
import json
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.utils import save_image
import nuscenes

from batch_3dmot.utils.config import ParamLib
import batch_3dmot.utils.nuscenes


def check_mkdir(dir_name: str):
    """
    Checks whether the directory dir_name exists. If not, it is created.

    Args:
        dir_name: String defining the directory that needs to exist.
    Returns:
        None
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def get_class_config(params: ParamLib, class_dict_name: str = "nuscenes_tracking_eval"):
    """

    Args:
        params:
        class_dict_name:

    Returns:

    """
    assert isinstance(class_dict_name, str), "Provide a valid class configuration string."

    try:
        if class_dict_name in vars(params.classes).keys():
            return vars(params.classes)[class_dict_name]
        else:
            raise NotImplementedError
    except NotImplementedError:
        print("The specified class configuration is not given.")


def class_to_int(class_dict: dict, search_str: str):
    """
    Function returning a class integer based on searching a passed/default class dictionary.

    Args:
        class_dict: Dictionary defining all object classes
        search_str: The string that is tried to match to one of the dict entries

    Returns:
        res: Integer specifying the class
    """

    # Check if argument is a valid class dictionary
    assert isinstance(class_dict, dict) and len(class_dict) > 0, "Pass a valid class dictionary"

    other_class = len(class_dict)

    res = [val for key, val in class_dict.items() if key in search_str]

    if len(res) == 1:
        res = res[0]
    elif len(res) == 0:
        res = other_class
    else:
        res = other_class
        print('ERROR: Multiple class assignments found.')

    return res

'''
def create_all_split_tokens(nusc: nuscenes.NuScenes, split_scene_meta_list: list):
    """
    Create lists of annotation tokens for all different dataset splits
    Args:
        nusc:
        split_scene_meta_list:

    Returns:
        tokens: list of all annotations belonging to that data split

    """
    tokens = []
    for scene in tqdm(split_scene_meta_list):

        tqdm.write('Processing "{}"'.format(nusc.get('scene', scene['token'])))

        next_sample_token = scene['first_sample_token']#

        sample = nusc.get('sample', next_sample_token)
        while sample['next'] is not '':
            tokens.extend(sample['anns'])
            sample = nusc.get('sample', sample['next'])
    return tokens
'''

class ImageDataset(torch.utils.data.Dataset):
    """Characterizes an image dataset to torch."""
    def __init__(self,
                 params: ParamLib,
                 class_dict: dict,
                 split_name: str,
                 transform=None):
        """
        Initialization
        """

        assert params.main.dataset in ['nuscenes', 'kitti'], "Dataset not supported."

        self.transform = transform
        self.class_dict = class_dict
        self.split_name = split_name
        self.params = params

        try:
            if params.main.dataset == "nuscenes":
                self.data_paths, \
                    self.labels, \
                    self.box_corners, \
                    self.valid_tokens = batch_3dmot.utils.nuscenes.load_image_data(params=params,
                                                                                 class_dict=self.class_dict,
                                                                                 split_name=self.split_name)
            else:
                raise NotImplementedError
        except NotImplementedError as exc:
            tqdm.write("Please provide compatible dataset.")

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index):
        """Generates one sample of data"""
        img_path = self.data_paths[index]
        corners = self.box_corners[index]

        X = Image.open(img_path)

        crop_corners = (round(corners[0]), corners[1], corners[2], corners[3])

        masked_img = X.crop(crop_corners)
        
        converter = PIL.ImageEnhance.Color(masked_img)
        masked_img = converter.enhance(2.0)

        if self.transform:
            X = self.transform(masked_img)

        save_image(X, os.path.join(self.params.paths.preprocessed_data, 'img/', str(self.valid_tokens[index])+'.png'))

        y = torch.LongTensor([self.labels[index]])
        return X, y


class PointCloudDataset(torch.utils.data.Dataset):
    """Characterizes a pointcloud dataset to torch."""
    def __init__(self,
                 params: ParamLib,
                 class_dict: dict,
                 split_name: str,
                 transform=None):
        """Initialization"""

        assert params.main.dataset in ['nuscenes', 'kitti'], "Dataset not supported."
        self.transform = transform
        self.class_dict = class_dict
        self.split_name = split_name
        try:
            if params.main.dataset == "nuscenes":
                self.data_paths, self.labels, self.distribution = batch_3dmot.utils.nuscenes.load_lidar_data(params=params,
                                                                                          class_dict=self.class_dict,
                                                                                          split_name=self.split_name)
                print(split_name + " Distribution of classes: \n", self.distribution)
            else:
                raise Exception("Please provide a method to load point cloud data for your dataset.")
        except Exception as exc:
            tqdm.write(exc)
            
    def get_stats(self):
        
        for path in self.data_paths:
            X = np.load(path)
        

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Generates one sample of data.
        Args:
            index: Index within the dataset that is asked for. (in
        Returns
            X:  Normalized and transformed point cloud
            y:  Corresponding class category label.
        """
        # Source point cloud from list of point clouds based on index
        X = np.load(self.data_paths[index])

        # Normalize: center & scale
        X = X - np.expand_dims(np.mean(X, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(X ** 2, axis=1)), 0)
        X = X / dist

        y = int(self.labels[index])
        return X, y


def collate_lidar(batch_list: list):
    """
    This function accomplishes a zero-padding of small point clouds to build fixed-size batches for training.
    If the point cloud is greater than pc_length it will be downsampled to pc_length.

    Args:
        batch_list: The list of point clouds.

    Returns:
        ret: Dict of point clouds with length pc_length each.
    """
    pc_length = 128

    batch_size = len(batch_list)

    pc_list = []
    label_list = []

    for pc_class_tuple in batch_list:

        key = pc_class_tuple[1]
        val = pc_class_tuple[0]

        try:
            label_list.append(torch.tensor([key]))
            if val.shape[1] < pc_length:
                num_pad = pc_length - val.shape[1]
                # print('------------ PRINTING POINT CLOUD, below 128')
                # print(val)
                padded_pc = np.pad(val, ((0, 0), (0, num_pad)), mode='constant')
                # print(padded_pc.shape)
                padded_tensor = torch.from_numpy(padded_pc[0:3,:])
                pc_list.append(padded_tensor)

            elif val.shape[1] == pc_length:
                pc_list.append(torch.from_numpy(val[0:3,:]))

            else:
                sample_idcs = random.sample(range(0, val.shape[1]), pc_length)
                pc_list.append(torch.from_numpy(np.take(val, sample_idcs, axis=1)[0:3,:]))
                # print(torch.from_numpy(np.take(val, sample_idcs, axis=1)).shape)

        except:
            print('Error in collate_lidar: key=%s' % key)
            raise TypeError

        # print(torch.stack(pc_list).shape) # right now: torch.Size([16, 4, 128]) CORRECT
    pc_list_tensor = torch.stack(pc_list)
    pc_list_tensor = torch.reshape(pc_list_tensor, (batch_size, 3, pc_length))
    # print(pc_list_tensor.shape)

    return pc_list_tensor, torch.tensor(label_list)


class RadarDataset(torch.utils.data.Dataset):
    """Characterizes a pointcloud dataset to torch."""
    def __init__(self, params: ParamLib, split_name: str, class_dict: dict, transform=None):
        """Initialization"""

        assert params.main.dataset in ['nuscenes', 'kitti'], "Dataset not supported."
        self.transform = transform
        self.class_dict = class_dict
        self.split_name = split_name

        try:
            if params.main.dataset == "nuscenes":
                self.data_paths, self.labels = batch_3dmot.utils.nuscenes.load_radar_data(params=params,
                                                                                          class_dict=self.class_dict,
                                                                                          split_name=self.split_name)
            else:
                raise NotImplementedError
        except NotImplementedError as exc:
            tqdm.write("Please provide a compatible dataset.")

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Generates one sample of data.
        Args:
            index: Index within the dataset that is asked for. (in
        Returns
            X:  Normalized and transformed point cloud
            y:  Corresponding class category label.
        """
        # Source all point clouds with a certain number of points as defined in the init-fct.
        masked_radarcloud_path = self.data_paths[index]

        X = np.load(masked_radarcloud_path)

        # Normalize: center & scale point coordinates (not velocity)
        X[0:3] = X[0:3] - np.expand_dims(np.mean(X[0:3], axis=0), 0)
        dist = np.max(np.sqrt(np.sum(X[0:3] ** 2, axis=1)), 0)

        X[0:3] = X[0:3] / dist

        radar_vector = X[[0,1,8,9], :]
        # print("radar vector -shape")
        # print(radar_vector.shape)

        y = int(self.labels[index])
        return radar_vector, y


def collate_radar(batch_list):
    """
    This function accomplishes a zero-padding of small point clouds to build fixed-size batches for training.
    If the point cloud is greater than pc_length it will be downsampled to pc_length.

    Args:
        batch_list: The list of point clouds.

    Returns:
        ret: Dict of point clouds with length pc_length each.
    """
    pc_length = 64
    batch_size = len(batch_list)

    pc_list = []
    label_list = []

    for pc_class_tuple in batch_list:

        # print(val[0].shape[0])
        # print(val)
        key = pc_class_tuple[1]
        val = pc_class_tuple[0]
        # print(val.shape)

        try:
            label_list.append(torch.tensor([key]))
            if val.shape[1] < pc_length:
                num_pad = pc_length - val.shape[1]
                # print('------------ PRINTING POINT CLOUD, below 64')
                # print(val)
                padded_pc = np.pad(val, ((0, 0), (0, num_pad)), mode='constant')
                # print(padded_pc.shape)
                padded_tensor = torch.from_numpy(padded_pc[0:4, :])
                pc_list.append(padded_tensor)

            elif val.shape[1] == pc_length:
                pc_list.append(torch.from_numpy(val[0:4, :]))

            else:
                sample_idcs = random.sample(range(0, val.shape[1]), pc_length)
                pc_list.append(torch.from_numpy(np.take(val, sample_idcs, axis=1)[0:4, :]))
                # print(torch.from_numpy(np.take(val, sample_idcs, axis=1)).shape)

        except:
            print('Error in collate_radar: key=%s' % key)
            raise TypeError

        # print(torch.stack(pc_list).shape) # right now: torch.Size([16, 4, 128]) CORRECT
    pc_list_tensor = torch.stack(pc_list)
    pc_list_tensor = torch.reshape(pc_list_tensor, (batch_size, 4, pc_length))
    # print(pc_list_tensor.shape)

    return pc_list_tensor, torch.tensor(label_list)