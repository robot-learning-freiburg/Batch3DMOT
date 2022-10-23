#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2022
@author: Martin Buechner, buechner@cs.uni-freiburg.de
"""
import json
import os
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Union, Optional, Dict

from typing import Type
from batch_3dmot.utils.config import ParamLib
import batch_3dmot.utils.dataset
#import nuscenes


import shapely.geometry

def category_to_tracking_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes tracking classes.
    :param category_name: Generic nuScenes class.
    :return: nuScenes tracking class.
    """
    tracking_mapping = {
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }

    if category_name in tracking_mapping:
        return tracking_mapping[category_name]
    else:
        return None
    


def load_image_data(params: ParamLib, class_dict: dict, split_name: str):
    """
    Loads the image data of the nuscenes dataset.
    Args:
        params:
        class_dict:
        split_tokens:
    Returns:

    """

    img_data_path = os.path.join(params.paths.preprocessed_data, 'img/')
    labels = []
    img_paths = []
    box_corners = []
    ann_tokens = []

    # Come up with indices from processed_anns.json that provide valid images.
    with open(params.paths.processed_img_anns, 'r') as json_file:
        processed_anns = json.load(json_file)
    tqdm.write("Successfully opened the processed img annotation JSONS.")

    # For all annotations create the image/ann metadata
    for ann_metadata in tqdm(processed_anns[split_name]):
        if ann_metadata["visibility_token"] is not "":
            img_paths.append(os.path.join(params.paths.data, ann_metadata['filename']))
            labels.append(class_dict[category_to_tracking_name(ann_metadata['category_name'])])
            box_corners.append(ann_metadata['bbox_corners'])
            ann_tokens.append(ann_metadata['sample_annotation_token'])

    return img_paths, labels, box_corners, ann_tokens


def load_lidar_data(params: ParamLib, class_dict, split_name: str):
    """
    Args:
        params:
        class_dict:
        split_name:
    Returns:
    """
    pc_data_path = os.path.join(params.paths.preprocessed_data, 'lidar/')
    pc_length = params.pointnet.num_points

    valid_anns = []
    labels = []

    sum_points = []

    pc_paths = []

    distribution = dict()

    for key, val in class_dict.items():
        distribution[key] = 0

    # Come up with indices from processed_anns.json that provide suitable point clouds (enough points).
    with open(params.paths.processed_lidar_anns, 'r') as json_file:
        processed_anns = json.load(json_file)

    # Filter out point clouds with too few points and use radius-based selection range
    for ann_metadata in tqdm(processed_anns[split_name]):
        if ann_metadata['num_lidar_pts'] > params.pointnet.min_lidar_pts \
                and params.pointnet.ego_rad_min < ann_metadata['ann_ego_radius'] < params.pointnet.ego_rad_max \
                and category_to_tracking_name(ann_metadata['category_name']) is not None:

            # Get the pointcloud path
            masked_pc_path = os.path.join(pc_data_path, str(ann_metadata['sample_annotation_token']) + '.npy')
            pc_paths.append(masked_pc_path)

            # Generate the respective label
            labels.append(class_dict[category_to_tracking_name(ann_metadata['category_name'])])

            distribution[category_to_tracking_name(ann_metadata['category_name'])] += 1

            # Retrieve dataset statistics: avg. number of points, valid annotations
            if ann_metadata['num_lidar_pts'] > pc_length:
                sum_points.append(pc_length)
            else:
                sum_points.append(ann_metadata['num_lidar_pts'])

            valid_anns.append(ann_metadata)

    # Calculate the number of valid masked point clouds.
    # tqdm.write(str(len(valid_anns)))

    # Compute average masked pointcloud size
    # tqdm.write(str(sum(sum_points) / len(valid_anns)))

    return pc_paths, labels, distribution


def load_radar_data(params: ParamLib, class_dict: dict, split_name: str):
    """
    Args:
        params:
        class_dict:
        split_name:
    Returns:
    """
    radar_data_path = os.path.join(params.paths.preprocessed_data, 'radar/')
    pc_length = params.radarnet.num_points

    radar_paths = []

    valid_anns = []
    labels = []

    sum_points = []

    # Come up with indices from processed_anns.json that provide suitable point clouds (enough points).
    with open(params.paths.processed_radar_anns, 'r') as json_file:
        processed_anns = json.load(json_file)

    # Filter out point clouds with too few points and use radius-based selection range
    for ann_metadata in tqdm(processed_anns[split_name]):

        # check whether the annotation is part of the train/val-tokens
        if ann_metadata['num_radar_pts'] >= params.radarnet.min_radar_pts \
                and params.radarnet.ego_rad_min < ann_metadata['ann_ego_radius'] < params.radarnet.ego_rad_max \
                and category_to_tracking_name(ann_metadata['category_name']) is not None:

            radar_paths.append(os.path.join(radar_data_path, str(ann_metadata['sample_annotation_token']) + '.npy'))

            class_str = category_to_tracking_name(ann_metadata['category_name'])
            labels.append(class_dict[class_str])

            if ann_metadata['num_radar_pts'] > pc_length:
                sum_points.append(pc_length)
            else:
                sum_points.append(ann_metadata['num_radar_pts'])
            valid_anns.append(ann_metadata)

    tqdm.write(str(len(valid_anns)))
    tqdm.write(str(sum(sum_points) / len(valid_anns)))

    return radar_paths, labels


def count_box_corners_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check if a box is visible inside an image without accounting for occlusions and count the number of corners
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    return np.sum(visible), np.sum(in_front)


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    try:
        polygon_from_2d_box = shapely.geometry.MultiPoint(corner_coords).convex_hull
        img_canvas = shapely.geometry.box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None
    except AttributeError:
        return None