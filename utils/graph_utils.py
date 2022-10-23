import numpy as np
from . import geo_utils

import torch


def compute_motion_edge_feats(cur_node: dict, oth_node: dict):
    """
    Compute the motion edge features between two nodes. The motion features contain:
    3D-distance (L2)
    Velocity-Distance (L2)
    Absolute yaw difference
    Volume difference of the two bounding boxes
    :param cur_node: dict (current node annotation, containing 'box'-key).
    :param oth_node: dict (other node annotation, containing 'box'-key).
    :return: List (containing all computed features).
    """
    # compute relative box differences
    l2_3d_dist = geo_utils.center_distance(cur_node['box'], oth_node['box'])

    # Currently: Neglect 3D velocity difference
    #l2_velocity_dist = geo_utils.velocity_l2(cur_node['box'], oth_node['box'])

    # 3D angle difference
    yaw_diff = np.abs(geo_utils.yaw_diff(cur_node['box'], oth_node['box']))

    # Log-difference
    vol_diff = np.log(geo_utils.box_volume(cur_node['box']) / geo_utils.box_volume(oth_node['box']))

    return [l2_3d_dist, yaw_diff, vol_diff]  # no velocity right now [l2_3d_dist, l2_velocity_dist, yaw_diff, vol_diff]


def get_knn_nodes_in_graph(cur_node: dict, other_nodes: list, k: int=50, use_img: bool = True, use_lidar: bool = True):
    """
    - Compute box of current node
    - Compute 2D position of current node
    """

    #print('Looking for kNN nodes to node: ' + str(cur_node['node_id']))

    # 3D Box
    cur_box = cur_node['box']

    transl_3d_dists = []
    vel_dists = []
    yaw_dists = []

    # compare translation
    for oth_node in other_nodes:

        # 2D/3D MOTION FEATURES
        #   For now neglect scale_IOU because we assume it is not a critical variable

        #   3D position difference = distance in global coordinates
        l2_3d_dist = geo_utils.center_distance(cur_box, oth_node['box'])

        #   3D velocity difference
        l2_vel_dist = geo_utils.velocity_l2(cur_box, oth_node['box'])

        #   3D angle difference
        yaw_diff = geo_utils.yaw_diff(cur_box, oth_node['box'])

        transl_3d_dists.append(l2_3d_dist)
        vel_dists.append(abs(l2_vel_dist))
        yaw_dists.append(abs(yaw_diff))

    # NORMALIZATION OF 2D/3D MOTION FEATURES
    transl_3d_dists = torch.tensor(transl_3d_dists)
    yaw_dists = torch.tensor(yaw_dists)
    vel_dists = torch.tensor(vel_dists)

    transl_3d_dists = transl_3d_dists/torch.max(transl_3d_dists)
    yaw_dists = yaw_dists/torch.max(yaw_dists)
    vel_dists = vel_dists/torch.max(vel_dists)

    # Combine different 2D/3D motion features
    motion_dists = (1/2) * transl_3d_dists + (1/4) * yaw_dists + (1/4) * vel_dists
    motion_dists = motion_dists/torch.max(motion_dists)

    combined_2d_3d_m_a_dists = motion_dists #(1/2)*motion_dists + (1/2)*appear_dists

    knn_classifier = torch.topk(combined_2d_3d_m_a_dists, k, largest=False)
    top_k_idcs = knn_classifier.indices.tolist()
    top_k_nodes = [other_nodes[i] for i in top_k_idcs]

    try:
        return top_k_nodes
    except IndexError:
        return other_nodes