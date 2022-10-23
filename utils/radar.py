import os
import numpy as np
from typing import Tuple, Dict
from nuscenes.utils.data_classes import Box, PointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from functools import reduce

from pyquaternion import Quaternion

# Taken from mrnabati/CenterFusion


class RadarPointCloudWithVelocity(RadarPointCloud):
    """
    A RadarPointCloud class where Radar velocity values are correctly
    transformed to the target coordinate system
    """
    @classmethod
    def rotate_velocity(cls, pointcloud, transform_matrix):
        n_points = pointcloud.shape[1]
        third_dim = np.zeros(n_points)
        pc_velocity = np.vstack((pointcloud[[8, 9], :], third_dim, np.ones(n_points)))
        pc_velocity = transform_matrix.dot(pc_velocity)

        ## in camera coordinates, x is right, z is front
        pointcloud[[8, 9], :] = pc_velocity[[0, 2], :]

        return pointcloud

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregate.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        all_times = np.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        ref_from_car_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)
        car_from_global_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(ref_pose_rec['rotation']),
                                               inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)
            global_from_car_rot = transform_matrix([0.0, 0.0, 0.0],
                                                   Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)
            car_from_current_rot = transform_matrix([0.0, 0.0, 0.0], Quaternion(current_cs_rec['rotation']),
                                                    inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            velocity_trans_matrix = reduce(np.dot, [ref_from_car_rot, car_from_global_rot, global_from_car_rot,
                                                    car_from_current_rot])
            current_pc.transform(trans_matrix)

            # Do the required rotations to the Radar velocity values
            current_pc.points = cls.rotate_velocity(current_pc.points, velocity_trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times


def points_in_box(box: Box, points, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = box.corners(wlh_factor=wlh_factor)


    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]

    i = p_x - p1
    j = p_y - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask = np.logical_and(mask_x, mask_y)

    return mask
