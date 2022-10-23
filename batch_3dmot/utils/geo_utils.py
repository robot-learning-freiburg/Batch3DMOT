import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from shapely.geometry import Polygon


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def center_distance(box_i: Box, box_j: Box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param box_i: Box i (annotation)
    :param box_j: Box j (annotation)
    :return: L2 distance.
    """
    return np.linalg.norm(box_i.center[:2] - box_j.center[:2])


def velocity_l2(box_i: Box, box_j: Box) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param box_i: nuscenes.utils.data_classes.Box i (annotation).
    :param box_j: nuscenes.utils.data_classes.Box j (annotation).
    :return: L2 distance.
    """
    return np.linalg.norm(box_i.velocity - box_j.velocity)


def yaw_diff(box_i: Box, box_j: Box, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param box_i: nuscenes.utils.data_classes.Box i (annotation).
    :param box_j: nuscenes.utils.data_classes.Box i (annotation).
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_box_i = quaternion_yaw(box_i.orientation)
    yaw_box_j = quaternion_yaw(box_j.orientation)

    return angle_diff(yaw_box_i, yaw_box_j, period)


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def scale_iou(box_i: Box, box_j: Box) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    s_i_size = box_i.wlh
    s_j_size = box_j.wlh
    assert all(s_i_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(s_j_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(s_i_size, s_j_size)
    volume_i = np.prod(s_i_size)
    volume_j = np.prod(s_j_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_i + volume_j - intersection  # type: float
    iou = intersection / union

    return iou


def box_volume(box: Box) -> float:
    """
    This method computes the volume of a 3D bounding box

    :param box: Some annotation parametrized as Box.
    :return: Volume of the box.
    """
    # Validate inputs.
    s_size = box.wlh

    assert all(s_size > 0), 'Error: sample_annotation sizes must be >0.'
    volume = np.prod(s_size)

    return volume


def boxes_to_sensor(nusc, boxes, pose_record, cs_record):
    """
    data-formats: nusc: Nuscenes, boxes: List[Box], pose_record: Dict, cs_record: Dict
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = Box(box.translation, box.size, Quaternion(box.rotation))
        box.velocity = nusc.box_velocity(box.token)

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out

def iou2d(det_box, ann_box):
    """
    Compute the 2D BEV intersection-over-union based on the bottom corners of two nuScenes bounding boxes.
    :param det_box: Candidate box.
    :param ann_box: Ground-truth box to be evaluated against.
    :return: 2D BEV IoU between the two provided boxes, neglecting the height coordinate.
    """
    
    box1 = det_box.copy()
    box2 = ann_box.copy()

    corners1 = box1.bottom_corners()[0:2,:]
    corners2 = box2.bottom_corners()[0:2,:]

    p1 = Polygon([(corners1[0,0],corners1[1,0]), (corners1[0,1],corners1[1,1]), (corners1[0,2],corners1[1,2]), (corners1[0,3],corners1[1,3])])
    p2 = Polygon([(corners2[0,0],corners2[1,0]), (corners2[0,1],corners2[1,1]), (corners2[0,2],corners2[1,2]), (corners2[0,3],corners2[1,3])])

    if p1.intersects(p2):
        return p1.intersection(p2).area/((p1.area - p1.intersection(p2).area) + p2.area)

    return 0.0