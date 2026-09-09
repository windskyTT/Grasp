from __future__ import annotations

import numpy as np
import trimesh

from .ik import InverseKinematicsUR5
from .rotations import axisangle2quat, quat2mat

def sample_rot_mats(hand_dir_x_w, num_samples, visible_points_w):
    reference = np.asarray(hand_dir_x_w).reshape(3)
    temporary = np.array([1.0, 0.0, 0.0]) if abs(reference[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(reference, temporary); first /= np.linalg.norm(first)
    second = np.cross(reference, first); second /= np.linalg.norm(second)
    perpendicular = []
    for theta in np.linspace(0, 2*np.pi, num_samples, endpoint=False):
        value = first*np.cos(theta) + second*np.sin(theta); value /= np.linalg.norm(value)
        perpendicular.append(-value if value[1] < 0 else value)
    centered = visible_points_w - visible_points_w.mean(axis=0)
    projection = np.asarray([centered @ value for value in perpendicular])
    lengths = projection.max(axis=1) - projection.min(axis=1)
    mats = []
    for value in perpendicular:
        y = np.cross(reference, value); y /= np.linalg.norm(y)
        mats.append(-np.stack((reference, y, value), axis=-1))
    return np.asarray(mats), lengths

def build_pregrasp(mesh_path, object_position, object_quat, camera_position, hand_center, sample_num, top_grasp, length_coeff, angle_coeff, affordance_points_obj):
    mesh = trimesh.load_mesh(mesh_path)
    points = np.asarray(affordance_points_obj)
    object_rotation = quat2mat(object_quat).reshape(3, 3)
    view_point_obj = object_rotation.T @ (np.asarray(camera_position) - object_position)
    directions = points - view_point_obj
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=np.broadcast_to(view_point_obj, points.shape),
        ray_directions=directions,
        multiple_hits=False,
    )
    expanded_locations = np.zeros((200, 3), dtype=np.float32)
    expanded_locations[:, :] = locations[0, :]
    expanded_locations[:locations.shape[0], :] = locations
    visible_points = expanded_locations @ object_rotation.T + object_position
    center = visible_points.mean(axis=0)
    direction = np.array([0.0, 0.0, 1.0]) if top_grasp else camera_position - center
    direction /= np.linalg.norm(direction)
    position = center + 0.25 * direction
    mats, projection_lengths = sample_rot_mats(direction, sample_num, visible_points)
    ik = InverseKinematicsUR5(); ik.setJointWeights([1,1,1,1,1,1]); ik.setJointLimits(-3.14, 3.14)
    ur5_to_world = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    theta0 = [0., -1.57, 1.57, 0., 1.57, -1.57]
    candidates = []
    for mat, length in zip(mats, projection_lengths):
        center_world = mat @ hand_center
        position_ur5 = np.array([position[0] + center_world[0], position[1] + center_world[1], position[2] - 0.771 + center_world[2]])
        wrist_ur5 = ur5_to_world.T @ mat
        goal = np.eye(4); goal[:3, :3] = wrist_ur5; goal[:3, 3] = ur5_to_world.T @ position_ur5
        solution = ik.findClosestIK(goal, theta0)
        if solution is not None:
            score = length * length_coeff + abs(solution[4] - 1.57) * angle_coeff + (abs(solution[4]) - 3.2) * angle_coeff * 0.5
            candidates.append((score if length < 0.18 else 10000.0, length, solution))
    if not candidates:
        raise RuntimeError(f"No valid UR5 IK solution for pre-grasp mesh: {mesh_path}")
    if min(projection_lengths) >= 0.18:
        return min(candidates, key=lambda item: abs(item[2][4]))[2], visible_points, center
    return min(candidates, key=lambda item: item[0])[2], visible_points, center

__all__ = ["sample_rot_mats", "build_pregrasp"]
