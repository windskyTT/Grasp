from __future__ import annotations

import torch

from .math_utils import quat_to_euler, rotate_world_to_object
from .rotations import quat_to_mat_torch
from .robot_cfg import HAND_BODY_PART_NAMES

RAW_OBSERVATION_DIM = 102
GLOBAL_STATE_DIM = 128
TEACHER_OBSERVATION_DIM = 153

def build_state(env):
    robot = env.robot.data
    obj = env.object.data
    hand_pos_w = robot.body_pos_w[:, env.hand_body_ids]
    arm_pos_w = robot.body_pos_w[:, env.arm_body_ids]
    wrist_pos_w = hand_pos_w[:, 0]
    wrist_quat_w = robot.body_quat_w[:, env.wrist_body_id]
    wrist_rot_w = quat_to_mat_torch(wrist_quat_w)
    wrist_euler = quat_to_euler(wrist_quat_w)
    wrist_lin_vel_w = robot.body_lin_vel_w[:, env.wrist_body_id]
    wrist_ang_vel_w = robot.body_ang_vel_w[:, env.wrist_body_id]
    wrist_lin_vel = torch.bmm(wrist_rot_w.transpose(1, 2), wrist_lin_vel_w.unsqueeze(-1)).squeeze(-1)
    wrist_ang_vel = torch.bmm(wrist_rot_w.transpose(1, 2), wrist_ang_vel_w.unsqueeze(-1)).squeeze(-1)

    object_pos_w = obj.body_pos_w[:, env.object_top_body_id]
    object_quat_w = obj.body_quat_w[:, env.object_top_body_id]
    object_rot_w = quat_to_mat_torch(object_quat_w)
    object_euler = quat_to_euler(object_quat_w)
    object_lin_vel = obj.body_lin_vel_w[:, env.object_top_body_id]
    object_ang_vel = obj.body_ang_vel_w[:, env.object_top_body_id]
    hand_pos_obj = rotate_world_to_object(hand_pos_w, object_pos_w, object_rot_w)
    wrist_pos_obj = hand_pos_obj[:, 0]

    obj_rot_in_wrist = torch.bmm(wrist_rot_w.transpose(1, 2), object_rot_w)
    obj_pose_wrist = matrix_to_euler(obj_rot_in_wrist)
    hand_pose_trans = matrix_to_euler(wrist_rot_w.transpose(1, 2))
    euler_diff = matrix_to_euler(torch.bmm(env.wrist_initial_rot_w.transpose(1, 2), wrist_rot_w))

    target_center_w = object_pos_w + torch.bmm(object_rot_w, env.affordance_center.unsqueeze(-1)).squeeze(-1)
    hand_center_w = wrist_pos_w + torch.bmm(wrist_rot_w, env.hand_center.view(1, 3, 1).expand(env.num_envs, -1, -1)).squeeze(-1)
    target_center_diff = torch.bmm(wrist_rot_w.transpose(1, 2), (target_center_w - hand_center_w).unsqueeze(-1)).squeeze(-1)
    joint_height = hand_pos_w[..., 2] - 0.771
    arm_height = arm_pos_w[..., 2] - 0.771

    object_impulse, non_affordance_impulse, table_impulse, arm_table_impulse, contacts_object, contacts_non_affordance, contacts_table, contacts_arm_table, contacts_arm_all = env._contact_step_data
    contact_aff = contacts_object.to(torch.float32)
    contact_non_aff = contacts_non_affordance.to(torch.float32)
    contact_table = contacts_table.to(torch.float32)
    return {
        "joint_pos": robot.joint_pos,
        "joint_vel": robot.joint_vel,
        "right_hand_torque": env.current_joint_target - robot.joint_pos,
        "wrist_pos_w": wrist_pos_w,
        "wrist_rot_w": wrist_rot_w,
        "wrist_euler": wrist_euler,
        "wrist_lin_vel": wrist_lin_vel,
        "wrist_ang_vel": wrist_ang_vel,
        "object_pos_w": object_pos_w,
        "object_rot_w": object_rot_w,
        "object_euler": object_euler,
        "object_lin_vel": object_lin_vel,
        "object_ang_vel": object_ang_vel,
        "wrist_pos_obj": wrist_pos_obj,
        "obj_pose_wrist": obj_pose_wrist,
        "hand_pos_w": hand_pos_w,
        "hand_pos_obj": hand_pos_obj,
        "joint_height": joint_height,
        "arm_height": arm_height,
        "hand_center_w": hand_center_w,
        "target_center_diff": target_center_diff,
        "euler_diff": euler_diff,
        "hand_pose_trans": hand_pose_trans,
        "contact_aff": contact_aff,
        "contact_non_aff": contact_non_aff,
        "contact_table": contact_table,
        "impulse_aff": object_impulse.norm(dim=-1),
        "impulse_aff_xy": object_impulse[..., :2].norm(dim=-1),
        "impulse_non_aff": non_affordance_impulse.norm(dim=-1),
        "impulse_table": table_impulse.norm(dim=-1),
        "impulse_aff_vector": object_impulse,
        "impulse_table_vector": table_impulse,
        "impulse_arm_table": arm_table_impulse.norm(dim=-1),
        "contacts_arm_table": contacts_arm_table.to(torch.float32),
        "contacts_arm_all": contacts_arm_all.to(torch.float32),
        "affordance_distances": env.affordance_distances,
        "object_initial_position": env.object_initial_position,
        "object_pose_obj": torch.zeros_like(wrist_pos_obj),
    }

def matrix_to_euler(matrix: torch.Tensor) -> torch.Tensor:
    y = torch.asin(matrix[..., 0, 2].clamp(-1.0, 1.0))
    x = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    z = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    return torch.stack((x, y, z), dim=-1)

def obj_pose_from_mat(matrix: torch.Tensor):
    return matrix_to_euler(matrix)

def build_observation(state: dict, previous_wrist_euler: torch.Tensor) -> torch.Tensor:
    unwrapped = state["wrist_euler_unwrapped"]
    euler_delta = state["euler_diff"]
    base = torch.cat((
        state["joint_pos"],
        state["right_hand_torque"],
        state["contact_aff"],
        state["impulse_aff"],
        state["joint_height"],
        state["arm_height"],
        state["hand_center_w"],
        euler_delta,
        unwrapped,
    ), dim=-1)
    affordance = state["affordance_vectors"].reshape(state["joint_pos"].shape[0], -1)
    return torch.cat((base, affordance), dim=-1)

def build_global_state(state: dict) -> torch.Tensor:
    arm_contacts = state["contacts_arm_all"]
    return torch.cat((
        state["obj_pose_wrist"],
        torch.zeros((state["joint_pos"].shape[0], 51), device=state["joint_pos"].device),
        state["hand_pos_obj"].reshape(state["joint_pos"].shape[0], -1),
        state["object_pos_w"],
        state["hand_pose_trans"],
        torch.zeros((state["joint_pos"].shape[0], 1), device=state["joint_pos"].device),
        state["wrist_pos_w"],
        state["target_center_diff"],
        state["object_euler"],
        state["wrist_pos_obj"],
        arm_contacts[:, 1:5],
    ), dim=-1)

__all__ = ["RAW_OBSERVATION_DIM", "GLOBAL_STATE_DIM", "TEACHER_OBSERVATION_DIM", "build_state", "build_observation", "build_global_state"]
