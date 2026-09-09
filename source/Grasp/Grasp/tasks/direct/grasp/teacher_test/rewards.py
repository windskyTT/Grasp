from __future__ import annotations

import torch

def weighted_contact_weights(device: torch.device) -> torch.Tensor:
    weights = torch.ones((13,), device=device)
    for index in (0, 3, 6, 9):
        weights[index] *= 3.0
    weights[0] = 0.0
    weights[10:13] *= 2.0
    weights *= 13.0 / weights.sum()
    return weights

def weighted_affordance_weights(device: torch.device) -> torch.Tensor:
    weights = torch.ones((17,), device=device)
    for index in (4, 8, 12, 16):
        weights[index] *= 4.0
    weights[16] *= 2.0
    weights /= weights.sum()
    weights[0] = 0.0
    return weights * 16.0

def compute_reward(env, state: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = env.cfg
    contact_weights = env.contact_weights
    reward_info: dict[str, torch.Tensor] = {}

    reward_info["affordance_contact_reward"] = (state["contact_aff"] * contact_weights).sum(1) / 13.0
    reward_info["affordance_impulse_reward"] = (state["impulse_aff_xy"].clamp(0.0, 0.1) * contact_weights).sum(1)
    push = (state["impulse_aff_vector"][..., 2] - torch.tensor([1.0] + [2.0]*12, device=env.device)).clamp_min(0.0).sum(1).clamp_max(10.0)
    reward_info["push_reward"] = push
    reward_info["table_contact_reward"] = (state["contact_table"] * contact_weights).sum(1) / 13.0
    reward_info["table_impulse_reward"] = (state["impulse_table"].clamp(0.0, 0.1) * contact_weights).sum(1)
    reward_info["arm_contact_reward"] = state["contacts_arm_table"].norm(dim=1)
    reward_info["arm_impulse_reward"] = state["impulse_arm_table"].norm(dim=1)
    reward_info["obj_displacement_reward"] = (state["object_pos_w"] - state["object_initial_position"]).norm(dim=1)
    reward_info["wrist_vel_reward_"] = state["wrist_lin_vel"].square().sum(1)
    reward_info["wrist_qvel_reward_"] = state["wrist_ang_vel"].square().sum(1)
    reward_info["obj_vel_reward_"] = state["object_lin_vel"].square().sum(1)
    reward_info["obj_qvel_reward_"] = state["object_ang_vel"].square().sum(1)
    wrist_speed = state["wrist_lin_vel"].norm(dim=1)
    reward_info["wrist_vel_reward_"] = torch.where(wrist_speed > 0.25, reward_info["wrist_vel_reward_"] * 10.0, reward_info["wrist_vel_reward_"])
    arm_velocity = state["joint_vel"][:, :6].clone()
    arm_velocity = torch.where(arm_velocity.abs() > 0.5, arm_velocity * 4.0, arm_velocity)
    reward_info["arm_joint_vel_reward_"] = arm_velocity.square().sum(1)
    reward_info["affordance_reward"] = -(state["affordance_distances"][:, 1:] * env.affordance_weights[:, 1:]).sum(1)
    reward_info["table_reward"] = -(torch.log((50.0 * state["joint_height"][:, :17].clamp(0.002, 0.02))) * env.finger_weights).sum(1)
    reward_info["arm_height_reward"] = -(torch.log((50.0 * state["arm_height"][:, 2:6].clamp(0.002, 0.02)))).sum(1)
    reward_info["arm_collision_reward"] = state["contacts_arm_all"][:, 1:5].sum(1)

    scales = cfg.reward_coefficients
    total = torch.zeros(env.num_envs, device=env.device)
    for name, value in reward_info.items():
        total = total + value * scales[name]
    reward_info["reward_sum"] = total
    return total.clamp_min(cfg.reward_clip), reward_info

__all__ = ["weighted_contact_weights", "weighted_affordance_weights", "compute_reward"]
