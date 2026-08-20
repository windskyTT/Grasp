"""Pure tensor reward calculation for the standalone Teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .env_cfg import TeacherRewardCfg


TEACHER_TERMINAL_REWARD = -10.0

TEACHER_REWARD_TERM_NAMES = (
    "affordance_reward",
    "affordance_contact_reward",
    "affordance_impulse_reward",
    "table_reward",
    "table_contact_reward",
    "table_impulse_reward",
    "arm_height_reward",
    "arm_contact_reward",
    "arm_impulse_reward",
    "arm_collision_reward",
    "push_reward",
    "wrist_vel_reward",
    "wrist_qvel_reward",
    "obj_vel_reward",
    "obj_qvel_reward",
    "obj_displacement_reward",
    "arm_joint_vel_reward",
)


def compute_teacher_reward_terms(
    *,
    nearest_affordance_distance: torch.Tensor,
    hand_body_height: torch.Tensor,
    arm_body_height: torch.Tensor,
    top_total_impulse_vector_w: torch.Tensor,
    top_normal_impulse_vector_w: torch.Tensor,
    top_friction_impulse_vector_w: torch.Tensor,
    bottom_total_impulse_vector_w: torch.Tensor,
    support_total_impulse_vector_w: torch.Tensor,
    arm_interaction_total_impulse_w: torch.Tensor,
    arm_all_contact: torch.Tensor,
    wrist_linear_velocity_w: torch.Tensor,
    wrist_angular_velocity_w: torch.Tensor,
    object_top_linear_velocity_w: torch.Tensor,
    object_top_angular_velocity_w: torch.Tensor,
    object_position_w: torch.Tensor,
    object_initial_position_w: torch.Tensor,
    arm_joint_velocity: torch.Tensor,
    arm_height_penalty_indices: torch.Tensor,
    arm_collision_indices: torch.Tensor,
    cfg: TeacherRewardCfg,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    dtype = nearest_affordance_distance.dtype
    device = nearest_affordance_distance.device
    geometry_weights = torch.tensor(
        cfg.hand_geometry_weights,
        dtype=dtype,
        device=device,
    )
    contact_weights = torch.tensor(
        cfg.hand_contact_weights,
        dtype=dtype,
        device=device,
    )
    impulse_upper = torch.tensor(
        cfg.hand_impulse_upper,
        dtype=dtype,
        device=device,
    )

    affordance_raw = -(
        nearest_affordance_distance * geometry_weights
    ).sum(dim=-1)
    table_raw = -(
        torch.log(
            50.0
            * torch.clamp(
                hand_body_height,
                min=0.002,
                max=0.02,
            )
        )
        * geometry_weights
    ).sum(dim=-1)
    selected_arm_height = arm_body_height.index_select(
        1,
        arm_height_penalty_indices,
    )
    arm_height_raw = -torch.log(
        50.0
        * torch.clamp(
            selected_arm_height,
            min=0.002,
            max=0.02,
        )
    ).sum(dim=-1)

    top_total_norm = torch.linalg.vector_norm(
        top_total_impulse_vector_w,
        dim=-1,
    )
    top_tangential_norm = torch.linalg.vector_norm(
        top_friction_impulse_vector_w,
        dim=-1,
    )
    top_normal_norm = torch.linalg.vector_norm(
        top_normal_impulse_vector_w,
        dim=-1,
    )
    top_contact = (
        top_total_norm > cfg.contact_impulse_threshold
    ).to(dtype)
    affordance_contact_raw = (
        top_contact * contact_weights
    ).sum(dim=-1) / 13.0
    affordance_impulse_raw = (
        torch.minimum(top_tangential_norm, impulse_upper)
        * contact_weights
    ).sum(dim=-1)

    bottom_norm = torch.linalg.vector_norm(
        bottom_total_impulse_vector_w,
        dim=-1,
    )
    bottom_contact = (
        bottom_norm > cfg.contact_impulse_threshold
    ).to(dtype)
    bottom_contact_raw = (
        bottom_contact * contact_weights
    ).sum(dim=-1) / 13.0
    bottom_impulse_raw = (
        torch.minimum(bottom_norm, impulse_upper)
        * contact_weights
    ).sum(dim=-1)

    support_norm = torch.linalg.vector_norm(
        support_total_impulse_vector_w,
        dim=-1,
    )
    support_contact = (
        support_norm > cfg.contact_impulse_threshold
    ).to(dtype)
    table_contact_raw = (
        support_contact * contact_weights
    ).sum(dim=-1) / 13.0
    table_impulse_raw = (
        torch.minimum(support_norm, impulse_upper)
        * contact_weights
    ).sum(dim=-1)

    arm_interaction_norm = torch.linalg.vector_norm(
        arm_interaction_total_impulse_w,
        dim=-1,
    )
    arm_interaction_contact = (
        arm_interaction_norm > cfg.contact_impulse_threshold
    ).to(dtype)
    arm_contact_raw = torch.linalg.vector_norm(
        arm_interaction_contact,
        dim=-1,
    )
    arm_impulse_raw = torch.linalg.vector_norm(
        arm_interaction_norm,
        dim=-1,
    )
    arm_collision_raw = arm_all_contact.index_select(
        1,
        arm_collision_indices,
    ).to(dtype).sum(dim=-1)

    push_raw = torch.clamp(
        top_normal_norm[:, 0] - cfg.push_palm_threshold,
        min=0.0,
    )
    push_raw += torch.clamp(
        top_normal_norm[:, 1:] - cfg.push_finger_threshold,
        min=0.0,
    ).sum(dim=-1)
    push_raw = torch.clamp(
        push_raw,
        max=cfg.push_maximum,
    )

    wrist_speed = torch.linalg.vector_norm(
        wrist_linear_velocity_w,
        dim=-1,
    )
    wrist_vel_raw = wrist_speed.square()
    wrist_vel_raw = torch.where(
        wrist_speed > cfg.wrist_velocity_threshold,
        wrist_vel_raw * cfg.wrist_velocity_multiplier,
        wrist_vel_raw,
    )
    wrist_qvel_raw = wrist_angular_velocity_w.square().sum(
        dim=-1
    )
    obj_vel_raw = object_top_linear_velocity_w.square().sum(
        dim=-1
    )
    obj_qvel_raw = object_top_angular_velocity_w.square().sum(
        dim=-1
    )
    obj_displacement_raw = torch.linalg.vector_norm(
        object_position_w - object_initial_position_w,
        dim=-1,
    )
    scaled_arm_joint_velocity = torch.where(
        arm_joint_velocity.abs()
        > cfg.arm_joint_velocity_threshold,
        arm_joint_velocity * cfg.arm_joint_velocity_multiplier,
        arm_joint_velocity,
    )
    arm_joint_vel_raw = scaled_arm_joint_velocity.square().sum(
        dim=-1
    )

    weighted_terms = {
        "affordance_reward": (
            affordance_raw * cfg.affordance_reward_scale
        ),
        "affordance_contact_reward": (
            affordance_contact_raw
            * cfg.affordance_contact_reward_scale
        ),
        "affordance_impulse_reward": (
            affordance_impulse_raw
            * cfg.affordance_impulse_reward_scale
        ),
        "table_reward": table_raw * cfg.table_reward_scale,
        "table_contact_reward": (
            table_contact_raw * cfg.table_contact_reward_scale
        ),
        "table_impulse_reward": (
            table_impulse_raw * cfg.table_impulse_reward_scale
        ),
        "arm_height_reward": (
            arm_height_raw * cfg.arm_height_reward_scale
        ),
        "arm_contact_reward": (
            arm_contact_raw * cfg.arm_contact_reward_scale
        ),
        "arm_impulse_reward": (
            arm_impulse_raw * cfg.arm_impulse_reward_scale
        ),
        "arm_collision_reward": (
            arm_collision_raw * cfg.arm_collision_reward_scale
        ),
        "push_reward": push_raw * cfg.push_reward_scale,
        "wrist_vel_reward": (
            wrist_vel_raw * cfg.wrist_vel_reward_scale
        ),
        "wrist_qvel_reward": (
            wrist_qvel_raw * cfg.wrist_qvel_reward_scale
        ),
        "obj_vel_reward": obj_vel_raw * cfg.obj_vel_reward_scale,
        "obj_qvel_reward": (
            obj_qvel_raw * cfg.obj_qvel_reward_scale
        ),
        "obj_displacement_reward": (
            obj_displacement_raw
            * cfg.obj_displacement_reward_scale
        ),
        "arm_joint_vel_reward": (
            arm_joint_vel_raw * cfg.arm_joint_vel_reward_scale
        ),
    }
    base_reward = torch.stack(
        tuple(weighted_terms.values()),
        dim=0,
    ).sum(dim=0)
    lift_success = (
        object_position_w[:, 2]
        - object_initial_position_w[:, 2]
        > cfg.lift_success_height
    ).to(dtype)
    raw_terms = {
        "affordance_distance_raw": affordance_raw,
        "top_contact_raw": affordance_contact_raw,
        "top_impulse_tangential_raw": affordance_impulse_raw,
        "bottom_contact_raw": bottom_contact_raw,
        "bottom_impulse_raw": bottom_impulse_raw,
        "table_height_raw": table_raw,
        "support_contact_raw": table_contact_raw,
        "support_impulse_raw": table_impulse_raw,
        "arm_height_raw": arm_height_raw,
        "arm_contact_raw": arm_contact_raw,
        "arm_impulse_raw": arm_impulse_raw,
        "arm_collision_raw": arm_collision_raw,
        "push_raw": push_raw,
        "wrist_vel_raw": wrist_vel_raw,
        "wrist_qvel_raw": wrist_qvel_raw,
        "obj_vel_raw": obj_vel_raw,
        "obj_qvel_raw": obj_qvel_raw,
        "obj_displacement_raw": obj_displacement_raw,
        "arm_joint_vel_raw": arm_joint_vel_raw,
        "lift_success": lift_success,
    }
    return base_reward, weighted_terms, raw_terms


__all__ = [
    "TEACHER_REWARD_TERM_NAMES",
    "TEACHER_TERMINAL_REWARD",
    "compute_teacher_reward_terms",
]
