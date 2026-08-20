from __future__ import annotations

import torch

from .math_utils import clamp_tensor


TEACHER_ACTION_DIM = 13
TEACHER_ARM_ACTION_DIM = 7
TEACHER_HAND_ACTION_DIM = 6


def compute_residual_active_target(
    actions: torch.Tensor,
    current_active_qpos: torch.Tensor,
    active_lower_limits: torch.Tensor,
    active_upper_limits: torch.Tensor,
    arm_scale: float,
    hand_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_scale = torch.full_like(actions, hand_scale)
    action_scale[:, :TEACHER_ARM_ACTION_DIM] = arm_scale

    active_target = (
        current_active_qpos + actions * action_scale
    )
    active_target = clamp_tensor(
        active_target,
        active_lower_limits,
        active_upper_limits,
    )
    return actions, active_target


def build_full_joint_target(
    current_joint_pos: torch.Tensor,
    active_target: torch.Tensor,
    active_joint_ids: torch.Tensor,
    passive_joint_ids: torch.Tensor,
    mimic_parent_joint_ids: torch.Tensor,
    mimic_multipliers: torch.Tensor,
    joint_lower_limits: torch.Tensor,
    joint_upper_limits: torch.Tensor,
) -> torch.Tensor:
    full_target = current_joint_pos.clone()
    full_target[:, active_joint_ids] = active_target

    passive_target = (
        full_target[:, mimic_parent_joint_ids]
        * mimic_multipliers
    )
    passive_target = clamp_tensor(
        passive_target,
        joint_lower_limits[passive_joint_ids],
        joint_upper_limits[passive_joint_ids],
    )
    full_target[:, passive_joint_ids] = passive_target

    return clamp_tensor(
        full_target,
        joint_lower_limits,
        joint_upper_limits,
    )


def sample_delay_mask(
    num_envs: int,
    probability: float,
    device: torch.device | str,
) -> torch.Tensor:
    return (
        torch.rand(num_envs, device=device) < probability
    )


def select_substep_joint_target(
    current_joint_target: torch.Tensor,
    previous_joint_target: torch.Tensor,
    delay_mask: torch.Tensor,
    physics_substep: int,
    decimation: int,
) -> torch.Tensor:
    if physics_substep == 0:
        return torch.where(
            delay_mask[:, None],
            previous_joint_target,
            current_joint_target,
        )
    return current_joint_target


__all__ = [
    "TEACHER_ACTION_DIM",
    "TEACHER_ARM_ACTION_DIM",
    "TEACHER_HAND_ACTION_DIM",
    "compute_residual_active_target",
    "build_full_joint_target",
    "sample_delay_mask",
    "select_substep_joint_target",
]
