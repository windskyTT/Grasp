from __future__ import annotations

import torch

def residual_joint_target(current_joint_pos: torch.Tensor, actions: torch.Tensor, arm_scale: float, hand_scale: float) -> torch.Tensor:
    scales = torch.full_like(actions, hand_scale)
    scales[:, :6] = arm_scale
    return current_joint_pos + actions * scales

def apply_joint_limits(target: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return target.clamp(lower, upper)

def sample_delay_mask(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.rand(num_envs, device=device) < 0.5

__all__ = ["residual_joint_target", "apply_joint_limits", "sample_delay_mask"]
