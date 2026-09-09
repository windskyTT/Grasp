from __future__ import annotations

import torch

from .rotations import quat_to_mat_torch

def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    mat = quat_to_mat_torch(quat)
    y = torch.asin(mat[..., 0, 2].clamp(-1.0, 1.0))
    x = torch.atan2(-mat[..., 1, 2], mat[..., 2, 2])
    z = torch.atan2(-mat[..., 0, 1], mat[..., 0, 0])
    return torch.stack((x, y, z), dim=-1)

def wrap_euler_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    delta = current - previous
    return (delta + torch.pi) % (2 * torch.pi) - torch.pi

def rotate_world_to_object(points_w: torch.Tensor, position_w: torch.Tensor, rotation_w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(rotation_w.transpose(-1, -2), (points_w - position_w[:, None, :]).unsqueeze(-1)).squeeze(-1)

__all__ = ["quat_to_euler", "wrap_euler_delta", "rotate_world_to_object"]
