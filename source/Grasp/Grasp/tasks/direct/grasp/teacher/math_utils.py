'''
tensor 逐元素上下限裁剪。
对象局部点云变换到世界坐标。
世界坐标点云变换回对象局部坐标。
'''

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def clamp_tensor(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    return torch.maximum(
        torch.minimum(value, upper),
        lower,
    )


def transform_points_to_world(
    points_object: torch.Tensor,
    object_pos_world: torch.Tensor,
    object_quat_world: torch.Tensor,
) -> torch.Tensor:
    if points_object.ndim != 3 or points_object.shape[-1] != 3:
        raise ValueError(
            "points_object must have shape [B, N, 3], "
            f"got {tuple(points_object.shape)}"
        )

    batch_size, num_points, _ = points_object.shape
    expanded_quat = object_quat_world[:, None, :].expand(
        batch_size,
        num_points,
        4,
    )
    rotated = quat_apply(
        expanded_quat.reshape(-1, 4),
        points_object.reshape(-1, 3),
    ).reshape(batch_size, num_points, 3)
    return rotated + object_pos_world[:, None, :]


def transform_points_to_object(
    points_world: torch.Tensor,
    object_pos_world: torch.Tensor,
    object_quat_world: torch.Tensor,
) -> torch.Tensor:
    if points_world.ndim != 3 or points_world.shape[-1] != 3:
        raise ValueError(
            "points_world must have shape [B, N, 3], "
            f"got {tuple(points_world.shape)}"
        )

    batch_size, num_points, _ = points_world.shape
    expanded_quat = object_quat_world[:, None, :].expand(
        batch_size,
        num_points,
        4,
    )
    centered = points_world - object_pos_world[:, None, :]
    return quat_apply_inverse(
        expanded_quat.reshape(-1, 4),
        centered.reshape(-1, 3),
    ).reshape(batch_size, num_points, 3)