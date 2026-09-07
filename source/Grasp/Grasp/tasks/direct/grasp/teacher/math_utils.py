"""
Teacher 数学工具。
本文件只包含纯 torch / IsaacLab quaternion 运算：
1. Tensor 逐元素上下限裁剪。
2. 对象局部点云 -> 世界坐标。
3. 世界坐标点云 -> 对象局部坐标。
GPU 约定：
    - 本文件不创建 CPU Tensor。
    - 不使用 NumPy。
    - 不调用 .cpu() / .numpy()。
    - 输入 Tensor 在 CUDA 上时，所有计算始终在同一 CUDA device 上完成。
    - Teacher Env 已在 env.py / env_cfg.py 中强制使用 CUDA，因此这里不重复
      做 device 检查，避免每个 step 增加无意义的 Python 分支。
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def clamp_tensor(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """逐元素把 value 限制在 [lower, upper]。
    数学形式：
        result = max(min(value, upper), lower)
    典型用途：
        - FR3 主动关节位置目标裁剪。
        - Inspire 主动/被动 mimic 关节目标裁剪
    value / lower / upper 都是 CUDA Tensor 时，
    torch.minimum / torch.maximum 会直接在 GPU 上执行。
    """

    return torch.maximum(
        torch.minimum(value, upper),
        lower,
    )


def transform_points_to_world(
    points_object: torch.Tensor,
    object_pos_world: torch.Tensor,
    object_quat_world: torch.Tensor,
) -> torch.Tensor:
    """把对象局部坐标系中的批量点云变换到世界坐标系。
    输入：
        points_object:
            [B, N, 3]
            B 个环境，每个环境 N 个对象局部点。
        object_pos_world:
            [B, 3]
            每个对象在世界坐标系中的位置
        object_quat_world:
            [B, 4]
            每个对象在世界坐标系中的四元数，IsaacLab 格式为 (w, x, y, z)。

    公式：
        p_world = R_world_object * p_object + t_world_object

    GPU 实现说明：
        object_quat_world[:, None, :] 先扩展成 [B, N, 4]。
        expand() 只是 view，不复制 N 份四元数数据。
        reshape() 后一次性调用 quat_apply()，整个 batch 在 CUDA 上完成。
    """

    batch_size, num_points, _ = points_object.shape

    # [B, 4] -> [B, N, 4]
    # expand 不复制实际数据，只建立广播 view。
    expanded_quat = object_quat_world[:, None, :].expand(
        batch_size,
        num_points,
        4,
    )

    # 批量执行：
    #     R * p_object
    rotated_points_world = quat_apply(
        expanded_quat.reshape(-1, 4),
        points_object.reshape(-1, 3),
    ).reshape(
        batch_size,
        num_points,
        3,
    )

    # 加上对象在世界坐标系中的平移：
    #     p_world = R * p_object + t
    return (
        rotated_points_world
        + object_pos_world[:, None, :]
    )


def transform_points_to_object(
    points_world: torch.Tensor,
    object_pos_world: torch.Tensor,
    object_quat_world: torch.Tensor,
) -> torch.Tensor:
    """把世界坐标系中的批量点云变换回对象局部坐标系。

    输入：
        points_world:
            [B, N, 3]

        object_pos_world:
            [B, 3]

        object_quat_world:
            [B, 4]

    公式：
        p_object = R_world_object^T * (p_world - t_world_object)

    quat_apply_inverse() 等价于应用四元数对应旋转的逆旋转。
    整个计算只使用 CUDA Tensor。
    """

    batch_size, num_points, _ = points_world.shape

    # 先去掉世界坐标中的对象平移。
    centered_points_world = (
        points_world
        - object_pos_world[:, None, :]
    )

    # [B, 4] -> [B, N, 4]
    expanded_quat = object_quat_world[:, None, :].expand(
        batch_size,
        num_points,
        4,
    )

    # 批量执行逆旋转：
    #     p_object = R^T * centered_world
    return quat_apply_inverse(
        expanded_quat.reshape(-1, 4),
        centered_points_world.reshape(-1, 3),
    ).reshape(
        batch_size,
        num_points,
        3,
    )


__all__ = [
    "clamp_tensor",
    "transform_points_to_world",
    "transform_points_to_object",
]
