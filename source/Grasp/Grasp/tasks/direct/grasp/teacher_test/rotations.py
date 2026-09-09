from __future__ import annotations

import numpy as np
import torch

def axisangle2quat(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = angle * 0.5
    small = np.abs(angle) < 1e-6
    ratio = np.empty_like(angle)
    ratio[~small] = np.sin(half[~small]) / angle[~small]
    ratio[small] = 0.5 - angle[small] * angle[small] / 48.0
    return np.concatenate((np.cos(half), axis_angle * ratio), axis=-1)

def quat2mat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
         2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
         2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=-1
    ).reshape(quat.shape[:-1] + (3, 3))

def euler2mat(euler: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(np.asarray(euler), -1, 0)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.stack((cy * cz, -cy * sz, sy,
                     sx * sy * cz + cx * sz, cx * cz - sx * sy * sz, -sx * cy,
                     sx * sz - cx * sy * cz, cx * sy * sz + sx * cz, cx * cy), axis=-1).reshape(np.asarray(euler).shape[:-1] + (3, 3))

def mat2euler(mat: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mat)
    y = np.arcsin(np.clip(matrix[..., 0, 2], -1.0, 1.0))
    x = np.arctan2(-matrix[..., 1, 2], matrix[..., 2, 2])
    z = np.arctan2(-matrix[..., 0, 1], matrix[..., 0, 0])
    return np.stack((x, y, z), axis=-1)

def quat_to_mat_torch(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), dim=-1).reshape(quat.shape[:-1] + (3, 3))

__all__ = ["axisangle2quat", "quat2mat", "euler2mat", "mat2euler", "quat_to_mat_torch"]
