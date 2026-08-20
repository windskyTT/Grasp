from __future__ import annotations

import os

import numpy as np
import torch
from isaaclab.utils.math import quat_conjugate, quat_mul


def tensor_clamp(tensor: torch.Tensor, min_tensor: torch.Tensor, max_tensor: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(tensor, max_tensor), min_tensor)


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (x - lower) / (upper - lower + 1e-8) - 1.0

COLORS_DICT = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "gray": [0.5, 0.5, 0.5],
    "light_gray": [0.75, 0.75, 0.75],
    "dark_gray": [0.25, 0.25, 0.25],
    "orange": [1.0, 0.65, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "pink": [1.0, 0.75, 0.8],
    "brown": [0.65, 0.16, 0.16],
    "olive": [0.5, 0.5, 0.0],
    "teal": [0.0, 0.5, 0.5],
    "navy": [0.0, 0.0, 0.5],
    "maroon": [0.5, 0.0, 0.0],
    "lime": [0.75, 1.0, 0.0],
    "gold": [1.0, 0.84, 0.0],
    "silver": [0.75, 0.75, 0.75],
    "bronze": [0.8, 0.5, 0.2],
    "sky_blue": [0.53, 0.81, 0.92],
    "forest_green": [0.13, 0.55, 0.13],
    "violet": [0.93, 0.51, 0.93],
    "coral": [1.0, 0.5, 0.31],
    "salmon": [0.98, 0.5, 0.45],
    "turquoise": [0.25, 0.88, 0.82],
    "indigo": [0.29, 0.0, 0.51],
    "beige": [0.96, 0.96, 0.86],
    "ivory": [1.0, 1.0, 0.94],
}
def batch_linear_interpolate_poses(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
    max_trans_step: float,
    max_rot_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate batched poses in IsaacLab format.

    pose1 and pose2 use [x, y, z, qw, qx, qy, qz].
    DemoGrasp source used [x, y, z, qx, qy, qz, qw], so callers must convert reference quaternions before calling.
    """
    batch_size = pose1.shape[0]
    device = pose1.device

    p1, q1 = pose1[:, :3], pose1[:, 3:]
    p2, q2 = pose2[:, :3], pose2[:, 3:]

    delta_p = p2 - p1
    trans_dist = torch.norm(delta_p, dim=1)
    n_trans = torch.ceil(trans_dist / max_trans_step).long().clamp(min=1)

    q_dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
    theta = 2.0 * torch.acos(q_dot)
    n_rot = torch.ceil(theta / max_rot_step).long().clamp(min=1)

    n_steps = torch.maximum(n_trans, n_rot)
    t_max = int(n_steps.max().item())
    step_idx = torch.arange(t_max + 1, device=device).expand(batch_size, -1)
    valid_mask = step_idx <= n_steps.unsqueeze(1)
    t = step_idx.float() / n_steps.unsqueeze(1).clamp(min=1)
    t = t * valid_mask.float()

    interp_p = p1.unsqueeze(1) + t.unsqueeze(-1) * delta_p.unsqueeze(1)
    interp_q = quat_slerp_batch(
        q1.unsqueeze(1).expand(-1, t_max + 1, -1),
        q2.unsqueeze(1).expand(-1, t_max + 1, -1),
        t.unsqueeze(-1),
    )
    return torch.cat([interp_p, interp_q], dim=-1), n_steps


def load_object_point_clouds(object_files: list[str], asset_root: str) -> list[np.ndarray]:
    point_clouds = []
    for object_file in object_files:
        parts = object_file.split("/")
        if len(parts) != 3:
            raise ValueError(f"Expected ObjDatasetName/urdf/name.urdf, got: {object_file}")
        point_cloud_file = os.path.join(parts[0], "pointclouds", parts[-1].replace(".urdf", ".npy"))
        point_clouds.append(np.load(os.path.join(asset_root, point_cloud_file)))
    return point_clouds


def transform_points(quat: torch.Tensor, pt_input: torch.Tensor) -> torch.Tensor:
    """Rotate points with IsaacLab wxyz quaternions.

    Args:
        quat: Rotation quaternion in [w, x, y, z] format.
        pt_input: Point quaternion in [0, x, y, z] format.

    Returns:
        Rotated point coordinates with shape [..., 3].
    """
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if pt_new.dim() == 3:
        return pt_new[:, :, 1:4]
    return pt_new[:, 1:4]

def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    device: torch.device | str,
    init: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Sample point indices with farthest point sampling.

    Args:
        xyz: Point cloud tensor with shape [B, N, 3].
        npoint: Number of sampled points.
        device: Target torch device.
        init: Optional initial farthest index for each batch item.

    Returns:
        Sampled point indices with shape [B, npoint].
    """
    batch_size, num_points, channels = xyz.size()
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1.0e10
    if init is not None:
        farthest = torch.as_tensor(init, dtype=torch.long, device=device).reshape(batch_size)
    else:
        farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, channels)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids

def index_points(points: torch.Tensor, idx: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Index batched point clouds.

    Args:
        points: Input points with shape [B, N, C].
        idx: Indices with shape [B, S] or higher.
        device: Target torch device.

    Returns:
        Gathered points with shape [B, S, C] for 2D indices.
    """
    batch_size = points.size(0)
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def quat_slerp_batch(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(dot < 0.0, -q2, q2)
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6

    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(t * theta) / (sin_theta + 1e-8)
    out = w1 * q1 + w2 * q2

    lerp = (1.0 - t) * q1 + t * q2
    out = torch.where(small, lerp, out)
    return out / torch.norm(out, dim=-1, keepdim=True).clamp_min(1e-8)