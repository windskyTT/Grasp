from __future__ import annotations

import numpy as np
import torch

def sample_object_xy(non_uniform_sampling: bool) -> tuple[float, float]:
    while True:
        if non_uniform_sampling and np.random.random() >= 0.5:
            normalized_angle = np.random.beta(0.5, 0.5)
            angle = -0.7 * np.pi + normalized_angle * 0.4 * np.pi
            distance = 0.45 + np.random.beta(0.5, 0.5) * 0.3
        else:
            angle = np.random.uniform(-0.7*np.pi, -0.3*np.pi)
            distance = np.random.uniform(0.45, 0.75)
        x, y = distance*np.cos(angle), distance*np.sin(angle)
        if -0.25 < x < 0.25:
            return float(x), float(y)

def reset_object_pose(env, env_id: int, lowest_point: float) -> np.ndarray:
    x, y = sample_object_xy(env.cfg.non_uniform_sampling)
    yaw = np.random.uniform(-np.pi, np.pi)
    quat = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)], dtype=np.float32)
    pose = np.concatenate((np.array([x, y, 0.773-lowest_point], dtype=np.float32), quat))
    env.object_initial_position[env_id] = torch.as_tensor(pose[:3], device=env.device)
    return pose

__all__ = ["sample_object_xy", "reset_object_pose"]
