from __future__ import annotations

from collections.abc import Callable

import torch

from .env_cfg import TeacherResetCfg


TeacherResetCandidateSampler = Callable[
    [torch.Tensor],
    torch.Tensor,
]
TeacherSelfCollisionChecker = Callable[
    [torch.Tensor],
    torch.Tensor,
]


def _sample_angle_and_distance_unit_interval(
    count: int,
    cfg: TeacherResetCfg,
    device: torch.device | str,
    uniform_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    uniform_angle = torch.rand(count, device=device)
    uniform_distance = torch.rand(count, device=device)
    if uniform_only or not cfg.non_uniform_sampling:
        return uniform_angle, uniform_distance

    beta_distribution = torch.distributions.Beta(
        torch.tensor(
            cfg.beta_alpha,
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            cfg.beta_beta,
            dtype=torch.float32,
            device=device,
        ),
    )
    beta_angle = beta_distribution.sample((count,))
    beta_distance = beta_distribution.sample((count,))
    use_uniform = (
        torch.rand(count, device=device)
        < cfg.uniform_sampling_probability
    )
    angle_unit = torch.where(
        use_uniform,
        uniform_angle,
        beta_angle,
    )
    distance_unit = torch.where(
        use_uniform,
        uniform_distance,
        beta_distance,
    )
    return angle_unit, distance_unit


def sample_teacher_object_xy(
    count: int,
    cfg: TeacherResetCfg,
    workspace_center_y: float,
    device: torch.device | str,
    uniform_only: bool = False,
) -> torch.Tensor:
    accepted_xy: list[torch.Tensor] = []
    accepted_count = 0

    while accepted_count < count:
        remaining = count - accepted_count
        draw_count = max(remaining * 2, 64)
        angle_unit, distance_unit = (
            _sample_angle_and_distance_unit_interval(
                count=draw_count,
                cfg=cfg,
                device=device,
                uniform_only=uniform_only,
            )
        )

        source_angle = (
            cfg.angle_range[0]
            + angle_unit
            * (cfg.angle_range[1] - cfg.angle_range[0])
        )
        distance = (
            cfg.distance_range[0]
            + distance_unit
            * (
                cfg.distance_range[1]
                - cfg.distance_range[0]
            )
        )
        fr3_angle = (
            source_angle + cfg.source_to_fr3_angle_offset
        )
        local_x = distance * torch.cos(fr3_angle)
        local_y = distance * torch.sin(fr3_angle)
        accepted = (
            (local_y > cfg.local_y_range[0])
            & (local_y < cfg.local_y_range[1])
        )

        candidate_xy = torch.stack(
            (
                local_x,
                local_y + workspace_center_y,
            ),
            dim=-1,
        )[accepted]
        take_count = min(remaining, candidate_xy.shape[0])
        if take_count > 0:
            accepted_xy.append(candidate_xy[:take_count])
            accepted_count += take_count

    return torch.cat(accepted_xy, dim=0)


def sample_teacher_object_pose(
    env_origins: torch.Tensor,
    lowest_points: torch.Tensor,
    support_height: float,
    workspace_center_y: float,
    cfg: TeacherResetCfg,
    uniform_only: bool = False,
) -> torch.Tensor:
    count = env_origins.shape[0]
    object_xy = sample_teacher_object_xy(
        count=count,
        cfg=cfg,
        workspace_center_y=workspace_center_y,
        device=env_origins.device,
        uniform_only=uniform_only,
    )
    yaw = torch.empty(
        count,
        dtype=torch.float32,
        device=env_origins.device,
    ).uniform_(cfg.yaw_range[0], cfg.yaw_range[1])

    object_pose_w = torch.zeros(
        (count, 7),
        dtype=torch.float32,
        device=env_origins.device,
    )
    object_pose_w[:, 0:2] = (
        object_xy + env_origins[:, 0:2]
    )
    object_pose_w[:, 2] = (
        env_origins[:, 2]
        + support_height
        - lowest_points.reshape(count)
    )
    object_pose_w[:, 3] = torch.cos(0.5 * yaw)
    object_pose_w[:, 6] = torch.sin(0.5 * yaw)
    return object_pose_w

def sample_teacher_stable_object_pose(
    env_origins: torch.Tensor,
    stable_states: torch.Tensor,
    support_height: float,
    source_support_height: float,
    clearance: float,
    workspace_center_y: float,
    cfg: TeacherResetCfg,
) -> torch.Tensor:
    count = env_origins.shape[0]
    object_xy = sample_teacher_object_xy(
        count=count,
        cfg=cfg,
        workspace_center_y=workspace_center_y,
        device=env_origins.device,
        uniform_only=True,
    )

    object_pose_w = torch.zeros(
        (count, 7),
        dtype=torch.float32,
        device=env_origins.device,
    )
    object_pose_w[:, 0:2] = (
        object_xy + env_origins[:, 0:2]
    )
    object_pose_w[:, 2] = (
        env_origins[:, 2]
        + support_height
        + stable_states[:, 2]
        - source_support_height
        + clearance
    )
    object_pose_w[:, 3:7] = stable_states[:, 3:7]
    return object_pose_w



def sample_collision_free_teacher_resets(
    env_ids: torch.Tensor,
    max_reset_rounds: int,
    sample_candidates: TeacherResetCandidateSampler,
    check_self_collision: TeacherSelfCollisionChecker,
) -> None:
    pending_env_ids = env_ids.clone()

    for _ in range(max_reset_rounds):
        candidate_valid = sample_candidates(pending_env_ids)
        invalid_env_ids = pending_env_ids[~candidate_valid]
        valid_env_ids = pending_env_ids[candidate_valid]

        if valid_env_ids.numel() > 0:
            self_collision = check_self_collision(valid_env_ids)
            collided_env_ids = valid_env_ids[self_collision]
        else:
            collided_env_ids = valid_env_ids

        pending_env_ids = torch.cat(
            (invalid_env_ids, collided_env_ids),
            dim=0,
        )
        if pending_env_ids.numel() == 0:
            return

    raise RuntimeError(
        "Teacher reset could not produce collision-free FR3 "
        "pregrasp states for env ids "
        f"{pending_env_ids.tolist()}"
    )


def reset_teacher_statistics(
    buffers: dict[str, torch.Tensor],
    env_ids: torch.Tensor,
) -> None:
    for value in buffers.values():
        value[env_ids] = 0.0


__all__ = [
    "TeacherResetCandidateSampler",
    "TeacherSelfCollisionChecker",
    "sample_teacher_object_xy",
    "sample_teacher_object_pose",
    "sample_collision_free_teacher_resets",
    "reset_teacher_statistics",
    "sample_teacher_stable_object_pose",

]