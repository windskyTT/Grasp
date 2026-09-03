from __future__ import annotations

from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from isaaclab.utils.math import (
    normalize,
    quat_from_angle_axis,
)

from .observations import (
    TeacherObservationFeatures,
    transform_object_points_to_world,
)


def make_teacher_affordance_marker_cfg() -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path="/Visuals/TeacherAffordance",
        markers={
            "hand_body": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.1, 0.1),
                ),
            ),
            "nearest_point": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 0.1),
                ),
            ),
            "connecting_line": sim_utils.CylinderCfg(
                radius=0.0015,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.8, 0.9),
                ),
            ),
            "affordance_center": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.8, 0.0),
                ),
            ),
            "palm_center": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.3, 1.0),
                ),
            ),
        },
    )


def _connecting_line_pose(
    start: torch.Tensor,
    end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = end - start
    length = torch.linalg.vector_norm(direction, dim=-1)
    midpoint = 0.5 * (start + end)
    direction_unit = normalize(direction)
    z_axis = torch.tensor(
        (0.0, 0.0, 1.0),
        dtype=start.dtype,
        device=start.device,
    ).expand_as(direction_unit)
    rotation_axis = torch.linalg.cross(z_axis, direction_unit)
    rotation_axis_norm = torch.linalg.vector_norm(
        rotation_axis,
        dim=-1,
    )
    non_parallel = rotation_axis_norm > 1.0e-6
    fallback_axis = torch.tensor(
        (1.0, 0.0, 0.0),
        dtype=start.dtype,
        device=start.device,
    ).expand_as(rotation_axis)
    rotation_axis = torch.where(
        non_parallel.unsqueeze(-1),
        normalize(rotation_axis),
        fallback_axis,
    )
    angle = torch.acos(
        torch.clamp(
            torch.sum(z_axis * direction_unit, dim=-1),
            -1.0,
            1.0,
        )
    )
    orientation = quat_from_angle_axis(angle, rotation_axis)
    return midpoint, orientation, length


class TeacherAffordanceVisualizer:
    def __init__(self) -> None:
        self.markers = VisualizationMarkers(
            make_teacher_affordance_marker_cfg()
        )

    def update(
        self,
        env: Any,
        features: TeacherObservationFeatures,
    ) -> None:
        hand_body_position_w = env.robot.data.body_pos_w[
            :, env.hand_body_ids, :
        ]
        nearest_point_w = features.nearest_affordance_point_world
        line_position, line_orientation, line_length = (
            _connecting_line_pose(
                hand_body_position_w.reshape(-1, 3),
                nearest_point_w.reshape(-1, 3),
            )
        )

        top_position_w = env.object.data.body_pos_w[
            :, env.object_top_body_id, :
        ]
        top_quaternion_w = env.object.data.body_quat_w[
            :, env.object_top_body_id, :
        ]
        top_points_w = transform_object_points_to_world(
            env.affordance_data.top_points_object,
            top_position_w,
            top_quaternion_w,
        )
        affordance_center_w = top_points_w.mean(dim=1)
        palm_center_global_w = (
            features.palm_center_world + env.scene.env_origins
        )

        hand_flat = hand_body_position_w.reshape(-1, 3)
        nearest_flat = nearest_point_w.reshape(-1, 3)
        translations = torch.cat(
            (
                hand_flat,
                nearest_flat,
                line_position,
                affordance_center_w,
                palm_center_global_w,
            ),
            dim=0,
        )

        identity_quaternion = torch.zeros(
            (translations.shape[0], 4),
            dtype=translations.dtype,
            device=translations.device,
        )
        identity_quaternion[:, 0] = 1.0
        orientations = identity_quaternion
        line_start = hand_flat.shape[0] + nearest_flat.shape[0]
        line_end = line_start + line_position.shape[0]
        orientations[line_start:line_end] = line_orientation

        scales = torch.ones_like(translations)
        scales[line_start:line_end, 2] = line_length

        marker_indices = torch.cat(
            (
                torch.zeros(
                    hand_flat.shape[0],
                    dtype=torch.long,
                    device=env.device,
                ),
                torch.ones(
                    nearest_flat.shape[0],
                    dtype=torch.long,
                    device=env.device,
                ),
                torch.full(
                    (line_position.shape[0],),
                    2,
                    dtype=torch.long,
                    device=env.device,
                ),
                torch.full(
                    (env.num_envs,),
                    3,
                    dtype=torch.long,
                    device=env.device,
                ),
                torch.full(
                    (env.num_envs,),
                    4,
                    dtype=torch.long,
                    device=env.device,
                ),
            ),
            dim=0,
        )
        self.markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )


__all__ = [
    "TeacherAffordanceVisualizer",
    "make_teacher_affordance_marker_cfg",
]
