from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.utils.math import matrix_from_quat


@dataclass(frozen=True)
class TeacherObservationSpec:
    active_qpos_dim: int = 13
    joint_target_error_dim: int = 13
    affordance_contact_dim: int = 13
    affordance_impulse_dim: int = 13
    hand_body_height_dim: int = 13
    arm_body_height_dim: int = 6
    palm_center_dim: int = 3
    wrist_delta_euler_dim: int = 3
    wrist_euler_dim: int = 3
    affordance_vector_dim: int = 39

    @property
    def base_dim(self) -> int:
        return (
            self.active_qpos_dim
            + self.joint_target_error_dim
            + self.affordance_contact_dim
            + self.affordance_impulse_dim
            + self.hand_body_height_dim
            + self.arm_body_height_dim
            + self.palm_center_dim
            + self.wrist_delta_euler_dim
            + self.wrist_euler_dim
        )

    @property
    def teacher_dim(self) -> int:
        return self.base_dim + self.affordance_vector_dim


@dataclass(frozen=True)
class TeacherObservationFeatures:
    active_qpos: torch.Tensor
    joint_target_error: torch.Tensor
    affordance_contact: torch.Tensor
    affordance_impulse: torch.Tensor
    hand_body_height: torch.Tensor
    arm_body_height: torch.Tensor
    palm_center_world: torch.Tensor
    wrist_delta_euler: torch.Tensor
    wrist_euler: torch.Tensor
    nearest_affordance_vector_world: torch.Tensor
    nearest_affordance_distance: torch.Tensor
    nearest_affordance_point_world: torch.Tensor


TEACHER_OBSERVATION_SPEC = TeacherObservationSpec()


def transform_object_points_to_world(
    points_object: torch.Tensor,
    body_position_world: torch.Tensor,
    body_quaternion_world: torch.Tensor,
) -> torch.Tensor:
    body_rotation_world = matrix_from_quat(
        body_quaternion_world
    )
    rotated_points_world = torch.matmul(
        body_rotation_world.unsqueeze(1),
        points_object.unsqueeze(-1),
    ).squeeze(-1)
    return rotated_points_world + body_position_world.unsqueeze(1)


def compute_affordance_geometry(
    hand_body_positions_world: torch.Tensor,
    top_points_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = torch.cdist(
        hand_body_positions_world,
        top_points_world,
    )
    nearest_distance, nearest_index = torch.min(
        distances,
        dim=2,
    )
    nearest_point_world = torch.gather(
        top_points_world,
        1,
        nearest_index.unsqueeze(-1).expand(-1, -1, 3),
    )
    nearest_vector_world = (
        nearest_point_world - hand_body_positions_world
    )
    return (
        nearest_distance,
        nearest_point_world,
        nearest_vector_world,
    )


def unwrap_euler_near_previous(
    current_euler: torch.Tensor,
    previous_euler: torch.Tensor,
) -> torch.Tensor:
    difference = current_euler - previous_euler
    return torch.where(
        difference > torch.pi,
        current_euler - 2.0 * torch.pi,
        torch.where(
            difference < -torch.pi,
            current_euler + 2.0 * torch.pi,
            current_euler,
        ),
    )


def build_teacher_observation(
    *,
    spec: TeacherObservationSpec,
    active_qpos: torch.Tensor,
    joint_target_error: torch.Tensor,
    affordance_contact: torch.Tensor,
    affordance_impulse: torch.Tensor,
    hand_body_height: torch.Tensor,
    arm_body_height: torch.Tensor,
    palm_center_world: torch.Tensor,
    wrist_delta_euler: torch.Tensor,
    wrist_euler: torch.Tensor,
    affordance_vector_world: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    batch_size = active_qpos.shape[0]
    flattened_affordance_vector = (
        affordance_vector_world.reshape(
            batch_size,
            spec.affordance_vector_dim,
        )
    )
    observation = torch.cat(
        (
            active_qpos,
            joint_target_error,
            affordance_contact,
            affordance_impulse,
            hand_body_height,
            arm_body_height,
            palm_center_world,
            wrist_delta_euler,
            wrist_euler,
            flattened_affordance_vector,
        ),
        dim=-1,
    )
    return torch.clamp(
        observation,
        min=-clip_value,
        max=clip_value,
    )


__all__ = [
    "TeacherObservationSpec",
    "TeacherObservationFeatures",
    "TEACHER_OBSERVATION_SPEC",
    "transform_object_points_to_world",
    "compute_affordance_geometry",
    "unwrap_euler_near_previous",
    "build_teacher_observation",
]
