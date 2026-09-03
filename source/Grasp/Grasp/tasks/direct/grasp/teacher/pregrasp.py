from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_euler,
    quat_from_matrix,
)

from .env_cfg import TeacherPregraspCfg


TeacherKinematicsEvaluator = Callable[
    [torch.Tensor],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]


FR3_JOINT_ORIGINS_XYZ = (
    (0.0, 0.0, 0.333),
    (0.0, 0.0, 0.0),
    (0.0, -0.316, 0.0),
    (0.0825, 0.0, 0.0),
    (-0.0825, 0.384, 0.0),
    (0.0, 0.0, 0.0),
    (0.088, 0.0, 0.0),
)
FR3_JOINT_ORIGINS_RPY = (
    (0.0, 0.0, 0.0),
    (-0.5 * np.pi, 0.0, 0.0),
    (0.5 * np.pi, 0.0, 0.0),
    (0.5 * np.pi, 0.0, 0.0),
    (-0.5 * np.pi, 0.0, 0.0),
    (0.5 * np.pi, 0.0, 0.0),
    (0.5 * np.pi, 0.0, 0.0),
)
FR3_WRIST_OFFSET = (0.0, 0.0, 0.107)


@dataclass(frozen=True)
class TeacherPregraspGeometry:
    visible_points_world: np.ndarray
    visible_points_object: np.ndarray
    affordance_center_world: np.ndarray
    approach_direction_world: np.ndarray
    wrist_target_positions_world: np.ndarray
    wrist_target_rotations_world: np.ndarray
    projection_lengths: np.ndarray


@dataclass(frozen=True)
class TeacherIKResult:
    arm_qpos: torch.Tensor
    converged: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor


@dataclass(frozen=True)
class TeacherPregraspSelection:
    arm_qpos: torch.Tensor
    selected_candidate_indices: torch.Tensor
    valid: torch.Tensor


def compute_visible_top_points(
    top_mesh: trimesh.Trimesh,
    sampled_top_points_object: np.ndarray,
    object_position_world: np.ndarray,
    object_rotation_world: np.ndarray,
    camera_position_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point_count = sampled_top_points_object.shape[0]
    camera_position_object = (
        object_rotation_world.T
        @ (camera_position_world - object_position_world)
    )
    ray_origins = np.repeat(
        camera_position_object.reshape(1, 3),
        point_count,
        axis=0,
    )
    ray_directions = sampled_top_points_object - ray_origins
    ray_directions = ray_directions / np.linalg.norm(
        ray_directions,
        axis=-1,
        keepdims=True,
    )

    locations, index_ray, _ = (
        top_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,
        )
    )
    expected_ray_indices = np.arange(
        point_count,
        dtype=index_ray.dtype,
    )
    if not np.array_equal(
        np.sort(index_ray),
        expected_ray_indices,
    ):
        raise RuntimeError(
            "Every sampled top point must produce one visible ray hit"
        )

    visible_points_object = np.empty(
        (point_count, 3),
        dtype=np.float32,
    )
    visible_points_object[index_ray] = locations.astype(
        np.float32,
        copy=False,
    )
    visible_points_world = (
        visible_points_object @ object_rotation_world.T
        + object_position_world
    )
    return (
        visible_points_world.astype(np.float32, copy=False),
        visible_points_object,
    )


def sample_rot_mats(
    approach_direction_world: np.ndarray,
    candidate_count: int,
    visible_points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_vector = approach_direction_world.reshape(3)
    if abs(reference_vector[0]) < 0.9:
        temporary_vector = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    else:
        temporary_vector = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )

    first_perpendicular = np.cross(
        reference_vector,
        temporary_vector,
    )
    first_perpendicular = (
        first_perpendicular
        / np.linalg.norm(first_perpendicular)
    )
    second_perpendicular = np.cross(
        reference_vector,
        first_perpendicular,
    )
    second_perpendicular = (
        second_perpendicular
        / np.linalg.norm(second_perpendicular)
    )

    thetas = np.linspace(
        0.0,
        2.0 * np.pi,
        candidate_count,
        endpoint=False,
    )
    perpendicular_vectors = np.zeros(
        (candidate_count, 3),
        dtype=np.float32,
    )
    for candidate_index, theta in enumerate(thetas):
        perpendicular_vector = (
            first_perpendicular * np.cos(theta)
            + second_perpendicular * np.sin(theta)
        )
        perpendicular_vector = (
            perpendicular_vector
            / np.linalg.norm(perpendicular_vector)
        )
        if perpendicular_vector[1] < 0.0:
            perpendicular_vector = -perpendicular_vector
        perpendicular_vectors[candidate_index] = (
            perpendicular_vector
        )

    centered_points = (
        visible_points_world
        - visible_points_world.mean(axis=0, keepdims=True)
    )
    projected_points = (
        centered_points @ perpendicular_vectors.T
    )
    projection_lengths = (
        projected_points.max(axis=0)
        - projected_points.min(axis=0)
    )

    rotation_matrices = np.zeros(
        (candidate_count, 3, 3),
        dtype=np.float32,
    )
    for candidate_index in range(candidate_count):
        y_direction = np.cross(
            reference_vector,
            perpendicular_vectors[candidate_index],
        )
        y_direction = y_direction / np.linalg.norm(y_direction)
        rotation_matrices[candidate_index] = -np.stack(
            (
                reference_vector,
                y_direction,
                perpendicular_vectors[candidate_index],
            ),
            axis=-1,
        )

    return (
        rotation_matrices,
        projection_lengths.astype(np.float32, copy=False),
    )


def build_teacher_pregrasp_geometry(
    top_mesh: trimesh.Trimesh,
    sampled_top_points_object: np.ndarray,
    object_position_world: np.ndarray,
    object_rotation_world: np.ndarray,
    env_origin_world: np.ndarray,
    palm_offset_body: np.ndarray,
    cfg: TeacherPregraspCfg,
) -> TeacherPregraspGeometry:
    camera_position_world = (
        np.asarray(
            cfg.camera_position,
            dtype=np.float32,
        )
        + env_origin_world
    )
    visible_points_world, visible_points_object = (
        compute_visible_top_points(
            top_mesh=top_mesh,
            sampled_top_points_object=(
                sampled_top_points_object
            ),
            object_position_world=object_position_world,
            object_rotation_world=object_rotation_world,
            camera_position_world=camera_position_world,
        )
    )
    affordance_center_world = visible_points_world.mean(axis=0)

    if cfg.top_grasp:
        approach_direction_world = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    else:
        approach_direction_world = (
            camera_position_world - affordance_center_world
        )
        approach_direction_world = (
            approach_direction_world
            / np.linalg.norm(approach_direction_world)
        )

    palm_center_target_world = (
        affordance_center_world
        + cfg.approach_distance * approach_direction_world
    )
    wrist_target_rotations_world, projection_lengths = (
        sample_rot_mats(
            approach_direction_world=approach_direction_world,
            candidate_count=cfg.candidate_count,
            visible_points_world=visible_points_world,
        )
    )
    rotated_palm_offsets_world = np.einsum(
        "nij,j->ni",
        wrist_target_rotations_world,
        palm_offset_body,
    )
    wrist_target_positions_world = (
        palm_center_target_world.reshape(1, 3)
        - rotated_palm_offsets_world
    ).astype(np.float32, copy=False)

    return TeacherPregraspGeometry(
        visible_points_world=visible_points_world,
        visible_points_object=visible_points_object,
        affordance_center_world=affordance_center_world,
        approach_direction_world=approach_direction_world,
        wrist_target_positions_world=(
            wrist_target_positions_world
        ),
        wrist_target_rotations_world=(
            wrist_target_rotations_world
        ),
        projection_lengths=projection_lengths,
    )


@torch.no_grad()
def compute_fr3_wrist_kinematics(
    arm_qpos: torch.Tensor,
    robot_base_positions_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = arm_qpos.shape[0]
    dtype = arm_qpos.dtype
    device = arm_qpos.device

    joint_origins_xyz = torch.tensor(
        FR3_JOINT_ORIGINS_XYZ,
        dtype=dtype,
        device=device,
    )
    joint_origins_rpy = torch.tensor(
        FR3_JOINT_ORIGINS_RPY,
        dtype=dtype,
        device=device,
    )
    joint_origin_rotations = matrix_from_euler(
        joint_origins_rpy,
        "XYZ",
    )
    wrist_offset = torch.tensor(
        FR3_WRIST_OFFSET,
        dtype=dtype,
        device=device,
    )

    position_world = robot_base_positions_world.clone()
    rotation_world = torch.eye(
        3,
        dtype=dtype,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1, -1).clone()
    joint_positions_world: list[torch.Tensor] = []
    joint_axes_world: list[torch.Tensor] = []

    for joint_index in range(arm_qpos.shape[1]):
        position_world = position_world + torch.bmm(
            rotation_world,
            joint_origins_xyz[joint_index]
            .view(1, 3, 1)
            .expand(batch_size, -1, -1),
        ).squeeze(-1)
        rotation_world = torch.bmm(
            rotation_world,
            joint_origin_rotations[joint_index]
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
        )
        joint_positions_world.append(position_world)
        joint_axes_world.append(rotation_world[:, :, 2])

        joint_angle = arm_qpos[:, joint_index]
        cosine = torch.cos(joint_angle)
        sine = torch.sin(joint_angle)
        zero = torch.zeros_like(joint_angle)
        one = torch.ones_like(joint_angle)
        joint_rotation = torch.stack(
            (
                cosine,
                -sine,
                zero,
                sine,
                cosine,
                zero,
                zero,
                zero,
                one,
            ),
            dim=-1,
        ).reshape(batch_size, 3, 3)
        rotation_world = torch.bmm(
            rotation_world,
            joint_rotation,
        )

    wrist_position_world = position_world + torch.bmm(
        rotation_world,
        wrist_offset.view(1, 3, 1).expand(batch_size, -1, -1),
    ).squeeze(-1)
    wrist_quaternion_world = quat_from_matrix(rotation_world)

    joint_positions_world_tensor = torch.stack(
        joint_positions_world,
        dim=1,
    )
    joint_axes_world_tensor = torch.stack(
        joint_axes_world,
        dim=1,
    )
    linear_jacobian = torch.linalg.cross(
        joint_axes_world_tensor,
        wrist_position_world.unsqueeze(1)
        - joint_positions_world_tensor,
        dim=-1,
    ).transpose(1, 2)
    angular_jacobian = joint_axes_world_tensor.transpose(1, 2)
    wrist_jacobian_world = torch.cat(
        (linear_jacobian, angular_jacobian),
        dim=1,
    )
    return (
        wrist_position_world,
        wrist_quaternion_world,
        wrist_jacobian_world,
    )


def compute_damped_least_squares_delta(
    jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    jacobian_transpose = jacobian.transpose(1, 2)
    identity = torch.eye(
        6,
        dtype=jacobian.dtype,
        device=jacobian.device,
    ).unsqueeze(0)
    damped_task_matrix = (
        jacobian @ jacobian_transpose
        + damping * damping * identity
    )
    solved_error = torch.linalg.solve(
        damped_task_matrix,
        pose_error.unsqueeze(-1),
    )
    return (
        jacobian_transpose @ solved_error
    ).squeeze(-1)


@torch.no_grad()
def solve_fr3_dls_ik(
    initial_arm_qpos: torch.Tensor,
    target_wrist_position_world: torch.Tensor,
    target_wrist_quaternion_world: torch.Tensor,
    arm_lower_limits: torch.Tensor,
    arm_upper_limits: torch.Tensor,
    cfg: TeacherPregraspCfg,
    evaluate_kinematics: TeacherKinematicsEvaluator,
) -> TeacherIKResult:
    arm_qpos = initial_arm_qpos.clone()

    for _ in range(cfg.ik_max_iterations):
        (
            wrist_position_world,
            wrist_quaternion_world,
            wrist_jacobian_world,
        ) = evaluate_kinematics(arm_qpos)
        position_error, rotation_error = compute_pose_error(
            wrist_position_world,
            wrist_quaternion_world,
            target_wrist_position_world,
            target_wrist_quaternion_world,
            rot_error_type="axis_angle",
        )
        converged = (
            torch.linalg.vector_norm(position_error, dim=-1)
            <= cfg.ik_position_tolerance
        ) & (
            torch.linalg.vector_norm(rotation_error, dim=-1)
            <= cfg.ik_rotation_tolerance
        )

        pose_error = torch.cat(
            (position_error, rotation_error),
            dim=-1,
        )
        arm_delta = compute_damped_least_squares_delta(
            jacobian=wrist_jacobian_world,
            pose_error=pose_error,
            damping=cfg.ik_damping,
        )
        next_arm_qpos = (
            arm_qpos + cfg.ik_step_scale * arm_delta
        )
        next_arm_qpos = torch.maximum(
            torch.minimum(next_arm_qpos, arm_upper_limits),
            arm_lower_limits,
        )
        arm_qpos = torch.where(
            converged.unsqueeze(-1),
            arm_qpos,
            next_arm_qpos,
        )

    (
        wrist_position_world,
        wrist_quaternion_world,
        _,
    ) = evaluate_kinematics(arm_qpos)
    position_error, rotation_error = compute_pose_error(
        wrist_position_world,
        wrist_quaternion_world,
        target_wrist_position_world,
        target_wrist_quaternion_world,
        rot_error_type="axis_angle",
    )
    converged = (
        torch.linalg.vector_norm(position_error, dim=-1)
        <= cfg.ik_position_tolerance
    ) & (
        torch.linalg.vector_norm(rotation_error, dim=-1)
        <= cfg.ik_rotation_tolerance
    )
    return TeacherIKResult(
        arm_qpos=arm_qpos,
        converged=converged,
        position_error=position_error,
        rotation_error=rotation_error,
    )


def select_teacher_pregrasp_candidate(
    candidate_arm_qpos: torch.Tensor,
    candidate_converged: torch.Tensor,
    projection_lengths: torch.Tensor,
    cfg: TeacherPregraspCfg,
) -> TeacherPregraspSelection:
    posture_joint = candidate_arm_qpos[
        :, :, cfg.posture_joint_index
    ]
    posture_error = torch.abs(
        posture_joint - cfg.posture_joint_target
    )
    posture_limit_score = (
        torch.abs(posture_joint) - cfg.posture_limit_target
    ) * cfg.posture_score_coeff * cfg.posture_limit_score_coeff

    short_projection = projection_lengths < cfg.projection_limit
    has_short_feasible = torch.any(
        candidate_converged & short_projection,
        dim=1,
    )

    short_scores = (
        projection_lengths * cfg.length_score_coeff
        + posture_error * cfg.posture_score_coeff
        + posture_limit_score
    )

    large_scores = projection_lengths
    candidate_scores = torch.where(
        has_short_feasible.unsqueeze(-1),
        short_scores,
        large_scores,
    )
    eligible = candidate_converged & torch.where(
        has_short_feasible.unsqueeze(-1),
        short_projection,
        torch.ones_like(short_projection),
    )
    candidate_scores = torch.where(
        eligible,
        candidate_scores,
        torch.full_like(candidate_scores, torch.inf),
    )

    selected_candidate_indices = torch.argmin(
        candidate_scores,
        dim=1,
    )
    valid = torch.any(eligible, dim=1)
    batch_indices = torch.arange(
        candidate_arm_qpos.shape[0],
        device=candidate_arm_qpos.device,
    )
    selected_arm_qpos = candidate_arm_qpos[
        batch_indices,
        selected_candidate_indices,
    ].clone()
    selected_arm_qpos[~valid] = torch.nan
    selected_candidate_indices = (
        selected_candidate_indices.clone()
    )
    selected_candidate_indices[~valid] = -1

    return TeacherPregraspSelection(
        arm_qpos=selected_arm_qpos,
        selected_candidate_indices=(
            selected_candidate_indices
        ),
        valid=valid,
    )


__all__ = [
    "TeacherKinematicsEvaluator",
    "TeacherPregraspGeometry",
    "TeacherIKResult",
    "TeacherPregraspSelection",
    "compute_visible_top_points",
    "sample_rot_mats",
    "build_teacher_pregrasp_geometry",
    "compute_fr3_wrist_kinematics",
    "compute_damped_least_squares_delta",
    "solve_fr3_dls_ik",
    "select_teacher_pregrasp_candidate",
]
