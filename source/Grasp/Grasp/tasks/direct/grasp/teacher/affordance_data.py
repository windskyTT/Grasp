from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

from .object_set import (
    NEW_TRAINING_SET_ROOT,
    TEACHER_OBJECT_NAMES,
)


TEACHER_ROOT_PRIM_PATH = "/TeacherObject"
TEACHER_BOTTOM_BODY_NAME = "bottom"
TEACHER_TOP_BODY_NAME = "top"
TEACHER_OBJECT_JOINT_NAME = "rotation"
AFFORDANCE_POINT_COUNT = 200
NON_AFFORDANCE_POINT_COUNT = 200


@dataclass(frozen=True)
class TeacherAffordanceFiles:
    object_name: str
    dynamic_urdf_path: Path
    top_mesh_path: Path
    bottom_mesh_path: Path
    lowest_point_path: Path
    top_usd_path: Path
    bottom_usd_path: Path


@dataclass(frozen=True)
class TeacherAffordanceData:
    object_names: tuple[str, ...]
    env_object_indices_cpu: torch.Tensor
    env_object_indices: torch.Tensor
    unique_lowest_points_cpu: torch.Tensor
    unique_top_points_object_cpu: torch.Tensor
    unique_top_normals_object_cpu: torch.Tensor
    unique_bottom_points_object_cpu: torch.Tensor
    unique_bottom_normals_object_cpu: torch.Tensor
    top_meshes: tuple[trimesh.Trimesh, ...]
    env_top_meshes: tuple[trimesh.Trimesh, ...]
    lowest_points: torch.Tensor
    top_points_object: torch.Tensor
    top_normals_object: torch.Tensor
    bottom_points_object: torch.Tensor
    bottom_normals_object: torch.Tensor
    stable_states: torch.Tensor | None


def _make_affordance_files(
    dataset_root: Path,
    object_name: str,
) -> TeacherAffordanceFiles:
    object_dir = dataset_root / object_name
    return TeacherAffordanceFiles(
        object_name=object_name,
        dynamic_urdf_path=object_dir / f"{object_name}.urdf",
        top_mesh_path=object_dir / "top_watertight_tiny.obj",
        bottom_mesh_path=(
            object_dir / "bottom_watertight_tiny.obj"
        ),
        lowest_point_path=object_dir / "lowest_point_new.txt",
        top_usd_path=object_dir / f"{object_name}_top.usd",
        bottom_usd_path=(
            object_dir / f"{object_name}_bottom.usd"
        ),
    )


TEACHER_AFFORDANCE_FILES: tuple[TeacherAffordanceFiles, ...] = (
    tuple(
        _make_affordance_files(
            NEW_TRAINING_SET_ROOT,
            object_name,
        )
        for object_name in TEACHER_OBJECT_NAMES
    )
)


TEACHER_AFFORDANCE_FILES_BY_NAME: dict[str, TeacherAffordanceFiles] = {
    item.object_name: item
    for item in TEACHER_AFFORDANCE_FILES
}


def get_teacher_affordance_files(
    object_name: str,
) -> TeacherAffordanceFiles:
    return TEACHER_AFFORDANCE_FILES_BY_NAME[object_name]


def _sample_mesh_surface(
    mesh_path: Path,
    point_count: int,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(str(mesh_path))
    points, face_indices = trimesh.sample.sample_surface(
        mesh,
        point_count,
    )
    normals = mesh.face_normals[face_indices]
    return (
        mesh,
        np.asarray(points, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
    )


def _make_env_object_indices(
    num_envs: int,
    weighted_object_indices: tuple[int, ...],
) -> torch.Tensor:
    weighted_indices = torch.tensor(
        weighted_object_indices,
        dtype=torch.long,
        device="cpu",
    )
    cycle_count = (
        num_envs + weighted_indices.numel() - 1
    ) // weighted_indices.numel()
    return weighted_indices.repeat(cycle_count)[:num_envs].clone()


def load_teacher_affordance_data(
    num_envs: int,
    device: torch.device | str,
    dataset_root: str,
    object_names: tuple[str, ...],
    weighted_object_indices: tuple[int, ...],
    stable_state_paths: tuple[str, ...],
) -> TeacherAffordanceData:
    files_by_object = tuple(
        _make_affordance_files(
            Path(dataset_root),
            object_name,
        )
        for object_name in object_names
    )
    lowest_points: list[float] = []
    top_points: list[np.ndarray] = []
    top_normals: list[np.ndarray] = []
    bottom_points: list[np.ndarray] = []
    bottom_normals: list[np.ndarray] = []
    top_meshes: list[trimesh.Trimesh] = []

    for files in files_by_object:
        lowest_points.append(
            float(
                files.lowest_point_path.read_text(
                    encoding="utf-8"
                ).strip()
            )
        )
        top_mesh, sampled_top_points, sampled_top_normals = (
            _sample_mesh_surface(
                files.top_mesh_path,
                AFFORDANCE_POINT_COUNT,
            )
        )
        _, sampled_bottom_points, sampled_bottom_normals = (
            _sample_mesh_surface(
                files.bottom_mesh_path,
                NON_AFFORDANCE_POINT_COUNT,
            )
        )
        top_meshes.append(top_mesh)
        top_points.append(sampled_top_points)
        top_normals.append(sampled_top_normals)
        bottom_points.append(sampled_bottom_points)
        bottom_normals.append(sampled_bottom_normals)

    unique_lowest_points_cpu = torch.tensor(
        lowest_points,
        dtype=torch.float32,
        device="cpu",
    ).unsqueeze(-1)
    unique_top_points_object_cpu = torch.from_numpy(
        np.stack(top_points, axis=0)
    ).contiguous()
    unique_top_normals_object_cpu = torch.from_numpy(
        np.stack(top_normals, axis=0)
    ).contiguous()
    unique_bottom_points_object_cpu = torch.from_numpy(
        np.stack(bottom_points, axis=0)
    ).contiguous()
    unique_bottom_normals_object_cpu = torch.from_numpy(
        np.stack(bottom_normals, axis=0)
    ).contiguous()

    env_object_indices_cpu = _make_env_object_indices(
        num_envs,
        weighted_object_indices,
    )
    env_object_indices = env_object_indices_cpu.to(device=device)

    gathered_lowest_points_cpu = (
        unique_lowest_points_cpu.index_select(
            0,
            env_object_indices_cpu,
        )
    )
    gathered_top_points_cpu = (
        unique_top_points_object_cpu.index_select(
            0,
            env_object_indices_cpu,
        )
    )
    gathered_top_normals_cpu = (
        unique_top_normals_object_cpu.index_select(
            0,
            env_object_indices_cpu,
        )
    )
    gathered_bottom_points_cpu = (
        unique_bottom_points_object_cpu.index_select(
            0,
            env_object_indices_cpu,
        )
    )
    gathered_bottom_normals_cpu = (
        unique_bottom_normals_object_cpu.index_select(
            0,
            env_object_indices_cpu,
        )
    )

    top_mesh_tuple = tuple(top_meshes)
    env_top_meshes = tuple(
        top_mesh_tuple[object_index]
        for object_index in env_object_indices_cpu.tolist()
    )

    if stable_state_paths:
        unique_stable_states_cpu = torch.from_numpy(
            np.stack(
                [
                    np.load(path)[-1, :7].astype(np.float32)
                    for path in stable_state_paths
                ],
                axis=0,
            )
        ).contiguous()
        stable_states = unique_stable_states_cpu.index_select(
            0,
            env_object_indices_cpu,
        ).to(device=device)
    else:
        stable_states = None

    return TeacherAffordanceData(
        object_names=object_names,
        env_object_indices_cpu=env_object_indices_cpu,
        env_object_indices=env_object_indices,
        unique_lowest_points_cpu=unique_lowest_points_cpu,
        unique_top_points_object_cpu=(
            unique_top_points_object_cpu
        ),
        unique_top_normals_object_cpu=(
            unique_top_normals_object_cpu
        ),
        unique_bottom_points_object_cpu=(
            unique_bottom_points_object_cpu
        ),
        unique_bottom_normals_object_cpu=(
            unique_bottom_normals_object_cpu
        ),
        top_meshes=top_mesh_tuple,
        env_top_meshes=env_top_meshes,
        lowest_points=gathered_lowest_points_cpu.to(
            device=device
        ),
        top_points_object=gathered_top_points_cpu.to(
            device=device
        ),
        top_normals_object=gathered_top_normals_cpu.to(
            device=device
        ),
        bottom_points_object=gathered_bottom_points_cpu.to(
            device=device
        ),
        bottom_normals_object=gathered_bottom_normals_cpu.to(
            device=device
        ),
        stable_states=stable_states,
    )

__all__ = [
    "TEACHER_ROOT_PRIM_PATH",
    "TEACHER_BOTTOM_BODY_NAME",
    "TEACHER_TOP_BODY_NAME",
    "TEACHER_OBJECT_JOINT_NAME",
    "AFFORDANCE_POINT_COUNT",
    "NON_AFFORDANCE_POINT_COUNT",
    "TeacherAffordanceFiles",
    "TeacherAffordanceData",
    "TEACHER_AFFORDANCE_FILES",
    "TEACHER_AFFORDANCE_FILES_BY_NAME",
    "get_teacher_affordance_files",
    "load_teacher_affordance_data",
]
