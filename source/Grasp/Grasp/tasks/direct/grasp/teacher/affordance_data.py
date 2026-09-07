from __future__ import annotations

import ast
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import warp as wp

from .object_set import (
    NEW_TRAINING_SET_ROOT,
    TEACHER_OBJECT_NAMES,
)

# =============================================================================
# RobustDexGrasp Teacher 物体语义
# =============================================================================
TEACHER_ROOT_PRIM_PATH = "/TeacherObject"
TEACHER_BOTTOM_BODY_NAME = "bottom"
TEACHER_TOP_BODY_NAME = "top"
TEACHER_OBJECT_JOINT_NAME = "rotation"
# top / bottom 各采样 200 个表面点
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
class TeacherGpuMesh:
    """完全驻留在 GPU 上的一份三角网格。
    vertices:[V, 3], float32, CUDA
    faces:[F, 3], int64, CUDA；用于 PyTorch GPU indexing / surface sampling。
    warp_face_indices:[F * 3], int32, CUDA；Warp Mesh 使用的三角面索引。
        单独保存 tensor 是为了保证 Warp 引用的 CUDA 内存生命周期。
    warp_mesh:Warp 在 CUDA 上构建的 mesh。
    """
    vertices: torch.Tensor
    faces: torch.Tensor
    warp_face_indices: torch.Tensor
    warp_mesh: wp.Mesh

@dataclass(frozen=True)
class TeacherAffordanceData:
    """Teacher 运行期使用的全部物体几何数据。
    unique_*:
        每种唯一物体一份数据。
    不带 unique_ 的字段:
        已按照 env_object_indices gather 成 [num_envs, ...]，
        可以直接供并行环境使用。
    """
    object_names: tuple[str, ...]
    env_object_indices: torch.Tensor # [num_envs]，每个环境对应哪一种唯一物体
    # 每种唯一物体的数据。
    unique_lowest_points: torch.Tensor
    unique_top_points_object: torch.Tensor
    unique_top_normals_object: torch.Tensor
    unique_bottom_points_object: torch.Tensor
    unique_bottom_normals_object: torch.Tensor
    # CUDA / Warp mesh。后续 GPU ray casting 使用。
    unique_top_meshes: tuple[TeacherGpuMesh, ...]
    unique_bottom_meshes: tuple[TeacherGpuMesh, ...]
    # 按环境 采集 后的数据
    lowest_points: torch.Tensor
    top_points_object: torch.Tensor
    top_normals_object: torch.Tensor
    bottom_points_object: torch.Tensor
    bottom_normals_object: torch.Tensor
    stable_states: torch.Tensor | None  # quantitative evaluation 定量评估使用；训练默认通常为 None


def _make_affordance_files(
    dataset_root: Path,
    object_name: str,
) -> TeacherAffordanceFiles:
    """只建立资产路径，不进行任何数值计算。"""
    object_dir = dataset_root / object_name
    return TeacherAffordanceFiles(
        object_name=object_name,
        dynamic_urdf_path=object_dir / f"{object_name}.urdf",
        top_mesh_path=object_dir / "top_watertight_tiny.obj",
        bottom_mesh_path=object_dir / "bottom_watertight_tiny.obj",
        lowest_point_path=object_dir / "lowest_point_new.txt",
        top_usd_path=object_dir / f"{object_name}_top.usd",
        bottom_usd_path=object_dir / f"{object_name}_bottom.usd",
    )

TEACHER_AFFORDANCE_FILES: tuple[TeacherAffordanceFiles, ...] = tuple(
    _make_affordance_files(
        NEW_TRAINING_SET_ROOT,
        object_name,
    )
    for object_name in TEACHER_OBJECT_NAMES
)

TEACHER_AFFORDANCE_FILES_BY_NAME: dict[str, TeacherAffordanceFiles] = {
    item.object_name: item
    for item in TEACHER_AFFORDANCE_FILES
}

def get_teacher_affordance_files(
    object_name: str,
) -> TeacherAffordanceFiles:
    return TEACHER_AFFORDANCE_FILES_BY_NAME[object_name]

def _parse_obj(
    mesh_path: Path,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """
    读取 OBJ 文本，返回 Python 元数据。
    这里仅负责磁盘文件解析。
    OBJ 顶点 / 面真正变成数值 Tensor 时，会在 _load_gpu_mesh()
    中通过 device="cuda" 直接创建在 GPU 上。
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for line in mesh_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            values = line.split()
            vertices.append(
                (
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                )
            )
        elif line.startswith("f "):
            tokens = line.split()[1:]
            indices = [
                int(token.split("/", 1)[0]) - 1
                for token in tokens
            ]
            # OBJ 可能存在多边形面。
            # 采用 fan triangulation：
            # (0,1,2), (0,2,3), ...
            for index in range(1, len(indices) - 1):
                faces.append(
                    (
                        indices[0],
                        indices[index],
                        indices[index + 1],
                    )
                )
    return vertices, faces

def _load_gpu_mesh(
    mesh_path: Path,
    device: torch.device,
) -> TeacherGpuMesh:
    """把 OBJ mesh 直接建立成 CUDA Tensor + CUDA Warp Mesh
    数值路径：

        OBJ 文本
          ↓
        Python 文件解析
          ↓
        torch.tensor(..., device="cuda")
          ↓
        CUDA vertices / faces
          ↓
        wp.from_torch(...)
          ↓
        CUDA Warp Mesh
    """
    vertex_values, face_values = _parse_obj(mesh_path)
    # 数值 Tensor 从创建开始就在 GPU。
    vertices = torch.tensor(
        vertex_values,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    faces = torch.tensor(
        face_values,
        dtype=torch.long,
        device=device,
    ).contiguous()
    # Warp 三角面索引要求 int32。
    # 这个类型转换同样发生在 GPU 上。
    warp_face_indices = faces.reshape(-1).to(
        dtype=torch.int32
    ).contiguous()
    # wp.from_torch 是零拷贝包装 CUDA Tensor。
    # 不经过 CPU。
    warp_vertices = wp.from_torch(
        vertices,
        dtype=wp.vec3,
    )
    warp_indices = wp.from_torch(
        warp_face_indices,
        dtype=wp.int32,
    )

    warp_mesh = wp.Mesh(
        points=warp_vertices,
        indices=warp_indices,
    )
    return TeacherGpuMesh(
        vertices=vertices,
        faces=faces,
        warp_face_indices=warp_face_indices,
        warp_mesh=warp_mesh,
    )

def _sample_mesh_surface_gpu(
    mesh: TeacherGpuMesh,
    point_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """完全在 GPU 上从三角网格表面均匀采样点，并计算面法向量。
    与 trimesh.sample.sample_surface() 的目标相同：
        1. 计算每个三角面的面积。
        2. 按面积概率选择三角面。
        3. 在选中的三角面内部做均匀重心采样。
        4. 返回采样点和对应三角面的单位法向量。
    返回：
        points  [point_count, 3] float32 CUDA
        normals [point_count, 3] float32 CUDA
    """

    # [F, 3, 3]
    triangles = mesh.vertices[mesh.faces]

    vertex_0 = triangles[:, 0]
    edge_01 = triangles[:, 1] - vertex_0
    edge_02 = triangles[:, 2] - vertex_0

    # cross 的模长等于三角形面积的 2 倍。
    # 作为 multinomial 权重时常数 0.5 会抵消，所以无需额外乘 0.5。
    face_cross = torch.linalg.cross(
        edge_01,
        edge_02,
        dim=-1,
    )
    double_area = torch.linalg.vector_norm(
        face_cross,
        dim=-1,
    )

    # 按三角形面积直接在 CUDA 上采样 face id。
    sampled_face_ids = torch.multinomial(
        double_area,
        num_samples=point_count,
        replacement=True,
    )

    selected_triangles = triangles[sampled_face_ids]

    # 三角形内部均匀随机采样。
    # sqrt(u) 是均匀重心采样的标准变换。
    random_uv = torch.rand(
        (point_count, 2),
        dtype=mesh.vertices.dtype,
        device=mesh.vertices.device,
    )
    sqrt_u = torch.sqrt(random_uv[:, 0:1])
    v = random_uv[:, 1:2]

    weight_0 = 1.0 - sqrt_u
    weight_1 = sqrt_u * (1.0 - v)
    weight_2 = sqrt_u * v

    sampled_points = (
        weight_0 * selected_triangles[:, 0]
        + weight_1 * selected_triangles[:, 1]
        + weight_2 * selected_triangles[:, 2]
    )

    # 对应面的单位法向量。
    sampled_cross = face_cross[sampled_face_ids]
    sampled_normals = (
        sampled_cross
        / torch.linalg.vector_norm(
            sampled_cross,
            dim=-1,
            keepdim=True,
        )
    )

    return sampled_points, sampled_normals


def _make_env_object_indices(
    num_envs: int,
    weighted_object_indices: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """直接在 GPU 上生成每个并行环境的 object index。"""

    weighted_indices = torch.tensor(
        weighted_object_indices,
        dtype=torch.long,
        device=device,
    )

    cycle_count = (
        num_envs + weighted_indices.numel() - 1
    ) // weighted_indices.numel()

    return weighted_indices.repeat(
        cycle_count
    )[:num_envs].clone()


def _read_npy_last_pose(
    path: str,
) -> tuple[float, ...]:
    """只用 Python 标准库读取 .npy 最后一行前 7 个值。
    函数只用于 quantitative evaluation 的磁盘资产读取。
    当前数据集使用普通 C-order float32 / float64 NPY。
    """
    with open(path, "rb") as file:
        file.read(6)  # \\x93NUMPY
        major, _ = struct.unpack("BB", file.read(2))

        if major == 1:
            header_length = struct.unpack("<H", file.read(2))[0]
        else:
            header_length = struct.unpack("<I", file.read(4))[0]

        header = ast.literal_eval(
            file.read(header_length).decode("latin1").strip()
        )

        dtype_description = header["descr"]
        shape = header["shape"]

        row_count = shape[0]
        column_count = shape[1]

        if dtype_description.endswith("f4"):
            item_size = 4
            item_format = "f"
        else:
            item_size = 8
            item_format = "d"

        byte_order = (
            ">"
            if dtype_description.startswith(">")
            else "<"
        )

        data_offset = file.tell()
        row_size_bytes = column_count * item_size
        last_row_offset = (
            data_offset
            + (row_count - 1) * row_size_bytes
        )
        file.seek(last_row_offset)

        row_values = struct.unpack(
            f"{byte_order}{column_count}{item_format}",
            file.read(row_size_bytes),
        )

    return tuple(float(value) for value in row_values[:7])


def load_teacher_affordance_data(
    num_envs: int,
    device: torch.device | str,
    dataset_root: str,
    object_names: tuple[str, ...],
    weighted_object_indices: tuple[int, ...],
    stable_state_paths: tuple[str, ...],
) -> TeacherAffordanceData:
    """加载 Teacher 物体几何，并建立纯 GPU 训练数据。 """
    device = torch.device(device)
    # Teacher 的数值训练链路强制使用 CUDA。
    if device.type != "cuda":
        raise RuntimeError(
            f"Teacher affordance data requires CUDA, got {device}"
        )

    files_by_object = tuple(
        _make_affordance_files(
            Path(dataset_root),
            object_name,
        )
        for object_name in object_names
    )
    unique_lowest_points: list[torch.Tensor] = []
    unique_top_points: list[torch.Tensor] = []
    unique_top_normals: list[torch.Tensor] = []
    unique_bottom_points: list[torch.Tensor] = []
    unique_bottom_normals: list[torch.Tensor] = []
    unique_top_meshes: list[TeacherGpuMesh] = []
    unique_bottom_meshes: list[TeacherGpuMesh] = []
    for files in files_by_object:
        # -------------------------------------------------------------
        # 1. OBJ -> CUDA mesh
        # -------------------------------------------------------------
        top_mesh = _load_gpu_mesh(
            files.top_mesh_path,
            device,
        )
        bottom_mesh = _load_gpu_mesh(
            files.bottom_mesh_path,
            device,
        )

        # -------------------------------------------------------------
        # 2. CUDA surface sampling
        # -------------------------------------------------------------
        sampled_top_points, sampled_top_normals = (
            _sample_mesh_surface_gpu(
                top_mesh,
                AFFORDANCE_POINT_COUNT,
            )
        )
        sampled_bottom_points, sampled_bottom_normals = (
            _sample_mesh_surface_gpu(
                bottom_mesh,
                NON_AFFORDANCE_POINT_COUNT,
            )
        )

        # -------------------------------------------------------------
        # 3. lowest point 直接由 CUDA mesh 计算
        # -------------------------------------------------------------
        # 原文件从 lowest_point_new.txt 读标量，再先建立 CPU tensor。
        # 这里直接从 top + bottom 几何的 z 最小值计算，避免 CPU 数值路径。
        lowest_point = torch.minimum(
            top_mesh.vertices[:, 2].amin(),
            bottom_mesh.vertices[:, 2].amin(),
        )

        unique_lowest_points.append(lowest_point)
        unique_top_points.append(sampled_top_points)
        unique_top_normals.append(sampled_top_normals)
        unique_bottom_points.append(sampled_bottom_points)
        unique_bottom_normals.append(sampled_bottom_normals)
        unique_top_meshes.append(top_mesh)
        unique_bottom_meshes.append(bottom_mesh)

    # 每种唯一物体堆叠成一个 CUDA batch。
    unique_lowest_points_tensor = torch.stack(
        unique_lowest_points,
        dim=0,
    ).unsqueeze(-1)

    unique_top_points_tensor = torch.stack(
        unique_top_points,
        dim=0,
    )
    unique_top_normals_tensor = torch.stack(
        unique_top_normals,
        dim=0,
    )
    unique_bottom_points_tensor = torch.stack(
        unique_bottom_points,
        dim=0,
    )
    unique_bottom_normals_tensor = torch.stack(
        unique_bottom_normals,
        dim=0,
    )

    # -------------------------------------------------------------
    # 4. env object mapping 直接在 GPU 上生成
    # -------------------------------------------------------------
    env_object_indices = _make_env_object_indices(
        num_envs=num_envs,
        weighted_object_indices=weighted_object_indices,
        device=device,
    )

    # -------------------------------------------------------------
    # 5. 按环境 gather，同样全部发生在 GPU
    # -------------------------------------------------------------
    lowest_points = unique_lowest_points_tensor.index_select(
        0,
        env_object_indices,
    )
    top_points_object = unique_top_points_tensor.index_select(
        0,
        env_object_indices,
    )
    top_normals_object = unique_top_normals_tensor.index_select(
        0,
        env_object_indices,
    )
    bottom_points_object = unique_bottom_points_tensor.index_select(
        0,
        env_object_indices,
    )
    bottom_normals_object = unique_bottom_normals_tensor.index_select(
        0,
        env_object_indices,
    )

    # -------------------------------------------------------------
    # 6. quantitative evaluation stable state
    # -------------------------------------------------------------
    if stable_state_paths:
        stable_state_values = [
            _read_npy_last_pose(path)
            for path in stable_state_paths
        ]

        # 数值 Tensor 直接创建到 CUDA，不经过 CPU torch.Tensor。
        unique_stable_states = torch.tensor(
            stable_state_values,
            dtype=torch.float32,
            device=device,
        )

        stable_states = unique_stable_states.index_select(
            0,
            env_object_indices,
        )
    else:
        stable_states = None

    return TeacherAffordanceData(
        object_names=object_names,
        env_object_indices=env_object_indices,
        unique_lowest_points=unique_lowest_points_tensor,
        unique_top_points_object=unique_top_points_tensor,
        unique_top_normals_object=unique_top_normals_tensor,
        unique_bottom_points_object=unique_bottom_points_tensor,
        unique_bottom_normals_object=unique_bottom_normals_tensor,
        unique_top_meshes=tuple(unique_top_meshes),
        unique_bottom_meshes=tuple(unique_bottom_meshes),
        lowest_points=lowest_points,
        top_points_object=top_points_object,
        top_normals_object=top_normals_object,
        bottom_points_object=bottom_points_object,
        bottom_normals_object=bottom_normals_object,
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
    "TeacherGpuMesh",
    "TeacherAffordanceData",
    "TEACHER_AFFORDANCE_FILES",
    "TEACHER_AFFORDANCE_FILES_BY_NAME",
    "get_teacher_affordance_files",
    "load_teacher_affordance_data",
]
