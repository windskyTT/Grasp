"""Teacher pregrasp：纯 GPU 几何、raycast 与 FR3 palm DLS IK。

本文件负责 Teacher reset 中最重的几何计算：

    sampled top affordance points
        ↓
    camera raycast
        ↓
    visible top surface points
        ↓
    affordance center / approach direction
        ↓
    10 个候选 palm orientation
        ↓
    FR3 + Inspire palm FK / Jacobian
        ↓
    batched damped-least-squares IK
        ↓
    candidate scoring / selection

GPU 约定：
    - 不使用 NumPy。
    - 不使用 trimesh。
    - 不创建 CPU torch.Tensor。
    - 不做 GPU -> CPU -> GPU 数值中转。
    - raycast 使用 IsaacLab 2.3.2 的 Warp CUDA mesh raycast。
    - candidate rotation / projection / IK 全部使用 batched Torch CUDA。

坐标系约定：
    - fr3_link8 不是 Inspire palm/base frame。
    - Inspire palm center 定义为：
          base_link position
          + base_link rotation * robot_spec.palm_offset
    - FK 显式包含：
          fr3_link8
          -> L_flange
          -> wrist
          -> base_link
          -> palm_offset
    - pregrasp target 直接定义为 palm center + base_link orientation，
      不再把 base_link 局部 palm_offset 当成 fr3_link8 局部 offset。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_euler,
    quat_from_matrix,
)
from isaaclab.utils.warp.ops import raycast_mesh

from .affordance_data import TeacherGpuMesh
from .env_cfg import TeacherPregraspCfg


TeacherKinematicsEvaluator = Callable[
    [torch.Tensor],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]


# =============================================================================
# FR3 7-DOF 几何参数
# =============================================================================
# 与当前 fr3_inspire_tac_L_right_safety.urdf 对齐。
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
    (-0.5 * math.pi, 0.0, 0.0),
    (0.5 * math.pi, 0.0, 0.0),
    (0.5 * math.pi, 0.0, 0.0),
    (-0.5 * math.pi, 0.0, 0.0),
    (0.5 * math.pi, 0.0, 0.0),
    (0.5 * math.pi, 0.0, 0.0),
)

# fr3_link7 -> fr3_link8 fixed joint。
FR3_LINK8_OFFSET = (0.0, 0.0, 0.107)


# =============================================================================
# FR3 link8 -> Inspire base_link 固定链
# =============================================================================
# URDF:
#
# fr3_link8
#   -> L_flange
#      xyz = (0, 0, 0)
#      rpy = (pi, 0, pi/2)
#
# L_flange
#   -> wrist
#      xyz = (0, 0.0415, -0.068767)
#      rpy = (pi/2, 0, pi)
#
# wrist
#   -> base_link
#      xyz = (0, 0, 0)
#      rpy = (3.14, 0, 0)
FR3_LINK8_TO_BASE_XYZ = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0415, -0.068767),
    (0.0, 0.0, 0.0),
)

FR3_LINK8_TO_BASE_RPY = (
    (math.pi, 0.0, 0.5 * math.pi),
    (0.5 * math.pi, 0.0, math.pi),
    (3.14, 0.0, 0.0),
)


# =============================================================================
# CUDA kinematics 常量缓存
# =============================================================================
# IK 在一次 reset 内最多调用 cfg.ik_max_iterations 次 FK/Jacobian。
# 这些机器人固定几何不能每次都重新创建 CUDA Tensor。
#
# 缓存中的 Tensor 第一次创建时就直接位于目标 CUDA device，
# 不存在 CPU Tensor -> CUDA。
_FR3_KINEMATIC_CACHE: dict[
    tuple[torch.device, torch.dtype],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}


@dataclass(frozen=True)
class TeacherPregraspGeometry:
    """一批环境的 GPU pregrasp geometry。"""

    # [B, 200, 3]
    visible_points_world: torch.Tensor
    visible_points_object: torch.Tensor

    # [B, 3]
    affordance_center_world: torch.Tensor
    approach_direction_world: torch.Tensor

    # [B, candidate_count, 3]
    palm_target_positions_world: torch.Tensor

    # [B, candidate_count, 3, 3]
    palm_target_rotations_world: torch.Tensor

    # [B, candidate_count]
    projection_lengths: torch.Tensor


@dataclass(frozen=True)
class TeacherIKResult:
    """Batched DLS IK 输出。"""

    # [B, 7]
    arm_qpos: torch.Tensor

    # [B]
    converged: torch.Tensor

    # [B, 3]
    position_error: torch.Tensor
    rotation_error: torch.Tensor


@dataclass(frozen=True)
class TeacherPregraspSelection:
    """每个环境最终选择的 pregrasp candidate。"""

    # [B, 7]
    arm_qpos: torch.Tensor

    # [B]
    selected_candidate_indices: torch.Tensor
    valid: torch.Tensor


def _get_fr3_kinematic_constants(
    reference: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """第一次使用时直接在 reference.device 上建立 FR3 固定几何。

    返回：
        joint_origins_xyz       [7, 3]
        joint_origin_rotations  [7, 3, 3]
        link8_offset            [3]
        link8_to_base_position  [3]
        link8_to_base_rotation  [3, 3]
        identity3               [3, 3]
    """

    key = (
        reference.device,
        reference.dtype,
    )

    cached = _FR3_KINEMATIC_CACHE.get(key)
    if cached is not None:
        return cached

    device = reference.device
    dtype = reference.dtype

    # 所有 Tensor 从创建开始就在 CUDA。
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

    link8_offset = torch.tensor(
        FR3_LINK8_OFFSET,
        dtype=dtype,
        device=device,
    )

    fixed_xyz = torch.tensor(
        FR3_LINK8_TO_BASE_XYZ,
        dtype=dtype,
        device=device,
    )

    fixed_rpy = torch.tensor(
        FR3_LINK8_TO_BASE_RPY,
        dtype=dtype,
        device=device,
    )

    fixed_rotations = matrix_from_euler(
        fixed_rpy,
        "XYZ",
    )

    # -------------------------------------------------------------
    # 组合：
    #     fr3_link8 -> L_flange -> wrist -> base_link
    # -------------------------------------------------------------
    link8_to_base_position = (
        fixed_xyz[0]
        + fixed_rotations[0] @ fixed_xyz[1]
        + (
            fixed_rotations[0]
            @ fixed_rotations[1]
            @ fixed_xyz[2]
        )
    )

    link8_to_base_rotation = (
        fixed_rotations[0]
        @ fixed_rotations[1]
        @ fixed_rotations[2]
    )

    identity3 = torch.eye(
        3,
        dtype=dtype,
        device=device,
    )

    cached = (
        joint_origins_xyz,
        joint_origin_rotations,
        link8_offset,
        link8_to_base_position,
        link8_to_base_rotation,
        identity3,
    )
    _FR3_KINEMATIC_CACHE[key] = cached

    return cached


@torch.no_grad()
def compute_visible_top_points_gpu(
    top_mesh: TeacherGpuMesh,
    sampled_top_points_object: torch.Tensor,
    object_position_world: torch.Tensor,
    object_rotation_world: torch.Tensor,
    camera_position_world: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """用 Warp CUDA raycast 得到 camera 可见的 top surface points。

    输入：
        sampled_top_points_object:
            [B, 200, 3]

        object_position_world:
            [B, 3]

        object_rotation_world:
            [B, 3, 3]

        camera_position_world:
            [B, 3]

    过程保持原 RobustDexGrasp 的语义：

        camera
          ↓ ray toward each sampled top point
        top mesh first intersection
          ↓
        visible point

    也就是说，返回的不一定是原 sampled point，
    而是从 camera 沿该方向看到的第一个 top mesh surface hit。
    """

    batch_size, point_count, _ = (
        sampled_top_points_object.shape
    )

    # -------------------------------------------------------------
    # camera world -> object local
    #
    # p_O = R_WO^T * (p_W - t_WO)
    # -------------------------------------------------------------
    camera_position_object = torch.bmm(
        object_rotation_world.transpose(1, 2),
        (
            camera_position_world
            - object_position_world
        ).unsqueeze(-1),
    ).squeeze(-1)

    # 每个环境的 200 条 ray 都从同一个 camera local position 出发。
    ray_starts_object = (
        camera_position_object[:, None, :]
        .expand(
            batch_size,
            point_count,
            3,
        )
        # raycast_mesh 内部先 view(-1, 3)，因此必须在传入前连续化。
        .contiguous()
    )

    # ray 指向对应 sampled top point。
    ray_directions_object = (
        sampled_top_points_object
        - ray_starts_object
    )
    ray_directions_object = (
        ray_directions_object
        / torch.linalg.vector_norm(
            ray_directions_object,
            dim=-1,
            keepdim=True,
        )
    )

    # -------------------------------------------------------------
    # IsaacLab 2.3.2 Warp CUDA raycast。
    #
    # top_mesh.warp_mesh 已在 affordance_data.py 中建立在 CUDA。
    # ray starts / directions 也是 CUDA Tensor。
    # 返回 hit tensor 仍留在同一 GPU。
    # -------------------------------------------------------------
    (
        visible_points_object,
        _,
        _,
        _,
    ) = raycast_mesh(
        ray_starts=ray_starts_object,
        ray_directions=ray_directions_object,
        mesh=top_mesh.warp_mesh,
        max_dist=1.0e6,
        return_distance=False,
        return_normal=False,
        return_face_id=False,
    )

    # -------------------------------------------------------------
    # object local -> world
    #
    # p_W = R_WO * p_O + t_WO
    # -------------------------------------------------------------
    visible_points_world = torch.matmul(
        object_rotation_world.unsqueeze(1),
        visible_points_object.unsqueeze(-1),
    ).squeeze(-1)

    visible_points_world = (
        visible_points_world
        + object_position_world[:, None, :]
    )

    return (
        visible_points_world,
        visible_points_object,
    )


@torch.no_grad()
def sample_rot_mats_gpu(
    approach_direction_world: torch.Tensor,
    candidate_count: int,
    visible_points_world: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """批量生成每个环境的 candidate end-effector rotations。

    完整保留原 RobustDexGrasp NumPy sample_rot_mats() 的几何逻辑，
    这里只做 Torch CUDA 化。

    重要：
        这里返回的 rotation 仍然保持原 wrist / arm-end candidate
        orientation 语义；FR3 + Inspire 的 base_link / palm orientation
        转换在 build_teacher_pregrasp_geometry_gpu() 中统一完成。

    输入：
        approach_direction_world:
            [B, 3]

        visible_points_world:
            [B, 200, 3]

    输出：
        rotation_matrices:
            [B, K, 3, 3]

        projection_lengths:
            [B, K]

    K = cfg.candidate_count，当前为 10。
    """

    batch_size = approach_direction_world.shape[0]
    dtype = approach_direction_world.dtype
    device = approach_direction_world.device

    reference_vector = approach_direction_world

    # -------------------------------------------------------------
    # 为 reference 选择一个不平行的 temporary vector。
    #
    # 原逻辑：
    #     if abs(reference.x) < 0.9:
    #         temp = x-axis
    #     else:
    #         temp = y-axis
    #
    # 这里用 CUDA mask 批量完成。
    # -------------------------------------------------------------
    basis_x = torch.zeros(
        (batch_size, 3),
        dtype=dtype,
        device=device,
    )
    basis_x[:, 0] = 1.0

    basis_y = torch.zeros_like(
        basis_x
    )
    basis_y[:, 1] = 1.0

    use_x_axis = (
        torch.abs(reference_vector[:, 0])
        < 0.9
    )

    temporary_vector = torch.where(
        use_x_axis[:, None],
        basis_x,
        basis_y,
    )

    # 第一个垂直方向。
    first_perpendicular = torch.linalg.cross(
        reference_vector,
        temporary_vector,
        dim=-1,
    )
    first_perpendicular = (
        first_perpendicular
        / torch.linalg.vector_norm(
            first_perpendicular,
            dim=-1,
            keepdim=True,
        )
    )

    # 第二个垂直方向。
    second_perpendicular = torch.linalg.cross(
        reference_vector,
        first_perpendicular,
        dim=-1,
    )
    second_perpendicular = (
        second_perpendicular
        / torch.linalg.vector_norm(
            second_perpendicular,
            dim=-1,
            keepdim=True,
        )
    )

    # -------------------------------------------------------------
    # theta = 0, 2*pi/K, ..., (K-1)*2*pi/K
    #
    # 直接在 CUDA 上生成。
    # -------------------------------------------------------------
    theta = torch.arange(
        candidate_count,
        dtype=dtype,
        device=device,
    )
    theta = theta * (
        2.0 * math.pi
        / candidate_count
    )

    cosine = torch.cos(theta).view(
        1,
        candidate_count,
        1,
    )
    sine = torch.sin(theta).view(
        1,
        candidate_count,
        1,
    )

    perpendicular_vectors = (
        first_perpendicular[:, None, :]
        * cosine
        + second_perpendicular[:, None, :]
        * sine
    )

    perpendicular_vectors = (
        perpendicular_vectors
        / torch.linalg.vector_norm(
            perpendicular_vectors,
            dim=-1,
            keepdim=True,
        )
    )

    # 保留原逻辑：
    #     if perpendicular_vector[1] < 0:
    #         perpendicular_vector *= -1
    perpendicular_vectors = torch.where(
        perpendicular_vectors[
            :, :, 1:2
        ] < 0.0,
        -perpendicular_vectors,
        perpendicular_vectors,
    )

    # -------------------------------------------------------------
    # candidate projection length
    #
    # visible points:
    #     [B, N, 3]
    #
    # perpendicular vectors:
    #     [B, K, 3]
    #
    # projected:
    #     [B, N, K]
    # -------------------------------------------------------------
    centered_points = (
        visible_points_world
        - visible_points_world.mean(
            dim=1,
            keepdim=True,
        )
    )

    projected_points = torch.bmm(
        centered_points,
        perpendicular_vectors.transpose(
            1,
            2,
        ),
    )

    projection_lengths = (
        projected_points.amax(dim=1)
        - projected_points.amin(dim=1)
    )

    # -------------------------------------------------------------
    # 原 rotation matrix：
    #
    #     -[reference, y_direction, perpendicular]
    #
    # 三个向量作为 matrix columns。
    # -------------------------------------------------------------
    reference_candidates = (
        reference_vector[:, None, :]
        .expand(
            batch_size,
            candidate_count,
            3,
        )
    )

    y_direction = torch.linalg.cross(
        reference_candidates,
        perpendicular_vectors,
        dim=-1,
    )
    y_direction = (
        y_direction
        / torch.linalg.vector_norm(
            y_direction,
            dim=-1,
            keepdim=True,
        )
    )

    rotation_matrices = -torch.stack(
        (
            reference_candidates,
            y_direction,
            perpendicular_vectors,
        ),
        dim=-1,
    )

    return (
        rotation_matrices,
        projection_lengths,
    )


@torch.no_grad()
def build_teacher_pregrasp_geometry_gpu(
    top_mesh: TeacherGpuMesh,
    sampled_top_points_object: torch.Tensor,
    object_position_world: torch.Tensor,
    object_rotation_world: torch.Tensor,
    env_origin_world: torch.Tensor,
    cfg: TeacherPregraspCfg,
) -> TeacherPregraspGeometry:
    """纯 GPU 构造一批环境的 pregrasp geometry。

    注意：
        sample_rot_mats_gpu() 保留原 RobustDexGrasp 的 candidate
        end-effector / wrist orientation 语义。

        对 FR3 + Inspire，IK 最终约束的是 base_link / palm orientation，
        因此必须显式加入固定链：

            R_WB,target
                =
            R_W8,target @ R_8B

        其中：
            R_W8,target
                = sample_rot_mats_gpu() 产生的 candidate orientation

            R_8B
                = fr3_link8 -> Inspire base_link 固定旋转

        不能把 candidate rotation 直接当成 base_link rotation。
    """

    batch_size = object_position_world.shape[0]
    dtype = object_position_world.dtype
    device = object_position_world.device

    # camera_position 是静态 Python 配置；
    # 第一个数值 Tensor 直接创建在 CUDA。
    camera_offset_world = torch.tensor(
        cfg.camera_position,
        dtype=dtype,
        device=device,
    )

    camera_position_world = (
        env_origin_world
        + camera_offset_world
    )

    (
        visible_points_world,
        visible_points_object,
    ) = compute_visible_top_points_gpu(
        top_mesh=top_mesh,
        sampled_top_points_object=(
            sampled_top_points_object
        ),
        object_position_world=(
            object_position_world
        ),
        object_rotation_world=(
            object_rotation_world
        ),
        camera_position_world=(
            camera_position_world
        ),
    )

    # 200 个 visible top points 的中心。
    affordance_center_world = (
        visible_points_world.mean(
            dim=1,
        )
    )

    # -------------------------------------------------------------
    # approach direction
    # -------------------------------------------------------------
    if cfg.top_grasp:
        approach_direction_world = torch.zeros(
            (batch_size, 3),
            dtype=dtype,
            device=device,
        )
        approach_direction_world[:, 2] = 1.0
    else:
        approach_direction_world = (
            camera_position_world
            - affordance_center_world
        )
        approach_direction_world = (
            approach_direction_world
            / torch.linalg.vector_norm(
                approach_direction_world,
                dim=-1,
                keepdim=True,
            )
        )

    # -------------------------------------------------------------
    # Palm center target：
    #
    #     p_target
    #       = affordance center
    #       + approach_distance * approach direction
    #
    # 这里已经是“真正 palm center”的目标位置。
    # 不再减去 base_link-local palm_offset。
    # -------------------------------------------------------------
    palm_center_target_world = (
        affordance_center_world
        + (
            cfg.approach_distance
            * approach_direction_world
        )
    )

    (
        link8_target_rotations_world,
        projection_lengths,
    ) = sample_rot_mats_gpu(
        approach_direction_world=(
            approach_direction_world
        ),
        candidate_count=(
            cfg.candidate_count
        ),
        visible_points_world=(
            visible_points_world
        ),
    )

    # -----------------------------------------------------------------
    # 关键坐标系修正：
    #
    # 原 RobustDexGrasp 的 sample_rot_mats() 生成的是机械臂
    # end-effector / wrist candidate orientation。
    #
    # 当前 FR3 + Inspire 的 DLS IK 却直接约束：
    #     Inspire base_link / palm orientation
    #
    # 二者之间存在固定链：
    #
    #     fr3_link8
    #        -> L_flange
    #        -> wrist
    #        -> base_link
    #
    # 因此：
    #
    #     R_WB,target
    #       =
    #     R_W8,target @ R_8B
    #
    # 旧代码漏掉 R_8B，导致 DLS IK 被要求去追一个错误的
    # base_link orientation，容易让大量 candidate 64 轮都不收敛。
    # -----------------------------------------------------------------
    (
        _,
        _,
        _,
        _,
        link8_to_base_rotation,
        _,
    ) = _get_fr3_kinematic_constants(
        object_position_world
    )

    palm_target_rotations_world = torch.matmul(
        link8_target_rotations_world,
        link8_to_base_rotation.view(
            1,
            1,
            3,
            3,
        ),
    )

    # 每个 candidate 共用相同 palm center，
    # 只改变 base_link / palm orientation。
    palm_target_positions_world = (
        palm_center_target_world[:, None, :]
        .expand(
            batch_size,
            cfg.candidate_count,
            3,
        )
    )

    return TeacherPregraspGeometry(
        visible_points_world=(
            visible_points_world
        ),
        visible_points_object=(
            visible_points_object
        ),
        affordance_center_world=(
            affordance_center_world
        ),
        approach_direction_world=(
            approach_direction_world
        ),
        palm_target_positions_world=(
            palm_target_positions_world
        ),
        palm_target_rotations_world=(
            palm_target_rotations_world
        ),
        projection_lengths=(
            projection_lengths
        ),
    )


@torch.no_grad()
def compute_fr3_palm_kinematics(
    arm_qpos: torch.Tensor,
    robot_base_positions_world: torch.Tensor,
    palm_offset_body: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """计算 FR3 + Inspire palm center 的 FK 与 6x7 Jacobian。

    输入：
        arm_qpos:
            [B, 7]

        robot_base_positions_world:
            [B, 3]

        palm_offset_body:
            [3]
            Inspire base_link 局部 palm center offset。

    输出：
        palm_position_world:
            [B, 3]

        palm_quaternion_world:
            [B, 4]

        palm_jacobian_world:
            [B, 6, 7]

    固定链：

        FR3 joint1..7
            ↓
        fr3_link8
            ↓
        L_flange
            ↓
        wrist
            ↓
        base_link
            ↓
        palm center
    """

    batch_size = arm_qpos.shape[0]

    (
        joint_origins_xyz,
        joint_origin_rotations,
        link8_offset,
        link8_to_base_position,
        link8_to_base_rotation,
        identity3,
    ) = _get_fr3_kinematic_constants(
        arm_qpos
    )

    # 当前 FR3 link frame 的 world pose。
    position_world = (
        robot_base_positions_world.clone()
    )

    rotation_world = (
        identity3.unsqueeze(0)
        .expand(
            batch_size,
            3,
            3,
        )
        .clone()
    )

    joint_positions_world: list[
        torch.Tensor
    ] = []
    joint_axes_world: list[
        torch.Tensor
    ] = []

    # -------------------------------------------------------------
    # 7 个关节只有固定 7 次 Python loop；
    # 每一次内部操作都是整个 batch 的 CUDA kernel。
    # -------------------------------------------------------------
    for joint_index in range(7):
        # parent link -> 当前 joint origin 平移。
        position_world = (
            position_world
            + torch.bmm(
                rotation_world,
                joint_origins_xyz[
                    joint_index
                ]
                .view(1, 3, 1)
                .expand(
                    batch_size,
                    3,
                    1,
                ),
            ).squeeze(-1)
        )

        # joint origin 固定 RPY。
        rotation_world = torch.bmm(
            rotation_world,
            joint_origin_rotations[
                joint_index
            ]
            .unsqueeze(0)
            .expand(
                batch_size,
                3,
                3,
            ),
        )

        # 当前 revolute joint 的 world origin / world z-axis。
        joint_positions_world.append(
            position_world
        )
        joint_axes_world.append(
            rotation_world[:, :, 2]
        )

        # 当前 joint 的 z-axis rotation。
        joint_angle = arm_qpos[
            :,
            joint_index,
        ]
        cosine = torch.cos(
            joint_angle
        )
        sine = torch.sin(
            joint_angle
        )

        joint_rotation = (
            identity3.unsqueeze(0)
            .expand(
                batch_size,
                3,
                3,
            )
            .clone()
        )
        joint_rotation[:, 0, 0] = cosine
        joint_rotation[:, 0, 1] = -sine
        joint_rotation[:, 1, 0] = sine
        joint_rotation[:, 1, 1] = cosine

        rotation_world = torch.bmm(
            rotation_world,
            joint_rotation,
        )

    # -------------------------------------------------------------
    # fr3_link7 -> fr3_link8
    # -------------------------------------------------------------
    link8_position_world = (
        position_world
        + torch.bmm(
            rotation_world,
            link8_offset
            .view(1, 3, 1)
            .expand(
                batch_size,
                3,
                1,
            ),
        ).squeeze(-1)
    )

    link8_rotation_world = (
        rotation_world
    )

    # -------------------------------------------------------------
    # fr3_link8 -> Inspire base_link
    # -------------------------------------------------------------
    base_position_world = (
        link8_position_world
        + torch.bmm(
            link8_rotation_world,
            link8_to_base_position
            .view(1, 3, 1)
            .expand(
                batch_size,
                3,
                1,
            ),
        ).squeeze(-1)
    )

    base_rotation_world = torch.bmm(
        link8_rotation_world,
        link8_to_base_rotation
        .unsqueeze(0)
        .expand(
            batch_size,
            3,
            3,
        ),
    )

    # -------------------------------------------------------------
    # base_link -> palm center
    #
    # p_palm = p_base + R_base * palm_offset_base
    # -------------------------------------------------------------
    palm_position_world = (
        base_position_world
        + torch.bmm(
            base_rotation_world,
            palm_offset_body
            .view(1, 3, 1)
            .expand(
                batch_size,
                3,
                1,
            ),
        ).squeeze(-1)
    )

    # Teacher palm orientation 使用 base_link orientation。
    palm_quaternion_world = (
        quat_from_matrix(
            base_rotation_world
        )
    )

    # -------------------------------------------------------------
    # Geometric Jacobian
    #
    # revolute joint:
    #
    #     Jv_i = z_i × (p_palm - p_joint_i)
    #     Jw_i = z_i
    # -------------------------------------------------------------
    joint_positions_world_tensor = (
        torch.stack(
            joint_positions_world,
            dim=1,
        )
    )

    joint_axes_world_tensor = (
        torch.stack(
            joint_axes_world,
            dim=1,
        )
    )

    linear_jacobian = (
        torch.linalg.cross(
            joint_axes_world_tensor,
            (
                palm_position_world.unsqueeze(1)
                - joint_positions_world_tensor
            ),
            dim=-1,
        )
        .transpose(
            1,
            2,
        )
    )

    angular_jacobian = (
        joint_axes_world_tensor.transpose(
            1,
            2,
        )
    )

    palm_jacobian_world = torch.cat(
        (
            linear_jacobian,
            angular_jacobian,
        ),
        dim=1,
    )

    return (
        palm_position_world,
        palm_quaternion_world,
        palm_jacobian_world,
    )


def compute_damped_least_squares_delta(
    jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """计算 DLS IK 关节增量。

    公式：

        dq
        =
        J^T
        (J J^T + lambda^2 I)^-1
        e

    实现使用 torch.linalg.solve()，
    不显式计算 matrix inverse。

    为减少每次 IK iteration 的临时 CUDA allocation，
    不再创建 6x6 identity tensor；
    直接给 task matrix 的 diagonal 加 lambda^2。
    """

    jacobian_transpose = (
        jacobian.transpose(
            1,
            2,
        )
    )

    damped_task_matrix = (
        jacobian
        @ jacobian_transpose
    )

    damped_task_matrix.diagonal(
        dim1=-2,
        dim2=-1,
    ).add_(
        damping * damping
    )

    solved_error = torch.linalg.solve(
        damped_task_matrix,
        pose_error.unsqueeze(-1),
    )

    return (
        jacobian_transpose
        @ solved_error
    ).squeeze(-1)


@torch.no_grad()
def solve_fr3_dls_ik(
    initial_arm_qpos: torch.Tensor,
    target_position_world: torch.Tensor,
    target_quaternion_world: torch.Tensor,
    arm_lower_limits: torch.Tensor,
    arm_upper_limits: torch.Tensor,
    cfg: TeacherPregraspCfg,
    evaluate_kinematics: TeacherKinematicsEvaluator,
) -> TeacherIKResult:
    """批量求解 FR3 7-DOF palm pose DLS IK。

    当前 env.py 会把：

        B environments
            ×
        K candidates

    展平为：

        [B*K, 7]

    因此 10 个候选不是逐个在 CPU 求 IK，
    而是作为一个 CUDA batch 同时迭代。
    """

    arm_qpos = (
        initial_arm_qpos.clone()
    )

    # 保留原 cfg.ik_max_iterations 固定迭代语义。
    # 不在 Python 中每轮读取 GPU bool 决定 early break，
    # 避免每轮产生 GPU synchronization。
    for _ in range(
        cfg.ik_max_iterations
    ):
        (
            current_position_world,
            current_quaternion_world,
            current_jacobian_world,
        ) = evaluate_kinematics(
            arm_qpos
        )

        (
            position_error,
            rotation_error,
        ) = compute_pose_error(
            current_position_world,
            current_quaternion_world,
            target_position_world,
            target_quaternion_world,
            rot_error_type="axis_angle",
        )

        converged = (
            torch.linalg.vector_norm(
                position_error,
                dim=-1,
            )
            <= cfg.ik_position_tolerance
        ) & (
            torch.linalg.vector_norm(
                rotation_error,
                dim=-1,
            )
            <= cfg.ik_rotation_tolerance
        )

        pose_error = torch.cat(
            (
                position_error,
                rotation_error,
            ),
            dim=-1,
        )

        arm_delta = (
            compute_damped_least_squares_delta(
                jacobian=(
                    current_jacobian_world
                ),
                pose_error=pose_error,
                damping=cfg.ik_damping,
            )
        )

        next_arm_qpos = (
            arm_qpos
            + (
                cfg.ik_step_scale
                * arm_delta
            )
        )

        # FR3 joint limits 全部为 CUDA Tensor。
        next_arm_qpos = torch.maximum(
            torch.minimum(
                next_arm_qpos,
                arm_upper_limits,
            ),
            arm_lower_limits,
        )

        # 已收敛 candidate 保持不动；
        # 未收敛 candidate 继续下一轮。
        arm_qpos = torch.where(
            converged.unsqueeze(-1),
            arm_qpos,
            next_arm_qpos,
        )

    # 最后一轮结束后统一重新计算最终误差。
    (
        current_position_world,
        current_quaternion_world,
        _,
    ) = evaluate_kinematics(
        arm_qpos
    )

    (
        position_error,
        rotation_error,
    ) = compute_pose_error(
        current_position_world,
        current_quaternion_world,
        target_position_world,
        target_quaternion_world,
        rot_error_type="axis_angle",
    )

    converged = (
        torch.linalg.vector_norm(
            position_error,
            dim=-1,
        )
        <= cfg.ik_position_tolerance
    ) & (
        torch.linalg.vector_norm(
            rotation_error,
            dim=-1,
        )
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
    """在 GPU 上为每个环境选择最终 pregrasp candidate。

    保持原评分逻辑：

    如果存在：
        converged
        AND projection_length < projection_limit

    则只在 short candidates 中按照：
        length score
        + posture error
        + posture limit score
    选最小。

    否则：
        在所有 converged candidates 中直接选最短 projection。
    """

    # [B, K]
    posture_joint = (
        candidate_arm_qpos[
            :,
            :,
            cfg.posture_joint_index,
        ]
    )

    posture_error = torch.abs(
        posture_joint
        - cfg.posture_joint_target
    )

    posture_limit_score = (
        torch.abs(
            posture_joint
        )
        - cfg.posture_limit_target
    ) * (
        cfg.posture_score_coeff
        * cfg.posture_limit_score_coeff
    )

    short_projection = (
        projection_lengths
        < cfg.projection_limit
    )

    has_short_feasible = torch.any(
        candidate_converged
        & short_projection,
        dim=1,
    )

    short_scores = (
        projection_lengths
        * cfg.length_score_coeff
        + posture_error
        * cfg.posture_score_coeff
        + posture_limit_score
    )

    large_scores = (
        projection_lengths
    )

    candidate_scores = torch.where(
        has_short_feasible.unsqueeze(-1),
        short_scores,
        large_scores,
    )

    eligible = (
        candidate_converged
        & torch.where(
            has_short_feasible.unsqueeze(-1),
            short_projection,
            torch.ones_like(
                short_projection
            ),
        )
    )

    candidate_scores = (
        candidate_scores.masked_fill(
            ~eligible,
            torch.inf,
        )
    )

    selected_candidate_indices = (
        torch.argmin(
            candidate_scores,
            dim=1,
        )
    )

    valid = torch.any(
        eligible,
        dim=1,
    )

    batch_indices = torch.arange(
        candidate_arm_qpos.shape[0],
        dtype=torch.long,
        device=candidate_arm_qpos.device,
    )

    selected_arm_qpos = (
        candidate_arm_qpos[
            batch_indices,
            selected_candidate_indices,
        ].clone()
    )

    # 无可用 candidate 的环境交给 reset round 重新采样。
    selected_arm_qpos[
        ~valid
    ] = torch.nan

    selected_candidate_indices = (
        selected_candidate_indices.clone()
    )
    selected_candidate_indices[
        ~valid
    ] = -1

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
    "compute_visible_top_points_gpu",
    "sample_rot_mats_gpu",
    "build_teacher_pregrasp_geometry_gpu",
    "compute_fr3_palm_kinematics",
    "compute_damped_least_squares_delta",
    "solve_fr3_dls_ik",
    "select_teacher_pregrasp_candidate",
]
