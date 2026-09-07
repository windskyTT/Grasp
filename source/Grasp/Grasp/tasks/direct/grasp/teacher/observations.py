"""Teacher observation 构造。

RobustDexGrasp Teacher 迁移到 FR3 + Inspire 后：

    FR3 active DOF      = 7
    Inspire active DOF  = 6
    active_qpos         = 13

    hand geometry bodies = 13
    arm height bodies    = 6

Teacher policy observation:

    13  active qpos
    13  joint target error
    13  affordance contact
    13  affordance impulse
    13  hand body height
     6  arm body height
     3  palm center
     3  wrist delta Euler
     3  wrist Euler
    39  nearest affordance vectors = 13 * 3
    --------------------------------
    119 dimensions

GPU 约定：
    - 本文件不使用 NumPy。
    - 不创建 CPU Tensor。
    - 不执行任何 GPU -> CPU 数值转换。
    - 所有 observation 数值都继承输入 Tensor 的 CUDA device。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .actions import TEACHER_ACTION_DIM
from .math_utils import transform_points_to_world


# =============================================================================
# Observation 结构常量
# =============================================================================
TEACHER_HAND_BODY_COUNT = 13
TEACHER_ARM_HEIGHT_BODY_COUNT = 6
TEACHER_VECTOR_DIM = 3


@dataclass(frozen=True)
class TeacherObservationSpec:
    """Teacher observation 每一段的维度定义。"""

    # FR3 7 + Inspire active 6 = 13。
    active_qpos_dim: int = TEACHER_ACTION_DIM

    # residual control:
    #     q_target_active - q_current_active
    joint_target_error_dim: int = TEACHER_ACTION_DIM

    # 13 个 hand geometry bodies 对 top affordance part 的 contact / impulse。
    affordance_contact_dim: int = TEACHER_HAND_BODY_COUNT
    affordance_impulse_dim: int = TEACHER_HAND_BODY_COUNT

    # 13 个 hand bodies 和 6 个 arm bodies 相对支撑面的高度。
    hand_body_height_dim: int = TEACHER_HAND_BODY_COUNT
    arm_body_height_dim: int = TEACHER_ARM_HEIGHT_BODY_COUNT

    # Palm center 和 wrist orientation。
    palm_center_dim: int = TEACHER_VECTOR_DIM
    wrist_delta_euler_dim: int = TEACHER_VECTOR_DIM
    wrist_euler_dim: int = TEACHER_VECTOR_DIM

    # 每个 hand body 一个 3D nearest-affordance vector：
    #     13 * xyz = 39。
    affordance_vector_dim: int = (
        TEACHER_HAND_BODY_COUNT
        * TEACHER_VECTOR_DIM
    )

    @property
    def base_dim(self) -> int:
        """不包含 privileged affordance vector 的基础 observation 维度。

        13 + 13 + 13 + 13 + 13 + 6 + 3 + 3 + 3 = 80
        """

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
        """Teacher privileged observation 总维度。

        80 base + 39 nearest-affordance vectors = 119。
        """

        return (
            self.base_dim
            + self.affordance_vector_dim
        )


@dataclass(frozen=True)
class TeacherObservationFeatures:
    """env.py 当前 physics step 已计算出的 Teacher 特征。

    这里仅保存 CUDA Tensor 引用，不做 CPU 中转。
    """

    # [B, 13]
    active_qpos: torch.Tensor
    joint_target_error: torch.Tensor
    affordance_contact: torch.Tensor
    affordance_impulse: torch.Tensor
    hand_body_height: torch.Tensor

    # [B, 6]
    arm_body_height: torch.Tensor

    # [B, 3]
    palm_center_world: torch.Tensor
    wrist_delta_euler: torch.Tensor
    wrist_euler: torch.Tensor

    # [B, 13, 3]
    nearest_affordance_vector_world: torch.Tensor

    # [B, 13]
    nearest_affordance_distance: torch.Tensor

    # [B, 13, 3]
    nearest_affordance_point_world: torch.Tensor


TEACHER_OBSERVATION_SPEC = TeacherObservationSpec()


def transform_object_points_to_world(
    points_object: torch.Tensor,
    body_position_world: torch.Tensor,
    body_quaternion_world: torch.Tensor,
) -> torch.Tensor:
    """把 object-local affordance points 批量变换到世界坐标。

    输入：
        points_object:
            [B, N, 3]

        body_position_world:
            [B, 3]

        body_quaternion_world:
            [B, 4], IsaacLab quaternion = (w, x, y, z)

    输出：
        [B, N, 3]

    这里直接复用 math_utils.transform_points_to_world()，
    避免 observation 和 pregrasp 各维护一份相同坐标变换实现。

    当输入位于 cuda:0 时，整个 quaternion transform 始终在 GPU。
    """

    return transform_points_to_world(
        points_object=points_object,
        object_pos_world=body_position_world,
        object_quat_world=body_quaternion_world,
    )


def compute_affordance_geometry(
    hand_body_positions_world: torch.Tensor,
    top_points_world: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """计算每个 hand body 到 top affordance 点云的最近几何关系。

    输入：
        hand_body_positions_world:
            [B, 13, 3]

        top_points_world:
            [B, 200, 3]

    第一步：
        torch.cdist() 一次性在 GPU 计算所有 pairwise distances：

            [B, 13, 3]
                ×
            [B, 200, 3]
                ↓
            [B, 13, 200]

    第二步：
        对 200 个 affordance points 取最小距离：

            nearest_distance [B, 13]
            nearest_index    [B, 13]

    第三步：
        根据 nearest_index 从 top_points_world gather 最近点：

            nearest_point_world [B, 13, 3]

    第四步：
        构造 Teacher privileged vector：

            nearest_vector_world
                = nearest_point_world
                - hand_body_positions_world

    这些 vector 会展平为 13 * 3 = 39 维，并拼到 Teacher observation。
    """

    # -------------------------------------------------------------------------
    # [B, 13, 200]
    # 对当前 13 个手部几何点与 200 个 top affordance 点做批量距离计算。
    # -------------------------------------------------------------------------
    distances = torch.cdist(
        hand_body_positions_world,
        top_points_world,
    )

    # 对每个 hand body 找到距离最近的 top affordance point。
    nearest_distance, nearest_index = torch.min(
        distances,
        dim=2,
    )

    # nearest_index:
    #     [B, 13]
    #
    # 扩展成 [B, 13, 3] 后，从 [B, 200, 3] 的 top point cloud 中 gather xyz。
    # expand() 只是 view，不复制实际索引数据。
    nearest_point_world = torch.gather(
        top_points_world,
        dim=1,
        index=nearest_index.unsqueeze(-1).expand(
            -1,
            -1,
            TEACHER_VECTOR_DIM,
        ),
    )

    # RobustDexGrasp Teacher privileged vector：
    #     从 hand body 指向最近 top-affordance point。
    nearest_vector_world = (
        nearest_point_world
        - hand_body_positions_world
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
    """把 Euler angle 保持在上一时刻附近，避免 ±pi 跳变。

    例如：
        previous = +3.13
        current  = -3.13

    如果直接相减会得到接近 -2*pi 的突变。
    这里把 current 加回 2*pi，使时间序列连续。

    输入 / 输出都是 [B, 3] CUDA Tensor。
    """

    difference = (
        current_euler
        - previous_euler
    )

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
) -> torch.Tensor:
    """按 RobustDexGrasp Teacher 顺序拼接 119 维 observation。

    拼接顺序必须固定：

        [0:13]    active qpos
        [13:26]   joint target error
        [26:39]   affordance contact
        [39:52]   affordance impulse
        [52:65]   hand body height
        [65:71]   arm body height
        [71:74]   palm center
        [74:77]   wrist delta Euler
        [77:80]   wrist Euler
        [80:119]  13 x 3 nearest-affordance vectors

    最后的 privileged geometry 是 39 维：

        [B, 13, 3]
            ↓ flatten
        [B, 39]

    torch.cat() 后得到：
        [B, 119]

    整个操作只处理现有 CUDA Tensor，不创建 CPU 中间数据。
    """

    batch_size = active_qpos.shape[0]

    # [B, 13, 3] -> [B, 39]
    flattened_affordance_vector = (
        affordance_vector_world.reshape(
            batch_size,
            spec.affordance_vector_dim,
        )
    )

    # 所有输入 Tensor 都在同一 env.device 上，
    # torch.cat() 直接在 GPU 构造最终 policy observation。
    return torch.cat(
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


__all__ = [
    "TEACHER_HAND_BODY_COUNT",
    "TEACHER_ARM_HEIGHT_BODY_COUNT",
    "TEACHER_VECTOR_DIM",
    "TeacherObservationSpec",
    "TeacherObservationFeatures",
    "TEACHER_OBSERVATION_SPEC",
    "transform_object_points_to_world",
    "compute_affordance_geometry",
    "unwrap_euler_near_previous",
    "build_teacher_observation",
]
