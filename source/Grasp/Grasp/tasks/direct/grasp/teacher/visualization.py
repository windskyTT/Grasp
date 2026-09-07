"""Teacher affordance visualization.

本文件只用于可视化 / evaluation，不参与 Teacher PPO 训练数值反馈。

显示内容：
    红色 sphere   = 13 个 hand geometry bodies
    绿色 sphere   = 每个 hand body 的最近 top-affordance point
    青色 cylinder = hand body -> nearest affordance point
    黄色 sphere   = top affordance point cloud center
    蓝色 sphere   = Inspire palm center

GPU 约定：
    - 本文件不使用 NumPy。
    - 所有几何计算都使用现有 CUDA Tensor。
    - 不主动执行 GPU -> CPU 数值转换。
    - IsaacLab VisualizationMarkers.visualize() 最终必须把数据写入 USD
      PointInstancer，因此官方实现会在“最终渲染输出边界”内部转换为 CPU/NumPy。
      这些输出数据不会再返回 Teacher policy / reward / physics 计算。

性能处理：
    - 静态 marker indices 只在第一次 visualize() 时提交。
    - line 的 z-axis / fallback-axis 在 CUDA 上缓存。
    - identity quaternion / base scales 在 CUDA 上缓存。
    - top affordance center 在 object-local frame 中只计算一次：
          mean(R p + t) = R mean(p) + t
      因此每个 visualization step 不再变换全部 200 个 top points，
      只变换每个环境 1 个 affordance-center point。
"""

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
    TEACHER_HAND_BODY_COUNT,
    TeacherObservationFeatures,
    transform_object_points_to_world,
)


# =============================================================================
# Marker prototype indices
# =============================================================================
HAND_BODY_MARKER_INDEX = 0
NEAREST_POINT_MARKER_INDEX = 1
CONNECTING_LINE_MARKER_INDEX = 2
AFFORDANCE_CENTER_MARKER_INDEX = 3
PALM_CENTER_MARKER_INDEX = 4


def make_teacher_affordance_marker_cfg() -> VisualizationMarkersCfg:
    """创建 Teacher affordance visualization 的 USD marker prototypes。

    这里都是静态 Isaac Sim / USD 配置，不属于训练数值路径。
    """

    return VisualizationMarkersCfg(
        prim_path="/Visuals/TeacherAffordance",
        markers={
            # -------------------------------------------------------------
            # 0: hand geometry body
            # -------------------------------------------------------------
            "hand_body": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            0.9,
                            0.1,
                            0.1,
                        ),
                    )
                ),
            ),

            # -------------------------------------------------------------
            # 1: nearest top-affordance point
            # -------------------------------------------------------------
            "nearest_point": sim_utils.SphereCfg(
                radius=0.006,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            0.1,
                            0.9,
                            0.1,
                        ),
                    )
                ),
            ),

            # -------------------------------------------------------------
            # 2: hand -> nearest point connection
            #
            # cylinder prototype 的默认 local z 长度为 1。
            # update() 中通过 scale[:, 2] = line_length 改成真实长度。
            # -------------------------------------------------------------
            "connecting_line": sim_utils.CylinderCfg(
                radius=0.0015,
                height=1.0,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            0.1,
                            0.8,
                            0.9,
                        ),
                    )
                ),
            ),

            # -------------------------------------------------------------
            # 3: affordance point cloud center
            # -------------------------------------------------------------
            "affordance_center": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            1.0,
                            0.8,
                            0.0,
                        ),
                    )
                ),
            ),

            # -------------------------------------------------------------
            # 4: Inspire palm center
            # -------------------------------------------------------------
            "palm_center": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            0.2,
                            0.3,
                            1.0,
                        ),
                    )
                ),
            ),
        },
    )


@torch.no_grad()
def _connecting_line_pose(
    start: torch.Tensor,
    end: torch.Tensor,
    z_axis: torch.Tensor,
    fallback_axis: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """计算连接线 cylinder 的位置、姿态和长度。

    输入：
        start:
            [M, 3]
            hand body world position。

        end:
            [M, 3]
            nearest affordance point world position。

        z_axis:
            [M, 3]
            缓存的 cylinder prototype local +z reference。

        fallback_axis:
            [M, 3]
            z 与 line direction 平行时使用的旋转轴。

    输出：
        midpoint:
            [M, 3]

        orientation:
            [M, 4], quaternion(wxyz)

        length:
            [M]

    CylinderCfg(height=1.0) 的 local z axis 要旋转到：

        direction = end - start

    然后 scale.z = length。
    """

    direction = (
        end
        - start
    )

    length = torch.linalg.vector_norm(
        direction,
        dim=-1,
    )

    midpoint = (
        0.5
        * (
            start
            + end
        )
    )

    direction_unit = normalize(
        direction
    )

    # cylinder local +z 旋转到目标 direction：
    #
    #     axis = z × direction
    rotation_axis = torch.linalg.cross(
        z_axis,
        direction_unit,
        dim=-1,
    )

    rotation_axis_norm = (
        torch.linalg.vector_norm(
            rotation_axis,
            dim=-1,
        )
    )

    non_parallel = (
        rotation_axis_norm
        > 1.0e-6
    )

    # 平行 / 反平行时 cross product 接近 0，
    # 使用预先缓存的 x-axis 作为旋转轴。
    rotation_axis = torch.where(
        non_parallel.unsqueeze(-1),
        normalize(
            rotation_axis
        ),
        fallback_axis,
    )

    # angle = acos(z · direction)
    angle = torch.acos(
        torch.clamp(
            torch.sum(
                z_axis
                * direction_unit,
                dim=-1,
            ),
            -1.0,
            1.0,
        )
    )

    orientation = (
        quat_from_angle_axis(
            angle,
            rotation_axis,
        )
    )

    return (
        midpoint,
        orientation,
        length,
    )


class TeacherAffordanceVisualizer:
    """Teacher affordance 的 GPU geometry + USD marker visualizer."""

    def __init__(self) -> None:
        self.markers = VisualizationMarkers(
            make_teacher_affordance_marker_cfg()
        )

        # -------------------------------------------------------------
        # 以下 CUDA cache 在第一次 update() 时根据当前 env 建立。
        #
        # 这样避免每一个 visualization step 都重复创建：
        #     z-axis
        #     fallback-axis
        #     identity quaternion
        #     base scales
        #     marker indices
        #     object-local affordance center
        # -------------------------------------------------------------
        self._line_z_axis: torch.Tensor | None = None
        self._line_fallback_axis: torch.Tensor | None = None
        self._identity_orientations: torch.Tensor | None = None
        self._base_scales: torch.Tensor | None = None
        self._marker_indices: torch.Tensor | None = None
        self._affordance_center_object: torch.Tensor | None = None

        # marker prototype indices 只需第一次提交给 PointInstancer。
        self._marker_indices_submitted = False

    @torch.no_grad()
    def _initialize_gpu_cache(
        self,
        env: Any,
        dtype: torch.dtype,
    ) -> None:
        """第一次 visualization update 时直接在 CUDA 建立静态缓存。"""

        device = env.device

        hand_marker_count = (
            env.num_envs
            * TEACHER_HAND_BODY_COUNT
        )

        total_marker_count = (
            hand_marker_count       # hand bodies
            + hand_marker_count     # nearest points
            + hand_marker_count     # connecting lines
            + env.num_envs          # affordance center
            + env.num_envs          # palm center
        )

        # -------------------------------------------------------------
        # Connecting-line reference axes。
        # -------------------------------------------------------------
        self._line_z_axis = torch.zeros(
            (
                hand_marker_count,
                3,
            ),
            dtype=dtype,
            device=device,
        )
        self._line_z_axis[:, 2] = 1.0

        self._line_fallback_axis = torch.zeros(
            (
                hand_marker_count,
                3,
            ),
            dtype=dtype,
            device=device,
        )
        self._line_fallback_axis[:, 0] = 1.0

        # -------------------------------------------------------------
        # 所有 sphere marker 默认 quaternion = identity。
        # connecting-line slice 每帧会覆盖成动态 quaternion。
        # -------------------------------------------------------------
        self._identity_orientations = torch.zeros(
            (
                total_marker_count,
                4,
            ),
            dtype=dtype,
            device=device,
        )
        self._identity_orientations[:, 0] = 1.0

        # 默认 scale = (1,1,1)。
        # line marker 的 z scale 每帧改成 line length。
        self._base_scales = torch.ones(
            (
                total_marker_count,
                3,
            ),
            dtype=dtype,
            device=device,
        )

        # -------------------------------------------------------------
        # Marker prototype indices。
        #
        # marker 顺序：
        #     hand bodies
        #     nearest points
        #     lines
        #     affordance centers
        #     palm centers
        # -------------------------------------------------------------
        self._marker_indices = torch.cat(
            (
                torch.full(
                    (hand_marker_count,),
                    HAND_BODY_MARKER_INDEX,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (hand_marker_count,),
                    NEAREST_POINT_MARKER_INDEX,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (hand_marker_count,),
                    CONNECTING_LINE_MARKER_INDEX,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (env.num_envs,),
                    AFFORDANCE_CENTER_MARKER_INDEX,
                    dtype=torch.long,
                    device=device,
                ),
                torch.full(
                    (env.num_envs,),
                    PALM_CENTER_MARKER_INDEX,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=0,
        )

        # -------------------------------------------------------------
        # 200 top points 的 object-local center 只计算一次。
        #
        # 因为 rigid transform 是线性的：
        #
        #     mean(R p_i + t)
        #       =
        #     R mean(p_i) + t
        #
        # 所以后续每帧只需要 transform 1 个 center point / env，
        # 不需要重新 transform 200 个点 / env。
        # -------------------------------------------------------------
        self._affordance_center_object = (
            env.affordance_data
            .top_points_object
            .mean(
                dim=1,
            )
        )

    @torch.no_grad()
    def update(
        self,
        env: Any,
        features: TeacherObservationFeatures,
    ) -> None:
        """更新当前 frame 的 Teacher affordance markers。

        所有 marker geometry 先在 CUDA 计算完成。
        最后的 self.markers.visualize() 是 USD/viewport 输出边界。
        """

        # ---------------------------------------------------------------------
        # 13 个 hand geometry bodies：
        #     [B, 13, 3]
        # ---------------------------------------------------------------------
        hand_body_position_w = (
            env.robot.data.body_pos_w[
                :,
                env.hand_body_ids,
                :,
            ]
        )

        # features 中已经由 observation geometry 得到：
        #     [B, 13, 3]
        nearest_point_w = (
            features.nearest_affordance_point_world
        )

        hand_flat = (
            hand_body_position_w.reshape(
                -1,
                3,
            )
        )

        nearest_flat = (
            nearest_point_w.reshape(
                -1,
                3,
            )
        )

        if self._line_z_axis is None:
            self._initialize_gpu_cache(
                env=env,
                dtype=hand_flat.dtype,
            )

        # ---------------------------------------------------------------------
        # hand -> nearest affordance 连接线。
        # ---------------------------------------------------------------------
        (
            line_position,
            line_orientation,
            line_length,
        ) = _connecting_line_pose(
            start=hand_flat,
            end=nearest_flat,
            z_axis=self._line_z_axis,
            fallback_axis=(
                self._line_fallback_axis
            ),
        )

        # ---------------------------------------------------------------------
        # Top affordance center。
        #
        # 不再每帧：
        #     [B,200,3] local points
        #         -> transform all 200 points
        #         -> mean
        #
        # 现在：
        #     cached local center [B,3]
        #         -> transform one point/env
        # ---------------------------------------------------------------------
        top_position_w = (
            env.object.data.body_pos_w[
                :,
                env.object_top_body_id,
                :,
            ]
        )

        top_quaternion_w = (
            env.object.data.body_quat_w[
                :,
                env.object_top_body_id,
                :,
            ]
        )

        affordance_center_w = (
            transform_object_points_to_world(
                (
                    self._affordance_center_object[
                        :,
                        None,
                        :,
                    ]
                ),
                top_position_w,
                top_quaternion_w,
            )
            .squeeze(1)
        )

        # ---------------------------------------------------------------------
        # features.palm_center_world 是 env-local observation quantity。
        #
        # visualization 需要 world position，所以加回 env origin。
        # ---------------------------------------------------------------------
        palm_center_global_w = (
            features.palm_center_world
            + env.scene.env_origins
        )

        # ---------------------------------------------------------------------
        # Marker translation 顺序必须与 _marker_indices 一致。
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Orientation：
        #
        # sphere marker = identity quaternion
        # line marker   = dynamic quaternion
        # ---------------------------------------------------------------------
        orientations = (
            self._identity_orientations.clone()
        )

        line_start = (
            hand_flat.shape[0]
            + nearest_flat.shape[0]
        )

        line_end = (
            line_start
            + line_position.shape[0]
        )

        orientations[
            line_start:line_end
        ] = line_orientation

        # ---------------------------------------------------------------------
        # Scale：
        #
        # sphere = (1,1,1)
        # line   = (1,1,length)
        # ---------------------------------------------------------------------
        scales = (
            self._base_scales.clone()
        )

        scales[
            line_start:line_end,
            2,
        ] = line_length

        # ---------------------------------------------------------------------
        # IsaacLab VisualizationMarkers 最终会把 Tensor 写入 USD。
        #
        # prototype indices 对于同一个 visualizer 是静态的，
        # 所以只在第一次 update 时提交。
        # 后续 marker 数量不变时，官方 PointInstancer 会保留原 indices。
        # ---------------------------------------------------------------------
        marker_indices = None

        if not self._marker_indices_submitted:
            marker_indices = (
                self._marker_indices
            )
            self._marker_indices_submitted = True

        self.markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )


__all__ = [
    "HAND_BODY_MARKER_INDEX",
    "NEAREST_POINT_MARKER_INDEX",
    "CONNECTING_LINE_MARKER_INDEX",
    "AFFORDANCE_CENTER_MARKER_INDEX",
    "PALM_CENTER_MARKER_INDEX",
    "TeacherAffordanceVisualizer",
    "make_teacher_affordance_marker_cfg",
]
