"""FR3 + Inspire Teacher robot configuration.

本文件负责：
1. 定义 FR3 + Inspire 的 joint / body 语义。
2. 定义 13 维 Teacher active action space：
       FR3 7 DOF
       +
       Inspire 6 active DOF
3. 定义 Inspire 6 个 passive mimic joints。
4. 定义 palm / wrist / fingertip / contact body。
5. 构造 IsaacLab ArticulationCfg 与 actuator 参数。

GPU 说明：
    这里是 IsaacLab / PhysX 的静态机器人配置，不生成 RL 训练 Tensor。
    因此不需要 NumPy，也不需要手动创建 CUDA Tensor。
    真正运行时的 joint state / target / observation / reward 会由
    IsaacLab Articulation.data 和 env.py 保持在 CUDA。

重要修正：
    原文件把 FR3 7 个关节统一设置为：
        effort_limit_sim = 87.0
        velocity_limit_sim = 2.175

    这与当前 FR3 URDF 不一致。

    当前 FR3 safety URDF：
        J1  effort 87   velocity 2.00
        J2  effort 87   velocity 1.00
        J3  effort 87   velocity 1.50
        J4  effort 87   velocity 1.25
        J5  effort 12   velocity 3.00
        J6  effort 12   velocity 1.50
        J7  effort 12   velocity 3.00

    IsaacLab 2.3.2 的 effort_limit_sim / velocity_limit_sim
    支持 dict[str, float]，所以这里按关节精确设置。
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# =============================================================================
# FR3 actuator limits
# =============================================================================
# 与当前：
# assets/robots/inspire_tac/fr3_inspire_tac_L_right_safety.urdf
# 保持一致。
#
# 这些是 PhysX simulation limits，不是 policy action scale。
FR3_ARM_EFFORT_LIMIT_SIM: dict[str, float] = {
    "fr3_joint1": 87.0,
    "fr3_joint2": 87.0,
    "fr3_joint3": 87.0,
    "fr3_joint4": 87.0,
    "fr3_joint5": 12.0,
    "fr3_joint6": 12.0,
    "fr3_joint7": 12.0,
}

FR3_ARM_VELOCITY_LIMIT_SIM: dict[str, float] = {
    "fr3_joint1": 2.0,
    "fr3_joint2": 1.0,
    "fr3_joint3": 1.5,
    "fr3_joint4": 1.25,
    "fr3_joint5": 3.0,
    "fr3_joint6": 1.5,
    "fr3_joint7": 3.0,
}


@dataclass(frozen=True)
class TeacherRobotSpec:
    """FR3 + Inspire Teacher 的机器人语义定义。

    这里区分三个重要概念：

    arm_joint_names:
        FR3 7 个主动关节。

    active_hand_joint_names:
        Inspire 6 个 policy 主动关节。

    passive_joint_mimic:
        Inspire 6 个不直接由 policy 输出的被动关节。
        它们由 actions.py 根据 parent joint * multiplier 计算。

    因此：
        full articulation joint count = 19
        policy action dim             = 13
    """

    name: str
    usd_path: str

    # FR3 7 DOF。
    arm_joint_names: tuple[str, ...]

    # Inspire 6 active DOF。
    active_hand_joint_names: tuple[str, ...]

    # passive joint -> (active parent joint, multiplier)
    passive_joint_mimic: dict[
        str,
        tuple[str, float],
    ]

    # Inspire base/palm 与 FR3 wrist frame。
    palm_link_name: str
    wrist_link_name: str

    fingertip_link_names: tuple[str, ...]
    hand_contact_body_names: tuple[str, ...]
    arm_height_body_names: tuple[str, ...]

    # 相对于 palm_link_name="base_link" 的局部 palm center offset。
    palm_offset: tuple[
        float,
        float,
        float,
    ]

    # 完整 articulation 的默认 joint position。
    # 当前为 19 joints：
    #     FR3 7 + Inspire 12。
    default_joint_pos: tuple[float, ...]

    @property
    def active_joint_names(
        self,
    ) -> tuple[str, ...]:
        """Teacher policy 实际控制的 13 个 joint。"""

        return (
            self.arm_joint_names
            + self.active_hand_joint_names
        )

    @property
    def action_dim(self) -> int:
        """FR3 7 + Inspire active 6 = 13。"""

        return len(
            self.active_joint_names
        )


FR3_INSPIRE_TEACHER_SPEC = TeacherRobotSpec(
    name="fr3_inspire_tac",

    usd_path=(
        "/home/windsky/project/Grasp/assets/robots/inspire_tac/"
        "fr3_inspire_tac_L_right_safety_visual_realistic.usd"
    ),

    # -------------------------------------------------------------------------
    # FR3 7 active joints
    # -------------------------------------------------------------------------
    arm_joint_names=(
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ),

    # -------------------------------------------------------------------------
    # Inspire 6 policy-controlled active joints
    #
    # 顺序必须与 actions / observation / policy action 的 6 hand dims 一致。
    # -------------------------------------------------------------------------
    active_hand_joint_names=(
        "right_little_1_joint",
        "right_ring_1_joint",
        "right_middle_1_joint",
        "right_index_1_joint",
        "right_thumb_2_joint",
        "right_thumb_1_joint",
    ),

    # -------------------------------------------------------------------------
    # Inspire passive mimic joints
    #
    # q_passive = multiplier * q_parent
    #
    # 当前没有非零 offset。
    # -------------------------------------------------------------------------
    passive_joint_mimic={
        "right_thumb_3_joint": (
            "right_thumb_2_joint",
            0.6,
        ),
        "right_thumb_4_joint": (
            "right_thumb_2_joint",
            0.8,
        ),
        "right_index_2_joint": (
            "right_index_1_joint",
            1.05,
        ),
        "right_middle_2_joint": (
            "right_middle_1_joint",
            1.05,
        ),
        "right_ring_2_joint": (
            "right_ring_1_joint",
            1.05,
        ),
        "right_little_2_joint": (
            "right_little_1_joint",
            1.18,
        ),
    },

    # -------------------------------------------------------------------------
    # Palm / wrist frames
    #
    # palm_link_name:
    #     Inspire base_link
    #
    # wrist_link_name:
    #     FR3 fr3_link8
    #
    # 两个 frame 不相同。
    # pregrasp.py 已改为显式包含：
    #     fr3_link8 -> L_flange -> wrist -> base_link
    # -------------------------------------------------------------------------
    palm_link_name="base_link",
    wrist_link_name="fr3_link8",

    fingertip_link_names=(
        "right_thumb_4",
        "right_index_2",
        "right_middle_2",
        "right_ring_2",
        "right_little_2",
    ),

    # -------------------------------------------------------------------------
    # 13 个 hand geometry / contact bodies
    #
    # 与 observations.py 的：
    #     TEACHER_HAND_BODY_COUNT = 13
    # 对齐。
    # -------------------------------------------------------------------------
    hand_contact_body_names=(
        "base_link",
        "right_thumb_1",
        "right_thumb_2",
        "right_thumb_3",
        "right_thumb_4",
        "right_index_1",
        "right_index_2",
        "right_middle_1",
        "right_middle_2",
        "right_ring_1",
        "right_ring_2",
        "right_little_1",
        "right_little_2",
    ),

    # 6 个 FR3 body height / contact bodies。
    arm_height_body_names=(
        "fr3_link1",
        "fr3_link2",
        "fr3_link3",
        "fr3_link4",
        "fr3_link5",
        "fr3_link6",
    ),

    # base_link local frame 中的 palm center offset。
    palm_offset=(
        0.0,
        0.02,
        -0.05,
    ),

    # -------------------------------------------------------------------------
    # 19-joint default pose
    #
    # 前 7 个：
    #     FR3 default configuration
    #
    # 后 12 个：
    #     Inspire open hand
    # -------------------------------------------------------------------------
    default_joint_pos=(
        0.0,
        0.0,
        0.0,
        -1.6,
        0.0,
        1.6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
)


def make_teacher_robot_cfg(
    spec: TeacherRobotSpec = (
        FR3_INSPIRE_TEACHER_SPEC
    ),
) -> ArticulationCfg:
    """构造 FR3 + Inspire Teacher 的 IsaacLab ArticulationCfg。

    本函数只创建静态 simulation configuration。
    不生成 runtime RL Tensor。
    """

    # -------------------------------------------------------------------------
    # Initial joint position
    # -------------------------------------------------------------------------
    arm_joint_count = len(
        spec.arm_joint_names
    )

    joint_pos = {
        name: value
        for name, value in zip(
            spec.arm_joint_names,
            spec.default_joint_pos[
                :arm_joint_count
            ],
            strict=True,
        )
    }

    # Inspire 所有 hand joints 初始为 open hand = 0。
    #
    # 这里使用 regex 是 IsaacLab ArticulationCfg 的静态配置，
    # 与 runtime GPU Tensor 无关。
    joint_pos[
        "right_.*_joint"
    ] = 0.0

    return ArticulationCfg(
        prim_path=(
            "/World/envs/env_.*/Robot"
        ),

        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,

            # Teacher reward / collision screening 需要 contact sensor。
            activate_contact_sensors=True,

            rigid_props=(
                sim_utils.RigidBodyPropertiesCfg(
                    # 当前迁移保持机器人 disable_gravity=True，
                    # 不在本次 actuator-limit 修正中改变算法语义。
                    disable_gravity=True,

                    angular_damping=0.01,

                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,

                    max_depenetration_velocity=1000.0,
                )
            ),

            articulation_props=(
                sim_utils.ArticulationRootPropertiesCfg(
                    # reset 阶段需要检测 FR3 自碰撞。
                    enabled_self_collisions=True,

                    # FR3 base 固定在每个 environment origin。
                    fix_root_link=True,

                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                )
            ),

            collision_props=(
                sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                )
            ),
        ),

        init_state=(
            ArticulationCfg.InitialStateCfg(
                pos=(
                    0.0,
                    0.0,
                    0.0,
                ),
                rot=(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                joint_pos=joint_pos,
            )
        ),

        actuators={
            # =================================================================
            # FR3 arm implicit actuator
            # =================================================================
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(
                    spec.arm_joint_names
                ),

                # -------------------------------------------------------------
                # 关键修正：
                # 不再把全部 7 个 FR3 joints 都设成 87 Nm。
                #
                # J1-J4 = 87 Nm
                # J5-J7 = 12 Nm
                #
                # IsaacLab 2.3.2 支持 dict[str, float]，
                # 因此 PhysX 会按 joint name 分别应用限制。
                # -------------------------------------------------------------
                effort_limit_sim=(
                    FR3_ARM_EFFORT_LIMIT_SIM
                ),

                # -------------------------------------------------------------
                # 关键修正：
                # 不再使用原来的统一速度上限标量。
                #
                # 使用当前 FR3 URDF 的逐关节速度限制：
                # [2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0]
                # -------------------------------------------------------------
                velocity_limit_sim=(
                    FR3_ARM_VELOCITY_LIMIT_SIM
                ),

                # 保持当前 Teacher 的 implicit PD 参数。
                # 这里暂不进行 controller 调参。
                stiffness=400.0,
                damping=40.0,

                friction=0.01,
                armature=0.001,
            ),

            # =================================================================
            # Inspire hand implicit actuator
            # =================================================================
            "hand": ImplicitActuatorCfg(
                # 包含 active + passive hand joints。
                # passive joint target 由 actions.py 的 mimic 逻辑生成。
                joint_names_expr=[
                    "right_.*_joint"
                ],

                # 不在这里覆盖 hand effort / velocity limit。
                #
                # 对 ImplicitActuatorCfg：
                # effort_limit_sim=None / velocity_limit_sim=None
                # 会使用 USD joint prim 中的限制。
                #
                # 这样避免在 robot_cfg.py 再维护一套 Inspire limit。
                stiffness=100.0,
                damping=10.0,

                friction=0.01,
                armature=0.001,
            ),
        },
    )


__all__ = [
    "FR3_ARM_EFFORT_LIMIT_SIM",
    "FR3_ARM_VELOCITY_LIMIT_SIM",
    "TeacherRobotSpec",
    "FR3_INSPIRE_TEACHER_SPEC",
    "make_teacher_robot_cfg",
]
