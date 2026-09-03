from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


@dataclass(frozen=True)
class TeacherRobotSpec:
    name: str
    usd_path: str
    arm_joint_names: tuple[str, ...]
    active_hand_joint_names: tuple[str, ...]
    passive_joint_mimic: dict[str, tuple[str, float]]
    palm_link_name: str
    wrist_link_name: str
    fingertip_link_names: tuple[str, ...]
    hand_contact_body_names: tuple[str, ...]
    arm_height_body_names: tuple[str, ...]
    palm_offset: tuple[float, float, float]
    default_joint_pos: tuple[float, ...]

    @property
    def active_joint_names(self) -> tuple[str, ...]:
        return self.arm_joint_names + self.active_hand_joint_names

    @property
    def action_dim(self) -> int:
        return len(self.active_joint_names)


FR3_INSPIRE_TEACHER_SPEC = TeacherRobotSpec(
    name="fr3_inspire_tac",
    usd_path=(
        "/home/windsky/project/Grasp/assets/robots/inspire_tac/"
        "fr3_inspire_tac_L_right_safety_visual_realistic.usd"
    ),
    arm_joint_names=(
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ),
    active_hand_joint_names=(
        "right_little_1_joint",
        "right_ring_1_joint",
        "right_middle_1_joint",
        "right_index_1_joint",
        "right_thumb_2_joint",
        "right_thumb_1_joint",
    ),
    passive_joint_mimic={
        "right_thumb_3_joint": ("right_thumb_2_joint", 0.6),
        "right_thumb_4_joint": ("right_thumb_2_joint", 0.8),
        "right_index_2_joint": ("right_index_1_joint", 1.05),
        "right_middle_2_joint": ("right_middle_1_joint", 1.05),
        "right_ring_2_joint": ("right_ring_1_joint", 1.05),
        "right_little_2_joint": ("right_little_1_joint", 1.18),
    },
    palm_link_name="base_link",
    wrist_link_name="fr3_link8",
    fingertip_link_names=(
        "right_thumb_4",
        "right_index_2",
        "right_middle_2",
        "right_ring_2",
        "right_little_2",
    ),
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
    arm_height_body_names=(
        "fr3_link1",
        "fr3_link2",
        "fr3_link3",
        "fr3_link4",
        "fr3_link5",
        "fr3_link6",
    ),
    palm_offset=(0.0, 0.02, -0.05),
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
    spec: TeacherRobotSpec = FR3_INSPIRE_TEACHER_SPEC,
) -> ArticulationCfg:
    joint_pos = {
        name: value
        for name, value in zip(
            spec.arm_joint_names,
            spec.default_joint_pos[:7],
            strict=True,
        )
    }
    joint_pos["right_.*_joint"] = 0.0

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=spec.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_pos,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(spec.arm_joint_names),
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=400.0,
                damping=40.0,
                friction=0.01,
                armature=0.001,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                stiffness=100.0,
                damping=10.0,
                friction=0.01,
                armature=0.001,
            ),
        },
    )


__all__ = [
    "TeacherRobotSpec",
    "FR3_INSPIRE_TEACHER_SPEC",
    "make_teacher_robot_cfg",
]
