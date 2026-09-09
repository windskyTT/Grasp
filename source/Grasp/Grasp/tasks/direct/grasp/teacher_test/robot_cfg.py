from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ARM_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
HAND_JOINT_NAMES = tuple(f"joint_{i}_0" for i in range(16))
ACTIVE_JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES

HAND_BODY_PART_NAMES = (
    "Flange_base_link", "link_1_0", "link_2_0", "link_3_0", "link_3_0_tip",
    "link_5_0", "link_6_0", "link_7_0", "link_7_0_tip", "link_9_0", "link_10_0",
    "link_11_0", "link_11_0_tip", "link_13_0", "link_14_0", "link_15_0", "link_15_0_tip",
)
HAND_CONTACT_BODY_NAMES = (
    "wrist_3_link", "link_1_0", "link_2_0", "link_3_0", "link_5_0", "link_6_0",
    "link_7_0", "link_9_0", "link_10_0", "link_11_0", "link_13_0", "link_14_0", "link_15_0",
)
ARM_CONTACT_BODY_NAMES = (
    "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link",
)
ARM_BODY_PART_NAMES = ARM_CONTACT_BODY_NAMES

INIT_FINGER_POSE = (0.2, 0.6, 0.2, 0.5, 0.2, 0.6, 0.2, 0.5, 0.2, 0.6, 0.2, 0.5, 1.3, 0.0, -0.1, 0.2)
DEFAULT_ARM_POSE = (0.0, -1.57, 1.57, 0.0, 1.57, -1.57)

@dataclass(frozen=True)
class RobustDexRobotSpec:
    asset_path: str = "/home/windsky/project/Grasp/rsc/ur5_allegro/ur5_allegro.urdf"
    action_dim: int = 22
    arm_action_scale: float = 0.005
    hand_action_scale: float = 0.015
    hand_center: tuple[float, float, float] = (-0.0091, 0.0, -0.095)

ROBOT_SPEC = RobustDexRobotSpec()

def make_robot_cfg() -> ArticulationCfg:
    joint_pos = dict(zip(ACTIVE_JOINT_NAMES, DEFAULT_ARM_POSE + INIT_FINGER_POSE))
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=ROBOT_SPEC.asset_path,
            fix_base=True,
            merge_fixed_joints=False,
            joint_drive=None,
            collider_type="convex_hull",
            self_collision=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_pos,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                stiffness=3000.0,
                damping=150.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.2,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(HAND_JOINT_NAMES),
                stiffness=60.0,
                damping=0.2,
                effort_limit_sim=1.0,
                velocity_limit_sim=10.0,
            ),
        },
    )

__all__ = [
    "ARM_JOINT_NAMES", "HAND_JOINT_NAMES", "ACTIVE_JOINT_NAMES",
    "HAND_BODY_PART_NAMES", "HAND_CONTACT_BODY_NAMES", "ARM_BODY_PART_NAMES",
    "ARM_CONTACT_BODY_NAMES", "INIT_FINGER_POSE", "DEFAULT_ARM_POSE",
    "RobustDexRobotSpec", "ROBOT_SPEC", "make_robot_cfg",
]
