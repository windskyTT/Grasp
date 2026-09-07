"""Standalone Teacher scene and reward configuration."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .actions import TEACHER_ACTION_DIM
from .object_pair_spawner import TeacherObjectPairSpawnerCfg
from .object_set import (
    NEW_TRAINING_SET_ROOT,
    TEACHER_FULL_ENV_COUNT,
    TEACHER_OBJECT_NAMES,
    TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS,
    TEACHER_WEIGHTED_OBJECT_INDICES,
    TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS,
)
from .observations import TEACHER_OBSERVATION_SPEC
from .robot_cfg import (
    FR3_INSPIRE_TEACHER_SPEC,
    TeacherRobotSpec,
    make_teacher_robot_cfg,
)

TEACHER_CONTACT_FILTER_PRIM_PATHS = (
    "{ENV_REGEX_NS}/Object/top",
    "{ENV_REGEX_NS}/Object/bottom",
    "{ENV_REGEX_NS}/Table",
    "{ENV_REGEX_NS}/Mat",
    "{ENV_REGEX_NS}/WoodenTable",
)

TEACHER_DEFAULT_FRICTION = 0.8
TEACHER_SUPPORT_FRICTION = 0.2
TEACHER_RESTITUTION = 0.0

# =============================================================================
# Teacher GPU / 时间步硬约束
# =============================================================================
TEACHER_SIM_DEVICE = "cuda:0"

# 与原 RobustDexGrasp Teacher 对齐：
# physics dt = 0.01 s
# 每个 RL action 执行 20 个 physics substeps
# 因此 control dt = 0.01 * 20 = 0.2 s
TEACHER_PHYSICS_DT = 0.01
TEACHER_DECIMATION = 20

# Teacher grasp rollout 长度为 70 个 policy steps：
# 70 * 0.2 s = 14 s
TEACHER_GRASP_STEPS = 70


@configclass
class TeacherAssetCfg:
    dataset_root: str = str(NEW_TRAINING_SET_ROOT)
    object_names: tuple[str, ...] = TEACHER_OBJECT_NAMES
    object_top_usd_paths: tuple[str, ...] = (
        TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS
    )
    object_bottom_usd_paths: tuple[str, ...] = (
        TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS
    )
    weighted_object_indices: tuple[int, ...] = (
        TEACHER_WEIGHTED_OBJECT_INDICES
    )
    stable_state_paths: tuple[str, ...] = ()
    stable_state_source_support_height: float = 0.773


@configclass
class TeacherGeometryCfg:
    table_height: float = 0.018
    table_size: tuple[float, float, float] = (0.9, 0.8, 0.3)
    mat_size: tuple[float, float, float] = (0.6, 0.8, 0.003)
    wooden_table_size: tuple[float, float, float] = (0.9, 1.5, 0.004)
    center_xy: tuple[float, float] = (0.51, -0.075)


@configclass
class TeacherActionCfg:
    arm_residual_scale: float = 0.005
    hand_residual_scale: float = 0.015
    delay_probability: float = 0.5


@configclass
class TeacherResetCfg:
    non_uniform_sampling: bool = True
    uniform_sampling_probability: float = 0.5
    beta_alpha: float = 0.5
    beta_beta: float = 0.5
    angle_range: tuple[float, float] = (
        -0.7 * math.pi,
        -0.3 * math.pi,
    )
    distance_range: tuple[float, float] = (0.45, 0.75)
    source_to_fr3_angle_offset: float = 0.5 * math.pi
    local_y_range: tuple[float, float] = (-0.25, 0.25)
    yaw_range: tuple[float, float] = (-math.pi, math.pi)
    biased: bool = False
    biased_distance_threshold: float = 0.07
    biased_position_range: float = 0.05


@configclass
class TeacherPregraspCfg:
    camera_position: tuple[float, float, float] = (
        0.035,
        -0.58,
        1.531,
    )
    top_grasp: bool = False
    candidate_count: int = 10
    approach_distance: float = 0.25
    projection_limit: float = 0.18
    length_score_coeff: float = 5.0
    posture_score_coeff: float = 1.0
    posture_joint_index: int = 5
    posture_joint_target: float = 1.6
    posture_limit_target: float = 3.2
    posture_limit_score_coeff: float = 0.5
    ik_damping: float = 0.05
    ik_step_scale: float = 0.5
    ik_max_iterations: int = 64
    ik_position_tolerance: float = 0.003
    ik_rotation_tolerance: float = 0.03
    max_reset_rounds: int = 32


@configclass
class TeacherRewardCfg:
    affordance_reward_scale: float = 0.5
    affordance_contact_reward_scale: float = 1.5
    affordance_impulse_reward_scale: float = 1.0
    table_reward_scale: float = -0.03
    table_contact_reward_scale: float = -1.0
    table_impulse_reward_scale: float = -0.5
    arm_height_reward_scale: float = -0.05
    arm_contact_reward_scale: float = -0.1
    arm_impulse_reward_scale: float = -0.1
    arm_collision_reward_scale: float = -1.0
    push_reward_scale: float = -0.0
    wrist_vel_reward_scale: float = -1.0
    wrist_qvel_reward_scale: float = -0.1
    obj_vel_reward_scale: float = -15.0
    obj_qvel_reward_scale: float = -0.2
    obj_displacement_reward_scale: float = -5.0
    arm_joint_vel_reward_scale: float = -1.0

    contact_impulse_threshold: float = 0.01
    wrist_velocity_threshold: float = 0.25
    wrist_velocity_multiplier: float = 10.0
    arm_joint_velocity_threshold: float = 0.5
    arm_joint_velocity_multiplier: float = 4.0
    push_palm_threshold: float = 1.0
    push_finger_threshold: float = 2.0
    push_maximum: float = 10.0
    lift_success_height: float = 0.10
    min_reward: float = -2.0

    hand_geometry_weights: tuple[float, ...] = tuple(
        value * 12.0 / 32.0
        for value in (0, 1, 1, 1, 8, 1, 4, 1, 4, 1, 4, 1, 4)
    )
    hand_contact_weights: tuple[float, ...] = tuple(
        value * 13.0 / 33.0
        for value in (0, 1, 2, 2, 12, 1, 3, 1, 3, 1, 3, 1, 3)
    )
    hand_impulse_upper: tuple[float, ...] = (
        0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    )
    arm_height_penalty_body_names: tuple[str, ...] = (
        "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6",
    )
    arm_collision_body_names: tuple[str, ...] = (
        "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5",
    )


def make_teacher_default_material_cfg() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=TEACHER_DEFAULT_FRICTION,
        dynamic_friction=TEACHER_DEFAULT_FRICTION,
        restitution=TEACHER_RESTITUTION,
        friction_combine_mode="min",
        restitution_combine_mode="min",
    )


def make_teacher_support_material_cfg() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=TEACHER_SUPPORT_FRICTION,
        dynamic_friction=TEACHER_SUPPORT_FRICTION,
        restitution=TEACHER_RESTITUTION,
        friction_combine_mode="min",
        restitution_combine_mode="min",
    )


def make_teacher_contact_sensor_cfgs(
    robot_spec: TeacherRobotSpec,
) -> dict[str, ContactSensorCfg]:
    sensor_cfgs: dict[str, ContactSensorCfg] = {}
    body_names = (
        robot_spec.hand_contact_body_names
        + robot_spec.arm_height_body_names
    )
    # Isaac Lab filtered contact supports one sensor body against many
    # filter bodies, so each robot body needs its own ContactSensor.
    for body_name in body_names:
        sensor_cfgs[f"contact__{body_name}"] = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            track_pose=False,
            track_contact_points=False,
            track_friction_forces=True,
            max_contact_data_count_per_prim=128,
            track_air_time=False,
            filter_prim_paths_expr=list(
                TEACHER_CONTACT_FILTER_PRIM_PATHS
            ),
        )
    return sensor_cfgs


def make_static_box_cfg(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=make_teacher_support_material_cfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def make_teacher_table_cfg(
    geometry: TeacherGeometryCfg,
) -> RigidObjectCfg:
    return make_static_box_cfg(
        prim_path="/World/envs/env_.*/Table",
        size=geometry.table_size,
        pos=(
            geometry.center_xy[0],
            geometry.center_xy[1],
            geometry.table_height
            - geometry.mat_size[2]
            - 0.5 * geometry.table_size[2],
        ),
        color=(0.10, 0.10, 0.10),
    )


def make_teacher_mat_cfg(
    geometry: TeacherGeometryCfg,
) -> RigidObjectCfg:
    return make_static_box_cfg(
        prim_path="/World/envs/env_.*/Mat",
        size=geometry.mat_size,
        pos=(
            geometry.center_xy[0],
            geometry.center_xy[1],
            geometry.table_height
            - 0.5 * geometry.mat_size[2],
        ),
        color=(0.86, 0.84, 0.76),
    )


def make_teacher_wooden_table_cfg(
    geometry: TeacherGeometryCfg,
) -> RigidObjectCfg:
    return make_static_box_cfg(
        prim_path="/World/envs/env_.*/WoodenTable",
        size=geometry.wooden_table_size,
        pos=(
            geometry.center_xy[0],
            geometry.center_xy[1],
            0.5 * geometry.wooden_table_size[2],
        ),
        color=(0.78, 0.74, 0.64),
    )


@configclass
class GraspTeacherEnvCfg(DirectRLEnvCfg):
    robot_spec: TeacherRobotSpec = FR3_INSPIRE_TEACHER_SPEC
    asset: TeacherAssetCfg = TeacherAssetCfg()
    geometry: TeacherGeometryCfg = TeacherGeometryCfg()
    action: TeacherActionCfg = TeacherActionCfg()
    reset: TeacherResetCfg = TeacherResetCfg()
    pregrasp: TeacherPregraspCfg = TeacherPregraspCfg()
    reward: TeacherRewardCfg = TeacherRewardCfg()

    # -------------------------------------------------------------------------
    # RL / physics 时间尺度
    # -------------------------------------------------------------------------
    # 一个 policy action 对应 TEACHER_DECIMATION 个 PhysX step。
    decimation: int = TEACHER_DECIMATION

    # 70 policy steps * 20 substeps * 0.01 s = 14 s。
    episode_length_s: float = (
        TEACHER_GRASP_STEPS
        * TEACHER_PHYSICS_DT
        * TEACHER_DECIMATION
    )

    is_finite_horizon: bool = False

    # FR3 7 active DOF + Inspire 6 active DOF = 13。
    # 直接复用 actions.py 的常量，防止两处配置以后改不同步。
    action_space: int = TEACHER_ACTION_DIM
    observation_space: int = TEACHER_OBSERVATION_SPEC.teacher_dim
    state_space: int = 0

    enable_grid_ground: bool = True
    grid_ground_prim_path: str = "/World/GridGround"
    grid_ground_size: float = 40.0
    grid_ground_spacing: float = 0.5
    grid_ground_line_width: float = 0.012
    grid_ground_line_height: float = 0.002
    grid_ground_z: float = 0.0
    grid_ground_base_color: tuple[float, float, float] = (
        0.005, 0.005, 0.005,
    )
    grid_ground_line_color: tuple[float, float, float] = (
        0.32, 0.32, 0.32,
    )
    grid_ground_axis_line_color: tuple[float, float, float] = (
        0.62, 0.62, 0.62,
    )

    sim: SimulationCfg = SimulationCfg(
        # 强制 PhysX 使用 CUDA。
        # 训练数值 tensor 也应由 env.device 继承到同一 CUDA device。
        device=TEACHER_SIM_DEVICE,

        # PhysX 单步 0.01 s；配合 decimation=20 得到 control_dt=0.2 s。
        dt=TEACHER_PHYSICS_DT,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=make_teacher_default_material_cfg(),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 当前 object set 的完整并行环境数量。
        num_envs=TEACHER_FULL_ENV_COUNT,
        env_spacing=1.2,
        replicate_physics=False,
        clone_in_fabric=False,
        lazy_sensor_update=True,
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.55, -1.65, 1.35),
        lookat=(0.45, -0.05, 0.35),
    )
    light: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(
        intensity=2000.0,
        color=(0.75, 0.75, 0.75),
    )
    robot: ArticulationCfg = make_teacher_robot_cfg(
        spec=FR3_INSPIRE_TEACHER_SPEC,
    )
    object: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        articulation_root_prim_path="/bottom",
        spawn=TeacherObjectPairSpawnerCfg(
            top_usd_paths=list(
                TEACHER_WEIGHTED_OBJECT_TOP_USD_PATHS
            ),
            bottom_usd_paths=list(
                TEACHER_WEIGHTED_OBJECT_BOTTOM_USD_PATHS
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            semantic_tags=[("class", "object")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    table: RigidObjectCfg = make_teacher_table_cfg(
        TeacherGeometryCfg()
    )
    mat: RigidObjectCfg = make_teacher_mat_cfg(
        TeacherGeometryCfg()
    )
    wooden_table: RigidObjectCfg = (
        make_teacher_wooden_table_cfg(
            TeacherGeometryCfg()
        )
    )
    contact_sensors: dict[str, ContactSensorCfg] = make_teacher_contact_sensor_cfgs(
        FR3_INSPIRE_TEACHER_SPEC,
    )

    def __post_init__(self) -> None:
        # 只根据最终 geometry 配置重新构造支撑面。
        # 这里不创建任何训练 tensor，也不存在 CPU -> GPU 数值搬运。
        self.table = make_teacher_table_cfg(self.geometry)
        self.mat = make_teacher_mat_cfg(self.geometry)
        self.wooden_table = (
            make_teacher_wooden_table_cfg(self.geometry)
        )


__all__ = [
    "TEACHER_SIM_DEVICE",
    "TEACHER_PHYSICS_DT",
    "TEACHER_DECIMATION",
    "TEACHER_GRASP_STEPS",
    "GraspTeacherEnvCfg",
    "TeacherActionCfg",
    "TeacherAssetCfg",
    "TeacherGeometryCfg",
    "TeacherPregraspCfg",
    "TeacherResetCfg",
    "TeacherRewardCfg",
]
