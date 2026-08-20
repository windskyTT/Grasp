from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.sensors import TiledCameraCfg

import numpy as np
from scipy.spatial.transform import Rotation


from .hand_cfg import FR3_INSPIRE_TAC_CFG, HandCfg
from .object_loader import load_multi_object_usd_cfgs


PROJECT_ROOT = "/home/windsky/project/Grasp"
ASSET_ROOT = f"{PROJECT_ROOT}/assets"
REFERENCE_ROOT = f"{PROJECT_ROOT}/source/Grasp/Grasp/reference"


@configclass
class ResetCfg:
    settle_object_after_reset: bool = True
    settle_time_s: float = 2.0
    
    reset_position_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.3, 0.8),
        (-0.35, 0.15),
        (0.1, 0.12),
    )
    reset_random_rot: str = "random"
    table_height_range: tuple[float, float] = (0.018, 0.018)
    reset_dof_pos_random_interval: float = 0.2
    reset_hand_dof_pos_full_range: bool = True
    ee_safe_workspace: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.15, -0.45, 0.05),
        (0.95, 0.25, 0.95),
    )

@configclass
class ControlCfg:
    use_relative_control: bool = False
    arm_controller: str = "qpos"
    actions_max_ang_vel_arm: float = 1.57
    actions_max_ang_vel_hand: float = 6.28
    delta_action_scale: tuple[float, ...] = (
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    control_frequency_inv: int = 1
    limit_control_error: bool = False
    max_pd_error_ee_pos: float = 0.03
    max_pd_error_hand: float = 0.35
    pd_param_scale: float = 1.0
    interpolation_step_scale: float = 1.0

@configclass
class ReferenceCfg:
    tracking_reference_file: str = f"{REFERENCE_ROOT}/grasp_ref_inspire.pkl"
    tracking_reference_lift_timestep: int = 13
    randomize_tracking_reference: bool = False
    randomize_tracking_reference_range: tuple[float, float, float, float, float, float] = (
        0.05,
        0.05,
        0.05,
        1.57,
        1.57,
        1.57,
    )
    randomize_grasp_pose: bool = False
    randomize_grasp_pose_range: float = 1.0


@configclass
class AssetCfg:
    asset_root: str = ASSET_ROOT
    multi_object: bool = True
    multi_object_list: str = "union_ycb_unidex/union_ycb_debugset.yaml"
    use_distractor_objects: bool = False
    num_distractor_objects: int = 5
    random_remove_distractor_objects: float = 0.5


@configclass
class RenderRandomizationCfg:
    camera_pos: tuple[float, float] = (0.0, 0.0)
    camera_quat: tuple[float, float] = (0.0, 0.0)
    depth_range: float = 0.05
    object_random_texture: bool = True
    texture_folder: str = "textures"
    object_color_choices: tuple[str, ...] = (
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "white",
        "black",
        "gray",
        "orange",
        "purple",
        "pink",
        "brown",
        "olive",
        "teal",
        "navy",
        "maroon",
        "lime",
        "gold",
        "silver",
        "bronze",
    )
    color: float = 0.2
    num_lights: int = 3
    light_intensity: tuple[float, float] = (0.1, 0.8)
    light_ambient: tuple[float, float] = (0.1, 0.8)
    table_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


@configclass
class RenderCfg:
    enable: bool = False
    appearance_realistic: bool = True
    camera_ids: tuple[int, ...] = (1, 2)
    data_type: str = "rgb"
    instruction_template: str = "Grasp the object."
    use_advanced_instruction: bool = False
    advanced_instruction_template: str = "Grasp the {COLOR} {OBJ}."
    object_name_list: str = "union_ycb_unidex/union_ycb_debugset_names.yaml"
    save_depth_range: tuple[float, float] = (0.15, 1.0)
    pcl_clip_workspace: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.25, -0.45, 0.0),
        (0.85, 0.3, 0.6),
    )
    n_pcl_downsample: int = 2048
    resize: tuple[int, int] = (256, 256)
    randomize: bool = False
    randomization_params: RenderRandomizationCfg = RenderRandomizationCfg()


@configclass
class CameraCalibrationCfg:
    type: str = MISSING
    mount: str = MISSING
    width: int = MISSING
    height: int = MISSING
    depth_range: tuple[float, float] = MISSING
    intrinsics: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = MISSING
    extrinsics: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = MISSING


CAMERA_1 = CameraCalibrationCfg(
    type="D435",
    mount="fixed",
    width=640,
    height=480,
    depth_range=(0.15, 3.0),
    intrinsics=((608.10601807, 0.0, 323.16036987), (0.0, 607.04846191, 245.16740417), (0.0, 0.0, 1.0)),
    extrinsics=(
        (-0.95458387, -0.22931432, 0.19022242, 0.25351984),
        (-0.29675754, 0.78866419, -0.53846426, 0.47084633),
        (-0.02654405, -0.57045923, -0.82089687, 0.62032344),
        (0.0, 0.0, 0.0, 1.0),
    ),
)

CAMERA_2 = CameraCalibrationCfg(
    type="D435",
    mount="fixed",
    width=640,
    height=480,
    depth_range=(0.15, 3.0),
    intrinsics=((610.1685791, 0.0, 327.0413208), (0.0, 609.07666016, 245.29025269), (0.0, 0.0, 1.0)),
    extrinsics=(
        (0.6464361, 0.58084152, -0.49471557, 0.89060788),
        (0.76291282, -0.4842839, 0.42829095, -0.44560813),
        (0.00918638, -0.65428758, -0.75619004, 0.49017396),
        (0.0, 0.0, 0.0, 1.0),
    ),
)

def camera_data_types(render_data_type: str) -> list[str]:
    data_types = ["rgb"]
    if "depth" in render_data_type or "pcl" in render_data_type:
        data_types.append("depth")
    if "seg" in render_data_type:
        data_types.append("semantic_segmentation")
    return data_types

def make_tiled_camera_cfg(camera_id: int, calibration: CameraCalibrationCfg, render_data_type: str = "rgb") -> TiledCameraCfg:
    extrinsics = np.asarray(calibration.extrinsics, dtype=np.float64)
    quat_xyzw = Rotation.from_matrix(extrinsics[:3, :3]).as_quat()
    quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    pos = (
        float(extrinsics[0, 3]),
        float(extrinsics[1, 3]),
        float(extrinsics[2, 3]),
    )
    intrinsic_matrix = [float(value) for row in calibration.intrinsics for value in row]
    data_types = camera_data_types(render_data_type)
    return TiledCameraCfg(
        prim_path=f"/World/envs/env_.*/Camera_{camera_id}",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=quat_wxyz, convention="world"),
        data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=calibration.width,
            height=calibration.height,
            clipping_range=calibration.depth_range,
        ),
        width=calibration.width,
        height=calibration.height,
    )


@configclass
class EventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.0, 0.01),
            "upper_limit_distribution_params": (0.0, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    robot_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (0.95, 1.05),
        },
    )

    object_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(720 * 0.01667 * 20, 720 * 0.01667 * 20),
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    object_visual_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mesh_name": "/.*",
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    object_visual_texture = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "event_name": "randomize_object_texture",
            "texture_paths": [
                f"{ASSET_ROOT}/textures/object/glass_glass_moderate_027_new.jpg",
                f"{ASSET_ROOT}/textures/object/leather_leather_object_009_new.jpg",
                f"{ASSET_ROOT}/textures/object/plastic_plastic_object_003_new.jpg",
                f"{ASSET_ROOT}/textures/object/wood_wood_object_004_new.jpg",
                f"{ASSET_ROOT}/textures/object/metal_metal_object_026_new.jpg",
            ],
            "texture_rotation": (0.0, 3.14159),
        },
    )

def make_robot_cfg(
    hand: HandCfg,
    use_robot_vhacd: bool,
    enable_self_collision: bool,
    pd_param_scale: float,
    appearance_realistic: bool = True,
) -> ArticulationCfg:
    robot_asset_file = hand.robot_asset_file_visual_realistic if appearance_realistic else hand.robot_asset_file
    joint_pos = {
        name: value
        for name, value in zip(hand.arm_dof_names, hand.default_dof_pos[: hand.num_arm_dofs])
    }
    joint_pos["right_.*_joint"] = 0.0

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/{robot_asset_file}",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=enable_self_collision,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_pos,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(hand.arm_dof_names),
                stiffness=16000.0 * pd_param_scale,
                damping=600.0 * pd_param_scale,
                friction=0.01,
                armature=0.001,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                stiffness=600.0 * pd_param_scale,
                damping=20.0 * pd_param_scale,
                friction=0.01,
                armature=0.001,
            ),
        },
    )


@configclass
class GraspEnvCfg(DirectRLEnvCfg):
    # ---- migrated from grasp.yaml/env ----
    hand: HandCfg = FR3_INSPIRE_TAC_CFG
    reset: ResetCfg = ResetCfg()
    control: ControlCfg = ControlCfg()
    reference: ReferenceCfg = ReferenceCfg()
    asset: AssetCfg = AssetCfg()
    render: RenderCfg = RenderCfg()
    camera_config: dict[str, CameraCalibrationCfg] = {"camera_1": CAMERA_1, "camera_2": CAMERA_2}

    # DemoGrasp grasp.yaml: episodeLength=50, sim.dt=0.01667, sim.decimation=20.
    # Old code actually runs one policy step per 20 physics steps, so episode length is about 16.67s.
    decimation = 20
    episode_length_s = 50 * 0.01667 * 20
    action_space = 13
    observation_type = "armdof+handdof+eefpose+objpose"
    observation_space = 27
    state_space = 0
    is_finite_horizon = False

    clip_observations = 5.0
    clip_actions = 1.0
    random_episode_length = False
    reward_type = "binary"
    randomize = False
    aggregate_mode = 1
    enable_debug_vis = False
    enable_point_cloud = False
    points_per_object = 512
    object_friction = 1.0
    enable_robot_table_collision = True
    enable_self_collision = False
    use_robot_vhacd = False
    
    enable_grid_ground = True
    grid_ground_prim_path = "/World/GridGround"
    grid_ground_size = 40.0
    grid_ground_spacing = 0.5
    grid_ground_line_width = 0.012
    grid_ground_line_height = 0.002
    grid_ground_z = 0.0
    grid_ground_base_color = (0.005, 0.005, 0.005)
    grid_ground_line_color = (0.32, 0.32, 0.32)
    grid_ground_axis_line_color = (0.62, 0.62, 0.62)


    tiled_camera_config: dict[str, TiledCameraCfg] = {
        "camera_1": make_tiled_camera_cfg(1, CAMERA_1),
        "camera_2": make_tiled_camera_cfg(2, CAMERA_2),
    }

    # ---- migrated from grasp.yaml/sim ----
    sim: SimulationCfg = SimulationCfg(
        dt=0.01667,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,
        ),
    )

    # multiObject=True means different envs can spawn different assets.
    # Start with replicate_physics=False for correctness.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=1.2,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    viewer = ViewerCfg(eye=(1.55, -1.65, 1.35), lookat=(0.45, -0.05, 0.35))
    
    robot: ArticulationCfg = make_robot_cfg(
        hand=FR3_INSPIRE_TAC_CFG,
        use_robot_vhacd=False,
        enable_self_collision=False,
        pd_param_scale=1.0,
        appearance_realistic=True,
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=load_multi_object_usd_cfgs(
            asset_root=ASSET_ROOT,
            multi_object_list="union_ycb_unidex/union_ycb_debugset.yaml",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.8, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.10, 0.10),
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51, -0.075, 0.018 - 0.003 - 0.3 / 2.0)),
    )

    mat: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mat",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.003),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.86, 0.84, 0.76),
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51, -0.075, 0.018 - 0.003 / 2.0)),
    )

    wooden_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/WoodenTable",
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 1.5, 0.004),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.78, 0.74, 0.64),
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.51, -0.075, 0.002)),
    )


    events: EventCfg | None = None
    # 如果要对应 grasp.yaml/task/randomize=True，把上一行改成：
    # events: EventCfg | None = EventCfg()

    # grasp.yaml/task/randomization_params/observations and actions.
    # 原始 grasp.yaml randomize=False，所以默认 None；如果打开随机化，再启用下面两项。
    observation_noise_model = None
    action_noise_model = None
    # observation_noise_model = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
    # )
    # action_noise_model = NoiseModelWithAdditiveBiasCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
    #     bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="add"),
    # )

@configclass
class GraspEnvRandomizedCfg(GraspEnvCfg):
    randomize = True
    events: EventCfg | None = EventCfg()
    observation_noise_model = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
    )
    action_noise_model = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="add"),
    )
