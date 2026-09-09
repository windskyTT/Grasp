from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .object_loader import make_object_spawner
from .object_set import TRAIN_OBJECT_ORDER, TRAIN_ROOT, TRAIN_URDF_PATHS
from .robot_cfg import make_robot_cfg

@configclass
class RobustDexTeacherEnvCfg(DirectRLEnvCfg):
    decimation = 20
    episode_length_s = 70 * 0.2
    action_space = 22
    observation_space = 153
    state_space = 0
    is_finite_horizon = False
    clip_observations = 10.0
    clip_actions = 1.0
    random_episode_length = False

    robot: ArticulationCfg = make_robot_cfg()
    object: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        articulation_root_prim_path="/bottom",
        spawn=make_object_spawner(TRAIN_URDF_PATHS),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.773), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        actuators={},
    )
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.0, 0.771),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.2, dynamic_friction=0.2, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), roughness=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, -0.75152, 0.3855)),
    )
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=20,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=88,
        env_spacing=1.2,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    viewer = ViewerCfg(eye=(1.2, -1.4, 1.1), lookat=(0.2, -0.4, 0.5))
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

    seed = 1
    grasp_steps = 70
    simulation_dt = 0.01
    control_dt = 0.2
    sample_num = 10
    camera_position = (0.035, -0.58, 1.531)
    non_uniform_sampling = True
    top_grasp = False
    hand_center = (-0.0091, 0.0, -0.095)
    finger_action_std = 0.015
    rot_action_std = 0.005
    reward_clip = -2.0
    lift_height = 0.1
    dataset_root: str = str(TRAIN_ROOT)
    object_names: tuple[str, ...] = TRAIN_OBJECT_ORDER

    reward_coefficients = {
        "affordance_reward": 0.5, "affordance_contact_reward": 1.5,
        "affordance_impulse_reward": 1.0, "table_reward": -0.03,
        "table_contact_reward": -1.0, "table_impulse_reward": -0.5,
        "arm_height_reward": -0.05, "arm_contact_reward": -0.1,
        "arm_impulse_reward": -0.1, "arm_collision_reward": -1.0,
        "push_reward": -0.0, "wrist_vel_reward_": -1.0,
        "wrist_qvel_reward_": -0.1, "obj_vel_reward_": -15.0,
        "obj_qvel_reward_": -0.2, "obj_displacement_reward": -5.0,
        "arm_joint_vel_reward_": -1.0,
    }

    ppo_policy_net = (128, 128)
    ppo_value_net = (128, 128)
    ppo_gamma = 0.996
    ppo_lambda = 0.95
    ppo_epochs = 4
    ppo_mini_batches = 4
    ppo_shuffle_batch = False

    def __post_init__(self):
        self.sim.device = "cuda:0"

__all__ = ["RobustDexTeacherEnvCfg"]
