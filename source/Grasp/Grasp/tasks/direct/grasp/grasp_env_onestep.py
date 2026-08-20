from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import math
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from isaaclab.sensors import TiledCameraCfg

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    normalize as quat_unit,
)

from .object_loader import load_object_urdf_file_list
from .utils import batch_linear_interpolate_poses, load_object_point_clouds, scale, tensor_clamp, transform_points, unscale
from .reward import REWARD_DICT
from .render_utils import build_instruction, load_object_display_names, sample_color_name
from .grasp_env_cfg_onestep import ASSET_ROOT, GraspEnvCfg, camera_data_types



class GraspEnv(DirectRLEnv):
    cfg: GraspEnvCfg

    def __init__(self, cfg: GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.reward_type not in REWARD_DICT:
            raise ValueError(f"Unsupported reward_type: {self.cfg.reward_type}")
        self.reward_function = REWARD_DICT[self.cfg.reward_type]

        self.hand_cfg = self.cfg.hand
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_targets = self.robot.data.default_joint_pos.clone()
        self.cur_targets = self.robot.data.default_joint_pos.clone()

        self._resolve_robot_indices()

        self.no_op_action = unscale(
            self.robot.data.default_joint_pos[:, self.active_robot_dof_ids],
            self.robot_dof_lower_limits[self.active_robot_dof_ids],
            self.robot_dof_upper_limits[self.active_robot_dof_ids],
        )
        self._interp_step = 0
        self._target_start = self.prev_targets.clone()
        self.table_thickness = 0.3
        self.mat_thickness = 0.003
        self.wooden_table_thickness = 0.004

        self._load_tracking_reference()
        self._init_episode_buffers()
        self._init_run_rl_grasp_compat()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        self.table = RigidObject(self.cfg.table)
        self.mat = RigidObject(self.cfg.mat)
        self.wooden_table = RigidObject(self.cfg.wooden_table)

        if self.cfg.render.enable:
            self._setup_cameras()

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["mat"] = self.mat
        self.scene.rigid_objects["wooden_table"] = self.wooden_table

        if self.cfg.render.enable:
            for name, camera in self.cameras.items():
                self.scene.sensors[name] = camera

        self._spawn_visual_grid_ground()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _spawn_visual_grid_ground(self) -> None:
        if not self.cfg.enable_grid_ground:
            return

        ground_root = self.cfg.grid_ground_prim_path
        visual_size = float(self.cfg.grid_ground_size)
        spacing = float(self.cfg.grid_ground_spacing)
        half_steps = math.ceil(visual_size / (2.0 * spacing))
        visual_size = 2.0 * half_steps * spacing
        line_width = float(self.cfg.grid_ground_line_width)
        line_height = float(self.cfg.grid_ground_line_height)
        z = float(self.cfg.grid_ground_z)

        sim_utils.create_prim(ground_root, "Xform")

        base_cfg = sim_utils.CuboidCfg(
            size=(visual_size, visual_size, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self.cfg.grid_ground_base_color,
                roughness=0.8,
            ),
        )
        base_cfg.func(f"{ground_root}/Base", base_cfg, translation=(0.0, 0.0, z - line_height))

        normal_line_cfg = sim_utils.CuboidCfg(
            size=(visual_size, line_width, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self.cfg.grid_ground_line_color,
                roughness=0.8,
            ),
        )
        axis_line_cfg = sim_utils.CuboidCfg(
            size=(visual_size, line_width * 1.5, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self.cfg.grid_ground_axis_line_color,
                roughness=0.8,
            ),
        )

        for index in range(-half_steps, half_steps + 1):
            coord = index * spacing
            line_cfg = axis_line_cfg if index == 0 else normal_line_cfg
            line_cfg.func(
                f"{ground_root}/Lines/X_{index + half_steps:04d}",
                line_cfg,
                translation=(0.0, coord, z),
            )
            line_cfg.func(
                f"{ground_root}/Lines/Y_{index + half_steps:04d}",
                line_cfg,
                translation=(coord, 0.0, z),
                orientation=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            )
    
    def _setup_cameras(self):
        self.cameras = {}
        for camera_id in self.cfg.render.camera_ids:
            name = f"camera_{camera_id}"
            camera_cfg = deepcopy(self.cfg.tiled_camera_config[name])
            camera_cfg.data_types = camera_data_types(self.cfg.render.data_type)
            self.cameras[name] = TiledCamera(camera_cfg)

    def _resolve_robot_indices(self):
        self.arm_dof_ids, self.arm_dof_names = self.robot.find_joints(list(self.hand_cfg.arm_dof_names))
        self.eef_body_ids, _ = self.robot.find_bodies(self.hand_cfg.eef_link)
        self.palm_body_ids, _ = self.robot.find_bodies(self.hand_cfg.palm_link)
        self.fingertip_body_ids, _ = self.robot.find_bodies(list(self.hand_cfg.fingertips_link))

        all_joint_names = self.robot.joint_names
        hand_names = [name for name in all_joint_names if name not in self.hand_cfg.arm_dof_names]
        self.hand_dof_ids, self.hand_dof_names = self.robot.find_joints(hand_names)

        passive_names = list(self.hand_cfg.passive_joints.keys())
        self.passive_hand_dof_ids, _ = self.robot.find_joints(passive_names)
        active_names = [name for name in hand_names if name not in passive_names]
        self.active_hand_dof_ids, self.active_hand_dof_names = self.robot.find_joints(active_names)
        self.active_robot_dof_ids = torch.tensor(self.arm_dof_ids + self.active_hand_dof_ids, device=self.device)

        self.mimic_parent_dof_ids = []
        self.mimic_multipliers = []
        for passive_name, mimic_cfg in self.hand_cfg.passive_joints.items():
            parent_ids, _ = self.robot.find_joints(mimic_cfg["mimic"])
            self.mimic_parent_dof_ids.append(parent_ids[0])
            self.mimic_multipliers.append(float(mimic_cfg["multiplier"]))
        self.mimic_parent_dof_ids = torch.tensor(self.mimic_parent_dof_ids, device=self.device, dtype=torch.long)
        self.mimic_multipliers = torch.tensor(self.mimic_multipliers, device=self.device)

        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = limits[0, :, 0]
        self.robot_dof_upper_limits = limits[0, :, 1]

    def _init_episode_buffers(self):
        self.successes = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_hit_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.table_heights = torch.full(
            (self.num_envs,),
            self.cfg.reset.table_height_range[0],
            dtype=torch.float32,
            device=self.device,
        )
        self.object_init_states = torch.zeros((self.num_envs, 13), dtype=torch.float32, device=self.device)
        self.reaching_plan_ee = torch.zeros((self.num_envs, self.max_episode_length, 7), device=self.device)
        self.reaching_plan_timesteps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.instructions = [self.cfg.render.instruction_template for _ in range(self.num_envs)]
        self.object_name_list = load_object_display_names(
            ASSET_ROOT,
            self.cfg.render.object_name_list,
            self.num_envs,
        )
        self.object_files = load_object_urdf_file_list(
            asset_root=ASSET_ROOT,
            multi_object_list=self.cfg.asset.multi_object_list,
        )
        self.object_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % len(self.object_files)
        if self.cfg.enable_point_cloud:
            point_clouds = load_object_point_clouds(self.object_files, ASSET_ROOT)
            self.obj_pcl_buf = torch.zeros(
                (self.num_envs, self.cfg.points_per_object, 3),
                dtype=torch.float32,
                device=self.device,
            )
            for env_id in range(self.num_envs):
                object_pcl = torch.as_tensor(
                    point_clouds[int(self.object_ids[env_id].item())],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.obj_pcl_buf[env_id] = object_pcl[: self.cfg.points_per_object]

    def _init_run_rl_grasp_compat(self):
        self.num_arm_dofs = len(self.arm_dof_ids)
        self.num_active_hand_dofs = len(self.active_hand_dof_ids)
        self.hand_dof_start_idx = 7
        self.num_actions = self.cfg.action_space
        self.hand_name = self.hand_cfg.name

        self.render_cfg = {
            "resize": self.cfg.render.resize,
            "data_type": self.cfg.render.data_type,
        }
        self.render_data_type = self.cfg.render.data_type
        self.camera_ids = list(self.cfg.render.camera_ids) if self.cfg.render.enable else []
        self.obs_dict = {}

        self.active_robot_dof_default_pos = self.robot.data.default_joint_pos[0, self.active_robot_dof_ids].clone()
        self.no_op_action = self._compute_no_op_action()

    def _compute_no_op_action(self) -> torch.Tensor:
        default_pos = self.robot.data.default_joint_pos[:, self.active_robot_dof_ids]
        return unscale(
            default_pos,
            self.robot_dof_lower_limits[self.active_robot_dof_ids],
            self.robot_dof_upper_limits[self.active_robot_dof_ids],
        )

    def _load_tracking_reference(self):
        with open(self.cfg.reference.tracking_reference_file, "rb") as f:
            self.tracking_reference = pickle.load(f)
        for key, value in self.tracking_reference.items():
            value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            # DemoGrasp/IsaacGym reference quaternions are usually xyzw.
            # IsaacLab state tensors use wxyz, so convert reference quaternions once here.
            if key.endswith("quat"):
                value = xyzw_to_wxyz(value)
            self.tracking_reference[key] = value.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.t_ref = self.tracking_reference["wrist_initobj_pos"].shape[1]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions).to(self.device)
        self._target_start[:] = self.prev_targets
        self._interp_step = 0
        self._compute_joint_targets_from_actions(self.actions)

    def _apply_action(self):
        alpha = float(self._interp_step + 1) / float(self.cfg.decimation)
        target = self._target_start + alpha * (self.cur_targets - self._target_start)
        self.robot.set_joint_position_target(target)
        self._interp_step += 1
        if self._interp_step >= self.cfg.decimation:
            self.prev_targets[:] = self.cur_targets

    def _compute_joint_targets_from_actions(self, actions: torch.Tensor):
        joint_pos = self.robot.data.joint_pos
        lower = self.robot_dof_lower_limits
        upper = self.robot_dof_upper_limits
        dt = self.cfg.sim.dt * self.cfg.decimation

        if self.cfg.control.use_relative_control:
            self.cur_targets[:, self.active_hand_dof_ids] = (
                self.prev_targets[:, self.active_hand_dof_ids]
                + actions[:, self.hand_action_slice] * torch.tensor(self.cfg.control.delta_action_scale[7:], device=self.device)
            )
        else:
            self.cur_targets[:, self.active_hand_dof_ids] = scale(
                actions[:, 7:],
                lower[self.active_hand_dof_ids],
                upper[self.active_hand_dof_ids],
            )

        self.cur_targets[:, self.active_hand_dof_ids] = tensor_clamp(
            self.cur_targets[:, self.active_hand_dof_ids],
            self.prev_targets[:, self.active_hand_dof_ids] - self.cfg.control.actions_max_ang_vel_hand * dt,
            self.prev_targets[:, self.active_hand_dof_ids] + self.cfg.control.actions_max_ang_vel_hand * dt,
        )

        if len(self.passive_hand_dof_ids) > 0:
            self.cur_targets[:, self.passive_hand_dof_ids] = (
                self.cur_targets[:, self.mimic_parent_dof_ids] * self.mimic_multipliers
            )

        if self.cfg.control.arm_controller == "qpos":
            self.cur_targets[:, self.arm_dof_ids] = scale(actions[:, :7], lower[self.arm_dof_ids], upper[self.arm_dof_ids])
        elif self.cfg.control.arm_controller == "pose":
            delta_arm = self.compute_arm_ik(actions[:, :7], is_delta_pose=False)
            self.cur_targets[:, self.arm_dof_ids] = joint_pos[:, self.arm_dof_ids] + delta_arm
        else:
            delta_arm = self.compute_arm_ik(
                actions[:, :6] * torch.tensor(self.cfg.control.delta_action_scale[:6], device=self.device),
                is_delta_pose=True,
                is_delta_pose_in_world=(self.cfg.control.arm_controller == "worlddpose"),
            )
            self.cur_targets[:, self.arm_dof_ids] = joint_pos[:, self.arm_dof_ids] + delta_arm

        self.cur_targets[:, self.arm_dof_ids] = tensor_clamp(
            self.cur_targets[:, self.arm_dof_ids],
            self.prev_targets[:, self.arm_dof_ids] - self.cfg.control.actions_max_ang_vel_arm * dt,
            self.prev_targets[:, self.arm_dof_ids] + self.cfg.control.actions_max_ang_vel_arm * dt,
        )
        self.cur_targets = tensor_clamp(self.cur_targets, lower, upper)

    @property
    def hand_action_slice(self) -> slice:
        return slice(7, 7 + len(self.active_hand_dof_ids))

    def compute_arm_ik(
        self,
        action: torch.Tensor,
        is_delta_pose: bool = True,
        is_delta_pose_in_world: bool = True,
    ) -> torch.Tensor:
        eef_state = self.robot.data.body_state_w[:, self.eef_body_ids[0], 0:7]
        if is_delta_pose:
            dtheta = torch.norm(action[:, 3:6], dim=-1, keepdim=True)
            axis = action[:, 3:6] / (dtheta + 1e-4)
            delta_quat = quat_from_angle_axis(dtheta.squeeze(-1), axis)
            if is_delta_pose_in_world:
                pos_err = action[:, 0:3]
                desired_quat = quat_mul(delta_quat, eef_state[:, 3:7])
            else:
                pos_err = quat_apply(eef_state[:, 3:7], action[:, 0:3])
                desired_quat = quat_mul(eef_state[:, 3:7], delta_quat)
            orn_err = orientation_error(desired_quat, eef_state[:, 3:7])
        else:
            pos_err = action[:, 0:3] - eef_state[:, 0:3]
            orn_err = orientation_error(quat_unit(action[:, 3:7]), eef_state[:, 3:7])

        dpose = torch.cat([pos_err, orn_err], dim=-1).unsqueeze(-1)
        jacobians = self.robot.root_physx_view.get_jacobians()
        j_eef = jacobians[:, self.eef_body_ids[0] - 1, 0:6, : len(self.arm_dof_ids)]
        j_eef_t = torch.transpose(j_eef, 1, 2)
        damping = 0.1
        lambda_matrix = torch.eye(6, device=self.device) * damping**2
        return (j_eef_t @ torch.inverse(j_eef @ j_eef_t + lambda_matrix) @ dpose).squeeze(-1)

    def _get_observations(self) -> dict:
        obs_parts = []
        joint_pos = self.robot.data.joint_pos

        if "armdof" in self.cfg.observation_type:
            obs_parts.append(
                unscale(
                    joint_pos[:, self.arm_dof_ids],
                    self.robot_dof_lower_limits[self.arm_dof_ids],
                    self.robot_dof_upper_limits[self.arm_dof_ids],
                )
            )

        if "handdof" in self.cfg.observation_type:
            obs_parts.append(
                unscale(
                    joint_pos[:, self.active_hand_dof_ids],
                    self.robot_dof_lower_limits[self.active_hand_dof_ids],
                    self.robot_dof_upper_limits[self.active_hand_dof_ids],
                )
            )

        if "eefpose" in self.cfg.observation_type:
            obs_parts.append(pose_wxyz_to_xyzw(self.robot.data.body_state_w[:, self.eef_body_ids[0], 0:7]))

        if "objpose" in self.cfg.observation_type:
            obs_parts.append(pose_wxyz_to_xyzw(self.object.data.root_state_w[:, 0:7]))

        if "fulldof" in self.cfg.observation_type:
            obs_parts.append(
                unscale(
                    joint_pos,
                    self.robot_dof_lower_limits,
                    self.robot_dof_upper_limits,
                )
            )

        if "ftpos" in self.cfg.observation_type:
            obs_parts.append(self.robot.data.body_state_w[:, self.fingertip_body_ids, 0:3].reshape(self.num_envs, -1))

        if "palmpose" in self.cfg.observation_type:
            obs_parts.append(pose_wxyz_to_xyzw(self.robot.data.body_state_w[:, self.palm_body_ids[0], 0:7]))

        if "lastact" in self.cfg.observation_type:
            obs_parts.append(self.actions)

        if "objxyz" in self.cfg.observation_type:
            obs_parts.append(self.object.data.root_state_w[:, 0:3])

        if "objinitpose" in self.cfg.observation_type:
            obs_parts.append(pose_wxyz_to_xyzw(self.object_init_states[:, 0:7]))

        if "objpcl" in self.cfg.observation_type:
            if "objpcl" in self.cfg.observation_type:
                if not self.cfg.enable_point_cloud:
                    raise RuntimeError("observation_type contains objpcl but cfg.enable_point_cloud is False")
                obs_parts.append(self.transform_obj_pcl_2_world().reshape(self.num_envs, -1))

        if len(obs_parts) == 0:
            raise RuntimeError(f"Unsupported observation_type: {self.cfg.observation_type}")

        obs = torch.cat(obs_parts, dim=-1)
        obs = torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        return {"policy": obs}
    
    def transform_obj_pcl_2_world(self) -> torch.Tensor:
        object_pos = self.object.data.root_state_w[:, 0:3].unsqueeze(1)
        object_quat = self.object.data.root_state_w[:, 3:7].unsqueeze(1)
        zeros = torch.zeros((self.num_envs, self.cfg.points_per_object, 1), dtype=torch.float32, device=self.device)
        point_quat = torch.cat([zeros, self.obj_pcl_buf], dim=-1)
        pcl = transform_points(object_quat.expand_as(point_quat), point_quat)
        return pcl + object_pos

    def _get_object_point_cloud_obs(self) -> torch.Tensor:
        obj_pose = self.object.data.root_state_w[:, 0:7]
        points = self.obj_pcl_buf
        zeros = torch.zeros((*points.shape[:-1], 1), device=self.device)
        point_quat = torch.cat([zeros, points], dim=-1)
        quat = obj_pose[:, None, 3:7].expand(-1, points.shape[1], -1)
        world_points = transform_points(quat, point_quat) + obj_pose[:, None, 0:3]
        return world_points.reshape(self.num_envs, -1)
    
    def _collect_camera_outputs(self) -> dict[str, torch.Tensor]:
        outputs = {}
        for name, camera in self.cameras.items():
            camera.update(self.physics_dt, force_recompute=True)
            if "rgb" in camera.data.output:
                outputs[f"{name}.rgb"] = camera.data.output["rgb"]
        return outputs

    def compute_real_observation_dict(self) -> dict:
        obs_dict = {
            "instruction": self.instructions,
            "right_arm_qpos": self.robot.data.joint_pos[:, self.arm_dof_ids].detach().cpu().numpy(),
            "right_arm_eef_pose": pose_wxyz_to_xyzw(self.robot.data.body_state_w[:, self.eef_body_ids[0], 0:7]).detach().cpu().numpy(),
            "right_hand_qpos": self.robot.data.joint_pos[:, self.active_hand_dof_ids].detach().cpu().numpy(),
        }

        if self.cfg.render.enable:
            camera_outputs = self._collect_camera_outputs()
            for camera_id in self.cfg.render.camera_ids:
                name = f"camera_{camera_id}"
                rgb_key = f"{name}.rgb"
                if rgb_key in camera_outputs and "rgb" in self.cfg.render.data_type:
                    rgb = camera_outputs[rgb_key][..., :3]
                    rgb = torch.nn.functional.interpolate(
                        rgb.permute(0, 3, 1, 2).to(torch.float32),
                        size=tuple(self.cfg.render.resize),
                        mode="area",
                    ).permute(0, 2, 3, 1)
                    obs_dict[rgb_key] = rgb.detach().cpu().numpy().astype("uint8")
                if "depth" in self.cfg.render.data_type:
                    raise NotImplementedError("Depth real observation is not implemented in the original DemoGrasp code")
                if "pcl" in self.cfg.render.data_type:
                    raise NotImplementedError("Point-cloud real observation is not implemented in the original DemoGrasp code")

        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        reward, _, _, self.successes, self.current_successes, self.has_hit_table, info = self.reward_function(
            reset_buf=self.reset_buf,
            progress_buf=self.episode_length_buf,
            successes=self.successes,
            current_successes=self.current_successes,
            has_hit_table=self.has_hit_table,
            max_episode_length=self.max_episode_length,
            table_heights=self.table_heights,
            object_pos=self.object.data.root_state_w[:, 0:3],
            palm_pos=self.robot.data.body_state_w[:, self.palm_body_ids[0], 0:3]
            + quat_apply(
                self.robot.data.body_state_w[:, self.palm_body_ids[0], 3:7],
                torch.tensor(self.hand_cfg.palm_offset, device=self.device).repeat(self.num_envs, 1),
            ),
            fingertip_pos=self.robot.data.body_state_w[:, self.fingertip_body_ids, 0:3],
            num_fingers=len(self.fingertip_body_ids),
            object_init_states=self.object_init_states,
        )
        self.extras["log"] = info # IsaacLab 日志兼容 额外适配
        self.extras.update(info)
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["has_hit_table"] = self.has_hit_table
        return reward
    
    def reset_idx(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        obs = self._get_observations()["policy"]
        self.obs_dict["obs"] = obs
        return self.obs_dict

    def get_state(self) -> torch.Tensor:
        if self.cfg.state_space:
            states = self._get_states()
            if states is not None and "critic" in states:
                return states["critic"]
        return torch.empty((self.num_envs, 0), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None, object_init_pose: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        self._reset_table(env_ids)
        self._reset_object(env_ids, object_init_pose=object_init_pose)
        self._reset_robot(env_ids)

        if self.cfg.reset.settle_object_after_reset:
            settle_steps = int(self.cfg.reset.settle_time_s / self.physics_dt)
            for _ in range(settle_steps):
                self.sim.step(render=False)
                self.scene.update(self.physics_dt)
        self.successes[env_ids] = 0.0
        self.current_successes[env_ids] = 0.0
        self.has_hit_table[env_ids] = False
        if self.cfg.random_episode_length:
            self.episode_length_buf[env_ids] = torch.randint(0, 10, (len(env_ids),), device=self.device)
        self.object_init_states[env_ids] = self.object.data.root_state_w[env_ids].clone()
        if self.cfg.render.enable:
            for env_id in env_ids.tolist():
                color_name = sample_color_name(self.cfg.render.randomization_params.object_color_choices)
                self.instructions[env_id] = build_instruction(
                    self.cfg.render.use_advanced_instruction,
                    self.cfg.render.advanced_instruction_template
                    if self.cfg.render.use_advanced_instruction
                    else self.cfg.render.instruction_template,
                    self.object_name_list[env_id],
                    color_name,
                )
        self.generate_reaching_plan_idx(env_ids)

    def _reset_object(self, env_ids: torch.Tensor, object_init_pose: torch.Tensor | None = None):
        root_state = self.object.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        if object_init_pose is None:
            ranges = torch.tensor(self.cfg.reset.reset_position_range, device=self.device)
            samples = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand((len(env_ids), 3), device=self.device)
            samples[:, 2] += self.table_heights[env_ids]
            root_state[:, :3] = self.scene.env_origins[env_ids] + samples

            if self.cfg.reset.reset_random_rot == "fixed":
                root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
            else:
                axis = torch.randn((len(env_ids), 3), device=self.device)
                if self.cfg.reset.reset_random_rot == "z":
                    axis[:] = torch.tensor((0.0, 0.0, 1.0), device=self.device)
                axis = axis / torch.norm(axis, dim=-1, keepdim=True)
                angle = -torch.pi + 2 * torch.pi * torch.rand((len(env_ids),), device=self.device)
                root_state[:, 3:7] = quat_from_angle_axis(angle, axis)
        else:
            pose = torch.as_tensor(object_init_pose, dtype=torch.float32, device=self.device)
            if pose.ndim == 1:
                pose = pose.unsqueeze(0).repeat(len(env_ids), 1)
            root_state[:, :7] = pose[:, :7]
            root_state[:, :3] += self.scene.env_origins[env_ids]

        root_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def _reset_table(self, env_ids: torch.Tensor):
        low, high = self.cfg.reset.table_height_range
        if self.cfg.render.appearance_realistic:
            self.table_heights[env_ids] = low
        else:
            self.table_heights[env_ids] = low + (high - low) * torch.rand((len(env_ids),), device=self.device)

        table_state = self.table.data.default_root_state[env_ids].clone()
        table_state[:, :3] += self.scene.env_origins[env_ids]
        table_state[:, 0] = self.scene.env_origins[env_ids, 0] + 0.51
        table_state[:, 1] = self.scene.env_origins[env_ids, 1] - 0.075
        table_state[:, 2] = self.table_heights[env_ids] - self.mat_thickness - self.table_thickness / 2.0
        table_state[:, 7:] = 0.0
        self.table.write_root_pose_to_sim(table_state[:, :7], env_ids=env_ids)
        self.table.write_root_velocity_to_sim(table_state[:, 7:], env_ids=env_ids)

        mat_state = self.mat.data.default_root_state[env_ids].clone()
        mat_state[:, :3] += self.scene.env_origins[env_ids]
        mat_state[:, 0] = self.scene.env_origins[env_ids, 0] + 0.51
        mat_state[:, 1] = self.scene.env_origins[env_ids, 1] - 0.075
        mat_state[:, 2] = self.table_heights[env_ids] - self.mat_thickness / 2.0
        mat_state[:, 7:] = 0.0
        self.mat.write_root_pose_to_sim(mat_state[:, :7], env_ids=env_ids)
        self.mat.write_root_velocity_to_sim(mat_state[:, 7:], env_ids=env_ids)

        wooden_state = self.wooden_table.data.default_root_state[env_ids].clone()
        wooden_state[:, :3] += self.scene.env_origins[env_ids]
        wooden_state[:, 0] = self.scene.env_origins[env_ids, 0] + 0.51
        wooden_state[:, 1] = self.scene.env_origins[env_ids, 1] - 0.075
        wooden_state[:, 2] = 0.002
        wooden_state[:, 7:] = 0.0
        self.wooden_table.write_root_pose_to_sim(wooden_state[:, :7], env_ids=env_ids)
        self.wooden_table.write_root_velocity_to_sim(wooden_state[:, 7:], env_ids=env_ids)

    def _reset_robot(self, env_ids: torch.Tensor):
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        default_pos = self.robot.data.default_joint_pos[env_ids].clone()
        default_vel = self.robot.data.default_joint_vel[env_ids].clone()
        delta_max = self.robot_dof_upper_limits - self.robot.data.default_joint_pos[0]
        delta_min = self.robot_dof_lower_limits - self.robot.data.default_joint_pos[0]
        rand_delta = delta_min + (delta_max - delta_min) * torch.rand((len(env_ids), self.robot.num_joints), device=self.device)
        pos = default_pos + self.cfg.reset.reset_dof_pos_random_interval * rand_delta
        if self.cfg.reset.reset_hand_dof_pos_full_range:
            pos[:, self.hand_dof_ids] = default_pos[:, self.hand_dof_ids] + rand_delta[:, self.hand_dof_ids]

        self.robot.write_joint_state_to_sim(pos, default_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(pos, env_ids=env_ids)
        self.prev_targets[env_ids] = pos
        self.cur_targets[env_ids] = pos

    def generate_reaching_plan_idx(self, env_ids: torch.Tensor, actions: torch.Tensor | None = None):
        def get_random_value(interval: slice) -> torch.Tensor:
            if actions is not None:
                return actions[env_ids, interval].to(self.device)
            return -1.0 + 2.0 * torch.rand((len(env_ids), interval.stop - interval.start), device=self.device)

        if not self.cfg.reference.randomize_tracking_reference:
            self.current_tracking_reference = self.tracking_reference
        else:
            self.current_tracking_reference = deepcopy(self.tracking_reference)
            rand_rot = get_random_value(slice(3, 6))
            ref_range = torch.tensor(self.cfg.reference.randomize_tracking_reference_range, device=self.device)
            rand_quat = quat_from_euler_xyz(
                rand_rot[:, 0] * ref_range[3],
                rand_rot[:, 1] * ref_range[4],
                rand_rot[:, 2] * ref_range[5],
            ).unsqueeze(1).expand(-1, self.t_ref, -1)
            self.current_tracking_reference["wrist_quat"][env_ids] = quat_mul(
                rand_quat, self.current_tracking_reference["wrist_quat"][env_ids]
            )
            self.current_tracking_reference["wrist_initobj_pos"][env_ids] = quat_apply(
                rand_quat, self.current_tracking_reference["wrist_initobj_pos"][env_ids]
            )

            rand_xyz = get_random_value(slice(0, 3))
            self.current_tracking_reference["wrist_initobj_pos"][env_ids] += (
                rand_xyz * ref_range[0:3]
            ).unsqueeze(1).expand(-1, self.t_ref, -1)

            lift_t = self.cfg.reference.tracking_reference_lift_timestep
            self.current_tracking_reference["wrist_initobj_pos"][env_ids, lift_t:, 0:3] = (
                self.tracking_reference["wrist_initobj_pos"][env_ids, lift_t:, 0:3]
                - self.tracking_reference["wrist_initobj_pos"][env_ids, lift_t - 1 : lift_t, 0:3]
                + self.current_tracking_reference["wrist_initobj_pos"][env_ids, lift_t - 1 : lift_t, 0:3]
            )

            if self.cfg.reference.randomize_grasp_pose:
                rand_hand = get_random_value(slice(6, 6 + len(self.active_hand_dof_ids)))
                hand_ref = self.current_tracking_reference["hand_qpos"][env_ids]
                grasp_pose = hand_ref[:, lift_t - 1] + rand_hand * self.cfg.reference.randomize_grasp_pose_range
                grasp_pose = torch.clamp(
                    grasp_pose,
                    self.robot_dof_lower_limits[self.active_hand_dof_ids],
                    self.robot_dof_upper_limits[self.active_hand_dof_ids],
                )
                hand_ref_t0 = hand_ref[:, 0].unsqueeze(1).repeat(1, lift_t - 1, 1)
                fraction = (grasp_pose - hand_ref[:, 0]) / (hand_ref[:, lift_t - 1] - hand_ref[:, 0] + 1e-6)
                hand_ref[:, 0 : lift_t - 1] = hand_ref_t0 + (hand_ref[:, 0 : lift_t - 1] - hand_ref_t0) * fraction.unsqueeze(1)
                hand_ref[:, lift_t - 1 :] = grasp_pose.unsqueeze(1).repeat(1, self.t_ref - lift_t + 1, 1)
                self.current_tracking_reference["hand_qpos"][env_ids] = torch.clamp(
                    hand_ref,
                    self.robot_dof_lower_limits[self.active_hand_dof_ids],
                    self.robot_dof_upper_limits[self.active_hand_dof_ids],
                )

        wrist_pose = self.robot.data.body_state_w[env_ids, self.eef_body_ids[0], 0:7]
        wrist_pose_target = torch.cat(
            [
                self.current_tracking_reference["wrist_initobj_pos"][env_ids, 0] + self.object_init_states[env_ids, 0:3],
                self.current_tracking_reference["wrist_quat"][env_ids, 0],
            ],
            dim=-1,
        )
        reaching_plan_ee, reaching_plan_timesteps = batch_linear_interpolate_poses(
            wrist_pose,
            wrist_pose_target,
            max_trans_step=0.04 * self.cfg.control.interpolation_step_scale,
            max_rot_step=0.1 * self.cfg.control.interpolation_step_scale,
        )
        reaching_plan_ee = reaching_plan_ee[:, 1 : min(self.max_episode_length, reaching_plan_ee.shape[1])]
        reaching_plan_timesteps -= 1
        self.reaching_plan_ee[env_ids, : reaching_plan_ee.shape[1]] = reaching_plan_ee
        self.reaching_plan_timesteps[env_ids] = reaching_plan_timesteps

    def compute_reference_actions(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        progress = self.episode_length_buf

        reaching_t = torch.minimum(progress, self.reaching_plan_timesteps)
        reaching_target = self.reaching_plan_ee[env_ids, reaching_t] * (progress < self.reaching_plan_timesteps).unsqueeze(-1)

        tracking_t = (progress - self.reaching_plan_timesteps).clamp(min=0, max=self.t_ref - 1)
        tracking_target = torch.cat(
            [
                self.current_tracking_reference["wrist_initobj_pos"][env_ids, tracking_t] + self.object_init_states[:, 0:3],
                self.current_tracking_reference["wrist_quat"][env_ids, tracking_t],
            ],
            dim=-1,
        ) * (progress >= self.reaching_plan_timesteps).unsqueeze(-1)
        wrist_pose_target = reaching_target + tracking_target
        hand_qpos_target = self.current_tracking_reference["hand_qpos"][env_ids, tracking_t]

        if self.cfg.control.limit_control_error:
            joint_pos = self.robot.data.joint_pos
            eef_pose = self.robot.data.body_state_w[:, self.eef_body_ids[0], 0:7]
            hand_qpos_target = tensor_clamp(
                hand_qpos_target,
                joint_pos[:, self.active_hand_dof_ids] - self.cfg.control.max_pd_error_hand,
                joint_pos[:, self.active_hand_dof_ids] + self.cfg.control.max_pd_error_hand,
            )
            wrist_pose_target[:, 0:3] = tensor_clamp(
                wrist_pose_target[:, 0:3],
                eef_pose[:, 0:3] - self.cfg.control.max_pd_error_ee_pos,
                eef_pose[:, 0:3] + self.cfg.control.max_pd_error_ee_pos,
            )

        if self.cfg.control.arm_controller == "qpos" and not self.cfg.control.use_relative_control:
            arm_qpos_target = self.compute_arm_ik(wrist_pose_target, is_delta_pose=False) + self.robot.data.joint_pos[:, self.arm_dof_ids]
            qpos_target = torch.cat([arm_qpos_target, hand_qpos_target], dim=-1)
            return unscale(
                qpos_target,
                self.robot_dof_lower_limits[self.active_robot_dof_ids],
                self.robot_dof_upper_limits[self.active_robot_dof_ids],
            )

        action = self.actions.clone()
        wrist_pose = self.robot.data.body_state_w[:, self.eef_body_ids[0], 0:7]
        if self.cfg.control.arm_controller == "worlddpose":
            dquat = quat_mul(wrist_pose_target[:, 3:], quat_conjugate(wrist_pose[:, 3:]))
            action[:, 0:3] = wrist_pose_target[:, 0:3] - wrist_pose[:, 0:3]
            action[:, 3:6] = axis_angle_from_quat(dquat)
            action[:, 6] = 0.0
        elif self.cfg.control.arm_controller == "eedpose":
            dquat = quat_mul(quat_conjugate(quat_unit(wrist_pose[:, 3:])), quat_unit(wrist_pose_target[:, 3:]))
            action[:, 0:3] = quat_apply(quat_conjugate(wrist_pose[:, 3:]), wrist_pose_target[:, 0:3] - wrist_pose[:, 0:3])
            action[:, 3:6] = axis_angle_from_quat(dquat).clamp(-0.3, 0.3)
            action[:, 6] = 0.0
        elif self.cfg.control.arm_controller == "pose":
            action[:, 0:7] = wrist_pose_target
        else:
            raise ValueError(f"Unsupported arm_controller: {self.cfg.control.arm_controller}")

        action[:, 7:] = unscale(
            hand_qpos_target,
            self.robot_dof_lower_limits[self.active_hand_dof_ids],
            self.robot_dof_upper_limits[self.active_hand_dof_ids],
        )
        return action


@torch.jit.script
def orientation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 1:4] * torch.sign(q_r[:, 0]).unsqueeze(-1)

def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 3:4], quat[..., 0:3]], dim=-1)

def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)

def pose_wxyz_to_xyzw(pose: torch.Tensor) -> torch.Tensor:
    return torch.cat([pose[..., 0:3], wxyz_to_xyzw(pose[..., 3:7])], dim=-1)