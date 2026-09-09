from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .actions import apply_joint_limits, residual_joint_target, sample_delay_mask
from .contact import collect_contact_data, make_contact_sensor_cfgs, make_self_collision_sensor_cfgs
from .env_cfg import RobustDexTeacherEnvCfg
from .observations import build_global_state, build_observation, build_state
from .rotations import quat_to_mat_torch
from .pregrasp import build_pregrasp
from .resets import reset_object_pose
from .rewards import compute_reward, weighted_affordance_weights, weighted_contact_weights
from .robot_cfg import ACTIVE_JOINT_NAMES, ARM_CONTACT_BODY_NAMES, HAND_BODY_PART_NAMES, INIT_FINGER_POSE
from .object_set import TRAIN_OBJECT_ORDER, TRAIN_ROOT, top_mesh_path, lowest_point_path

class RobustDexTeacherEnv(DirectRLEnv):
    cfg: RobustDexTeacherEnvCfg

    def __init__(self, cfg: RobustDexTeacherEnvCfg, render_mode: str | None = None, **kwargs):
        if torch.device(cfg.sim.device).type != "cuda":
            raise RuntimeError(f"RobustDexTeacherEnv requires CUDA, got {cfg.sim.device}")
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_indices()
        self._load_affordance_data()
        self.hand_center = torch.tensor(cfg.hand_center, dtype=torch.float32, device=self.device)
        self.current_joint_target = self.robot.data.joint_pos.clone()
        self.previous_joint_target = self.current_joint_target.clone()
        self.applied_joint_target = self.current_joint_target.clone()
        self.delay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift = False
        self.lift_num = 0
        self.arm_gc_lift = torch.zeros((self.num_envs, 6), device=self.device)
        self.affordance_center = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_initial_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.wrist_initial_rot_w = torch.eye(3, device=self.device).expand(self.num_envs, -1, -1).clone()
        self.previous_wrist_euler = torch.zeros((self.num_envs, 3), device=self.device)
        self.affordance_distances = torch.zeros((self.num_envs, 17), device=self.device)
        self.finger_weights = weighted_affordance_weights(self.device).expand(self.num_envs, -1)
        self.affordance_weights = self.finger_weights
        self.contact_weights = weighted_contact_weights(self.device).expand(self.num_envs, -1)
        self._contact_step_data = self._zero_contact_data()
        self._physics_substep = 0
        self._state = None
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        self._update_state()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = Articulation(self.cfg.object)
        self.table = RigidObject(self.cfg.table)
        self.contact_sensors = {}
        env_regex_ns = self.scene.env_regex_ns
        for name, source_cfg in make_contact_sensor_cfgs().items():
            sensor_cfg = deepcopy(source_cfg)
            sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
            sensor_cfg.filter_prim_paths_expr = [p.format(ENV_REGEX_NS=env_regex_ns) for p in sensor_cfg.filter_prim_paths_expr]
            sensor = ContactSensor(sensor_cfg)
            self.contact_sensors[name] = sensor
        for name, source_cfg in make_self_collision_sensor_cfgs().items():
            sensor_cfg = deepcopy(source_cfg)
            sensor_cfg.prim_path = sensor_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
            sensor_cfg.filter_prim_paths_expr = [p.format(ENV_REGEX_NS=env_regex_ns) for p in sensor_cfg.filter_prim_paths_expr]
            self.contact_sensors[name] = ContactSensor(sensor_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        for name, sensor in self.contact_sensors.items():
            self.scene.sensors[name] = sensor
        self.cfg.light.func("/World/Light", self.cfg.light)

    def _resolve_indices(self):
        arm_ids, arm_names = self.robot.find_bodies(list(ARM_CONTACT_BODY_NAMES), preserve_order=True)
        hand_ids, hand_names = self.robot.find_bodies(list(HAND_BODY_PART_NAMES), preserve_order=True)
        wrist_ids, wrist_names = self.robot.find_bodies("Flange_base_link", preserve_order=True)
        top_ids, top_names = self.object.find_bodies("top", preserve_order=True)
        bottom_ids, bottom_names = self.object.find_bodies("bottom", preserve_order=True)
        joint_ids, joint_names = self.robot.find_joints(list(ACTIVE_JOINT_NAMES), preserve_order=True)
        if tuple(arm_names) != ARM_CONTACT_BODY_NAMES or tuple(hand_names) != HAND_BODY_PART_NAMES:
            raise RuntimeError(f"UR5+Allegro body order mismatch: arm={arm_names}, hand={hand_names}")
        if len(wrist_ids) != 1 or wrist_names[0] != "Flange_base_link":
            raise RuntimeError(f"Flange_base_link was not resolved: {wrist_names}")
        if len(top_ids) != 1 or len(bottom_ids) != 1:
            raise RuntimeError(f"object top/bottom was not resolved: top={top_names}, bottom={bottom_names}")
        if len(joint_ids) != 22:
            raise RuntimeError(f"UR5+Allegro action dimension is {len(joint_ids)}, expected 22")
        if tuple(joint_names) != ACTIVE_JOINT_NAMES:
            raise RuntimeError(f"UR5+Allegro joint order mismatch: {joint_names}")
        self.arm_body_ids = torch.tensor(arm_ids, dtype=torch.long, device=self.device)
        self.hand_body_ids = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
        self.wrist_body_id = wrist_ids[0]
        self.object_top_body_id = top_ids[0]
        self.object_bottom_body_id = bottom_ids[0]
        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()

    def _load_affordance_data(self):
        object_order = self.cfg.object_names
        dataset_root = Path(self.cfg.dataset_root)
        env_names = tuple(object_order[i % len(object_order)] for i in range(self.num_envs))
        unique = {}
        for name in sorted(set(env_names)):
            mesh = trimesh.load_mesh(top_mesh_path(dataset_root, name))
            points, _ = trimesh.sample.sample_surface(mesh, 200)
            unique[name] = (points.astype(np.float32), np.asarray(mesh.centroid, dtype=np.float32), float(open(lowest_point_path(dataset_root, name), encoding="utf-8").read()))
        self.object_names = env_names
        point_data = np.stack([unique[name][0] for name in env_names])
        centers = np.stack([unique[name][1] for name in env_names])
        lowest = np.asarray([unique[name][2] for name in env_names], dtype=np.float32)
        self.affordance_points_obj = torch.as_tensor(point_data, device=self.device)
        self.affordance_center_obj = torch.as_tensor(centers, device=self.device)
        self.lowest_points = torch.as_tensor(lowest, device=self.device)

    def _zero_contact_data(self):
        zeros13 = torch.zeros((self.num_envs, 13, 3), device=self.device)
        zeros6 = torch.zeros((self.num_envs, 6, 3), device=self.device)
        false13 = torch.zeros((self.num_envs, 13), dtype=torch.bool, device=self.device)
        false6 = torch.zeros((self.num_envs, 6), dtype=torch.bool, device=self.device)
        return zeros13, zeros13.clone(), zeros13.clone(), zeros6, false13, false13.clone(), false13.clone(), false6, false6.clone()

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self.previous_joint_target.copy_(self.current_joint_target)
        self.current_joint_target = residual_joint_target(self.robot.data.joint_pos, actions, self.cfg.rot_action_std, self.cfg.finger_action_std)
        if self.lift:
            self.lift_num = min(self.lift_num + 1, 80)
            self.current_joint_target[:, :6] = self.arm_gc_lift + (actions[:, :6] - self.arm_gc_lift) * self.lift_num / 80.0
        self.current_joint_target = apply_joint_limits(self.current_joint_target, self.joint_lower_limits, self.joint_upper_limits)
        self.delay_mask = sample_delay_mask(self.num_envs, self.device)
        self._physics_substep = 0

    def _apply_action(self):
        target = self.current_joint_target
        if self._physics_substep == 0:
            target = torch.where(self.delay_mask[:, None], self.previous_joint_target, self.current_joint_target)
        self.applied_joint_target.copy_(target)
        self.robot.set_joint_position_target(target)
        if self._physics_substep == self.cfg.decimation - 1:
            self.previous_joint_target.copy_(self.current_joint_target)
        self._physics_substep += 1

    def _update_state(self):
        self._contact_step_data = collect_contact_data(self)
        state = build_state(self)
        hand_pos_obj = state["hand_pos_obj"]
        distances = torch.cdist(hand_pos_obj, self.affordance_points_obj)
        self.affordance_distances, indices = distances.min(dim=-1)
        selected = torch.gather(self.affordance_points_obj, 1, indices[..., None].expand(-1, -1, 3))
        state["affordance_vectors"] = torch.matmul(
            state["object_rot_w"].unsqueeze(1), (selected - hand_pos_obj).unsqueeze(-1)
        ).squeeze(-1)
        current_wrist_euler = state["wrist_euler"]
        unwrapped_wrist_euler = self.previous_wrist_euler + (current_wrist_euler - self.previous_wrist_euler + torch.pi).remainder(2.0 * torch.pi) - torch.pi
        state["wrist_euler_unwrapped"] = unwrapped_wrist_euler
        self.previous_wrist_euler.copy_(unwrapped_wrist_euler)
        self._state = state
        self._global_state = build_global_state(state)
        self._observation = build_observation(state, self.previous_wrist_euler)

    def _get_observations(self):
        self._update_state()
        if self._observation.shape[-1] != 153:
            raise RuntimeError(f"Teacher observation dimension is {self._observation.shape[-1]}, expected 153")
        return {"policy": self._observation}

    def _get_rewards(self):
        reward, reward_info = compute_reward(self, self._state)
        self.extras["log"] = {name: value.mean() for name, value in reward_info.items()}
        return reward

    def _get_dones(self):
        self._update_state()
        invalid = (~torch.isfinite(self._observation).all(dim=1)) | (~torch.isfinite(self._global_state).all(dim=1))
        below_table = (self._state["joint_height"] < 0.0).any(dim=1)
        terminated = invalid | below_table
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        for env_id in env_ids.tolist():
            pose = reset_object_pose(self, env_id, float(self.lowest_points[env_id].item()))
            arm, _, center = build_pregrasp(
                top_mesh_path(Path(self.cfg.dataset_root), self.object_names[env_id]), pose[:3], pose[3:], np.asarray(self.cfg.camera_position), np.asarray(self.cfg.hand_center),
                self.cfg.sample_num, self.cfg.top_grasp, 5.0, 1.0, self.affordance_points_obj[env_id].cpu().numpy(),
            )
            qpos = torch.tensor(tuple(arm) + INIT_FINGER_POSE, dtype=self.robot.data.joint_pos.dtype, device=self.device)
            self.robot.write_joint_state_to_sim(qpos.unsqueeze(0), torch.zeros_like(qpos).unsqueeze(0), env_ids=env_ids[env_ids == env_id])
            object_pose = torch.tensor(pose, device=self.device).unsqueeze(0)
            object_pose[:, :3] += self.scene.env_origins[env_id]
            self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids[env_ids == env_id])
            self.object.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=env_ids[env_ids == env_id])
            self.object.write_joint_state_to_sim(torch.zeros((1, self.object.num_joints), device=self.device), torch.zeros((1, self.object.num_joints), device=self.device), env_ids=env_ids[env_ids == env_id])
            self.object_initial_position[env_id] = object_pose[0, :3]
            self.affordance_center[env_id] = 0.0
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)
        self.object_initial_position[env_ids] = self.object.data.body_pos_w[env_ids, self.object_top_body_id]
        self.wrist_initial_rot_w[env_ids] = quat_to_mat_torch(
            self.robot.data.body_quat_w[env_ids][:, self.wrist_body_id]
        )
        self.previous_wrist_euler[env_ids] = 0.0
        self.current_joint_target[env_ids] = self.robot.data.joint_pos[env_ids]
        self.previous_joint_target[env_ids] = self.robot.data.joint_pos[env_ids]
        self.applied_joint_target[env_ids] = self.robot.data.joint_pos[env_ids]
        self.delay_mask[env_ids] = False

    def switch_root_guidance(self, enabled: bool):
        self.lift = enabled
        if enabled:
            self.arm_gc_lift = self.robot.data.joint_pos[:, :6].clone()
            self.lift_num = 0

    def switch_obj_pos(self, bias: torch.Tensor):
        pose = self.object.data.root_pose_w.clone()
        pose[:, :3] += bias
        self.object.write_root_pose_to_sim(pose)

    def get_global_state(self):
        self._update_state()
        return self._global_state

__all__ = ["RobustDexTeacherEnv"]
