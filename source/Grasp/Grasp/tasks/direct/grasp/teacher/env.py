from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_matrix,
)


from .actions import (
    build_full_joint_target,
    compute_residual_active_target,
    sample_delay_mask,
    select_substep_joint_target,
)
from .affordance_data import (
    AFFORDANCE_POINT_COUNT,
    TEACHER_BOTTOM_BODY_NAME,
    TEACHER_OBJECT_JOINT_NAME,
    TEACHER_TOP_BODY_NAME,
    load_teacher_affordance_data,
)
from .env_cfg import GraspTeacherEnvCfg
from .observations import (
    TEACHER_OBSERVATION_SPEC,
    TeacherObservationFeatures,
    build_teacher_observation,
    compute_affordance_geometry,
    transform_object_points_to_world,
    unwrap_euler_near_previous,
)

from .pregrasp import (
    build_teacher_pregrasp_geometry,
    compute_fr3_wrist_kinematics,
    select_teacher_pregrasp_candidate,
    solve_fr3_dls_ik,
)
from .resets import (
    sample_collision_free_teacher_resets,
    sample_teacher_object_pose,
    sample_teacher_stable_object_pose,
)
from .rewards import (
    TEACHER_REWARD_TERM_NAMES,
    TEACHER_TERMINAL_REWARD,
    compute_teacher_reward_terms,
)


class GraspTeacherEnv(DirectRLEnv):
    """Standalone RobustDexGrasp-style Teacher environment."""

    cfg: GraspTeacherEnvCfg

    def __init__(
        self,
        cfg: GraspTeacherEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        self.obs_spec = TEACHER_OBSERVATION_SPEC
        self._resolve_teacher_joint_indices()
        self._resolve_teacher_body_indices()
        self._resolve_teacher_object_indices()
        self.affordance_data = load_teacher_affordance_data(
            num_envs=self.num_envs,
            device=self.device,
            dataset_root=self.cfg.asset.dataset_root,
            object_names=self.cfg.asset.object_names,
            weighted_object_indices=(
                self.cfg.asset.weighted_object_indices
            ),
            stable_state_paths=self.cfg.asset.stable_state_paths,
        )
        self.palm_offset_b = torch.tensor(
            self.cfg.robot_spec.palm_offset,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).expand(self.num_envs, -1)
        self._initialize_teacher_reset_buffers()
        self._initialize_teacher_action_buffers()
        self._initialize_teacher_reward_state()

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.object = Articulation(self.cfg.object)
        self.table = RigidObject(self.cfg.table)
        self.mat = RigidObject(self.cfg.mat)
        self.wooden_table = RigidObject(self.cfg.wooden_table)

        self.contact_sensors: dict[str, ContactSensor] = {}
        env_regex_ns = self.scene.env_regex_ns
        for sensor_name, source_cfg in self.cfg.contact_sensors.items():
            sensor_cfg = deepcopy(source_cfg)
            sensor_cfg.prim_path = sensor_cfg.prim_path.format(
                ENV_REGEX_NS=env_regex_ns,
            )
            sensor_cfg.filter_prim_paths_expr = [
                path.format(ENV_REGEX_NS=env_regex_ns)
                for path in sensor_cfg.filter_prim_paths_expr
            ]
            sensor = ContactSensor(sensor_cfg)
            self.contact_sensors[sensor_name] = sensor

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.articulations["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["mat"] = self.mat
        self.scene.rigid_objects["wooden_table"] = self.wooden_table
        for sensor_name, sensor in self.contact_sensors.items():
            self.scene.sensors[sensor_name] = sensor

        self._spawn_visual_grid_ground()
        self.cfg.light.func("/World/Light", self.cfg.light)

    def _spawn_visual_grid_ground(self) -> None:
        if not self.cfg.enable_grid_ground:
            return

        ground_root = self.cfg.grid_ground_prim_path
        spacing = float(self.cfg.grid_ground_spacing)
        half_steps = math.ceil(
            float(self.cfg.grid_ground_size) / (2.0 * spacing)
        )
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
        base_cfg.func(
            f"{ground_root}/Base",
            base_cfg,
            translation=(0.0, 0.0, z - line_height),
        )

        line_cfg = sim_utils.CuboidCfg(
            size=(visual_size, line_width, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=self.cfg.grid_ground_line_color,
                roughness=0.8,
            ),
        )
        axis_cfg = sim_utils.CuboidCfg(
            size=(visual_size, line_width * 1.5, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(
                    self.cfg.grid_ground_axis_line_color
                ),
                roughness=0.8,
            ),
        )

        for index in range(-half_steps, half_steps + 1):
            coord = index * spacing
            current_cfg = axis_cfg if index == 0 else line_cfg
            current_cfg.func(
                f"{ground_root}/Lines/X_{index + half_steps:04d}",
                current_cfg,
                translation=(0.0, coord, z),
            )
            current_cfg.func(
                f"{ground_root}/Lines/Y_{index + half_steps:04d}",
                current_cfg,
                translation=(coord, 0.0, z),
                orientation=(
                    math.sqrt(0.5),
                    0.0,
                    0.0,
                    math.sqrt(0.5),
                ),
            )

    def _resolve_teacher_joint_indices(self) -> None:
        spec = self.cfg.robot_spec

        arm_ids, arm_names = self.robot.find_joints(
            list(spec.arm_joint_names),
            preserve_order=True,
        )
        active_hand_ids, active_hand_names = self.robot.find_joints(
            list(spec.active_hand_joint_names),
            preserve_order=True,
        )
        passive_names = tuple(spec.passive_joint_mimic.keys())
        passive_ids, resolved_passive_names = self.robot.find_joints(
            list(passive_names),
            preserve_order=True,
        )

        if tuple(arm_names) != spec.arm_joint_names:
            raise RuntimeError(
                "Resolved FR3 joint order differs from Teacher spec: "
                f"resolved={tuple(arm_names)}, "
                f"expected={spec.arm_joint_names}"
            )
        if tuple(active_hand_names) != spec.active_hand_joint_names:
            raise RuntimeError(
                "Resolved Inspire active joint order differs from "
                "Teacher spec: "
                f"resolved={tuple(active_hand_names)}, "
                f"expected={spec.active_hand_joint_names}"
            )
        if tuple(resolved_passive_names) != passive_names:
            raise RuntimeError(
                "Resolved Inspire passive joint order differs from "
                "Teacher spec: "
                f"resolved={tuple(resolved_passive_names)}, "
                f"expected={passive_names}"
            )

        self.arm_joint_ids = torch.tensor(
            arm_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.active_hand_joint_ids = torch.tensor(
            active_hand_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.active_joint_ids = torch.cat(
            (self.arm_joint_ids, self.active_hand_joint_ids),
            dim=0,
        )
        self.passive_joint_ids = torch.tensor(
            passive_ids,
            dtype=torch.long,
            device=self.device,
        )

        mimic_parent_ids: list[int] = []
        mimic_multipliers: list[float] = []
        for passive_name in passive_names:
            parent_name, multiplier = (
                spec.passive_joint_mimic[passive_name]
            )
            parent_ids, parent_names = self.robot.find_joints(
                parent_name,
                preserve_order=True,
            )
            if len(parent_ids) != 1 or parent_names[0] != parent_name:
                raise RuntimeError(
                    "Teacher mimic parent did not resolve exactly once: "
                    f"passive={passive_name}, parent={parent_name}, "
                    f"resolved={parent_names}"
                )
            mimic_parent_ids.append(parent_ids[0])
            mimic_multipliers.append(float(multiplier))

        self.mimic_parent_joint_ids = torch.tensor(
            mimic_parent_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.mimic_multipliers = torch.tensor(
            mimic_multipliers,
            dtype=torch.float32,
            device=self.device,
        )

        self.joint_lower_limits = (
            self.robot.data.soft_joint_pos_limits[0, :, 0]
            .to(self.device)
            .clone()
        )
        self.joint_upper_limits = (
            self.robot.data.soft_joint_pos_limits[0, :, 1]
            .to(self.device)
            .clone()
        )
        self.active_lower_limits = self.joint_lower_limits[
            self.active_joint_ids
        ]
        self.active_upper_limits = self.joint_upper_limits[
            self.active_joint_ids
        ]
        self.arm_lower_limits = self.joint_lower_limits[
            self.arm_joint_ids
        ]
        self.arm_upper_limits = self.joint_upper_limits[
            self.arm_joint_ids
        ]

    def _resolve_teacher_body_indices(self) -> None:
        spec = self.cfg.robot_spec

        hand_ids, hand_names = self.robot.find_bodies(
            list(spec.hand_contact_body_names),
            preserve_order=True,
        )
        arm_ids, arm_names = self.robot.find_bodies(
            list(spec.arm_height_body_names),
            preserve_order=True,
        )
        palm_ids, palm_names = self.robot.find_bodies(
            spec.palm_link_name,
            preserve_order=True,
        )
        wrist_ids, wrist_names = self.robot.find_bodies(
            spec.wrist_link_name,
            preserve_order=True,
        )

        if tuple(hand_names) != spec.hand_contact_body_names:
            raise RuntimeError(
                "Resolved Teacher hand body order differs from robot spec: "
                f"resolved={tuple(hand_names)}, "
                f"expected={spec.hand_contact_body_names}"
            )
        if tuple(arm_names) != spec.arm_height_body_names:
            raise RuntimeError(
                "Resolved Teacher arm body order differs from robot spec: "
                f"resolved={tuple(arm_names)}, "
                f"expected={spec.arm_height_body_names}"
            )
        if len(palm_ids) != 1 or palm_names[0] != spec.palm_link_name:
            raise RuntimeError(
                "Teacher palm body did not resolve exactly once: "
                f"resolved={palm_names}, expected={spec.palm_link_name}"
            )
        if len(wrist_ids) != 1 or wrist_names[0] != spec.wrist_link_name:
            raise RuntimeError(
                "Teacher wrist body did not resolve exactly once: "
                f"resolved={wrist_names}, expected={spec.wrist_link_name}"
            )

        self.hand_body_ids = torch.tensor(
            hand_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.arm_height_body_ids = torch.tensor(
            arm_ids,
            dtype=torch.long,
            device=self.device,
        )
        self.palm_body_id = palm_ids[0]
        self.wrist_body_id = wrist_ids[0]
        self.wrist_jacobian_body_index = self.wrist_body_id - 1

    def _resolve_teacher_object_indices(self) -> None:
        bottom_ids, bottom_names = self.object.find_bodies(
            TEACHER_BOTTOM_BODY_NAME,
            preserve_order=True,
        )
        top_ids, top_names = self.object.find_bodies(
            TEACHER_TOP_BODY_NAME,
            preserve_order=True,
        )
        rotation_ids, rotation_names = self.object.find_joints(
            TEACHER_OBJECT_JOINT_NAME,
            preserve_order=True,
        )

        if len(bottom_ids) != 1 or bottom_names[0] != TEACHER_BOTTOM_BODY_NAME:
            raise RuntimeError(
                "Teacher object bottom body did not resolve exactly once: "
                f"resolved={bottom_names}"
            )
        if len(top_ids) != 1 or top_names[0] != TEACHER_TOP_BODY_NAME:
            raise RuntimeError(
                "Teacher object top body did not resolve exactly once: "
                f"resolved={top_names}"
            )
        if (
            len(rotation_ids) != 1
            or rotation_names[0] != TEACHER_OBJECT_JOINT_NAME
        ):
            raise RuntimeError(
                "Teacher object rotation joint did not resolve exactly once: "
                f"resolved={rotation_names}"
            )

        self.object_bottom_body_id = bottom_ids[0]
        self.object_top_body_id = top_ids[0]
        self.object_rotation_joint_id = rotation_ids[0]


    def _initialize_teacher_reset_buffers(self) -> None:
        self.reset_object_root_state = (
            self.object.data.default_root_state.clone()
        )
        self.reset_object_joint_pos = (
            self.object.data.default_joint_pos.clone()
        )
        self.reset_object_joint_vel = torch.zeros_like(
            self.object.data.default_joint_vel
        )
        self.reset_robot_joint_pos = (
            self.robot.data.default_joint_pos.clone()
        )
        self.reset_robot_joint_vel = torch.zeros_like(
            self.robot.data.default_joint_vel
        )

        self.object_initial_root_state = torch.zeros_like(
            self.object.data.default_root_state
        )
        self.object_position_bias = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.object_bias_applied = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.visible_top_points_object = torch.zeros(
            (
                self.num_envs,
                AFFORDANCE_POINT_COUNT,
                3,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.visible_top_points_w = torch.zeros_like(
            self.visible_top_points_object
        )
        self.affordance_center_w = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.selected_pregrasp_candidate = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self.wrist_initial_quat_w = torch.zeros(
            (self.num_envs, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self.wrist_initial_quat_w[:, 0] = 1.0
        self.wrist_initial_euler = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_wrist_euler = torch.zeros_like(
            self.wrist_initial_euler
        )

    def _initialize_teacher_action_buffers(self) -> None:
        joint_pos = self.robot.data.joint_pos.clone()

        self.current_joint_target = joint_pos.clone()
        self.previous_joint_target = joint_pos.clone()
        self.applied_joint_target = joint_pos.clone()
        self.delay_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._physics_substep = 0


    def _write_teacher_reset_state(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        self.object.write_root_pose_to_sim(
            self.reset_object_root_state[env_ids, 0:7],
            env_ids=env_ids,
        )
        self.object.write_root_velocity_to_sim(
            self.reset_object_root_state[env_ids, 7:13],
            env_ids=env_ids,
        )
        self.object.write_joint_state_to_sim(
            self.reset_object_joint_pos[env_ids],
            self.reset_object_joint_vel[env_ids],
            env_ids=env_ids,
        )

        self.robot.write_joint_state_to_sim(
            self.reset_robot_joint_pos[env_ids],
            self.reset_robot_joint_vel[env_ids],
            env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            self.reset_robot_joint_pos[env_ids],
            env_ids=env_ids,
        )
        self.robot.set_joint_velocity_target(
            self.reset_robot_joint_vel[env_ids],
            env_ids=env_ids,
        )

    def _refresh_teacher_reset_kinematics(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=0.0)

    def _build_teacher_pregrasp_batch(
        self,
        env_ids: torch.Tensor,
        object_pose_w: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        object_rotation_w = matrix_from_quat(
            object_pose_w[:, 3:7]
        )
        object_pose_cpu = object_pose_w.detach().cpu().numpy()
        object_rotation_cpu = (
            object_rotation_w.detach().cpu().numpy()
        )
        env_origins_cpu = (
            self.scene.env_origins[env_ids]
            .detach()
            .cpu()
            .numpy()
        )
        env_ids_cpu = env_ids.detach().cpu()
        object_indices_cpu = (
            self.affordance_data.env_object_indices_cpu.index_select(
                0,
                env_ids_cpu,
            )
        )
        sampled_top_points_object_cpu = (
            self.affordance_data.unique_top_points_object_cpu
            .index_select(0, object_indices_cpu)
            .numpy()
        )
        palm_offset_body_cpu = np.asarray(
            self.cfg.robot_spec.palm_offset,
            dtype=np.float32,
        )

        visible_points_w: list[np.ndarray] = []
        visible_points_object: list[np.ndarray] = []
        affordance_centers_w: list[np.ndarray] = []
        target_positions_w: list[np.ndarray] = []
        target_rotations_w: list[np.ndarray] = []
        projection_lengths: list[np.ndarray] = []

        for local_index, env_id_value in enumerate(
            env_ids_cpu.tolist()
        ):
            geometry = build_teacher_pregrasp_geometry(
                top_mesh=(
                    self.affordance_data.env_top_meshes[env_id_value]
                ),
                sampled_top_points_object=(
                    sampled_top_points_object_cpu[local_index]
                ),
                object_position_world=object_pose_cpu[
                    local_index,
                    0:3,
                ],
                object_rotation_world=object_rotation_cpu[
                    local_index
                ],
                env_origin_world=env_origins_cpu[local_index],
                palm_offset_body=palm_offset_body_cpu,
                cfg=self.cfg.pregrasp,
            )
            visible_points_w.append(geometry.visible_points_world)
            visible_points_object.append(
                geometry.visible_points_object
            )
            affordance_centers_w.append(
                geometry.affordance_center_world
            )
            target_positions_w.append(
                geometry.wrist_target_positions_world
            )
            target_rotations_w.append(
                geometry.wrist_target_rotations_world
            )
            projection_lengths.append(geometry.projection_lengths)

        visible_points_w_tensor = torch.as_tensor(
            np.stack(visible_points_w, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        visible_points_object_tensor = torch.as_tensor(
            np.stack(visible_points_object, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        affordance_centers_w_tensor = torch.as_tensor(
            np.stack(affordance_centers_w, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        target_positions_w_tensor = torch.as_tensor(
            np.stack(target_positions_w, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        target_rotation_matrices_w = torch.as_tensor(
            np.stack(target_rotations_w, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        target_quaternions_w_tensor = quat_from_matrix(
            target_rotation_matrices_w.reshape(-1, 3, 3)
        ).reshape(
            env_ids.numel(),
            self.cfg.pregrasp.candidate_count,
            4,
        )
        projection_lengths_tensor = torch.as_tensor(
            np.stack(projection_lengths, axis=0),
            dtype=torch.float32,
            device=self.device,
        )

        return (
            visible_points_w_tensor,
            visible_points_object_tensor,
            affordance_centers_w_tensor,
            target_positions_w_tensor,
            target_quaternions_w_tensor,
            projection_lengths_tensor,
        )

    def _sample_teacher_reset_candidates(
        self,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.affordance_data.stable_states is None:
            object_pose_w = sample_teacher_object_pose(
                env_origins=self.scene.env_origins[env_ids],
                lowest_points=(
                    self.affordance_data.lowest_points[env_ids]
                ),
                support_height=self.cfg.geometry.table_height,
                workspace_center_y=(
                    self.cfg.geometry.center_xy[1]
                ),
                cfg=self.cfg.reset,
                uniform_only=(
                    not self.cfg.reset.non_uniform_sampling
                ),
            )
        else:
            object_pose_w = sample_teacher_stable_object_pose(
                env_origins=self.scene.env_origins[env_ids],
                stable_states=(
                    self.affordance_data.stable_states[env_ids]
                ),
                support_height=self.cfg.geometry.table_height,
                source_support_height=(
                    self.cfg.asset
                    .stable_state_source_support_height
                ),
                workspace_center_y=(
                    self.cfg.geometry.center_xy[1]
                ),
                cfg=self.cfg.reset,
            )

        (
            visible_points_w,
            visible_points_object,
            affordance_centers_w,
            target_positions_w,
            target_quaternions_w,
            projection_lengths,
        ) = self._build_teacher_pregrasp_batch(
            env_ids=env_ids,
            object_pose_w=object_pose_w,
        )

        initial_arm_qpos = self.robot.data.default_joint_pos[
            env_ids
        ][:, self.arm_joint_ids]
        candidate_count = self.cfg.pregrasp.candidate_count
        arm_joint_count = initial_arm_qpos.shape[1]
        robot_base_positions_world = self.scene.env_origins[env_ids]
        batched_initial_arm_qpos = (
            initial_arm_qpos.unsqueeze(1)
            .expand(-1, candidate_count, -1)
            .reshape(-1, arm_joint_count)
        )
        batched_robot_base_positions_world = (
            robot_base_positions_world.unsqueeze(1)
            .expand(-1, candidate_count, -1)
            .reshape(-1, 3)
        )
        ik_result = solve_fr3_dls_ik(
            initial_arm_qpos=batched_initial_arm_qpos,
            target_wrist_position_world=target_positions_w.reshape(
                -1, 3
            ),
            target_wrist_quaternion_world=(
                target_quaternions_w.reshape(-1, 4)
            ),
            arm_lower_limits=self.arm_lower_limits,
            arm_upper_limits=self.arm_upper_limits,
            cfg=self.cfg.pregrasp,
            evaluate_kinematics=(
                lambda arm_qpos: compute_fr3_wrist_kinematics(
                    arm_qpos=arm_qpos,
                    robot_base_positions_world=(
                        batched_robot_base_positions_world
                    ),
                )
            ),
        )
        candidate_arm_qpos = ik_result.arm_qpos.reshape(
            env_ids.numel(),
            candidate_count,
            arm_joint_count,
        )
        candidate_converged = ik_result.converged.reshape(
            env_ids.numel(),
            candidate_count,
        )

        selection = select_teacher_pregrasp_candidate(
            candidate_arm_qpos=candidate_arm_qpos,
            candidate_converged=candidate_converged,
            projection_lengths=projection_lengths,
            cfg=self.cfg.pregrasp,
        )
        valid_env_ids = env_ids[selection.valid]
        if valid_env_ids.numel() == 0:
            return selection.valid

        valid_object_pose_w = object_pose_w[selection.valid]
        object_root_state = self.object.data.default_root_state[
            valid_env_ids
        ].clone()
        object_root_state[:, 0:7] = valid_object_pose_w
        object_root_state[:, 7:13] = 0.0
        self.reset_object_root_state[valid_env_ids] = object_root_state

        object_joint_pos = self.object.data.default_joint_pos[
            valid_env_ids
        ].clone()
        object_joint_pos[:, self.object_rotation_joint_id] = 0.0
        self.reset_object_joint_pos[valid_env_ids] = object_joint_pos
        self.reset_object_joint_vel[valid_env_ids] = 0.0

        robot_joint_pos = self.robot.data.default_joint_pos[
            valid_env_ids
        ].clone()
        robot_joint_pos[:, self.arm_joint_ids] = (
            selection.arm_qpos[selection.valid]
        )
        self.reset_robot_joint_pos[valid_env_ids] = robot_joint_pos
        self.reset_robot_joint_vel[valid_env_ids] = 0.0

        self.visible_top_points_w[valid_env_ids] = (
            visible_points_w[selection.valid]
        )
        self.visible_top_points_object[valid_env_ids] = (
            visible_points_object[selection.valid]
        )
        self.affordance_center_w[valid_env_ids] = (
            affordance_centers_w[selection.valid]
        )
        self.selected_pregrasp_candidate[valid_env_ids] = (
            selection.selected_candidate_indices[selection.valid]
        )

        return selection.valid

    def _check_teacher_initial_self_collision(
        self,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        sensor_names = tuple(
            f"contact__{body_name}"
            for body_name in self.cfg.reward.arm_collision_body_names
        )
        for sensor_name in sensor_names:
            self.contact_sensors[sensor_name].reset(env_ids)

        self._write_teacher_reset_state(env_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        self_collision = torch.zeros(
            env_ids.numel(),
            dtype=torch.bool,
            device=self.device,
        )
        for sensor_name in sensor_names:
            net_forces_w = self.contact_sensors[
                sensor_name
            ].data.net_forces_w
            contact_force_w = net_forces_w[
                env_ids,
                0,
                :,
            ]
            self_collision |= (
                torch.linalg.vector_norm(
                    contact_force_w,
                    dim=-1,
                ) > 0.0
            )
        return self_collision


    # 这里必须读取 robot.data.joint_pos，因为 RobustDexGrasp 每个 step
    # 结束后把最新 gc_r_ 作为下一 action mean。不能用 previous_joint_target
    # 代替真实 qpos，否则执行误差会在 target 中累积。
    def _pre_physics_step(
        self,
        actions: torch.Tensor,
    ) -> None:
        current_joint_pos = self.robot.data.joint_pos
        current_active_qpos = current_joint_pos[
            :,
            self.active_joint_ids,
        ]

        policy_actions, active_target = (
            compute_residual_active_target(
                actions=actions.to(self.device),
                current_active_qpos=current_active_qpos,
                active_lower_limits=self.active_lower_limits,
                active_upper_limits=self.active_upper_limits,
                arm_scale=self.cfg.action.arm_residual_scale,
                hand_scale=self.cfg.action.hand_residual_scale,
            )
        )
        self.actions.copy_(policy_actions)

        self.current_joint_target.copy_(
            build_full_joint_target(
                current_joint_pos=current_joint_pos,
                active_target=active_target,
                active_joint_ids=self.active_joint_ids,
                passive_joint_ids=self.passive_joint_ids,
                mimic_parent_joint_ids=(
                    self.mimic_parent_joint_ids
                ),
                mimic_multipliers=self.mimic_multipliers,
                joint_lower_limits=self.joint_lower_limits,
                joint_upper_limits=self.joint_upper_limits,
            )
        )
        self.delay_mask.copy_(
            sample_delay_mask(
                num_envs=self.num_envs,
                probability=self.cfg.action.delay_probability,
                device=self.device,
            )
        )
        self._physics_substep = 0
        self._begin_teacher_contact_step()

    # Isaac Lab 的 DirectRLEnv.step() 会在每个 _apply_action() 后自动调用
    # scene.write_data_to_sim()、sim.step() 和 scene.update()。
    def _apply_action(self) -> None:
        selected_target = select_substep_joint_target(
            current_joint_target=self.current_joint_target,
            previous_joint_target=self.previous_joint_target,
            delay_mask=self.delay_mask,
            physics_substep=self._physics_substep,
            decimation=self.cfg.decimation,
        )
        self.applied_joint_target.copy_(selected_target)
        self.robot.set_joint_position_target(
            self.applied_joint_target
        )

        if self._physics_substep == self.cfg.decimation - 1:
            self.previous_joint_target.copy_(
                self.current_joint_target
            )
        self._physics_substep += 1

    # reset 动作状态
    def _reset_teacher_action_state(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        current_joint_pos = self.robot.data.joint_pos[env_ids]
        self.current_joint_target[env_ids] = current_joint_pos
        self.previous_joint_target[env_ids] = current_joint_pos
        self.applied_joint_target[env_ids] = current_joint_pos
        self.actions[env_ids] = 0.0
        self.delay_mask[env_ids] = False

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        resolved_env_ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        )
        super()._reset_idx(resolved_env_ids)

        self.selected_pregrasp_candidate[resolved_env_ids] = -1
        sample_collision_free_teacher_resets(
            env_ids=resolved_env_ids,
            max_reset_rounds=self.cfg.pregrasp.max_reset_rounds,
            sample_candidates=self._sample_teacher_reset_candidates,
            check_self_collision=(
                self._check_teacher_initial_self_collision
            ),
        )

        self._write_teacher_reset_state(resolved_env_ids)
        self._refresh_teacher_reset_kinematics()

        self.object_initial_root_state[resolved_env_ids] = (
            self.object.data.root_state_w[resolved_env_ids].clone()
        )
        self.object_position_bias[resolved_env_ids] = torch.empty(
            (resolved_env_ids.numel(), 3),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(
            -self.cfg.reset.biased_position_range,
            self.cfg.reset.biased_position_range,
        )
        self.object_bias_applied[resolved_env_ids] = False
        wrist_quaternion_w = self.robot.data.body_quat_w[
            resolved_env_ids,
            self.wrist_body_id,
        ].clone()
        wrist_roll, wrist_pitch, wrist_yaw = (
            euler_xyz_from_quat(wrist_quaternion_w)
        )
        wrist_euler = torch.stack(
            (wrist_roll, wrist_pitch, wrist_yaw),
            dim=-1,
        )
        self.wrist_initial_quat_w[resolved_env_ids] = (
            wrist_quaternion_w
        )
        self.wrist_initial_euler[resolved_env_ids] = wrist_euler
        self.previous_wrist_euler[resolved_env_ids] = wrist_euler

        self._reset_teacher_action_state(resolved_env_ids)
        self._reset_teacher_reward_state(resolved_env_ids)
        for sensor in self.contact_sensors.values():
            sensor.reset(resolved_env_ids)

        self._teacher_step_features = None
        self._teacher_step_observation = None

    def _reset_teacher_reward_state(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        contact_tensors = (
            self.hand_normal_impulse_by_filter_w,
            self.hand_friction_impulse_by_filter_w,
            self.arm_normal_impulse_by_filter_w,
            self.arm_friction_impulse_by_filter_w,
            self.top_normal_impulse_vector_w,
            self.top_friction_impulse_vector_w,
            self.top_impulse_vector_w,
            self.bottom_normal_impulse_vector_w,
            self.bottom_friction_impulse_vector_w,
            self.bottom_impulse_vector_w,
            self.support_normal_impulse_vector_w,
            self.support_friction_impulse_vector_w,
            self.support_impulse_vector_w,
            self.arm_interaction_normal_impulse_w,
            self.arm_interaction_friction_impulse_w,
            self.arm_interaction_impulse_w,
            self.affordance_contact,
            self.affordance_impulse,
        )
        for tensor in contact_tensors:
            tensor[env_ids] = 0.0

        self.arm_all_contact[env_ids] = False
        self.successes[env_ids] = 0.0
        self.current_successes[env_ids] = 0.0
        for value in self.teacher_reward_terms.values():
            value[env_ids] = 0.0

        self._contact_step_finalized = True


    def _compute_teacher_observation_features(
        self,
        commit_wrist_history: bool,
    ) -> TeacherObservationFeatures:
        active_qpos = self.robot.data.joint_pos[
            :,
            self.active_joint_ids,
        ].clone()
        current_active_target = self.current_joint_target[
            :,
            self.active_joint_ids,
        ]
        joint_target_error = (
            current_active_target - active_qpos
        )

        hand_body_positions_w = self.robot.data.body_pos_w[
            :,
            self.hand_body_ids,
            :,
        ]
        arm_body_positions_w = self.robot.data.body_pos_w[
            :,
            self.arm_height_body_ids,
            :,
        ]
        support_height_w = (
            self.scene.env_origins[:, 2:3]
            + self.cfg.geometry.table_height
        )
        hand_body_height = (
            hand_body_positions_w[:, :, 2]
            - support_height_w
        )
        arm_body_height = (
            arm_body_positions_w[:, :, 2]
            - support_height_w
        )

        palm_position_w = self.robot.data.body_pos_w[
            :,
            self.palm_body_id,
            :,
        ]
        palm_quaternion_w = self.robot.data.body_quat_w[
            :,
            self.palm_body_id,
            :,
        ]
        palm_center_global_w = (
            palm_position_w
            + quat_apply(
                palm_quaternion_w,
                self.palm_offset_b,
            )
        )
        palm_center_world = (
            palm_center_global_w - self.scene.env_origins
        )

        object_top_position_w = self.object.data.body_pos_w[
            :,
            self.object_top_body_id,
            :,
        ]
        object_top_quaternion_w = self.object.data.body_quat_w[
            :,
            self.object_top_body_id,
            :,
        ]
        top_points_world = transform_object_points_to_world(
            points_object=(
                self.affordance_data.top_points_object
            ),
            body_position_world=object_top_position_w,
            body_quaternion_world=object_top_quaternion_w,
        )
        (
            nearest_affordance_distance,
            nearest_affordance_point_world,
            nearest_affordance_vector_world,
        ) = compute_affordance_geometry(
            hand_body_positions_world=hand_body_positions_w,
            top_points_world=top_points_world,
        )

        wrist_quaternion_w = self.robot.data.body_quat_w[
            :,
            self.wrist_body_id,
            :,
        ]
        wrist_roll, wrist_pitch, wrist_yaw = (
            euler_xyz_from_quat(wrist_quaternion_w)
        )
        raw_wrist_euler = torch.stack(
            (wrist_roll, wrist_pitch, wrist_yaw),
            dim=-1,
        )
        wrist_euler = unwrap_euler_near_previous(
            current_euler=raw_wrist_euler,
            previous_euler=self.previous_wrist_euler,
        )
        if commit_wrist_history:
            self.previous_wrist_euler.copy_(wrist_euler)

        initial_wrist_rotation_w = matrix_from_quat(
            self.wrist_initial_quat_w
        )
        current_wrist_rotation_w = matrix_from_quat(
            wrist_quaternion_w
        )
        wrist_delta_rotation = torch.matmul(
            initial_wrist_rotation_w.transpose(-1, -2),
            current_wrist_rotation_w,
        )
        wrist_delta_quaternion = quat_from_matrix(
            wrist_delta_rotation
        )
        delta_roll, delta_pitch, delta_yaw = (
            euler_xyz_from_quat(wrist_delta_quaternion)
        )
        wrist_delta_euler = torch.stack(
            (delta_roll, delta_pitch, delta_yaw),
            dim=-1,
        )

        return TeacherObservationFeatures(
            active_qpos=active_qpos,
            joint_target_error=joint_target_error,
            affordance_contact=self.affordance_contact.clone(),
            affordance_impulse=self.affordance_impulse.clone(),
            hand_body_height=hand_body_height,
            arm_body_height=arm_body_height,
            palm_center_world=palm_center_world,
            wrist_delta_euler=wrist_delta_euler,
            wrist_euler=wrist_euler,
            nearest_affordance_vector_world=(
                nearest_affordance_vector_world
            ),
            nearest_affordance_distance=(
                nearest_affordance_distance
            ),
            nearest_affordance_point_world=(
                nearest_affordance_point_world
            ),
        )

    def _apply_teacher_object_bias(
        self,
        features: TeacherObservationFeatures,
    ) -> None:
        if not self.cfg.reset.biased:
            return
        minimum_distance = torch.min(
            features.nearest_affordance_distance,
            dim=1,
        ).values
        trigger = (
            ~self.object_bias_applied
            & (
                minimum_distance
                < self.cfg.reset.biased_distance_threshold
            )
        )
        env_ids = torch.nonzero(trigger, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        object_root_pose = self.object.data.root_state_w[
            env_ids, 0:7
        ].clone()
        object_root_pose[:, 0:3] += self.object_position_bias[env_ids]
        self.object.write_root_pose_to_sim(
            object_root_pose,
            env_ids=env_ids,
        )
        self.object_bias_applied[env_ids] = True

    def _get_observations(self) -> dict[str, torch.Tensor]:
        features = self._compute_teacher_observation_features(
            commit_wrist_history=True,
        )
        policy_observation = (
            self._build_teacher_policy_observation(features)
        )
        self._apply_teacher_object_bias(features)
        self._teacher_step_features = None
        self._teacher_step_observation = None
        return {"policy": policy_observation}

    def _initialize_teacher_reward_state(self) -> None:
        spec = self.cfg.robot_spec
        self.hand_contact_sensor_names = tuple(
            f"contact__{body_name}"
            for body_name in spec.hand_contact_body_names
        )
        self.arm_contact_sensor_names = tuple(
            f"contact__{body_name}"
            for body_name in spec.arm_height_body_names
        )

        arm_body_names = spec.arm_height_body_names
        self.arm_height_penalty_indices = torch.tensor(
            [
                arm_body_names.index(body_name)
                for body_name in (
                    self.cfg.reward.arm_height_penalty_body_names
                )
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_collision_indices = torch.tensor(
            [
                arm_body_names.index(body_name)
                for body_name in (
                    self.cfg.reward.arm_collision_body_names
                )
            ],
            dtype=torch.long,
            device=self.device,
        )

        first_sensor = self.contact_sensors[
            self.hand_contact_sensor_names[0]
        ]
        self.contact_filter_count = (
            first_sensor.data.force_matrix_w.shape[2]
        )
        self.top_contact_filter_index = 0
        self.bottom_contact_filter_index = 1
        self.support_contact_filter_slice = slice(2, 5)
        hand_filter_shape = (
            self.num_envs,
            13,
            self.contact_filter_count,
            3,
        )
        arm_filter_shape = (
            self.num_envs,
            6,
            self.contact_filter_count,
            3,
        )
        self.hand_normal_impulse_by_filter_w = torch.zeros(
            hand_filter_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.hand_friction_impulse_by_filter_w = (
            torch.zeros_like(
                self.hand_normal_impulse_by_filter_w
            )
        )
        self.arm_normal_impulse_by_filter_w = torch.zeros(
            arm_filter_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.arm_friction_impulse_by_filter_w = (
            torch.zeros_like(
                self.arm_normal_impulse_by_filter_w
            )
        )

        hand_shape = (self.num_envs, 13, 3)
        arm_shape = (self.num_envs, 6, 3)
        self.top_normal_impulse_vector_w = torch.zeros(
            hand_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.top_friction_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.top_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.bottom_normal_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.bottom_friction_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.bottom_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.support_normal_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.support_friction_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.support_impulse_vector_w = torch.zeros_like(
            self.top_normal_impulse_vector_w
        )
        self.arm_interaction_normal_impulse_w = torch.zeros(
            arm_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.arm_interaction_friction_impulse_w = (
            torch.zeros_like(
                self.arm_interaction_normal_impulse_w
            )
        )
        self.arm_interaction_impulse_w = torch.zeros_like(
            self.arm_interaction_normal_impulse_w
        )
        self.arm_all_contact = torch.zeros(
            (self.num_envs, 6),
            dtype=torch.bool,
            device=self.device,
        )

        self.affordance_contact = torch.zeros(
            (self.num_envs, 13),
            dtype=torch.float32,
            device=self.device,
        )
        self.affordance_impulse = torch.zeros_like(
            self.affordance_contact
        )
        self.successes = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.current_successes = torch.zeros_like(self.successes)
        self.teacher_reward_terms = {
            name: torch.zeros_like(self.successes)
            for name in TEACHER_REWARD_TERM_NAMES
        }

        self._contact_step_finalized = True
        self._teacher_step_features = None
        self._teacher_step_observation = None

    def _begin_teacher_contact_step(self) -> None:
        self.hand_normal_impulse_by_filter_w.zero_()
        self.hand_friction_impulse_by_filter_w.zero_()
        self.arm_normal_impulse_by_filter_w.zero_()
        self.arm_friction_impulse_by_filter_w.zero_()
        self.arm_all_contact.zero_()
        self._contact_step_finalized = False

    def _read_teacher_contact_forces(
        self,
        sensor_names: tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal_forces_w: list[torch.Tensor] = []
        friction_forces_w: list[torch.Tensor] = []
        net_forces_w: list[torch.Tensor] = []
        for sensor_name in sensor_names:
            sensor_data = self.contact_sensors[sensor_name].data
            normal_forces_w.append(
                sensor_data.force_matrix_w[:, 0, :, :]
            )
            friction_forces_w.append(
                sensor_data.friction_forces_w[:, 0, :, :]
            )
            net_forces_w.append(
                sensor_data.net_forces_w[:, 0, :]
            )
        return (
            torch.stack(normal_forces_w, dim=1),
            torch.stack(friction_forces_w, dim=1),
            torch.stack(net_forces_w, dim=1),
        )

    def _accumulate_teacher_contact_substep(self) -> None:
        (
            hand_normal_force_w,
            hand_friction_force_w,
            _,
        ) = self._read_teacher_contact_forces(
            self.hand_contact_sensor_names
        )
        (
            arm_normal_force_w,
            arm_friction_force_w,
            arm_net_force_w,
        ) = self._read_teacher_contact_forces(
            self.arm_contact_sensor_names
        )

        self.hand_normal_impulse_by_filter_w.add_(
            hand_normal_force_w * self.physics_dt
        )
        self.hand_friction_impulse_by_filter_w.add_(
            hand_friction_force_w * self.physics_dt
        )
        self.arm_normal_impulse_by_filter_w.add_(
            arm_normal_force_w * self.physics_dt
        )
        self.arm_friction_impulse_by_filter_w.add_(
            arm_friction_force_w * self.physics_dt
        )

        self.arm_all_contact.logical_or_(
            torch.linalg.vector_norm(
                arm_net_force_w,
                dim=-1,
            ) > 0.0
        )

    def _finalize_teacher_contact_step_data(self) -> None:
        hand_total_impulse_by_filter_w = (
            self.hand_normal_impulse_by_filter_w
            + self.hand_friction_impulse_by_filter_w
        )
        arm_total_impulse_by_filter_w = (
            self.arm_normal_impulse_by_filter_w
            + self.arm_friction_impulse_by_filter_w
        )

        top_index = self.top_contact_filter_index
        bottom_index = self.bottom_contact_filter_index
        support_slice = self.support_contact_filter_slice

        self.top_normal_impulse_vector_w.copy_(
            self.hand_normal_impulse_by_filter_w[
                :, :, top_index, :
            ]
        )
        self.top_friction_impulse_vector_w.copy_(
            self.hand_friction_impulse_by_filter_w[
                :, :, top_index, :
            ]
        )
        self.top_impulse_vector_w.copy_(
            hand_total_impulse_by_filter_w[
                :, :, top_index, :
            ]
        )

        self.bottom_normal_impulse_vector_w.copy_(
            self.hand_normal_impulse_by_filter_w[
                :, :, bottom_index, :
            ]
        )
        self.bottom_friction_impulse_vector_w.copy_(
            self.hand_friction_impulse_by_filter_w[
                :, :, bottom_index, :
            ]
        )
        self.bottom_impulse_vector_w.copy_(
            hand_total_impulse_by_filter_w[
                :, :, bottom_index, :
            ]
        )

        self.support_normal_impulse_vector_w.copy_(
            self.hand_normal_impulse_by_filter_w[
                :, :, support_slice, :
            ].sum(dim=2)
        )
        self.support_friction_impulse_vector_w.copy_(
            self.hand_friction_impulse_by_filter_w[
                :, :, support_slice, :
            ].sum(dim=2)
        )
        self.support_impulse_vector_w.copy_(
            hand_total_impulse_by_filter_w[
                :, :, support_slice, :
            ].sum(dim=2)
        )
        self.arm_interaction_normal_impulse_w.copy_(
            self.arm_normal_impulse_by_filter_w.sum(dim=2)
        )
        self.arm_interaction_friction_impulse_w.copy_(
            self.arm_friction_impulse_by_filter_w.sum(dim=2)
        )
        self.arm_interaction_impulse_w.copy_(
            arm_total_impulse_by_filter_w.sum(dim=2)
        )

        top_total_impulse = torch.linalg.vector_norm(
            self.top_impulse_vector_w,
            dim=-1,
        )
        self.affordance_contact.copy_(
            (
                top_total_impulse
                > self.cfg.reward.contact_impulse_threshold
            ).to(torch.float32)
        )
        self.affordance_impulse.copy_(top_total_impulse)

    def _update_teacher_contact_step_data(self) -> None:
        if not self._contact_step_finalized:
            self._accumulate_teacher_contact_substep()
            self._finalize_teacher_contact_step_data()
            self._contact_step_finalized = True


    def _build_teacher_policy_observation(
        self,
        features: TeacherObservationFeatures,
    ) -> torch.Tensor:
        return build_teacher_observation(
            spec=self.obs_spec,
            active_qpos=features.active_qpos,
            joint_target_error=features.joint_target_error,
            affordance_contact=features.affordance_contact,
            affordance_impulse=features.affordance_impulse,
            hand_body_height=features.hand_body_height,
            arm_body_height=features.arm_body_height,
            palm_center_world=features.palm_center_world,
            wrist_delta_euler=features.wrist_delta_euler,
            wrist_euler=features.wrist_euler,
            affordance_vector_world=(
                features.nearest_affordance_vector_world
            ),
        )


    def _update_teacher_step_data(self) -> None:
        self._update_teacher_contact_step_data()
        features = self._compute_teacher_observation_features(
            commit_wrist_history=False,
        )
        self._teacher_step_features = features
        self._teacher_step_observation = (
            self._build_teacher_policy_observation(features)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_teacher_step_data()
        features = self._teacher_step_features
        policy_observation = self._teacher_step_observation

        hand_too_low = torch.any(
            features.hand_body_height < 0.0,
            dim=-1,
        )
        non_finite = ~torch.isfinite(
            policy_observation
        ).all(dim=-1)
        critical_tensors = (
            features.active_qpos,
            features.joint_target_error,
            features.nearest_affordance_distance,
            features.nearest_affordance_vector_world,
            features.hand_body_height,
            features.arm_body_height,
            features.palm_center_world,
            features.wrist_delta_euler,
            features.wrist_euler,
            self.robot.data.joint_pos[
                :, self.active_joint_ids
            ],
            self.robot.data.joint_vel[
                :, self.active_joint_ids
            ],
            self.object.data.root_state_w,
            self.object.data.body_state_w[
                :, self.object_top_body_id, :
            ],
            self.top_normal_impulse_vector_w,
            self.top_friction_impulse_vector_w,
            self.top_impulse_vector_w,
            self.bottom_normal_impulse_vector_w,
            self.bottom_friction_impulse_vector_w,
            self.bottom_impulse_vector_w,
            self.support_normal_impulse_vector_w,
            self.support_friction_impulse_vector_w,
            self.support_impulse_vector_w,
            self.arm_interaction_normal_impulse_w,
            self.arm_interaction_friction_impulse_w,
            self.arm_interaction_impulse_w,
        )
        for tensor in critical_tensors:
            non_finite |= ~torch.isfinite(
                tensor.reshape(self.num_envs, -1)
            ).all(dim=-1)

        terminated = hand_too_low | non_finite
        truncated = torch.zeros_like(terminated)
        return terminated, truncated


    def _get_rewards(self) -> torch.Tensor:
        features = self._teacher_step_features
        if features is None:
            raise RuntimeError(
                "Teacher reward requested before current step snapshot"
            )

        base_reward, weighted_terms, raw_terms = (
            compute_teacher_reward_terms(
                nearest_affordance_distance=(
                    features.nearest_affordance_distance
                ),
                hand_body_height=features.hand_body_height,
                arm_body_height=features.arm_body_height,
                top_total_impulse_vector_w=(
                    self.top_impulse_vector_w
                ),
                top_normal_impulse_vector_w=(
                    self.top_normal_impulse_vector_w
                ),
                top_friction_impulse_vector_w=(
                    self.top_friction_impulse_vector_w
                ),
                bottom_total_impulse_vector_w=(
                    self.bottom_impulse_vector_w
                ),
                support_total_impulse_vector_w=(
                    self.support_impulse_vector_w
                ),
                arm_interaction_total_impulse_w=(
                    self.arm_interaction_impulse_w
                ),
                arm_all_contact=self.arm_all_contact,
                wrist_linear_velocity_w=(
                    self.robot.data.body_lin_vel_w[
                        :, self.wrist_body_id, :
                    ]
                ),
                wrist_angular_velocity_w=(
                    self.robot.data.body_ang_vel_w[
                        :, self.wrist_body_id, :
                    ]
                ),
                object_top_linear_velocity_w=(
                    self.object.data.body_lin_vel_w[
                        :, self.object_top_body_id, :
                    ]
                ),
                object_top_angular_velocity_w=(
                    self.object.data.body_ang_vel_w[
                        :, self.object_top_body_id, :
                    ]
                ),
                object_position_w=self.object.data.root_pos_w,
                object_initial_position_w=(
                    self.object_initial_root_state[:, 0:3]
                ),
                arm_joint_velocity=self.robot.data.joint_vel[
                    :, self.arm_joint_ids
                ],
                arm_height_penalty_indices=(
                    self.arm_height_penalty_indices
                ),
                arm_collision_indices=(
                    self.arm_collision_indices
                ),
                cfg=self.cfg.reward,
            )
        )

        finite_terminal_base_reward = torch.where(
            self.reset_terminated,
            torch.nan_to_num(
                base_reward,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            base_reward,
        )
        terminal_penalty = (
            self.reset_terminated.to(base_reward.dtype)
            * TEACHER_TERMINAL_REWARD
        )
        total_reward = torch.clamp(
            finite_terminal_base_reward + terminal_penalty,
            min=self.cfg.reward.min_reward,
        )

        for name in TEACHER_REWARD_TERM_NAMES:
            self.teacher_reward_terms[name].copy_(
                weighted_terms[name]
            )

        self.successes.copy_(raw_terms["lift_success"])
        self.current_successes.copy_(
            torch.maximum(
                self.current_successes,
                self.successes,
            )
        )

        log_values = {
            name: value.mean()
            for name, value in weighted_terms.items()
        }
        log_values.update(
            {
                "teacher_reward_before_terminal": (
                    finite_terminal_base_reward.mean()
                ),
                "terminal_penalty": terminal_penalty.mean(),
                "teacher_reward": total_reward.mean(),
                "top_contact_raw": raw_terms[
                    "top_contact_raw"
                ].mean(),
                "top_impulse_tangential_raw": raw_terms[
                    "top_impulse_tangential_raw"
                ].mean(),
                "bottom_contact_raw": raw_terms[
                    "bottom_contact_raw"
                ].mean(),
                "bottom_impulse_raw": raw_terms[
                    "bottom_impulse_raw"
                ].mean(),
                "support_contact_raw": raw_terms[
                    "support_contact_raw"
                ].mean(),
                "support_impulse_raw": raw_terms[
                    "support_impulse_raw"
                ].mean(),
                "arm_contact_raw": raw_terms[
                    "arm_contact_raw"
                ].mean(),
                "arm_impulse_raw": raw_terms[
                    "arm_impulse_raw"
                ].mean(),
                "arm_collision_raw": raw_terms[
                    "arm_collision_raw"
                ].mean(),
                "lift_success_rate": self.successes.mean(),
            }
        )
        self.extras["log"] = log_values
        return total_reward
