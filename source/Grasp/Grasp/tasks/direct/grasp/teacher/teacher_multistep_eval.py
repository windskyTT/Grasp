"""Shared Teacher visual and quantitative evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import (
    TEACHER_ACTION_DIM,
    TEACHER_ARM_ACTION_DIM,
    TEACHER_HAND_ACTION_DIM,
)
from .object_set import (
    NEW_TRAINING_SET_ROOT,
    TEACHER_OBJECT_NAMES,
)
from .observations import TEACHER_OBSERVATION_SPEC


SHAPENET_30_ROOT = Path(
    "/home/windsky/project/Grasp/assets/shapenet-30obj"
)

SHAPENET_30_OBJECT_NAMES: tuple[str, ...] = (
    "Bear_34",
    "Black_mug",
    "Blue_camera",
    "Blue_teapot",
    "Camera_brown",
    "Camera_yellow",
    "CellPhone_4e",
    "Donut",
    "DrinkBottle_blue_1ef",
    "Gun",
    "Hammer_40",
    "Knife",
    "Mug_8b_red",
    "Mug_gray_d7",
    "Mug_yellow_18",
    "Pan_gray",
    "Plate_gold_69",
    "Purse_brown_d58",
    "Purse_red_f58",
    "Red_bottle",
    "Red_chair",
    "Red_scissor",
    "Stapler_gray_d9",
    "Teapot_blue_high_3d2",
    "Teapot_brown",
    "Vase_red_a1",
    "Watch_9f",
    "WineGlass_gray_9d",
    "Wine_glass2_1_blue",
    "Wine_glass_body_blue",
)


@dataclass(frozen=True)
class TeacherEvaluationMode:
    name: str
    dataset_root: Path
    object_names: tuple[str, ...]
    grasp_steps: int
    lift_steps: int
    default_episodes: int
    use_stable_states: bool


@dataclass(frozen=True)
class TeacherObjectSelection:
    mode: TeacherEvaluationMode
    names: tuple[str, ...]
    usd_paths: tuple[str, ...]
    stable_state_paths: tuple[str, ...]


@dataclass(frozen=True)
class TeacherEvaluationEpisode:
    success: torch.Tensor
    interrupted: torch.Tensor
    lift_height: torch.Tensor
    mean_grasp_reward: float
    mean_grasp_logs: dict[str, float]


VISUAL_MODE = TeacherEvaluationMode(
    name="visual",
    dataset_root=NEW_TRAINING_SET_ROOT,
    object_names=TEACHER_OBJECT_NAMES,
    grasp_steps=70,
    lift_steps=30,
    default_episodes=10,
    use_stable_states=False,
)

QUANTITATIVE_MODE = TeacherEvaluationMode(
    name="quantitative",
    dataset_root=SHAPENET_30_ROOT,
    object_names=SHAPENET_30_OBJECT_NAMES,
    grasp_steps=100,
    lift_steps=100,
    default_episodes=5,
    use_stable_states=True,
)

TEACHER_EVALUATION_MODES = {
    VISUAL_MODE.name: VISUAL_MODE,
    QUANTITATIVE_MODE.name: QUANTITATIVE_MODE,
}


def resolve_teacher_object_selection(
    mode_name: str,
    value: str,
) -> TeacherObjectSelection:
    mode = TEACHER_EVALUATION_MODES[mode_name]
    if value.strip() == "all":
        names = mode.object_names
    else:
        names = tuple(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )

    unknown_names = tuple(
        name for name in names if name not in mode.object_names
    )
    if not names or unknown_names or len(set(names)) != len(names):
        raise ValueError(
            "Teacher object selection must contain unique names "
            f"from mode={mode.name}: names={names}, "
            f"unknown={unknown_names}"
        )

    usd_paths = tuple(
        str(
            mode.dataset_root
            / name
            / "teacher_object.usd"
        )
        for name in names
    )
    if mode.use_stable_states:
        stable_state_paths = tuple(
            str(mode.dataset_root / name / f"{name}.npy")
            for name in names
        )
    else:
        stable_state_paths = ()

    return TeacherObjectSelection(
        mode=mode,
        names=names,
        usd_paths=usd_paths,
        stable_state_paths=stable_state_paths,
    )


def configure_teacher_evaluation_objects(
    env_cfg: Any,
    selection: TeacherObjectSelection,
    repeats_per_object: int,
) -> tuple[str, ...]:
    env_cfg.asset.dataset_root = str(
        selection.mode.dataset_root
    )
    env_cfg.asset.object_names = selection.names
    env_cfg.asset.object_usd_paths = selection.usd_paths
    env_cfg.asset.weighted_object_indices = tuple(
        range(len(selection.names))
    )
    env_cfg.asset.stable_state_paths = (
        selection.stable_state_paths
    )
    env_cfg.object.spawn.usd_path = list(selection.usd_paths)
    env_cfg.scene.num_envs = (
        len(selection.names) * repeats_per_object
    )
    env_cfg.reset.non_uniform_sampling = False

    return tuple(
        selection.names[index % len(selection.names)]
        for index in range(env_cfg.scene.num_envs)
    )


def get_teacher_env_object_names(env: Any) -> tuple[str, ...]:
    return tuple(
        env.affordance_data.object_names[index]
        for index in (
            env.affordance_data.env_object_indices_cpu.tolist()
        )
    )


def assert_teacher_evaluation_contract(env: Any) -> None:
    if type(env).__module__ != (
        "Grasp.tasks.direct.grasp.teacher.env"
    ):
        raise RuntimeError(
            "Evaluation requires the standalone Teacher environment"
        )
    if env.obs_spec != TEACHER_OBSERVATION_SPEC:
        raise RuntimeError(
            "Teacher evaluation observation spec mismatch"
        )
    if env.cfg.action_space != TEACHER_ACTION_DIM:
        raise RuntimeError(
            "Teacher evaluation action dimension must be 13"
        )
    if env.arm_joint_ids.numel() != TEACHER_ARM_ACTION_DIM:
        raise RuntimeError(
            "Teacher evaluation requires seven FR3 joints"
        )
    if (
        env.active_hand_joint_ids.numel()
        != TEACHER_HAND_ACTION_DIM
    ):
        raise RuntimeError(
            "Teacher evaluation requires six active hand joints"
        )


def compute_damped_least_squares_delta(
    jacobian: torch.Tensor,
    target_twist: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    jacobian_transpose = jacobian.transpose(1, 2)
    identity = torch.eye(
        6,
        dtype=jacobian.dtype,
        device=jacobian.device,
    ).unsqueeze(0)
    solved_twist = torch.linalg.solve(
        jacobian @ jacobian_transpose
        + damping * damping * identity,
        target_twist.unsqueeze(-1),
    )
    return (
        jacobian_transpose @ solved_twist
    ).squeeze(-1)


def capture_closed_hand_target(env: Any) -> torch.Tensor:
    return env.current_joint_target[
        :, env.active_hand_joint_ids
    ].clone()


def build_multistep_lift_action(
    env: Any,
    closed_hand_target: torch.Tensor,
    lift_delta_z: float,
    damping: float,
) -> torch.Tensor:
    full_jacobians = env.robot.root_physx_view.get_jacobians()
    wrist_jacobian = full_jacobians[
        :, env.wrist_body_id - 1, 0:6, :
    ].index_select(-1, env.arm_joint_ids)

    target_twist = torch.zeros(
        (env.num_envs, 6),
        dtype=wrist_jacobian.dtype,
        device=env.device,
    )
    target_twist[:, 2] = lift_delta_z
    arm_joint_delta = compute_damped_least_squares_delta(
        jacobian=wrist_jacobian,
        target_twist=target_twist,
        damping=damping,
    )

    arm_action = (
        arm_joint_delta / env.cfg.action.arm_residual_scale
    )
    current_hand_qpos = env.robot.data.joint_pos[
        :, env.active_hand_joint_ids
    ]
    hand_action = (
        closed_hand_target - current_hand_qpos
    ) / env.cfg.action.hand_residual_scale
    return torch.cat((arm_action, hand_action), dim=-1)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(
        np.float32,
        copy=False,
    )


def run_teacher_evaluation_episode(
    env: Any,
    actor: Any,
    grasp_steps: int,
    lift_steps: int,
    lift_delta_z: float,
    damping: float,
) -> TeacherEvaluationEpisode:
    observation_dict, _ = env.reset()
    observation = _tensor_to_numpy(observation_dict["policy"])
    initial_object_z = env.object_initial_root_state[:, 2].clone()
    interrupted = torch.zeros(
        env.num_envs,
        dtype=torch.bool,
        device=env.device,
    )
    reward_sum = 0.0
    log_sums: dict[str, float] = {}

    with torch.inference_mode():
        for _ in range(grasp_steps):
            action = actor.noiseless_action(observation)
            (
                next_observation_dict,
                reward,
                terminated,
                truncated,
                extras,
            ) = env.step(action)
            interrupted |= terminated | truncated
            reward_sum += float(reward.mean().item())
            for name, value in extras["log"].items():
                log_sums[name] = (
                    log_sums.get(name, 0.0)
                    + float(value.detach().item())
                )
            observation = _tensor_to_numpy(
                next_observation_dict["policy"]
            )

        closed_hand_target = capture_closed_hand_target(env)
        for _ in range(lift_steps):
            lift_action = build_multistep_lift_action(
                env=env,
                closed_hand_target=closed_hand_target,
                lift_delta_z=lift_delta_z,
                damping=damping,
            )
            _, _, terminated, truncated, _ = env.step(lift_action)
            interrupted |= terminated | truncated

    lift_height = (
        env.object.data.root_pos_w[:, 2] - initial_object_z
    )
    success = (
        lift_height > env.cfg.reward.lift_success_height
    ) & ~interrupted
    return TeacherEvaluationEpisode(
        success=success,
        interrupted=interrupted,
        lift_height=lift_height,
        mean_grasp_reward=reward_sum / grasp_steps,
        mean_grasp_logs={
            name: value / grasp_steps
            for name, value in log_sums.items()
        },
    )


class TeacherEvaluationTotals:
    def __init__(self, object_names: tuple[str, ...]) -> None:
        unique_names = tuple(dict.fromkeys(object_names))
        self.attempts = 0
        self.successes = 0
        self.lift_height_sum = 0.0
        self.object_attempts = {
            name: 0 for name in unique_names
        }
        self.object_successes = {
            name: 0 for name in unique_names
        }
        self.object_lift_height_sum = {
            name: 0.0 for name in unique_names
        }

    def add_episode(
        self,
        env_object_names: tuple[str, ...],
        episode: TeacherEvaluationEpisode,
    ) -> None:
        success_values = episode.success.detach().cpu().tolist()
        lift_values = episode.lift_height.detach().cpu().tolist()
        for name, success, height in zip(
            env_object_names,
            success_values,
            lift_values,
            strict=True,
        ):
            success_value = int(success)
            height_value = float(height)
            self.attempts += 1
            self.successes += success_value
            self.lift_height_sum += height_value
            self.object_attempts[name] += 1
            self.object_successes[name] += success_value
            self.object_lift_height_sum[name] += height_value

    @property
    def failures(self) -> int:
        return self.attempts - self.successes

    def flat_metrics(self) -> dict[str, float]:
        metrics = {
            "attempts": float(self.attempts),
            "successes": float(self.successes),
            "failures": float(self.failures),
            "success_rate": self.successes / self.attempts,
            "failure_rate": self.failures / self.attempts,
            "mean_lift_height": (
                self.lift_height_sum / self.attempts
            ),
        }
        for name, attempts in self.object_attempts.items():
            successes = self.object_successes[name]
            failures = attempts - successes
            prefix = f"object/{name}"
            metrics[f"{prefix}/attempts"] = float(attempts)
            metrics[f"{prefix}/successes"] = float(successes)
            metrics[f"{prefix}/failures"] = float(failures)
            metrics[f"{prefix}/success_rate"] = (
                successes / attempts
            )
            metrics[f"{prefix}/failure_rate"] = (
                failures / attempts
            )
            metrics[f"{prefix}/mean_lift_height"] = (
                self.object_lift_height_sum[name] / attempts
            )
        return metrics


def print_teacher_evaluation_summary(
    mode_name: str,
    totals: TeacherEvaluationTotals,
) -> None:
    metrics = totals.flat_metrics()
    print(f"teacher_evaluation_mode={mode_name}")
    print(f"attempts={int(metrics['attempts'])}")
    print(f"successes={int(metrics['successes'])}")
    print(f"failures={int(metrics['failures'])}")
    print(f"success_rate={metrics['success_rate']:.6f}")
    print(f"failure_rate={metrics['failure_rate']:.6f}")
    print(
        "mean_lift_height="
        f"{metrics['mean_lift_height']:.6f}"
    )
    for name in totals.object_attempts:
        prefix = f"object/{name}"
        print(
            f"object={name} "
            f"attempts={int(metrics[f'{prefix}/attempts'])} "
            f"successes={int(metrics[f'{prefix}/successes'])} "
            f"failures={int(metrics[f'{prefix}/failures'])} "
            "success_rate="
            f"{metrics[f'{prefix}/success_rate']:.6f} "
            "failure_rate="
            f"{metrics[f'{prefix}/failure_rate']:.6f} "
            "mean_lift_height="
            f"{metrics[f'{prefix}/mean_lift_height']:.6f}"
        )


__all__ = [
    "QUANTITATIVE_MODE",
    "SHAPENET_30_OBJECT_NAMES",
    "SHAPENET_30_ROOT",
    "TEACHER_EVALUATION_MODES",
    "TeacherEvaluationEpisode",
    "TeacherEvaluationMode",
    "TeacherEvaluationTotals",
    "TeacherObjectSelection",
    "VISUAL_MODE",
    "assert_teacher_evaluation_contract",
    "build_multistep_lift_action",
    "capture_closed_hand_target",
    "compute_damped_least_squares_delta",
    "configure_teacher_evaluation_objects",
    "get_teacher_env_object_names",
    "print_teacher_evaluation_summary",
    "resolve_teacher_object_selection",
    "run_teacher_evaluation_episode",
]
