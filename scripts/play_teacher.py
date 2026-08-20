"""Play and evaluate the standalone multi-step Teacher policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Play and evaluate a Grasp multi-step Teacher checkpoint."
    )
)
parser.add_argument(
    "--task",
    type=str,
    default="Grasp-Teacher-Direct-v0",
)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--mode",
    type=str,
    choices=("visual", "quantitative"),
    default="visual",
)
parser.add_argument("--objects", type=str, default="all")
parser.add_argument("--repeats_per_object", type=int, default=1)
parser.add_argument("--episodes", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lift_delta_z", type=float, default=0.005)
parser.add_argument("--dls_damping", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.seed import configure_seed

import Grasp.tasks  # noqa: F401
from Grasp.algo import ppo_teacher
from Grasp.algo.checkpoint_contract import (
    TEACHER_PLAY_KEYS,
    validate_teacher_checkpoint,
)
from Grasp.tasks.direct.grasp.teacher.teacher_multistep_eval import (
    TEACHER_EVALUATION_MODES,
    TeacherEvaluationTotals,
    assert_teacher_evaluation_contract,
    configure_teacher_evaluation_objects,
    print_teacher_evaluation_summary,
    resolve_teacher_object_selection,
    run_teacher_evaluation_episode,
)



GRASP_TEACHER_CHECKPOINT_VERSION = 1
GRASP_TEACHER_TASK_ID = "Grasp-Teacher-Direct-v0"
GRASP_TEACHER_ALGORITHM = "ppo_teacher"
GRASP_TEACHER_PLAY_KEYS = (
    *TEACHER_PLAY_KEYS,
    "checkpoint_version",
    "task_id",
    "algorithm_name",
    "grasp_steps",
)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(
        np.float32,
        copy=False,
    )


def serialize_teacher_observation_spec(env) -> dict[str, int]:
    observation_spec = asdict(env.obs_spec)
    observation_spec["base_dim"] = env.obs_spec.base_dim
    observation_spec["teacher_dim"] = env.obs_spec.teacher_dim
    return observation_spec


def validate_arguments() -> None:
    if args_cli.task != GRASP_TEACHER_TASK_ID:
        raise ValueError(
            "Teacher playback task must be "
            f"{GRASP_TEACHER_TASK_ID}, got {args_cli.task}"
        )
    if args_cli.repeats_per_object <= 0:
        raise ValueError(
            "repeats_per_object must be positive, got "
            f"{args_cli.repeats_per_object}"
        )
    if args_cli.episodes is not None and args_cli.episodes <= 0:
        raise ValueError(
            f"episodes must be positive, got {args_cli.episodes}"
        )
    if args_cli.lift_delta_z <= 0.0:
        raise ValueError(
            "lift_delta_z must be positive, got "
            f"{args_cli.lift_delta_z}"
        )
    if args_cli.dls_damping <= 0.0:
        raise ValueError(
            "dls_damping must be positive, got "
            f"{args_cli.dls_damping}"
        )



def build_actor(env, agent_cfg, seed: int):
    if agent_cfg.activation != "lrelu":
        raise RuntimeError(
            "Teacher activation must remain lrelu, got "
            f"{agent_cfg.activation}"
        )

    return ppo_teacher.Actor(
        ppo_teacher.MLP(
            agent_cfg.policy_net,
            nn.LeakyReLU,
            env.obs_spec.teacher_dim,
            env.cfg.action_space,
        ),
        ppo_teacher.MultivariateGaussianDiagonalCovariance(
            env.cfg.action_space,
            env.num_envs,
            agent_cfg.init_std,
            ppo_teacher.TorchNormalSampler(env.cfg.action_space),
            seed=seed,
        ),
        env.device,
    )


def load_actor_checkpoint(
    path: str,
    actor,
    env,
    task_id: str,
    algorithm_name: str,
    grasp_steps: int,
) -> None:
    checkpoint = torch.load(
        path,
        map_location=env.device,
        weights_only=True,
    )
    validate_teacher_checkpoint(
        checkpoint,
        required_keys=GRASP_TEACHER_PLAY_KEYS,
        expected_teacher_dim=env.obs_spec.teacher_dim,
        expected_action_dim=env.cfg.action_space,
        path=path,
    )

    if (
        type(checkpoint["checkpoint_version"]) is not int
        or checkpoint["checkpoint_version"]
        != GRASP_TEACHER_CHECKPOINT_VERSION
    ):
        raise RuntimeError(
            "Teacher checkpoint version mismatch: "
            f"checkpoint={checkpoint['checkpoint_version']}, "
            f"required={GRASP_TEACHER_CHECKPOINT_VERSION}, "
            f"path={path}"
        )
    if (
        type(checkpoint["task_id"]) is not str
        or checkpoint["task_id"] != task_id
    ):
        raise RuntimeError(
            "Teacher checkpoint task mismatch: "
            f"checkpoint={checkpoint['task_id']}, "
            f"required={task_id}, path={path}"
        )
    if (
        type(checkpoint["algorithm_name"]) is not str
        or checkpoint["algorithm_name"] != algorithm_name
    ):
        raise RuntimeError(
            "Teacher checkpoint algorithm mismatch: "
            f"checkpoint={checkpoint['algorithm_name']}, "
            f"required={algorithm_name}, path={path}"
        )
    if (
        type(checkpoint["grasp_steps"]) is not int
        or checkpoint["grasp_steps"] != grasp_steps
    ):
        raise RuntimeError(
            "Teacher checkpoint rollout length mismatch: "
            f"checkpoint={checkpoint['grasp_steps']}, "
            f"required={grasp_steps}, path={path}"
        )

    expected_spec = serialize_teacher_observation_spec(env)
    checkpoint_spec = checkpoint["obs_spec"]
    if set(checkpoint_spec) != set(expected_spec):
        raise RuntimeError(
            "Teacher checkpoint observation spec fields differ "
            f"from the environment: path={path}"
        )
    for field, expected_value in expected_spec.items():
        checkpoint_value = checkpoint_spec[field]
        if (
            type(checkpoint_value) is not type(expected_value)
            or checkpoint_value != expected_value
        ):
            raise RuntimeError(
                "Teacher checkpoint observation spec mismatch: "
                f"field={field}, checkpoint={checkpoint_value}, "
                f"required={expected_value}, path={path}"
            )

    actor.architecture.load_state_dict(
        checkpoint["actor_architecture_state_dict"],
        strict=True,
    )
    actor.distribution.load_state_dict(
        checkpoint["actor_distribution_state_dict"],
        strict=True,
    )
    actor.update()
    actor.architecture.architecture.eval()
    print(f"loaded Teacher playback checkpoint: {path}")

def main() -> None:
    validate_arguments()
    seed = configure_seed(args_cli.seed)
    env_cfg = load_cfg_from_registry(
        args_cli.task,
        "env_cfg_entry_point",
    )
    agent_cfg = load_cfg_from_registry(
        args_cli.task,
        "ppo_teacher_cfg_entry_point",
    )
    if agent_cfg.name != GRASP_TEACHER_ALGORITHM:
        raise RuntimeError(
            "Teacher algorithm name must be "
            f"{GRASP_TEACHER_ALGORITHM}, got {agent_cfg.name}"
        )

    selection = resolve_teacher_object_selection(
        mode_name=args_cli.mode,
        value=args_cli.objects,
    )
    env_object_names = configure_teacher_evaluation_objects(
        env_cfg=env_cfg,
        selection=selection,
        repeats_per_object=args_cli.repeats_per_object,
    )
    mode = TEACHER_EVALUATION_MODES[args_cli.mode]
    episode_count = (
        mode.default_episodes
        if args_cli.episodes is None
        else args_cli.episodes
    )

    env_cfg.seed = seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = (
        (mode.grasp_steps + mode.lift_steps + 2)
        * env_cfg.sim.dt
        * env_cfg.decimation
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        assert_teacher_evaluation_contract(env)
        actor = build_actor(env, agent_cfg, seed)
        load_actor_checkpoint(
            path=args_cli.checkpoint,
            actor=actor,
            env=env,
            task_id=args_cli.task,
            algorithm_name=agent_cfg.name,
            grasp_steps=agent_cfg.grasp_steps,
        )

        totals = TeacherEvaluationTotals(env_object_names)
        for episode_index in range(1, episode_count + 1):
            episode = run_teacher_evaluation_episode(
                env=env,
                actor=actor,
                grasp_steps=mode.grasp_steps,
                lift_steps=mode.lift_steps,
                lift_delta_z=args_cli.lift_delta_z,
                damping=args_cli.dls_damping,
            )
            totals.add_episode(env_object_names, episode)
            attempts = episode.success.numel()
            successes = int(episode.success.sum().item())
            failures = attempts - successes
            print(
                f"episode={episode_index} "
                f"attempts={attempts} "
                f"successes={successes} "
                f"failures={failures} "
                f"success_rate={successes / attempts:.6f} "
                "failure_rate="
                f"{failures / attempts:.6f} "
                "mean_lift_height="
                f"{episode.lift_height.mean().item():.6f}"
            )

        print_teacher_evaluation_summary(
            mode_name=mode.name,
            totals=totals,
        )
    finally:
        env.close()



if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()