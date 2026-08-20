from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Play and evaluate a RobustDexGrasp-style multi-step Student "
        "checkpoint in Grasp IsaacLab."
    )
)
parser.add_argument(
    "--task",
    type=str,
    default="Grasp-Student-Direct-v0",
)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--lift_delta_z", type=float, default=0.002)
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
from Grasp.algo import ppo_student
from Grasp.tasks.direct.grasp.teacher_multistep_eval import (
    build_multistep_lift_action,
)
from Grasp.algo.checkpoint_contract import (
    STUDENT_PLAY_KEYS,
    validate_student_checkpoint,
)


METRIC_NAMES = (
    "affordance_reward",
    "table_reward",
    "arm_height_reward",
    "arm_collision_reward",
    "lift_success",
    "teacher_reward",
)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def assert_multistep_environment(env) -> None:
    modules = tuple(cls.__module__ for cls in type(env).__mro__)
    cfg_modules = tuple(cls.__module__ for cls in type(env.cfg).__mro__)
    forbidden = tuple(
        module
        for module in modules + cfg_modules
        if module.endswith(".grasp_env_onestep")
        or module.endswith(".grasp_env_cfg_onestep")
    )
    if forbidden:
        raise RuntimeError(
            "Student playback still depends on the forbidden OneStep "
            f"path: {forbidden}"
        )


def build_student_actor(env, agent_cfg, student_actor_obs_dim, seed):
    return ppo_student.Actor(
        ppo_student.MLP(
            agent_cfg.policy_net,
            nn.LeakyReLU,
            student_actor_obs_dim,
            env.cfg.action_space,
        ),
        ppo_student.MultivariateGaussianDiagonalCovariance(
            env.cfg.action_space,
            env.num_envs,
            agent_cfg.init_std,
            ppo_student.TorchNormalSampler(env.cfg.action_space),
            seed=seed,
        ),
        env.device,
    )


def load_student_checkpoint(
    path,
    actor_student,
    prop_latent_encoder,
    env,
    student_actor_obs_dim,
):
    checkpoint = torch.load(
        path,
        map_location=env.device,
        weights_only=True,
    )

    validate_student_checkpoint(
        checkpoint,
        required_keys=STUDENT_PLAY_KEYS,
        expected_obs_spec=env.obs_spec,
        expected_student_actor_obs_dim=student_actor_obs_dim,
        expected_action_dim=env.cfg.action_space,
        role="Student play",
        path=path,
    )

    actor_student.architecture.load_state_dict(
        checkpoint["actor_architecture_state_dict"],
        strict=True,
    )
    actor_student.distribution.load_state_dict(
        checkpoint["actor_distribution_state_dict"],
        strict=True,
    )
    prop_latent_encoder.load_state_dict(
        checkpoint["prop_latent_encoder_state_dict"],
        strict=True,
    )

    actor_student.update()
    actor_student.architecture.architecture.eval()
    prop_latent_encoder.eval()

    print(f"loaded Student playback checkpoint: {path}")

def add_metrics(totals, counts, extras):
    for name in METRIC_NAMES:
        value = extras[name]
        totals[name] += float(value.sum().item())
        counts[name] += value.numel()


def main():
    seed = configure_seed(args_cli.seed)
    env_cfg = load_cfg_from_registry(
        args_cli.task,
        "env_cfg_entry_point",
    )
    agent_cfg = load_cfg_from_registry(
        args_cli.task,
        "ppo_student_cfg_entry_point",
    )

    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    evaluation_steps = agent_cfg.grasp_steps + agent_cfg.lift_steps
    env_cfg.episode_length_s = (
        (evaluation_steps + 2)
        * env_cfg.sim.dt
        * env_cfg.decimation
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    assert_multistep_environment(env)

    if env.cfg.control.arm_controller != "pose":
        raise ValueError("Student lift requires pose arm controller")
    if env.obs_spec.student_total_dim != 517:
        raise ValueError("Student total observation must be 517")
    if env.obs_spec.history_len != agent_cfg.history_len:
        raise ValueError("Student history length config mismatch")
    if env.obs_spec.prop_latent_dim != agent_cfg.prop_latent_dim:
        raise ValueError("Student latent config mismatch")
    if env.obs_spec.aff_vec_dim != 18:
        raise ValueError("Student affordance dimension must be 18")
    if env.cfg.action_space != 13:
        raise ValueError("Student action dimension must be 13")
    if agent_cfg.grasp_steps <= 0 or agent_cfg.lift_steps <= 0:
        raise ValueError("Student grasp/lift steps must be positive")

    student_actor_obs_dim = (
        ppo_student.compute_student_actor_obs_dim(
            env.obs_spec.history_dim,
            agent_cfg.prop_latent_dim,
            env.obs_spec.aff_vec_dim,
        )
    )
    actor_student = build_student_actor(
        env,
        agent_cfg,
        student_actor_obs_dim,
        seed,
    )
    prop_latent_encoder = ppo_student.LSTM_StateHistoryEncoder(
        env.obs_spec.history_dim,
        agent_cfg.prop_latent_dim,
        agent_cfg.history_len,
        env.device,
    ).to(env.device)
    load_student_checkpoint(
        args_cli.checkpoint,
        actor_student,
        prop_latent_encoder,
        env,
        student_actor_obs_dim,
    )

    metric_totals = {name: 0.0 for name in METRIC_NAMES}
    metric_counts = {name: 0 for name in METRIC_NAMES}
    success_count = 0
    trial_count = 0
    lift_height_sum = 0.0

    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            obs_dict, _ = env.reset()
            total_obs = obs_dict["policy"]
            object_init_z = env.object_init_states[:, 2].clone()
            last_grasp_action = None

            for _ in range(agent_cfg.grasp_steps):
                student_actor_obs = (
                    ppo_student.build_student_actor_observation(
                        total_obs=total_obs,
                        prop_latent_encoder=prop_latent_encoder,
                        history_len=env.obs_spec.history_len,
                        history_dim=env.obs_spec.history_dim,
                        student_affordance=env.last_affordance_vec,
                    )
                )
                action = actor_student.noiseless_action(
                    tensor_to_numpy(student_actor_obs)
                )
                next_obs_dict, _, _, _, extras = env.step(action)
                add_metrics(metric_totals, metric_counts, extras)
                total_obs = next_obs_dict["policy"]
                last_grasp_action = action

            for _ in range(agent_cfg.lift_steps):
                current_eef_pose_w = env.robot.data.body_state_w[
                    :, env.eef_body_ids[0], 0:7
                ]
                lift_action = build_multistep_lift_action(
                    current_eef_pose_w=current_eef_pose_w,
                    last_grasp_action=last_grasp_action,
                    lift_delta_z=args_cli.lift_delta_z,
                )
                _, _, _, _, extras = env.step(lift_action)
                add_metrics(metric_totals, metric_counts, extras)

            lift_height = (
                env.object.data.root_state_w[:, 2]
                - object_init_z
            )
            lifted = lift_height > 0.10
            success_count += int(lifted.sum().item())
            trial_count += env.num_envs
            lift_height_sum += float(lift_height.sum().item())
            print(
                f"episode={episode} "
                f"success_rate={lifted.float().mean().item():.6f} "
                f"mean_lift_height={lift_height.mean().item():.6f}"
            )

    print("student multi-step evaluation summary")
    print(f"episodes={args_cli.episodes}")
    print(f"trials={trial_count}")
    print(f"success_rate={success_count / trial_count:.6f}")
    print(f"mean_lift_height={lift_height_sum / trial_count:.6f}")
    for name in METRIC_NAMES:
        print(
            f"mean_{name}="
            f"{metric_totals[name] / metric_counts[name]:.6f}"
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()