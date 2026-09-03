
"""Train the standalone RobustDexGrasp-style Teacher in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Train the standalone multi-step Teacher PPO."
)
parser.add_argument(
    "--task",
    type=str,
    default="Grasp-Teacher-Direct-v0",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--torch_deterministic",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--biased",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--enable_evaluation",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--evaluation_interval",
    type=int,
    default=None,
)
parser.add_argument(
    "--evaluation_rollouts",
    type=int,
    default=None,
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

import Grasp.tasks  # noqa: F401
from Grasp.algo import ppo_teacher
from Grasp.algo.checkpoint_contract import (
    TEACHER_RESUME_KEYS,
    get_resume_start_update,
    validate_teacher_checkpoint,
)
from Grasp.tasks.direct.grasp.teacher.teacher_multistep_eval import (
    TeacherEvaluationTotals,
    get_teacher_env_object_names,
    run_teacher_evaluation_episode,
)


GRASP_TEACHER_CHECKPOINT_VERSION = 1
GRASP_TEACHER_TASK_ID = "Grasp-Teacher-Direct-v0"
GRASP_TEACHER_ALGORITHM = "ppo_teacher"
GRASP_TEACHER_OBSERVATION_DIM = 119
GRASP_TEACHER_ACTION_DIM = 13
GRASP_TEACHER_GRASP_STEPS = 70
GRASP_TEACHER_REWARD_CLIP = -2.0

GRASP_TEACHER_RESUME_KEYS = (
    *TEACHER_RESUME_KEYS,
    "checkpoint_version",
    "task_id",
    "algorithm_name",
    "grasp_steps",
    "reward_clip",
    "ppo_learning_rate",
    "ppo_tot_timesteps",
)


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def serialize_teacher_observation_spec(env) -> dict[str, int]:
    observation_spec = asdict(env.obs_spec)
    observation_spec["base_dim"] = env.obs_spec.base_dim
    observation_spec["teacher_dim"] = env.obs_spec.teacher_dim
    return observation_spec


def resolve_resume_path(checkpoint_argument: str) -> str | None:
    if checkpoint_argument == "":
        return None
    return os.path.abspath(
        os.path.expanduser(checkpoint_argument)
    )


def build_actor_critic(env, agent_cfg, seed: int):
    if agent_cfg.activation != "lrelu":
        raise RuntimeError(
            "Teacher activation must remain lrelu, got "
            f"{agent_cfg.activation}"
        )

    obs_dim = env.obs_spec.teacher_dim
    action_dim = env.cfg.action_space
    actor = ppo_teacher.Actor(
        ppo_teacher.MLP(
            agent_cfg.policy_net,
            nn.LeakyReLU,
            obs_dim,
            action_dim,
        ),
        ppo_teacher.MultivariateGaussianDiagonalCovariance(
            action_dim,
            agent_cfg.init_std,
            ppo_teacher.TorchNormalSampler(action_dim),
            seed=seed,
        ),
        env.device,
    )
    critic = ppo_teacher.Critic(
        ppo_teacher.MLP(
            agent_cfg.value_net,
            nn.LeakyReLU,
            obs_dim,
            1,
        ),
        env.device,
    )
    return actor, critic


def build_ppo(env, actor, critic, agent_cfg, log_dir: str):
    return ppo_teacher.PPO(
        actor=actor,
        critic=critic,
        num_envs=env.num_envs,
        num_transitions_per_env=agent_cfg.grasp_steps,
        num_learning_epochs=agent_cfg.num_learning_epochs,
        num_mini_batches=agent_cfg.num_mini_batches,
        clip_param=agent_cfg.clip_param,
        gamma=agent_cfg.gamma,
        lam=agent_cfg.lam,
        learning_rate=agent_cfg.learning_rate,
        desired_kl=agent_cfg.desired_kl,
        log_dir=log_dir,
        device=env.device,
        shuffle_batch=agent_cfg.shuffle_batch,
    )


def save_checkpoint(
    path: str,
    actor,
    critic,
    ppo,
    env,
    update: int,
    task_id: str,
    algorithm_name: str,
    grasp_steps: int,
    reward_clip: float,
) -> None:
    torch.save(
        {
            "checkpoint_version": (
                GRASP_TEACHER_CHECKPOINT_VERSION
            ),
            "task_id": task_id,
            "algorithm_name": algorithm_name,
            "actor_architecture_state_dict": (
                actor.architecture.state_dict()
            ),
            "actor_distribution_state_dict": (
                actor.distribution.state_dict()
            ),
            "critic_architecture_state_dict": (
                critic.architecture.state_dict()
            ),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "obs_spec": serialize_teacher_observation_spec(env),
            "action_dim": env.cfg.action_space,
            "grasp_steps": grasp_steps,
            "reward_clip": reward_clip,
            "ppo_learning_rate": float(ppo.learning_rate),
            "ppo_tot_timesteps": int(ppo.tot_timesteps),
            "update": update,
        },
        path,
    )
    print(f"saved Teacher checkpoint: {path}")


def load_checkpoint(
    path: str,
    actor,
    critic,
    ppo,
    env,
    max_iterations: int,
    task_id: str,
    algorithm_name: str,
    grasp_steps: int,
    reward_clip: float,
) -> int:
    checkpoint = torch.load(
        path,
        map_location=env.device,
        weights_only=True,
    )

    validate_teacher_checkpoint(
        checkpoint,
        required_keys=GRASP_TEACHER_RESUME_KEYS,
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

    expected_observation_spec = (
        serialize_teacher_observation_spec(env)
    )
    checkpoint_observation_spec = checkpoint["obs_spec"]
    if set(checkpoint_observation_spec) != set(
        expected_observation_spec
    ):
        raise RuntimeError(
            "Teacher checkpoint observation spec fields differ "
            f"from the current environment: path={path}"
        )
    for field, expected_value in (
        expected_observation_spec.items()
    ):
        checkpoint_value = checkpoint_observation_spec[field]
        if (
            type(checkpoint_value) is not type(expected_value)
            or checkpoint_value != expected_value
        ):
            raise RuntimeError(
                "Teacher checkpoint observation spec mismatch: "
                f"field={field}, checkpoint={checkpoint_value}, "
                f"required={expected_value}, path={path}"
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
    if (
        type(checkpoint["reward_clip"]) is not float
        or checkpoint["reward_clip"] != reward_clip
    ):
        raise RuntimeError(
            "Teacher checkpoint reward lower bound mismatch: "
            f"checkpoint={checkpoint['reward_clip']}, "
            f"required={reward_clip}, path={path}"
        )

    ppo_learning_rate = checkpoint["ppo_learning_rate"]
    if type(ppo_learning_rate) is not float:
        raise RuntimeError(
            "Teacher checkpoint ppo_learning_rate must be float: "
            f"path={path}"
        )
    ppo_tot_timesteps = checkpoint["ppo_tot_timesteps"]
    if type(ppo_tot_timesteps) is not int:
        raise RuntimeError(
            "Teacher checkpoint ppo_tot_timesteps must be int: "
            f"path={path}"
        )

    actor.architecture.load_state_dict(
        checkpoint["actor_architecture_state_dict"],
        strict=True,
    )
    actor.distribution.load_state_dict(
        checkpoint["actor_distribution_state_dict"],
        strict=True,
    )
    critic.architecture.load_state_dict(
        checkpoint["critic_architecture_state_dict"],
        strict=True,
    )
    ppo.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    ppo.learning_rate = ppo_learning_rate
    ppo.tot_timesteps = ppo_tot_timesteps

    optimizer_learning_rates = {
        float(group["lr"])
        for group in ppo.optimizer.param_groups
    }
    if optimizer_learning_rates != {ppo_learning_rate}:
        raise RuntimeError(
            "Teacher optimizer learning rate differs from PPO state: "
            f"optimizer={optimizer_learning_rates}, "
            f"ppo={ppo_learning_rate}, path={path}"
        )

    start_update = get_resume_start_update(
        checkpoint,
        role="Teacher resume",
        path=path,
    )
    if start_update >= max_iterations:
        raise RuntimeError(
            "Teacher checkpoint update is outside the configured "
            "training range: "
            f"start_update={start_update}, "
            f"max_iterations={max_iterations}"
        )
    print(
        f"loaded Teacher checkpoint: {path}, "
        f"loaded_update={start_update - 1}, "
        f"next_update={start_update}"
    )
    return start_update


def create_log_dir(agent_cfg) -> str:
    log_root_name = os.path.basename(
        os.path.normpath(agent_cfg.log_dir)
    )
    if log_root_name != "runs_teacher":
        raise RuntimeError(
            "Teacher log root must end with runs_teacher, got "
            f"{agent_cfg.log_dir}"
        )

    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.run_name
    if run_name is None:
        run_name = f"{args_cli.task}_{time_string}"
    if os.path.basename(run_name) != run_name:
        raise RuntimeError(
            "Teacher run_name must be one directory name, got "
            f"{run_name}"
        )

    log_dir = os.path.abspath(
        os.path.join(agent_cfg.log_dir, run_name)
    )
    os.makedirs(log_dir, exist_ok=False)
    return log_dir


def write_run_config(
    log_dir: str,
    env_cfg,
    agent_cfg,
    seed: int,
    resume_path: str | None,
) -> None:
    dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "agent.yaml"), agent_cfg)
    with open(
        os.path.join(log_dir, "run.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "task": args_cli.task,
                "algorithm": agent_cfg.name,
                "seed": seed,
                "resume_checkpoint": resume_path,
                "num_envs": env_cfg.scene.num_envs,
                "max_iterations": agent_cfg.max_iterations,
                "grasp_steps": agent_cfg.grasp_steps,
                "reward_clip": agent_cfg.reward_clip,
                "evaluation_enabled": (
                    agent_cfg.evaluation_enabled
                ),
                "evaluation_interval": (
                    agent_cfg.evaluation_interval
                ),
                "evaluation_rollouts": (
                    agent_cfg.evaluation_rollouts
                ),
            },
            file,
            indent=2,
        )


def validate_training_contract(
    env,
    obs_dict: dict[str, torch.Tensor],
    agent_cfg,
) -> None:
    if env.obs_spec.teacher_dim != GRASP_TEACHER_OBSERVATION_DIM:
        raise RuntimeError(
            "Teacher observation spec must be 119, got "
            f"{env.obs_spec.teacher_dim}"
        )
    if env.cfg.observation_space != GRASP_TEACHER_OBSERVATION_DIM:
        raise RuntimeError(
            "Teacher environment observation space must be 119, got "
            f"{env.cfg.observation_space}"
        )
    expected_observation_shape = (
        env.num_envs,
        GRASP_TEACHER_OBSERVATION_DIM,
    )
    if tuple(obs_dict["policy"].shape) != expected_observation_shape:
        raise RuntimeError(
            "Teacher policy observation shape mismatch: "
            f"actual={tuple(obs_dict['policy'].shape)}, "
            f"required={expected_observation_shape}"
        )
    if env.cfg.action_space != GRASP_TEACHER_ACTION_DIM:
        raise RuntimeError(
            "Teacher action dimension must be 13, got "
            f"{env.cfg.action_space}"
        )
    if agent_cfg.grasp_steps != GRASP_TEACHER_GRASP_STEPS:
        raise RuntimeError(
            "Teacher rollout must contain 70 steps, got "
            f"{agent_cfg.grasp_steps}"
        )
    if env.max_episode_length != agent_cfg.grasp_steps:
        raise RuntimeError(
            "Teacher episode and rollout lengths differ: "
            f"episode={env.max_episode_length}, "
            f"rollout={agent_cfg.grasp_steps}"
        )
    if agent_cfg.reward_clip != GRASP_TEACHER_REWARD_CLIP:
        raise RuntimeError(
            "Teacher reward lower bound must be -2.0, got "
            f"{agent_cfg.reward_clip}"
        )
    if env.cfg.reward.min_reward != agent_cfg.reward_clip:
        raise RuntimeError(
            "Teacher environment and PPO reward lower bounds differ: "
            f"environment={env.cfg.reward.min_reward}, "
            f"ppo={agent_cfg.reward_clip}"
        )
    if agent_cfg.shuffle_batch:
        raise RuntimeError(
            "Teacher PPO must use ordered mini-batches"
        )
    if agent_cfg.max_iterations <= 0:
        raise RuntimeError(
            "Teacher max_iterations must be positive"
        )
    if agent_cfg.save_interval <= 0:
        raise RuntimeError(
            "Teacher save_interval must be positive"
        )
    if agent_cfg.evaluation_interval <= 0:
        raise RuntimeError(
            "Teacher evaluation_interval must be positive"
        )
    if agent_cfg.evaluation_rollouts <= 0:
        raise RuntimeError(
            "Teacher evaluation_rollouts must be positive"
        )


def run_rollout(
    env,
    ppo,
    grasp_steps: int,
    collect_for_update: bool,
) -> tuple[
    torch.Tensor,
    float,
    dict[str, float],
    float,
    float,
]:
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    rollout_reward = torch.zeros((), device=env.device)
    log_sums: dict[str, torch.Tensor] = {}
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(
        env.num_envs,
        dtype=torch.int32,
        device=env.device,
    )
    completed_reward_sum = torch.zeros((), device=env.device)
    completed_length_sum = torch.zeros((), device=env.device)
    completed_episode_count = torch.zeros((), device=env.device)

    for _ in range(grasp_steps):
        action = ppo.act(obs)
        next_obs_dict, reward, terminated, truncated, extras = env.step(
            action
        )

        episode_rewards += reward
        episode_lengths += 1
        episode_done = terminated | truncated
        completed_reward_sum += torch.where(
            episode_done,
            episode_rewards,
            0.0,
        ).sum()
        completed_length_sum += torch.where(
            episode_done,
            episode_lengths,
            0,
        ).sum()
        completed_episode_count += episode_done.sum()
        episode_rewards.masked_fill_(episode_done, 0.0)
        episode_lengths.masked_fill_(episode_done, 0)

        if collect_for_update:
            ppo.step(
                value_obs=obs,
                rews=reward,
                dones=terminated,
            )

        for name, value in extras["log"].items():
            if name not in log_sums:
                log_sums[name] = torch.zeros((), device=env.device)
            log_sums[name] += value.detach()

        obs = next_obs_dict["policy"]
        rollout_reward += reward.mean()

    remaining = episode_lengths > 0
    completed_reward_sum += torch.where(
        remaining,
        episode_rewards,
        0.0,
    ).sum()
    completed_length_sum += torch.where(
        remaining,
        episode_lengths,
        0,
    ).sum()
    completed_episode_count += remaining.sum()

    mean_logs = {
        name: (value / grasp_steps).item()
        for name, value in log_sums.items()
    }
    return (
        obs,
        (rollout_reward / grasp_steps).item(),
        mean_logs,
        (completed_reward_sum / completed_episode_count).item(),
        (completed_length_sum / completed_episode_count).item(),
    )


def run_periodic_evaluation(
    env,
    ppo,
    agent_cfg,
) -> dict[str, float]:
    storage_step_before = ppo.storage.step
    env_object_names = get_teacher_env_object_names(env)
    totals = TeacherEvaluationTotals(env_object_names)
    grasp_reward_sum = 0.0
    grasp_log_sums: dict[str, float] = {}

    training_non_uniform_sampling = (
        env.cfg.reset.non_uniform_sampling
    )
    env.cfg.reset.non_uniform_sampling = False
    try:
        for _ in range(agent_cfg.evaluation_rollouts):
            episode = run_teacher_evaluation_episode(
                env=env,
                actor=ppo.actor,
                grasp_steps=agent_cfg.grasp_steps,
                lift_steps=agent_cfg.lift_steps,
                lift_delta_z=0.005,
                damping=0.05,
            )
            totals.add_episode(env_object_names, episode)
            grasp_reward_sum += episode.mean_grasp_reward
            for name, value in episode.mean_grasp_logs.items():
                grasp_log_sums[name] = (
                    grasp_log_sums.get(name, 0.0) + value
                )
    finally:
        env.cfg.reset.non_uniform_sampling = (
            training_non_uniform_sampling
        )

    if ppo.storage.step != storage_step_before:
        raise RuntimeError(
            "Periodic evaluation changed Teacher PPO storage: "
            f"before={storage_step_before}, "
            f"after={ppo.storage.step}"
        )

    rollout_count = agent_cfg.evaluation_rollouts
    metrics = totals.flat_metrics()
    metrics["mean_grasp_reward"] = (
        grasp_reward_sum / rollout_count
    )
    for name, value in grasp_log_sums.items():
        metrics[f"grasp/{name}"] = value / rollout_count
    return metrics


def write_training_logs(
    ppo,
    update: int,
    mean_reward: float,
    mean_logs: dict[str, float],
) -> None:
    ppo.writer.add_scalar(
        "Teacher/rollout_mean_reward",
        mean_reward,
        update,
    )
    for name, value in mean_logs.items():
        ppo.writer.add_scalar(
            f"Teacher/{name}",
            value,
            update,
        )


def write_evaluation_logs(
    ppo,
    update: int,
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        ppo.writer.add_scalar(
            f"Evaluation/{name}",
            value,
            update,
        )


def print_teacher_iteration(
    update: int,
    max_iterations: int,
    steps_per_iteration: int,
    collection_time: float,
    learning_time: float,
    mean_std: float,
    mean_value_loss: float,
    mean_surrogate_loss: float,
    mean_entropy: float,
    mean_episode_reward: float,
    mean_episode_length: float,
    total_timesteps: int,
    learning_rate: float,
    elapsed_time: float,
    eta: float,
) -> None:
    width = 80
    iteration_time = collection_time + learning_time
    steps_per_second = int(steps_per_iteration / iteration_time)
    title = (
        f"Learning iteration {update + 1}/{max_iterations}"
    )

    print("#" * width)
    print(f"{title:^{width}}")
    print()
    print(
        f"{'Computation:':>30} {steps_per_second} steps/s "
        f"(collection: {collection_time:.3f}s, "
        f"learning: {learning_time:.3f}s)"
    )
    print(f"{'Mean action noise std:':>30} {mean_std:.4f}")
    print(
        f"{'Mean value function loss:':>30} "
        f"{mean_value_loss:.4f}"
    )
    print(
        f"{'Mean surrogate loss:':>30} "
        f"{mean_surrogate_loss:.4f}"
    )
    print(f"{'Mean entropy loss:':>30} {mean_entropy:.4f}")
    print(
        f"{'Mean reward:':>30} {mean_episode_reward:.4f}"
    )
    print(
        f"{'Mean episode length:':>30} "
        f"{mean_episode_length:.2f}"
    )
    print("-" * width)
    print(f"{'Total timesteps:':>30} {total_timesteps}")
    print(f"{'Learning rate:':>30} {learning_rate:.8f}")
    print(f"{'Iteration time:':>30} {iteration_time:.2f}s")
    print(
        f"{'Time elapsed:':>30} "
        f"{format_duration(elapsed_time)}"
    )
    print(f"{'ETA:':>30} {format_duration(eta)}")
    print()


def main() -> None:
    seed = configure_seed(
        args_cli.seed,
        torch_deterministic=args_cli.torch_deterministic,
    )
    env_cfg = load_cfg_from_registry(
        args_cli.task,
        "env_cfg_entry_point",
    )
    agent_cfg = load_cfg_from_registry(
        args_cli.task,
        "ppo_teacher_cfg_entry_point",
    )

    if args_cli.task != GRASP_TEACHER_TASK_ID:
        raise RuntimeError(
            "Teacher training only accepts "
            "Grasp-Teacher-Direct-v0, got "
            f"{args_cli.task}"
        )
    if agent_cfg.name != GRASP_TEACHER_ALGORITHM:
        raise RuntimeError(
            "Teacher algorithm must be ppo_teacher, got "
            f"{agent_cfg.name}"
        )

    env_cfg.seed = seed
    env_cfg.reset.biased = args_cli.biased
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.enable_evaluation:
        agent_cfg.evaluation_enabled = True
    if args_cli.evaluation_interval is not None:
        agent_cfg.evaluation_interval = (
            args_cli.evaluation_interval
        )
    if args_cli.evaluation_rollouts is not None:
        agent_cfg.evaluation_rollouts = (
            args_cli.evaluation_rollouts
        )

    resume_path = resolve_resume_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs_dict, _ = env.reset()
    validate_training_contract(env, obs_dict, agent_cfg)

    log_dir = create_log_dir(agent_cfg)
    write_run_config(
        log_dir,
        env_cfg,
        agent_cfg,
        seed,
        resume_path,
    )

    actor, critic = build_actor_critic(env, agent_cfg, seed)
    ppo = build_ppo(env, actor, critic, agent_cfg, log_dir)

    if resume_path is None:
        start_update = 0
        print(
            "starting Teacher training from newly initialized "
            "networks"
        )
    else:
        start_update = load_checkpoint(
            path=resume_path,
            actor=actor,
            critic=critic,
            ppo=ppo,
            env=env,
            max_iterations=agent_cfg.max_iterations,
            task_id=args_cli.task,
            algorithm_name=agent_cfg.name,
            grasp_steps=agent_cfg.grasp_steps,
            reward_clip=agent_cfg.reward_clip,
        )

    last_update = start_update - 1
    last_saved_update = start_update - 1
    training_start_time = time.perf_counter()
    completed_updates = 0
    steps_per_iteration = env.num_envs * agent_cfg.grasp_steps

    for update in range(
        start_update,
        agent_cfg.max_iterations,
    ):
        collection_start_time = time.perf_counter()
        (
            final_obs,
            mean_reward,
            mean_logs,
            mean_episode_reward,
            mean_episode_length,
        ) = run_rollout(
            env=env,
            ppo=ppo,
            grasp_steps=agent_cfg.grasp_steps,
            collect_for_update=True,
        )
        collection_time = (
            time.perf_counter() - collection_start_time
        )
        if ppo.storage.step != agent_cfg.grasp_steps:
            raise RuntimeError(
                "Teacher PPO storage does not contain exactly "
                "70 steps: "
                f"actual={ppo.storage.step}, "
                f"required={agent_cfg.grasp_steps}"
            )

        learning_start_time = time.perf_counter()
        ppo_metrics = ppo.update(
            actor_obs=final_obs,
            value_obs=final_obs,
            log_this_iteration=True,
            update=update,
        )
        learning_time = time.perf_counter() - learning_start_time
        if ppo.check_exploding_gradient():
            raise RuntimeError(
                f"Exploding gradient detected at update {update}"
            )
        (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy,
        ) = ppo_metrics

        min_std = torch.full(
            (env.cfg.action_space,),
            agent_cfg.min_std,
            dtype=torch.float32,
            device=env.device,
        )
        actor.distribution.enforce_minimum_std(min_std)

        write_training_logs(
            ppo,
            update,
            mean_reward,
            mean_logs,
        )
        completed_updates += 1
        elapsed_time = time.perf_counter() - training_start_time
        remaining_updates = agent_cfg.max_iterations - update - 1
        eta = (
            elapsed_time
            / completed_updates
            * remaining_updates
        )
        print_teacher_iteration(
            update=update,
            max_iterations=agent_cfg.max_iterations,
            steps_per_iteration=steps_per_iteration,
            collection_time=collection_time,
            learning_time=learning_time,
            mean_std=actor.distribution.std.mean().item(),
            mean_value_loss=mean_value_loss,
            mean_surrogate_loss=mean_surrogate_loss,
            mean_entropy=mean_entropy,
            mean_episode_reward=mean_episode_reward,
            mean_episode_length=mean_episode_length,
            total_timesteps=ppo.tot_timesteps,
            learning_rate=ppo.learning_rate,
            elapsed_time=elapsed_time,
            eta=eta,
        )

        last_update = update
        if update % agent_cfg.save_interval == 0:
            checkpoint_path = os.path.join(
                log_dir,
                f"full_{update}_r.pt",
            )
            save_checkpoint(
                path=checkpoint_path,
                actor=actor,
                critic=critic,
                ppo=ppo,
                env=env,
                update=update,
                task_id=args_cli.task,
                algorithm_name=agent_cfg.name,
                grasp_steps=agent_cfg.grasp_steps,
                reward_clip=agent_cfg.reward_clip,
            )
            last_saved_update = update

        if (
            agent_cfg.evaluation_enabled
            and (update + 1)
            % agent_cfg.evaluation_interval
            == 0
        ):
            evaluation_metrics = run_periodic_evaluation(
                env,
                ppo,
                agent_cfg,
            )
            write_evaluation_logs(
                ppo,
                update,
                evaluation_metrics,
            )
            print(
                f"evaluation_update={update} "
                "attempts="
                f"{int(evaluation_metrics['attempts'])} "
                "successes="
                f"{int(evaluation_metrics['successes'])} "
                "failures="
                f"{int(evaluation_metrics['failures'])} "
                "success_rate="
                f"{evaluation_metrics['success_rate']:.6f} "
                "failure_rate="
                f"{evaluation_metrics['failure_rate']:.6f} "
                "mean_lift_height="
                f"{evaluation_metrics['mean_lift_height']:.6f}"
            )


    if (
        last_update >= start_update
        and last_saved_update != last_update
    ):
        checkpoint_path = os.path.join(
            log_dir,
            f"full_{last_update}_r.pt",
        )
        save_checkpoint(
            path=checkpoint_path,
            actor=actor,
            critic=critic,
            ppo=ppo,
            env=env,
            update=last_update,
            task_id=args_cli.task,
            algorithm_name=agent_cfg.name,
            grasp_steps=agent_cfg.grasp_steps,
            reward_clip=agent_cfg.reward_clip,
        )

    ppo.writer.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
