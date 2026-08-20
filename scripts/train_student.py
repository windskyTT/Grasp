from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Train RobustDexGrasp-style multi-step Student DAgger/PPO "
        "in Grasp IsaacLab."
    )
)
parser.add_argument(
    "--task",
    type=str,
    default="Grasp-Student-Direct-v0",
)
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument("--student_checkpoint", type=str, default="")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--torch_deterministic",
    action="store_true",
    default=False,
)
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
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

import Grasp.tasks  # noqa: F401
from Grasp.algo import ppo_student
from Grasp.algo.checkpoint_contract import (
    STUDENT_RESUME_KEYS,
    TEACHER_EXPERT_KEYS,
    get_resume_start_update,
    validate_student_checkpoint,
    validate_teacher_checkpoint,
)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def build_actor(input_dim, env, agent_cfg, seed):
    return ppo_student.Actor(
        ppo_student.MLP(
            agent_cfg.policy_net,
            nn.LeakyReLU,
            input_dim,
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


def load_teacher_checkpoint(
    path,
    actor_expert,
    actor_student,
    critic_student,
    env,
):
    checkpoint = torch.load(
        path,
        map_location=env.device,
        weights_only=True,
    )

    validate_teacher_checkpoint(
        checkpoint,
        required_keys=TEACHER_EXPERT_KEYS,
        expected_teacher_dim=env.obs_spec.teacher_dim,
        expected_action_dim=env.cfg.action_space,
        role="Student Teacher expert",
        path=path,
    )

    actor_expert.architecture.load_state_dict(
        checkpoint["actor_architecture_state_dict"],
        strict=True,
    )
    actor_expert.distribution.load_state_dict(
        checkpoint["actor_distribution_state_dict"],
        strict=True,
    )
    critic_student.architecture.load_state_dict(
        checkpoint["critic_architecture_state_dict"],
        strict=True,
    )

    ppo_student.copy_compatible_teacher_actor_layers(
        actor_expert.architecture,
        actor_student.architecture,
    )
    actor_student.distribution.load_state_dict(
        checkpoint["actor_distribution_state_dict"],
        strict=True,
    )

    actor_expert.update()
    actor_student.update()
    actor_expert.architecture.architecture.eval()
    for parameter in actor_expert.architecture.parameters():
        parameter.requires_grad_(False)

    print(f"loaded frozen Teacher expert checkpoint: {path}")


def save_student_checkpoint(
    path,
    actor_student,
    critic_student,
    prop_latent_encoder,
    dagger,
    env,
    student_actor_obs_dim,
    update,
):
    torch.save(
        {
            "actor_architecture_state_dict": (
                actor_student.architecture.state_dict()
            ),
            "actor_distribution_state_dict": (
                actor_student.distribution.state_dict()
            ),
            "critic_architecture_state_dict": (
                critic_student.architecture.state_dict()
            ),
            "optimizer_state_dict": dagger.optimizer.state_dict(),
            "prop_latent_encoder_state_dict": (
                prop_latent_encoder.state_dict()
            ),
            "obs_spec": asdict(env.obs_spec),
            "student_actor_obs_dim": student_actor_obs_dim,
            "action_dim": env.cfg.action_space,
            "update": update,
        },
        path,
    )
    print(f"saved checkpoint: {path}")


def load_student_checkpoint(
    path,
    actor_student,
    critic_student,
    prop_latent_encoder,
    dagger,
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
        required_keys=STUDENT_RESUME_KEYS,
        expected_obs_spec=env.obs_spec,
        expected_student_actor_obs_dim=student_actor_obs_dim,
        expected_action_dim=env.cfg.action_space,
        role="Student resume",
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
    critic_student.architecture.load_state_dict(
        checkpoint["critic_architecture_state_dict"],
        strict=True,
    )
    prop_latent_encoder.load_state_dict(
        checkpoint["prop_latent_encoder_state_dict"],
        strict=True,
    )
    dagger.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    actor_student.update()

    start_update = get_resume_start_update(
        checkpoint,
        role="Student resume",
        path=path,
    )
    print(
        f"loaded Student checkpoint: {path}, "
        f"loaded_update={start_update - 1}, "
        f"next_update={start_update}"
    )
    return start_update

def create_log_dir(agent_cfg):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.run_name
    if run_name is None:
        run_name = f"{args_cli.task}_{time_str}"
    log_dir = os.path.abspath(os.path.join(agent_cfg.log_dir, run_name))
    os.makedirs(log_dir, exist_ok=False)
    return log_dir


def main():
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
        "ppo_student_cfg_entry_point",
    )
    if agent_cfg.name != "ppo_student":
        raise ValueError(f"Unsupported algorithm: {agent_cfg.name}")
    if agent_cfg.activation != "lrelu":
        raise ValueError(f"Unsupported activation: {agent_cfg.activation}")
    if agent_cfg.student_driven_ratio != 1.0:
        raise ValueError(
            "This PPO-integrated DAgger route requires "
            "student_driven_ratio=1.0"
        )

    env_cfg.seed = seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = create_log_dir(agent_cfg)
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
                "teacher_checkpoint": args_cli.teacher_checkpoint,
                "student_checkpoint": args_cli.student_checkpoint,
                "seed": seed,
            },
            file,
            indent=2,
        )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs_dict, _ = env.reset()

    if env.obs_spec.student_total_dim != 517:
        raise ValueError("Student total observation must be 517")
    if env.obs_spec.teacher_dim != 77:
        raise ValueError("Teacher observation must be 77")
    if env.cfg.action_space != 13:
        raise ValueError("Student action dimension must be 13")
    if env.max_episode_length != agent_cfg.grasp_steps:
        raise ValueError("Student episode and rollout length mismatch")
    if agent_cfg.history_len != env.obs_spec.history_len:
        raise ValueError("Student history length mismatch")
    if agent_cfg.prop_latent_dim != env.obs_spec.prop_latent_dim:
        raise ValueError("Student property latent dimension mismatch")

    student_actor_obs_dim = (
        ppo_student.compute_student_actor_obs_dim(
            env.obs_spec.history_dim,
            agent_cfg.prop_latent_dim,
            env.obs_spec.aff_vec_dim,
        )
    )
    actor_expert = build_actor(
        env.obs_spec.teacher_dim,
        env,
        agent_cfg,
        seed,
    )
    actor_student = build_actor(
        student_actor_obs_dim,
        env,
        agent_cfg,
        seed + 1,
    )
    critic_student = ppo_student.Critic(
        ppo_student.MLP(
            agent_cfg.value_net,
            nn.LeakyReLU,
            env.obs_spec.teacher_dim,
            1,
        ),
        env.device,
    )
    prop_latent_encoder = ppo_student.LSTM_StateHistoryEncoder(
        env.obs_spec.history_dim,
        agent_cfg.prop_latent_dim,
        agent_cfg.history_len,
        env.device,
    ).to(env.device)

    load_teacher_checkpoint(
        args_cli.teacher_checkpoint,
        actor_expert,
        actor_student,
        critic_student,
        env,
    )

    dagger = ppo_student.Dagger(
        expert_policy=actor_expert.architecture,
        actor_student=actor_student,
        critic_student=critic_student,
        prop_latent_encoder=prop_latent_encoder,
        tobeEncode_dim=env.obs_spec.history_dim,
        prop_latent_dim=agent_cfg.prop_latent_dim,
        aff_vec_dim=env.obs_spec.aff_vec_dim,
        total_obs_dim=env.obs_spec.student_total_dim,
        teacher_obs_dim=env.obs_spec.teacher_dim,
        t_steps=agent_cfg.history_len,
        num_envs=env.num_envs,
        num_transitions_per_env=agent_cfg.grasp_steps,
        num_learning_epochs=agent_cfg.num_learning_epochs,
        num_mini_batches=agent_cfg.num_mini_batches,
        clip_param=agent_cfg.clip_param,
        gamma=agent_cfg.gamma,
        lam=agent_cfg.lam,
        learning_rate=agent_cfg.learning_rate,
        desired_kl=agent_cfg.desired_kl,
        device=env.device,
        log_dir=log_dir,
        shuffle_batch=False,
        update_mlp=agent_cfg.update_mlp,
        ppo_ratio=agent_cfg.ppo_ratio,
    )

    start_update = 0
    if args_cli.student_checkpoint:
        start_update = load_student_checkpoint(
            args_cli.student_checkpoint,
            actor_student,
            critic_student,
            prop_latent_encoder,
            dagger,
            env,
            student_actor_obs_dim,
        )

    last_update = start_update - 1
    last_saved_update = start_update - 1

    for update in range(start_update, agent_cfg.max_iterations):
        if agent_cfg.curriculum:
            dagger.update_ppo_ratio(min(update * 0.0005, 1.0))

        obs_dict, _ = env.reset()
        total_obs = tensor_to_numpy(obs_dict["policy"])
        teacher_obs = tensor_to_numpy(obs_dict["teacher"])
        rollout_reward = 0.0

        for _ in range(agent_cfg.grasp_steps):
            student_aff = tensor_to_numpy(env.last_affordance_vec)
            action_numpy, _ = dagger.act(
                total_obs,
                student_driven_ratio=agent_cfg.student_driven_ratio,
                student_aff=student_aff,
            )
            action_tensor = torch.from_numpy(action_numpy).to(
                device=env.device,
                dtype=torch.float32,
            )

            next_obs_dict, reward, terminated, truncated, _ = env.step(
                action_tensor
            )
            reward_numpy = np.maximum(
                tensor_to_numpy(reward),
                agent_cfg.reward_clip,
            )
            done_numpy = (
                terminated | truncated
            ).detach().cpu().numpy().astype(np.bool_, copy=False)

            dagger.step(
                total_obs=total_obs,
                rews=reward_numpy,
                dones=done_numpy,
                value_obs=teacher_obs,
            )

            total_obs = tensor_to_numpy(next_obs_dict["policy"])
            teacher_obs = tensor_to_numpy(next_obs_dict["teacher"])
            rollout_reward += float(reward_numpy.mean())

        prop_mse, action_mse = dagger.update(value_obs=teacher_obs)
        if dagger.check_exploding_gradient():
            raise RuntimeError(
                f"Non-finite Student gradient/parameter at update {update}"
            )

        min_std = torch.full(
            (env.cfg.action_space,),
            agent_cfg.min_std,
            dtype=torch.float32,
            device=env.device,
        )
        actor_student.distribution.enforce_minimum_std(min_std)
        actor_student.update()

        average_reward = rollout_reward / agent_cfg.grasp_steps
        print(
            f"update={update} "
            f"ppo_ratio={dagger.ppo_ratio:.6f} "
            f"prop mse loss={prop_mse:.6f} "
            f"action mse loss={action_mse:.6f} "
            f"average reward={average_reward:.6f}"
        )
        dagger.writer.add_scalar("Loss/prop_mse", prop_mse, update)
        dagger.writer.add_scalar("Loss/action_mse", action_mse, update)
        dagger.writer.add_scalar("Train/ppo_ratio", dagger.ppo_ratio, update)
        dagger.writer.add_scalar("Train/average_reward", average_reward, update)

        last_update = update
        if update % agent_cfg.save_interval == 0:
            path = os.path.join(log_dir, f"full_{update}_r.pt")
            save_student_checkpoint(
                path,
                actor_student,
                critic_student,
                prop_latent_encoder,
                dagger,
                env,
                student_actor_obs_dim,
                update,
            )
            last_saved_update = update

    if last_update >= start_update and last_saved_update != last_update:
        path = os.path.join(log_dir, f"full_{last_update}_r.pt")
        save_student_checkpoint(
            path,
            actor_student,
            critic_student,
            prop_latent_encoder,
            dagger,
            env,
            student_actor_obs_dim,
            last_update,
        )

    dagger.writer.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()