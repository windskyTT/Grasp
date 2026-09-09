"""Train the standalone RobustDexGrasp Teacher with the original PPO contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=50001)
parser.add_argument("--run_dir", type=Path, default=Path("runs_teacher_test"))
parser.add_argument("--checkpoint", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn

import Grasp.tasks  # noqa: F401
from Grasp.tasks.direct.grasp.teacher_test.env_cfg import RobustDexTeacherEnvCfg
from Grasp.tasks.direct.grasp.teacher_test.ppo import Actor, Critic, MLP, MultivariateGaussianDiagonalCovariance, PPO


def make_agent(env):
    actor = Actor(
        MLP(env.cfg.ppo_policy_net, nn.LeakyReLU, 153, 22),
        MultivariateGaussianDiagonalCovariance(22, 1.0),
        env.device,
    )
    critic = Critic(MLP(env.cfg.ppo_value_net, nn.LeakyReLU, 153, 1), env.device)
    ppo = PPO(
        actor=actor,
        critic=critic,
        num_envs=env.num_envs,
        num_transitions_per_env=env.cfg.grasp_steps,
        num_learning_epochs=env.cfg.ppo_epochs,
        num_mini_batches=env.cfg.ppo_mini_batches,
        gamma=env.cfg.ppo_gamma,
        lam=env.cfg.ppo_lambda,
        device=env.device,
        shuffle_batch=env.cfg.ppo_shuffle_batch,
    )
    return actor, critic, ppo


def save_checkpoint(path: Path, actor, critic, ppo, update: int, env):
    torch.save(
        {
            "task_id": "Grasp-RobustDexTeacher-UR5-Allegro-v0",
            "algorithm_name": "ppo",
            "observation_dim": 153,
            "action_dim": 22,
            "grasp_steps": 70,
            "actor": actor.architecture.state_dict(),
            "distribution": actor.distribution.state_dict(),
            "critic": critic.architecture.state_dict(),
            "optimizer": ppo.optimizer.state_dict(),
            "update": update,
        },
        path,
    )


def main():
    cfg = RobustDexTeacherEnvCfg()
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Grasp-RobustDexTeacher-UR5-Allegro-v0", cfg=cfg).unwrapped
    actor, critic, ppo = make_agent(env)
    if args_cli.checkpoint is not None:
        checkpoint = torch.load(args_cli.checkpoint, map_location=env.device, weights_only=True)
        actor.architecture.load_state_dict(checkpoint["actor"], strict=True)
        actor.distribution.load_state_dict(checkpoint["distribution"], strict=True)
        critic.architecture.load_state_dict(checkpoint["critic"], strict=True)
        ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["update"] + 1
    else:
        start = 0
    obs, _ = env.reset()
    shape_checked = False
    for update in range(start, args_cli.max_iterations):
        for _ in range(env.cfg.grasp_steps):
            value_obs = obs["policy"]
            action = ppo.act(value_obs)
            if not shape_checked:
                print(f"observation_shape={tuple(value_obs.shape)}")
                print(f"action_shape={tuple(action.shape)}")
                if value_obs.shape[-1] != 153 or action.shape[-1] != 22:
                    raise RuntimeError(
                        f"Teacher tensor shapes do not match observation=153/action=22: "
                        f"observation={tuple(value_obs.shape)}, action={tuple(action.shape)}"
                    )
                shape_checked = True
            obs, reward, terminated, truncated, _ = env.step(action)
            ppo.step(value_obs, reward, terminated | truncated)
        ppo.update(obs["policy"], obs["policy"])
        if update % 100 == 0:
            args_cli.run_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args_cli.run_dir / f"model_{update:06d}.pt", actor, critic, ppo, update, env)
            print(f"update={update} reward={reward.mean().item():.6f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
