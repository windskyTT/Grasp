"""Run quantitative Teacher evaluation on the original shapenet-30obj set."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn

import Grasp.tasks  # noqa: F401
from Grasp.tasks.direct.grasp.teacher_test.env_cfg import RobustDexTeacherEnvCfg
from Grasp.tasks.direct.grasp.teacher_test.object_loader import make_object_spawner
from Grasp.tasks.direct.grasp.teacher_test.object_set import EVAL_OBJECT_NAMES, EVAL_ROOT, EVAL_URDF_PATHS
from Grasp.tasks.direct.grasp.teacher_test.ppo import Actor, MLP, MultivariateGaussianDiagonalCovariance


def main():
    cfg = RobustDexTeacherEnvCfg()
    cfg.dataset_root = str(EVAL_ROOT)
    cfg.object_names = EVAL_OBJECT_NAMES
    cfg.non_uniform_sampling = False
    cfg.object.spawn = make_object_spawner(EVAL_URDF_PATHS)
    cfg.scene.num_envs = args_cli.num_envs or len(EVAL_OBJECT_NAMES)
    env = gym.make("Grasp-RobustDexTeacher-UR5-Allegro-v0", cfg=cfg).unwrapped
    actor = Actor(MLP(cfg.ppo_policy_net, nn.LeakyReLU, 153, 22), MultivariateGaussianDiagonalCovariance(22, 1.0), env.device)
    checkpoint = torch.load(args_cli.checkpoint, map_location=env.device, weights_only=True)
    if checkpoint["observation_dim"] != 153 or checkpoint["action_dim"] != 22:
        raise RuntimeError("Teacher checkpoint dimensions do not match 153/22")
    actor.architecture.load_state_dict(checkpoint["actor"], strict=True)
    actor.distribution.load_state_dict(checkpoint["distribution"], strict=True)
    actor.architecture.eval()
    successes = 0
    episodes = 0
    for _ in range(args_cli.repeats):
        obs, _ = env.reset()
        initial_z = env.object_initial_position[:, 2].clone()
        for _ in range(cfg.grasp_steps):
            action = actor.noiseless_action(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(action)
        env.switch_root_guidance(True)
        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, 22), device=env.device))
        successes += int((env.object.data.root_pos_w[:, 2] - initial_z > cfg.lift_height).sum().item())
        episodes += env.num_envs
        print(f"repeat={episodes // env.num_envs} success={successes}/{episodes}")
    print(f"quantitative_success_rate={successes / episodes:.6f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
