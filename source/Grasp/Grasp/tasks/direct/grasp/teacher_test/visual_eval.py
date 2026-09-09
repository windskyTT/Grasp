"""Run a visual Teacher rollout with the original evaluation objects."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=70)
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
    cfg.object_names = EVAL_OBJECT_NAMES[:1]
    cfg.non_uniform_sampling = False
    cfg.object.spawn = make_object_spawner(EVAL_URDF_PATHS[:1])
    cfg.scene.num_envs = 1
    env = gym.make("Grasp-RobustDexTeacher-UR5-Allegro-v0", cfg=cfg).unwrapped
    actor = Actor(MLP(cfg.ppo_policy_net, nn.LeakyReLU, 153, 22), MultivariateGaussianDiagonalCovariance(22, 1.0), env.device)
    checkpoint = torch.load(args_cli.checkpoint, map_location=env.device, weights_only=True)
    actor.architecture.load_state_dict(checkpoint["actor"], strict=True)
    actor.distribution.load_state_dict(checkpoint["distribution"], strict=True)
    obs, _ = env.reset()
    for step in range(args_cli.steps):
        obs, reward, terminated, truncated, _ = env.step(actor.noiseless_action(obs["policy"]))
        env.render()
        print(f"step={step + 1} reward={reward.item():.6f} terminated={terminated.item()} truncated={truncated.item()}")
    env.switch_root_guidance(True)
    for step in range(100):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 22), device=env.device))
        env.render()
        print(f"lift_step={step + 1} reward={reward.item():.6f} terminated={terminated.item()} truncated={truncated.item()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
