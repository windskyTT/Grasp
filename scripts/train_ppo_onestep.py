import argparse
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train DemoGrasp PPOOneStep in IsaacLab.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="grasp")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--torch_deterministic", action="store_true", default=False)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--run_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

import Grasp.tasks  # noqa: F401
from Grasp.algo import ppo_onestep

def compute_observation_dim(env_cfg):
    return sum(
        dim
        for name, dim in env_cfg.hand.num_obs_dict.items()
        if name in env_cfg.observation_type
    )

def main():
    seed = configure_seed(args_cli.seed, torch_deterministic=args_cli.torch_deterministic)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "ppo_onestep_cfg_entry_point")
    env_cfg.reference.randomize_tracking_reference = True
    env_cfg.control.arm_controller = "pose"
    env_cfg.reference.randomize_tracking_reference = True
    env_cfg.reference.randomize_grasp_pose = True
    env_cfg.observation_type = "eefpose+objinitpose+objpcl"
    env_cfg.observation_space = compute_observation_dim(env_cfg)
    env_cfg.episode_length_s = 40 * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.enable_point_cloud = True
    env_cfg.enable_robot_table_collision = False
    agent_cfg.is_vision = True

    env_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.test = args_cli.test

    if not agent_cfg.test:
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{args_cli.task}_{time_str}"
        if args_cli.run_name:
            run_name = args_cli.run_name
        log_dir = os.path.abspath(os.path.join(agent_cfg.log_dir, run_name))
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "task": args_cli.task,
                    "env_cfg": env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else str(env_cfg),
                    "agent_cfg": agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else str(agent_cfg),
                },
                f,
                indent=4,
            )
        dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "agent.yaml"), agent_cfg)
    else:
        log_dir = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()

    if agent_cfg.name != "ppo_onestep":
        raise ValueError(f"Unsupported algorithm: {agent_cfg.name}")

    if not env.cfg.reference.randomize_tracking_reference:
        raise RuntimeError(
            "PPOOneStep training requires cfg.reference.randomize_tracking_reference=True. "
            "Otherwise policy actions will not affect generate_reaching_plan_idx()."
        )

    act_dim = 6
    if env.cfg.reference.randomize_grasp_pose:
        act_dim += len(env.active_hand_dof_ids)

    runner = ppo_onestep.PPO(
        vec_env=env,
        step_env=env,
        actor_critic_class=ppo_onestep.ActorCritic,
        train_param=agent_cfg,
        log_dir=log_dir,
        apply_reset=False,
        action_dim=act_dim,
    )

    if args_cli.test and args_cli.checkpoint:
        print(f"Loading model from {args_cli.checkpoint}")
        runner.test(args_cli.checkpoint)
    elif args_cli.checkpoint:
        print(f"Loading model from {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    runner.run()
    env.close()



if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()