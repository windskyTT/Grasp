import argparse
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run DemoGrasp PPOOneStep in IsaacLab.")
parser.add_argument("--task", type=str, default="grasp")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--torch_deterministic", action="store_true", default=False)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--debug", type=str, default=None, choices=["check_joint", "test_demo_replay", "collect_real_dataset", "noop"])
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--print_config", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=100)
parser.add_argument("--video_interval", type=int, default=1464)
parser.add_argument("--force_render", action="store_true", default=False)
parser.add_argument("--randomize_tracking_reference", action="store_true", default=None)
parser.add_argument("--randomize_grasp_pose", action="store_true", default=None)
parser.add_argument("--arm_controller", type=str, default=None, choices=["qpos", "worlddpose", "eedpose", "pose"])
parser.add_argument("--observation_type", type=str, default=None)
parser.add_argument("--episode_length", type=int, default=None)
parser.add_argument("--tracking_reference_file", type=str, default=None)
parser.add_argument("--tracking_reference_lift_timestep", type=int, default=None)
parser.add_argument("--multi_object_list", type=str, default=None)
parser.add_argument("--enable_point_cloud", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--is_vision", action=argparse.BooleanOptionalAction, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.video or args_cli.force_render:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import Grasp.tasks  # noqa: F401
from Grasp.algo import ppo_onestep
def print_debug_config(env_cfg, agent_cfg):
    print("========== DEBUG CONFIG ==========")
    print("task:", args_cli.task)
    print("test:", args_cli.test)
    print("checkpoint:", args_cli.checkpoint)
    print("num_envs:", env_cfg.scene.num_envs)
    print("hand:", env_cfg.hand.name)
    print("observation_type:", env_cfg.observation_type)
    print("arm_controller:", env_cfg.control.arm_controller)
    print("multi_object_list:", env_cfg.asset.multi_object_list)
    print("randomize_tracking_reference:", env_cfg.reference.randomize_tracking_reference)
    print("randomize_grasp_pose:", env_cfg.reference.randomize_grasp_pose)
    print("tracking_reference_file:", env_cfg.reference.tracking_reference_file)
    print("episode_length_s:", env_cfg.episode_length_s)
    print("enable_point_cloud:", env_cfg.enable_point_cloud)
    print("train is_vision:", getattr(agent_cfg, "is_vision", None))
    print("==================================")

def cfg_to_plain(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)

def compute_observation_dim(env_cfg):
    return sum(
        dim
        for name, dim in env_cfg.hand.num_obs_dict.items()
        if name in env_cfg.observation_type
    )


def resolve_reference_file(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return path
    if path.startswith("tasks/"):
        candidate = os.path.join(
            os.getcwd(),
            "source",
            "Grasp",
            "Grasp",
            "reference",
            os.path.basename(path),
        )
        if os.path.exists(candidate):
            return candidate
    return path


def apply_demograsp_cli_overrides(env_cfg, agent_cfg):
    if args_cli.arm_controller is not None:
        env_cfg.control.arm_controller = args_cli.arm_controller
    if args_cli.observation_type is not None:
        env_cfg.observation_type = args_cli.observation_type
        env_cfg.observation_space = compute_observation_dim(env_cfg)
    if args_cli.episode_length is not None:
        env_cfg.episode_length_s = args_cli.episode_length * env_cfg.sim.dt * env_cfg.decimation
    if args_cli.tracking_reference_file is not None:
        env_cfg.reference.tracking_reference_file = resolve_reference_file(args_cli.tracking_reference_file)
    if args_cli.tracking_reference_lift_timestep is not None:
        env_cfg.reference.tracking_reference_lift_timestep = args_cli.tracking_reference_lift_timestep
    if args_cli.multi_object_list is not None:
        env_cfg.asset.multi_object_list = args_cli.multi_object_list
    if args_cli.randomize_tracking_reference is not None:
        env_cfg.reference.randomize_tracking_reference = args_cli.randomize_tracking_reference
    if args_cli.randomize_grasp_pose is not None:
        env_cfg.reference.randomize_grasp_pose = args_cli.randomize_grasp_pose
    if args_cli.enable_point_cloud is not None:
        env_cfg.enable_point_cloud = args_cli.enable_point_cloud
    if args_cli.is_vision is not None:
        agent_cfg.is_vision = args_cli.is_vision

def build_log_dir(agent_cfg):
    if args_cli.test:
        return None
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.run_name if args_cli.run_name else f"{args_cli.task}_{time_str}"
    log_dir = os.path.abspath(os.path.join(agent_cfg.log_dir, run_name))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def build_runner(agent_cfg, env, step_env, log_dir):
    print("[RUN_CKPT_STAGE] build_runner start", flush=True)
    if agent_cfg.name != "ppo_onestep":
        raise ValueError(f"Unrecognized algorithm: {agent_cfg.name}")

    if not env.cfg.reference.randomize_tracking_reference:
        raise RuntimeError(
            "PPOOneStep requires env.cfg.reference.randomize_tracking_reference=True. "
            "Otherwise policy actions will not affect generate_reaching_plan_idx()."
        )

    act_dim = 6
    if env.cfg.reference.randomize_grasp_pose:
        act_dim += len(env.active_hand_dof_ids)

    state_shape = env.state_space.shape if env.state_space is not None else (0,)
    print(
        "[RUN_CKPT_STAGE] build_runner dimensions "
        f"obs_shape={env.observation_space.shape} state_shape={state_shape} "
        f"act_dim={act_dim} active_hand_dofs={len(env.active_hand_dof_ids)}",
        flush=True,
    )

    runner = ppo_onestep.PPO(
        vec_env=env,
        step_env=step_env,
        actor_critic_class=ppo_onestep.ActorCritic,
        train_param=agent_cfg,
        log_dir=log_dir,
        apply_reset=False,
        action_dim=act_dim,
    )
    print("[RUN_CKPT_STAGE] PPO runner created", flush=True)

    if args_cli.test and args_cli.checkpoint:
        print(f"[RUN_CKPT_STAGE] before runner.test checkpoint={args_cli.checkpoint}", flush=True)
        runner.test(args_cli.checkpoint)
        print("[RUN_CKPT_STAGE] after runner.test", flush=True)
    elif args_cli.checkpoint:
        print(f"[RUN_CKPT_STAGE] before runner.load checkpoint={args_cli.checkpoint}", flush=True)
        runner.load(args_cli.checkpoint)
        print("[RUN_CKPT_STAGE] after runner.load", flush=True)

    print("[RUN_CKPT_STAGE] build_runner return", flush=True)
    return runner

def check_joint(env):
    per_joint_duration = 10
    for t in range(100000):
        act = env.no_op_action.clone()
        i_joint = int(t / per_joint_duration) % env.num_actions
        print(i_joint)
        t_ = t % per_joint_duration
        if t_ < per_joint_duration // 2:
            target = env.robot_dof_lower_limits[env.active_robot_dof_ids[i_joint]]
        else:
            target = env.robot_dof_upper_limits[env.active_robot_dof_ids[i_joint]]

        lower = env.robot_dof_lower_limits[env.active_robot_dof_ids[i_joint]]
        upper = env.robot_dof_upper_limits[env.active_robot_dof_ids[i_joint]]
        act[:, i_joint] = 2.0 * (target - lower) / (upper - lower + 1.0e-6) - 1.0
        env.step(act)

def test_demo_replay(env):
    for t in range(100000):
        action = env.compute_reference_actions()
        obs, reward, terminated, truncated, extras = env.step(action)
        reset = terminated | truncated
        if (t + 1) % env.max_episode_length == 0:
            print("success rate:", env.current_successes.mean())
        env_ids = reset.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            env.reset_idx(env_ids)

def collect_real_dataset(agent_cfg, env, log_dir):
    num_episodes = args_cli.num_episodes
    play_policy = True

    if play_policy:
        agent_cfg.test = True
        runner = build_runner(agent_cfg, env, step_env, log_dir)
        policy = runner.actor_critic
        policy.eval()
    else:
        policy = None

    from Grasp.data.dataset_utils import LerobotDatasetWriter

    dataset_writer = LerobotDatasetWriter(
        output_path=f"{env.hand_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        camera_ids=env.camera_ids,
        data_type=env.render_data_type,
        image_shape=(*env.render_cfg["resize"], 3),
        fps=round(1 / env.step_dt),
    )

    n_saved_episodes = 0
    while n_saved_episodes < num_episodes:
        obs_dict = env.reset_idx(torch.arange(env.num_envs, device=env.device))
        obs = obs_dict["obs"].clone()
        if play_policy:
            with torch.no_grad():
                plan = policy(obs, inference=True)
                env.generate_reaching_plan_idx(torch.arange(env.num_envs, device=env.device), actions=plan)

        episode_data_buffer = []
        reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        extras = {}
        for _ in range(env.max_episode_length):
            real_obs = env.compute_real_observation_dict()
            action = env.compute_reference_actions()
            _, _, terminated, truncated, extras = env.step(action)
            reset = terminated | truncated
            real_obs["action"] = action.cpu().numpy()
            episode_data_buffer.append(real_obs)

        assert reset.all()
        success = extras["current_successes"] > 0.5
        for env_id in range(env.num_envs):
            if n_saved_episodes >= num_episodes:
                break
            if success[env_id]:
                for t, frame in enumerate(episode_data_buffer):
                    episode_end = t == len(episode_data_buffer) - 1
                    dataset_writer.append_step({k: v[env_id : env_id + 1] for k, v in frame.items()}, episode_end)
                n_saved_episodes += 1
                print(f"Saved episode {n_saved_episodes} from env {env_id}")


def main():
    print("[RUN_CKPT_STAGE] main start", flush=True)
    seed = configure_seed(args_cli.seed, torch_deterministic=args_cli.torch_deterministic)
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "ppo_onestep_cfg_entry_point")
    print("[RUN_CKPT_STAGE] configs loaded", flush=True)

    apply_demograsp_cli_overrides(env_cfg, agent_cfg)
    env_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.test = args_cli.test
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.randomize_tracking_reference is not None:
        env_cfg.reference.randomize_tracking_reference = args_cli.randomize_tracking_reference
    if args_cli.randomize_grasp_pose is not None:
        env_cfg.reference.randomize_grasp_pose = args_cli.randomize_grasp_pose
    print(
        "[RUN_CKPT_STAGE] overrides applied "
        f"test={agent_cfg.test} checkpoint={args_cli.checkpoint} "
        f"obs={env_cfg.observation_type} pcl={env_cfg.enable_point_cloud} "
        f"is_vision={getattr(agent_cfg, 'is_vision', None)} "
        f"randomize_ref={env_cfg.reference.randomize_tracking_reference} "
        f"randomize_grasp={env_cfg.reference.randomize_grasp_pose}",
        flush=True,
    )

    if args_cli.print_config:
        print_debug_config(env_cfg, agent_cfg)

    log_dir = build_log_dir(agent_cfg)
    if log_dir is not None:
        dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump({"env_cfg": cfg_to_plain(env_cfg), "agent_cfg": cfg_to_plain(agent_cfg)}, f, indent=4)

    print("[RUN_CKPT_STAGE] before gym.make", flush=True)
    env_wrapper = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[RUN_CKPT_STAGE] after gym.make", flush=True)
    if args_cli.video:
        video_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_kwargs = {
            "video_folder": os.path.join(log_dir or os.getcwd(), "videos", "run_rl_grasp"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"rl-video-{video_time}",
            "disable_logger": True,
        }
        env_wrapper = gym.wrappers.RecordVideo(env_wrapper, **video_kwargs)
    env = env_wrapper.unwrapped
    print("[RUN_CKPT_STAGE] before env.reset", flush=True)
    env_wrapper.reset()
    print("[RUN_CKPT_STAGE] after env.reset", flush=True)

    try:
        if args_cli.debug == "check_joint":
            print("[RUN_CKPT_STAGE] debug check_joint start", flush=True)
            check_joint(env)
        elif args_cli.debug == "test_demo_replay":
            print("[RUN_CKPT_STAGE] debug test_demo_replay start", flush=True)
            test_demo_replay(env)
        elif args_cli.debug == "collect_real_dataset":
            print("[RUN_CKPT_STAGE] debug collect_real_dataset start", flush=True)
            collect_real_dataset(agent_cfg, env, env_wrapper, log_dir)
        elif args_cli.debug == "noop":
            print("[RUN_CKPT_STAGE] debug noop start", flush=True)
            for _ in range(100000):
                env.step(env.no_op_action)
        else:
            print("[RUN_CKPT_STAGE] before build_runner", flush=True)
            runner = build_runner(agent_cfg, env, env_wrapper, log_dir)
            print("[RUN_CKPT_STAGE] after build_runner before runner.run", flush=True)
            runner.run()
            print("[RUN_CKPT_STAGE] after runner.run", flush=True)
    finally:
        print("[RUN_CKPT_STAGE] closing env_wrapper", flush=True)
        env_wrapper.close()

if __name__ == "__main__":
    main()
    simulation_app.close()