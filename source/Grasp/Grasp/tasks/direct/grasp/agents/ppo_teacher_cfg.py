"""Configuration for the standalone multi-step Teacher PPO."""

from isaaclab.utils import configclass


@configclass
class PPOTeacherRunnerCfg:
    name: str = "ppo_teacher"
    log_dir: str = "./runs_teacher"

    max_iterations: int = 50001
    save_interval: int = 500

    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    gamma: float = 0.996
    lam: float = 0.95
    learning_rate: float = 5.0e-4
    clip_param: float = 0.2
    desired_kl: float = 0.01
    shuffle_batch: bool = False

    init_std: float = 1.0
    min_std: float = 0.2
    policy_net: list[int] = [128, 128]
    value_net: list[int] = [128, 128]
    activation: str = "lrelu"

    grasp_steps: int = 70
    lift_steps: int = 30
    reward_clip: float = -2.0

    evaluation_enabled: bool = False
    evaluation_interval: int = 500
    evaluation_rollouts: int = 1


__all__ = ["PPOTeacherRunnerCfg"]
