from isaaclab.utils import configclass


@configclass
class PPOStudentRunnerCfg:
    name: str = "ppo_student"
    log_dir: str = "./runs_student"

    test: bool = False
    max_iterations: int = 50001
    save_interval: int = 500

    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    gamma: float = 0.996
    lam: float = 0.95
    learning_rate: float = 5.0e-4
    clip_param: float = 0.2
    desired_kl: float = 0.01

    init_std: float = 1.0
    min_std: float = 0.2
    policy_net: list[int] = [128, 128]
    value_net: list[int] = [128, 128]
    activation: str = "lrelu"

    grasp_steps: int = 70
    lift_steps: int = 80

    history_len: int = 10
    prop_latent_dim: int = 26
    update_mlp: bool = True
    student_driven_ratio: float = 1.0
    ppo_ratio: float = 0.5
    curriculum: bool = True
    reward_clip: float = -2.0