from isaaclab.utils import configclass


@configclass
class PPOOneStepPolicyCfg:
    backbone_type: str = "pn"
    freeze_backbone: bool = False
    pi_hid_sizes: list[int] = [1024, 1024, 512, 512]
    vf_hid_sizes: list[int] = [1024, 1024, 512, 512]
    activation: str = "elu"
    pc_shape: list[int] = [512, 3]
    pc_emb_dim: int = 128


@configclass
class PPOOneStepRunnerCfg:
    name: str = "ppo_onestep"
    log_dir: str = "./runs_ppo"

    is_vision: bool = False
    policy: PPOOneStepPolicyCfg = PPOOneStepPolicyCfg()

    test: bool = False
    resume: int = 0
    save_interval: int = 100
    print_log: bool = True

    max_iterations: int = 20000

    cliprange: float = 0.2
    ent_coef: float = 0.0
    nsteps: int = 1
    noptepochs: int = 5
    nminibatches: int = 4
    max_grad_norm: float = 1.0
    optim_stepsize: float = 3.0e-4
    schedule: str = "adaptive"
    desired_kl: float = 0.016
    gamma: float = 0.96
    lam: float = 0.95
    init_noise_std: float = 0.8

    surrogate_loss_coef: float = 1.0
    value_loss_coef: float = 2.0

    discard_invalid_resets: bool = False