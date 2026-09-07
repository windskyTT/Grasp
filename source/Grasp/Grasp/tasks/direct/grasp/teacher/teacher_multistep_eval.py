"""Teacher 多步抓取 / 抬升评估：GPU 数值链版本。

本文件负责：
1. visual / quantitative 两种评估模式。
2. 选择评估对象并修改 env 配置。
3. 运行 Teacher grasp policy。
4. 抓取结束后保持 Inspire 闭合目标，并用 FR3 palm Jacobian 向上抬升。
5. 在 GPU 上累计 success / lift height / reward / log。
6. 最终打印时才把标量转换成 Python 数值。

GPU 约定：
    - 不使用 NumPy。
    - 不访问 affordance_data 的 CPU object index。
    - grasp loop 内不做 reward/log 的 .item() 同步。
    - success / lift height / per-object totals 在 CUDA 上累计。
    - 最终 print / Python metrics 属于输出边界，才允许读取 CUDA scalar。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .actions import (
    TEACHER_ACTION_DIM,
    TEACHER_ARM_ACTION_DIM,
    TEACHER_HAND_ACTION_DIM,
)
from .object_set import (
    NEW_TRAINING_SET_ROOT,
    TEACHER_OBJECT_NAMES,
)
from .observations import TEACHER_OBSERVATION_SPEC
from .pregrasp import compute_fr3_palm_kinematics
from .visualization import TeacherAffordanceVisualizer


# =============================================================================
# ShapeNet quantitative evaluation set
# =============================================================================
SHAPENET_30_ROOT = Path(
    "/home/windsky/project/Grasp/assets/shapenet-30obj"
)

SHAPENET_30_OBJECT_NAMES: tuple[str, ...] = (
    "Bear_34",
    "Black_mug",
    "Blue_camera",
    "Blue_teapot",
    "Camera_brown",
    "Camera_yellow",
    "CellPhone_4e",
    "Donut",
    "DrinkBottle_blue_1ef",
    "Gun",
    "Hammer_40",
    "Knife",
    "Mug_8b_red",
    "Mug_gray_d7",
    "Mug_yellow_18",
    "Pan_gray",
    "Plate_gold_69",
    "Purse_brown_d58",
    "Purse_red_f58",
    "Red_bottle",
    "Red_chair",
    "Red_scissor",
    "Stapler_gray_d9",
    "Teapot_blue_high_3d2",
    "Teapot_brown",
    "Vase_red_a1",
    "Watch_9f",
    "WineGlass_gray_9d",
    "Wine_glass2_1_blue",
    "Wine_glass_body_blue",
)


@dataclass(frozen=True)
class TeacherEvaluationMode:
    """一种 Teacher evaluation protocol。"""

    name: str
    dataset_root: Path
    object_names: tuple[str, ...]

    # policy grasp 阶段步数。
    grasp_steps: int

    # 闭手后向上 lift 的步数。
    lift_steps: int

    default_episodes: int

    # quantitative ShapeNet 使用预先保存的 stable state。
    use_stable_states: bool


@dataclass(frozen=True)
class TeacherObjectSelection:
    """一次评估实际选择的 object subset。"""

    mode: TeacherEvaluationMode
    names: tuple[str, ...]
    top_usd_paths: tuple[str, ...]
    bottom_usd_paths: tuple[str, ...]
    stable_state_paths: tuple[str, ...]


@dataclass(frozen=True)
class TeacherEvaluationEpisode:
    """一个 evaluation episode 的结果。

    success / interrupted / lift_height 保持为 CUDA Tensor。

    mean_grasp_reward / mean_grasp_logs 是最终报告值；
    grasp loop 内部不会逐 step 做 GPU -> CPU 同步。
    """

    success: torch.Tensor
    interrupted: torch.Tensor
    lift_height: torch.Tensor
    mean_grasp_reward: float
    mean_grasp_logs: dict[str, float]


# =============================================================================
# Evaluation protocol
# =============================================================================
VISUAL_MODE = TeacherEvaluationMode(
    name="visual",
    dataset_root=NEW_TRAINING_SET_ROOT,
    object_names=TEACHER_OBJECT_NAMES,
    grasp_steps=70,
    lift_steps=30,
    default_episodes=10,
    use_stable_states=False,
)

QUANTITATIVE_MODE = TeacherEvaluationMode(
    name="quantitative",
    dataset_root=SHAPENET_30_ROOT,
    object_names=SHAPENET_30_OBJECT_NAMES,
    grasp_steps=100,
    lift_steps=100,
    default_episodes=5,
    use_stable_states=True,
)

TEACHER_EVALUATION_MODES = {
    VISUAL_MODE.name: VISUAL_MODE,
    QUANTITATIVE_MODE.name: QUANTITATIVE_MODE,
}


def resolve_teacher_object_selection(
    mode_name: str,
    value: str,
) -> TeacherObjectSelection:
    """把 CLI 的 object selection 转成静态 USD path 配置。

    这里只处理 Python 字符串 / Path，
    不属于运行期 RL 数值数据。
    """

    mode = TEACHER_EVALUATION_MODES[
        mode_name
    ]

    if value.strip() == "all":
        names = mode.object_names
    else:
        names = tuple(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )

    # 保留原 evaluation contract。
    unknown_names = tuple(
        name
        for name in names
        if name not in mode.object_names
    )

    if (
        not names
        or unknown_names
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "Teacher object selection must contain unique names "
            f"from mode={mode.name}: names={names}, "
            f"unknown={unknown_names}"
        )

    top_usd_paths = tuple(
        str(
            mode.dataset_root
            / name
            / f"{name}_top.usd"
        )
        for name in names
    )

    bottom_usd_paths = tuple(
        str(
            mode.dataset_root
            / name
            / f"{name}_bottom.usd"
        )
        for name in names
    )

    if mode.use_stable_states:
        stable_state_paths = tuple(
            str(
                mode.dataset_root
                / name
                / f"{name}.npy"
            )
            for name in names
        )
    else:
        stable_state_paths = ()

    return TeacherObjectSelection(
        mode=mode,
        names=names,
        top_usd_paths=top_usd_paths,
        bottom_usd_paths=bottom_usd_paths,
        stable_state_paths=stable_state_paths,
    )


def configure_teacher_evaluation_objects(
    env_cfg: Any,
    selection: TeacherObjectSelection,
    repeats_per_object: int,
) -> tuple[str, ...]:
    """把评估 object subset 写入 env config。

    这里仍然只是静态配置：
        object names
        USD paths
        stable-state paths
        num_envs

    真正 object index Tensor 会由 affordance_data.py 直接在 CUDA 建立。
    """

    env_cfg.asset.dataset_root = str(
        selection.mode.dataset_root
    )

    env_cfg.asset.object_names = (
        selection.names
    )

    env_cfg.asset.object_top_usd_paths = (
        selection.top_usd_paths
    )

    env_cfg.asset.object_bottom_usd_paths = (
        selection.bottom_usd_paths
    )

    # 每种 selected object 各保留一个 weighted entry。
    env_cfg.asset.weighted_object_indices = tuple(
        range(
            len(selection.names)
        )
    )

    env_cfg.asset.stable_state_paths = (
        selection.stable_state_paths
    )

    env_cfg.object.spawn.top_usd_paths = list(
        selection.top_usd_paths
    )

    env_cfg.object.spawn.bottom_usd_paths = list(
        selection.bottom_usd_paths
    )

    env_cfg.scene.num_envs = (
        len(selection.names)
        * repeats_per_object
    )

    # evaluation 使用 uniform workspace sampling。
    env_cfg.reset.non_uniform_sampling = False

    # 环境对象顺序与 affordance_data 的 weighted-index repeat 顺序一致：
    #
    #     object0, object1, ..., objectN,
    #     object0, object1, ..., objectN, ...
    return tuple(
        selection.names[
            index % len(selection.names)
        ]
        for index in range(
            env_cfg.scene.num_envs
        )
    )


def get_teacher_env_object_names(
    env: Any,
) -> tuple[str, ...]:
    """从静态 env 配置恢复每个 environment 的 object name。

    旧实现：
        旧版 CPU object-index 列表转换

    这要求 affordance_data 额外维护 CPU object-index Tensor，
    与当前 GPU-only affordance_data API 冲突。

    新实现直接使用：
        object_names              Python metadata
        weighted_object_indices  Python metadata
        num_envs                 config metadata

    因此不需要读取任何 CUDA 数值。
    """

    object_names = (
        env.affordance_data.object_names
    )

    weighted_indices = (
        env.cfg.asset.weighted_object_indices
    )

    weighted_count = len(
        weighted_indices
    )

    return tuple(
        object_names[
            weighted_indices[
                env_index
                % weighted_count
            ]
        ]
        for env_index in range(
            env.num_envs
        )
    )


def assert_teacher_evaluation_contract(
    env: Any,
) -> None:
    """保留原 evaluation entry point 的环境契约检查。"""

    if type(env).__module__ != (
        "Grasp.tasks.direct.grasp.teacher.env"
    ):
        raise RuntimeError(
            "Evaluation requires the standalone Teacher environment"
        )

    if (
        env.obs_spec
        != TEACHER_OBSERVATION_SPEC
    ):
        raise RuntimeError(
            "Teacher evaluation observation spec mismatch"
        )

    if (
        env.cfg.action_space
        != TEACHER_ACTION_DIM
    ):
        raise RuntimeError(
            "Teacher evaluation action dimension must be 13"
        )

    if (
        env.arm_joint_ids.numel()
        != TEACHER_ARM_ACTION_DIM
    ):
        raise RuntimeError(
            "Teacher evaluation requires seven FR3 joints"
        )

    if (
        env.active_hand_joint_ids.numel()
        != TEACHER_HAND_ACTION_DIM
    ):
        raise RuntimeError(
            "Teacher evaluation requires six active hand joints"
        )


@torch.no_grad()
def compute_damped_least_squares_delta(
    jacobian: torch.Tensor,
    target_twist: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    """计算 lift controller 的 DLS joint delta。

    dq =
        J^T
        (J J^T + lambda^2 I)^-1
        dx

    与 pregrasp.py 一样，不显式求 inverse；
    也不每次创建 6x6 identity Tensor，
    而是直接给 task matrix 的 diagonal 加 lambda^2。
    """

    jacobian_transpose = (
        jacobian.transpose(
            1,
            2,
        )
    )

    damped_task_matrix = (
        jacobian
        @ jacobian_transpose
    )

    damped_task_matrix.diagonal(
        dim1=-2,
        dim2=-1,
    ).add_(
        damping * damping
    )

    solved_twist = torch.linalg.solve(
        damped_task_matrix,
        target_twist.unsqueeze(-1),
    )

    return (
        jacobian_transpose
        @ solved_twist
    ).squeeze(-1)


@torch.no_grad()
def capture_closed_hand_target(
    env: Any,
) -> torch.Tensor:
    """保存 grasp 结束时 6 个 Inspire active hand joint target。

    lift 阶段继续追踪这份闭手 target，
    防止 lift controller 把手重新张开。
    """

    return env.current_joint_target[
        :,
        env.active_hand_joint_ids,
    ].clone()


@torch.no_grad()
def build_multistep_lift_action(
    env: Any,
    closed_hand_target: torch.Tensor,
    lift_delta_z: float,
    damping: float,
) -> torch.Tensor:
    """构造一个 lift step 的 13D residual action。

    关键修正：
        旧实现使用 PhysX 的 fr3_link8 Jacobian。

        但当前 pregrasp / observation 已统一到：
            Inspire base_link + palm_offset
        定义的真实 palm center。

        因此这里也使用 compute_fr3_palm_kinematics()
        得到真正 palm center 的 6x7 Jacobian。

    目标：
        palm 在世界 z 方向上移 lift_delta_z，
        orientation delta = 0。

    hand：
        始终追踪 grasp 结束时保存的 closed_hand_target。
    """

    # 当前 FR3 7D qpos，已经位于 CUDA。
    arm_qpos = env.robot.data.joint_pos[
        :,
        env.arm_joint_ids,
    ]

    # 与 pregrasp IK 使用同一个 palm frame 定义：
    #
    # fr3_link8
    #   -> L_flange
    #   -> wrist
    #   -> base_link
    #   -> palm_offset
    (
        _,
        _,
        palm_jacobian,
    ) = compute_fr3_palm_kinematics(
        arm_qpos=arm_qpos,
        robot_base_positions_world=(
            env.scene.env_origins
        ),
        palm_offset_body=(
            env.palm_offset_local_b
        ),
    )

    # [B,6]：
    # xyz delta + axis-angle delta。
    target_twist = torch.zeros(
        (
            env.num_envs,
            6,
        ),
        dtype=palm_jacobian.dtype,
        device=env.device,
    )

    # 只要求 world z 向上移动。
    target_twist[:, 2] = (
        lift_delta_z
    )

    arm_joint_delta = (
        compute_damped_least_squares_delta(
            jacobian=palm_jacobian,
            target_twist=target_twist,
            damping=damping,
        )
    )

    # env action 是 residual action：
    #
    #     target_delta
    #       =
    #     action
    #       *
    #     arm_residual_scale
    arm_action = (
        arm_joint_delta
        / env.cfg.action.arm_residual_scale
    )

    # -------------------------------------------------------------
    # Inspire hand：
    # 继续追踪 grasp 结束时保存的 closed target。
    # -------------------------------------------------------------
    current_hand_qpos = (
        env.robot.data.joint_pos[
            :,
            env.active_hand_joint_ids,
        ]
    )

    hand_action = (
        closed_hand_target
        - current_hand_qpos
    ) / env.cfg.action.hand_residual_scale

    # FR3 7 + Inspire 6 = 13。
    return torch.cat(
        (
            arm_action,
            hand_action,
        ),
        dim=-1,
    )


@torch.no_grad()
def update_teacher_affordance_visualization(
    env: Any,
    visualizer: (
        TeacherAffordanceVisualizer
        | None
    ),
) -> None:
    """更新 affordance visualization。

    关键接口修正：
        当前 env._compute_teacher_observation_features()
        的参数是：
            commit_wrist_history: bool

        旧代码却传入一组 environment ids，
        与当前 env.py 接口不匹配。

    visualization 只读取当前 feature，
    不应该提交 wrist Euler history。
    """

    if visualizer is None:
        return

    features = (
        env._compute_teacher_observation_features(
            commit_wrist_history=False,
        )
    )

    visualizer.update(
        env,
        features,
    )


@torch.no_grad()
def run_teacher_evaluation_episode(
    env: Any,
    actor: Any,
    grasp_steps: int,
    lift_steps: int,
    lift_delta_z: float,
    damping: float,
    visualizer: (
        TeacherAffordanceVisualizer
        | None
    ) = None,
) -> TeacherEvaluationEpisode:
    """运行一次完整 Teacher grasp + lift evaluation episode。

    grasp 阶段：
        policy noiseless action

    lift 阶段：
        保持 hand closed
        +
        palm Jacobian DLS 向上移动

    性能修正：
        reward / log 在整个 grasp loop 内都用 CUDA scalar 累加。
        不再每个 step 调 .item() 强制 GPU 同步。
    """

    observation_dict, _ = env.reset()

    update_teacher_affordance_visualization(
        env,
        visualizer,
    )

    observation = (
        observation_dict["policy"]
    )

    initial_object_z = (
        env.object_initial_root_state[
            :,
            2,
        ].clone()
    )

    interrupted = torch.zeros(
        (env.num_envs,),
        dtype=torch.bool,
        device=env.device,
    )

    # -------------------------------------------------------------
    # GPU scalar accumulators。
    # -------------------------------------------------------------
    reward_sum = torch.zeros(
        (),
        dtype=observation.dtype,
        device=env.device,
    )

    log_sums: dict[
        str,
        torch.Tensor,
    ] = {}

    # =================================================================
    # Phase 1: Teacher grasp
    # =================================================================
    with torch.inference_mode():
        for _ in range(
            grasp_steps
        ):
            action = (
                actor.noiseless_action(
                    observation
                )
            )

            (
                next_observation_dict,
                reward,
                terminated,
                truncated,
                extras,
            ) = env.step(
                action
            )

            update_teacher_affordance_visualization(
                env,
                visualizer,
            )

            interrupted.logical_or_(
                terminated
                | truncated
            )

            # 只在 CUDA scalar 上累加。
            reward_sum.add_(
                reward.mean()
            )

            for (
                name,
                value,
            ) in extras["log"].items():
                detached_value = (
                    value.detach()
                )

                if name not in log_sums:
                    log_sums[name] = (
                        torch.zeros_like(
                            detached_value
                        )
                    )

                log_sums[name].add_(
                    detached_value
                )

            observation = (
                next_observation_dict[
                    "policy"
                ]
            )

        # =============================================================
        # Phase 2: close-hand lift
        # =============================================================
        closed_hand_target = (
            capture_closed_hand_target(
                env
            )
        )

        for _ in range(
            lift_steps
        ):
            lift_action = (
                build_multistep_lift_action(
                    env=env,
                    closed_hand_target=(
                        closed_hand_target
                    ),
                    lift_delta_z=(
                        lift_delta_z
                    ),
                    damping=damping,
                )
            )

            (
                _,
                _,
                terminated,
                truncated,
                _,
            ) = env.step(
                lift_action
            )

            update_teacher_affordance_visualization(
                env,
                visualizer,
            )

            interrupted.logical_or_(
                terminated
                | truncated
            )

    # =================================================================
    # Episode result
    # =================================================================
    lift_height = (
        env.object.data.root_pos_w[
            :,
            2,
        ]
        - initial_object_z
    )

    success = (
        (
            lift_height
            > env.cfg.reward.lift_success_height
        )
        & ~interrupted
    )

    # 最终报告边界才读取 CUDA scalar。
    mean_grasp_reward = float(
        (
            reward_sum
            / grasp_steps
        ).item()
    )

    mean_grasp_logs = {
        name: float(
            (
                value
                / grasp_steps
            ).item()
        )
        for name, value in (
            log_sums.items()
        )
    }

    return TeacherEvaluationEpisode(
        success=success,
        interrupted=interrupted,
        lift_height=lift_height,
        mean_grasp_reward=(
            mean_grasp_reward
        ),
        mean_grasp_logs=(
            mean_grasp_logs
        ),
    )


class TeacherEvaluationTotals:
    """在 GPU 上累计整个 evaluation 的统计量。

    object name 是 Python 静态 metadata；
    attempts / successes / lift-height sums 是 CUDA Tensor。

    只有 flat_metrics() / print 阶段才转成 Python float。
    """

    def __init__(
        self,
        object_names: tuple[str, ...],
        device: torch.device | str = "cuda:0",
    ) -> None:
        self.object_names = tuple(
            dict.fromkeys(
                object_names
            )
        )

        self.object_name_to_index = {
            name: index
            for index, name in enumerate(
                self.object_names
            )
        }

        self.device = torch.device(
            device
        )

        object_count = len(
            self.object_names
        )

        # 全局 totals。
        self.attempts = torch.zeros(
            (),
            dtype=torch.int64,
            device=self.device,
        )

        self.successes = torch.zeros(
            (),
            dtype=torch.int64,
            device=self.device,
        )

        self.lift_height_sum = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )

        # 每个 object totals。
        self.object_attempts_tensor = (
            torch.zeros(
                (object_count,),
                dtype=torch.int64,
                device=self.device,
            )
        )

        self.object_successes_tensor = (
            torch.zeros_like(
                self.object_attempts_tensor
            )
        )

        self.object_lift_height_sum_tensor = (
            torch.zeros(
                (object_count,),
                dtype=torch.float32,
                device=self.device,
            )
        )

    @torch.no_grad()
    def add_episode(
        self,
        env_object_names: tuple[str, ...],
        episode: TeacherEvaluationEpisode,
    ) -> None:
        """把一个 episode 的所有环境统计量直接在 GPU 聚合。

        旧实现：
            GPU episode 结果转成 CPU/Python 列表
            Python for-loop 逐 env 累加

        新实现：
            Python object names
                -> CUDA object-index Tensor
            success / lift-height CUDA
                -> torch.bincount()
            per-object totals CUDA
        """

        # env_object_names 是静态字符串 metadata。
        # 第一份数值 index Tensor 直接建立在 CUDA。
        env_object_indices = torch.tensor(
            [
                self.object_name_to_index[
                    name
                ]
                for name in env_object_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        success_int = (
            episode.success.to(
                dtype=torch.int64,
                device=self.device,
            )
        )

        lift_height = (
            episode.lift_height.to(
                dtype=torch.float32,
                device=self.device,
            )
        )

        env_count = (
            episode.success.numel()
        )

        self.attempts.add_(
            env_count
        )

        self.successes.add_(
            success_int.sum()
        )

        self.lift_height_sum.add_(
            lift_height.sum()
        )

        object_count = len(
            self.object_names
        )

        episode_object_attempts = (
            torch.bincount(
                env_object_indices,
                minlength=object_count,
            )
        )

        episode_object_successes = (
            torch.bincount(
                env_object_indices,
                weights=(
                    success_int.to(
                        torch.float32
                    )
                ),
                minlength=object_count,
            ).to(
                torch.int64
            )
        )

        episode_object_lift_sum = (
            torch.bincount(
                env_object_indices,
                weights=lift_height,
                minlength=object_count,
            )
        )

        self.object_attempts_tensor.add_(
            episode_object_attempts
        )

        self.object_successes_tensor.add_(
            episode_object_successes
        )

        self.object_lift_height_sum_tensor.add_(
            episode_object_lift_sum
        )

    @property
    def failures(
        self,
    ) -> torch.Tensor:
        return (
            self.attempts
            - self.successes
        )

    def flat_metrics(
        self,
    ) -> dict[str, float]:
        """最终报告阶段把 GPU totals 转成 Python metrics。"""

        attempts = float(
            self.attempts.item()
        )

        successes = float(
            self.successes.item()
        )

        failures = float(
            self.failures.item()
        )

        lift_height_sum = float(
            self.lift_height_sum.item()
        )

        metrics = {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "success_rate": (
                successes
                / attempts
            ),
            "failure_rate": (
                failures
                / attempts
            ),
            "mean_lift_height": (
                lift_height_sum
                / attempts
            ),
        }

        for (
            object_index,
            name,
        ) in enumerate(
            self.object_names
        ):
            object_attempts = float(
                self.object_attempts_tensor[
                    object_index
                ].item()
            )

            object_successes = float(
                self.object_successes_tensor[
                    object_index
                ].item()
            )

            object_failures = (
                object_attempts
                - object_successes
            )

            object_lift_sum = float(
                self.object_lift_height_sum_tensor[
                    object_index
                ].item()
            )

            prefix = (
                f"object/{name}"
            )

            metrics[
                f"{prefix}/attempts"
            ] = object_attempts

            metrics[
                f"{prefix}/successes"
            ] = object_successes

            metrics[
                f"{prefix}/failures"
            ] = object_failures

            metrics[
                f"{prefix}/success_rate"
            ] = (
                object_successes
                / object_attempts
            )

            metrics[
                f"{prefix}/failure_rate"
            ] = (
                object_failures
                / object_attempts
            )

            metrics[
                f"{prefix}/mean_lift_height"
            ] = (
                object_lift_sum
                / object_attempts
            )

        return metrics


def print_teacher_evaluation_summary(
    mode_name: str,
    totals: TeacherEvaluationTotals,
) -> None:
    """打印最终 evaluation summary。

    print 是最终输出边界，因此这里使用 Python float 是正常的；
    这些值不会再送回 GPU 参与策略或物理计算。
    """

    metrics = totals.flat_metrics()

    print(
        f"teacher_evaluation_mode={mode_name}"
    )

    print(
        f"attempts={int(metrics['attempts'])}"
    )

    print(
        f"successes={int(metrics['successes'])}"
    )

    print(
        f"failures={int(metrics['failures'])}"
    )

    print(
        "success_rate="
        f"{metrics['success_rate']:.6f}"
    )

    print(
        "failure_rate="
        f"{metrics['failure_rate']:.6f}"
    )

    print(
        "mean_lift_height="
        f"{metrics['mean_lift_height']:.6f}"
    )

    for name in totals.object_names:
        prefix = (
            f"object/{name}"
        )

        print(
            f"object={name} "
            f"attempts="
            f"{int(metrics[f'{prefix}/attempts'])} "
            f"successes="
            f"{int(metrics[f'{prefix}/successes'])} "
            f"failures="
            f"{int(metrics[f'{prefix}/failures'])} "
            "success_rate="
            f"{metrics[f'{prefix}/success_rate']:.6f} "
            "failure_rate="
            f"{metrics[f'{prefix}/failure_rate']:.6f} "
            "mean_lift_height="
            f"{metrics[f'{prefix}/mean_lift_height']:.6f}"
        )


__all__ = [
    "QUANTITATIVE_MODE",
    "SHAPENET_30_OBJECT_NAMES",
    "SHAPENET_30_ROOT",
    "TEACHER_EVALUATION_MODES",
    "TeacherEvaluationEpisode",
    "TeacherEvaluationMode",
    "TeacherEvaluationTotals",
    "TeacherObjectSelection",
    "VISUAL_MODE",
    "assert_teacher_evaluation_contract",
    "build_multistep_lift_action",
    "capture_closed_hand_target",
    "compute_damped_least_squares_delta",
    "configure_teacher_evaluation_objects",
    "get_teacher_env_object_names",
    "print_teacher_evaluation_summary",
    "resolve_teacher_object_selection",
    "run_teacher_evaluation_episode",
    "update_teacher_affordance_visualization",
]
