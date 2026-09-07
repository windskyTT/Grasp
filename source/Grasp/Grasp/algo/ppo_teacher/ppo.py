"""Teacher PPO：CUDA-only 运行期数值版本。

保持 RobustDexGrasp Teacher PPO 核心语义：
    clip_param        = 0.2
    gamma             = cfg value
    lambda            = 0.95
    adaptive KL       = desired_kl
    ordered mini-batch
    clipped value loss
    Adam optimizer

GPU 约定：
    - RolloutStorage 全部位于 CUDA。
    - Actor / Critic 参数位于 CUDA。
    - loss / entropy / KL / advantage / return 全部在 CUDA。
    - adaptive learning-rate 使用 CUDA scalar Tensor。
    - KL 不通过 Python ``if tensor`` 做每 mini-batch GPU 同步。
    - 只有 TensorBoard / checkpoint / terminal 输出边界读取 Python float。

项目要求“不进行防御性编程”：
    原代码每个 mini-batch 遍历所有 Actor weight 并检查 NaN，
    会反复触发 CUDA -> Python bool 同步。
    该扫描已删除，不再在 PPO 数值主循环中做防御式参数检查。
"""

from __future__ import annotations

from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .storage import RolloutStorage


class PPO:
    """Teacher Proximal Policy Optimization。"""

    def __init__(
        self,
        actor,
        critic,
        num_envs,
        num_transitions_per_env,
        num_learning_epochs,
        num_mini_batches,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        learning_rate=5.0e-4,
        max_grad_norm=0.5,
        learning_rate_schedule="adaptive",
        desired_kl=0.01,
        use_clipped_value_loss=True,
        log_dir="run",
        device="cuda:0",
        shuffle_batch=True,
    ):
        self.device = torch.device(
            device
        )

        self.actor = actor
        self.critic = critic

        # =============================================================
        # CUDA rollout storage
        # =============================================================
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor.obs_shape,
            critic.obs_shape,
            actor.action_shape,
            self.device,
        )

        if shuffle_batch:
            self.batch_sampler = (
                self.storage
                .mini_batch_generator_shuffle
            )
        else:
            self.batch_sampler = (
                self.storage
                .mini_batch_generator_inorder
            )

        # =============================================================
        # Adaptive LR CUDA scalar
        # =============================================================
        # 运行期 learning rate 从创建第一刻就在 CUDA。
        #
        # Adam 使用 capturable=True 后允许 CUDA Tensor LR。
        # adaptive KL 只原位修改这一份 Tensor，
        # optimizer param_group 始终引用它。
        self._learning_rate_tensor = (
            torch.tensor(
                learning_rate,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.optimizer = optim.Adam(
            [
                *self.actor.parameters(),
                *self.critic.parameters(),
            ],
            lr=self._learning_rate_tensor,
            capturable=True,
        )

        # =============================================================
        # PPO parameters
        # =============================================================
        self.num_transitions_per_env = (
            num_transitions_per_env
        )
        self.num_envs = num_envs

        self.clip_param = clip_param
        self.num_learning_epochs = (
            num_learning_epochs
        )
        self.num_mini_batches = (
            num_mini_batches
        )
        self.value_loss_coef = (
            value_loss_coef
        )
        self.entropy_coef = (
            entropy_coef
        )
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = (
            max_grad_norm
        )
        self.use_clipped_value_loss = (
            use_clipped_value_loss
        )

        self.desired_kl = desired_kl
        self.schedule = (
            learning_rate_schedule
        )

        # =============================================================
        # Logging：host-side 输出边界
        # =============================================================
        self.log_dir = os.path.join(
            log_dir,
            datetime.now().strftime(
                "%b%d_%H-%M-%S"
            ),
        )

        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            flush_secs=10,
        )

        self.tot_timesteps = 0
        self.tot_time = 0

        # act() 与 step() 之间保存当前 transition。
        self.actions: (
            torch.Tensor
            | None
        ) = None

        self.actions_log_prob: (
            torch.Tensor
            | None
        ) = None

        self.actor_obs: (
            torch.Tensor
            | None
        ) = None

    # =================================================================
    # Learning-rate public interface
    # =================================================================
    @property
    def learning_rate(
        self,
    ) -> float:
        """只在日志 / checkpoint 输出边界读取 Python float。"""

        return float(
            self._learning_rate_tensor.item()
        )

    @learning_rate.setter
    def learning_rate(
        self,
        value: float,
    ) -> None:
        """Checkpoint resume 时把静态 host 标量写入 CUDA LR Tensor。

        随后 optimizer 继续引用同一份 CUDA scalar。
        """

        with torch.no_grad():
            self._learning_rate_tensor.fill_(
                value
            )

        # optimizer.load_state_dict() 可能恢复 param-group metadata；
        # resume setter 后重新指向当前 CUDA LR Tensor。
        for param_group in (
            self.optimizer.param_groups
        ):
            param_group["lr"] = (
                self._learning_rate_tensor
            )

    # =================================================================
    # Rollout
    # =================================================================
    @torch.no_grad()
    def act(
        self,
        actor_obs,
        student_driven_ratio=0,
    ):
        """根据当前 observation 在 CUDA 上采样 action。"""

        self.actor_obs = actor_obs

        (
            self.actions,
            self.actions_log_prob,
        ) = self.actor.sample(
            actor_obs
        )

        return self.actions

    @torch.no_grad()
    def step(
        self,
        value_obs,
        rews,
        dones,
    ) -> None:
        """把一步 CUDA transition 写入 rollout storage。

        train_teacher.py 必须传：
            dones = terminated | truncated
        """

        self.storage.add_transitions(
            self.actor_obs,
            value_obs,
            self.actions,
            self.actor.action_mean,
            self.actor.distribution.std,
            rews,
            dones,
            self.actions_log_prob,
        )

    # =================================================================
    # PPO update
    # =================================================================
    def update(
        self,
        actor_obs,
        value_obs,
        log_this_iteration,
        update,
    ):
        """计算 GAE 并执行 PPO gradient update。"""

        # 最后 observation 的 bootstrap value。
        last_values = (
            self.critic.predict(
                value_obs
            )
        )

        self.storage.compute_returns(
            last_values,
            self.critic,
            self.gamma,
            self.lam,
        )

        (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy,
        ) = self._train_step(
            log_this_iteration
        )

        # PPO 是 on-policy：
        # 当前 rollout 使用一次后，下个 iteration 从 slot 0 重新覆盖。
        self.storage.clear()

        if log_this_iteration:
            self.log(
                update=update,
                mean_value_loss=(
                    mean_value_loss
                ),
                mean_surrogate_loss=(
                    mean_surrogate_loss
                ),
            )

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy,
        )

    def log(
        self,
        update: int,
        mean_value_loss: float,
        mean_surrogate_loss: float,
    ) -> None:
        """TensorBoard 输出边界。

        这里的 scalar 不会再回到 GPU 参与 PPO 数值计算。
        """

        self.tot_timesteps += (
            self.num_transitions_per_env
            * self.num_envs
        )

        mean_std = float(
            self.actor.distribution
            .std
            .mean()
            .item()
        )

        self.writer.add_scalar(
            "PPO/value_function",
            mean_value_loss,
            update,
        )

        self.writer.add_scalar(
            "PPO/surrogate",
            mean_surrogate_loss,
            update,
        )

        self.writer.add_scalar(
            "PPO/mean_noise_std",
            mean_std,
            update,
        )

        self.writer.add_scalar(
            "PPO/learning_rate",
            self.learning_rate,
            update,
        )

    # =================================================================
    # Core PPO gradient computation
    # =================================================================
    def _train_step(
        self,
        log_this_iteration,
    ):
        """对当前 CUDA rollout 执行多 epoch / mini-batch PPO update。"""

        mean_value_loss = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )

        mean_surrogate_loss = (
            torch.zeros(
                (),
                dtype=torch.float32,
                device=self.device,
            )
        )

        mean_entropy = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )

        for _ in range(
            self.num_learning_epochs
        ):
            for (
                actor_obs_batch,
                critic_obs_batch,
                actions_batch,
                old_sigma_batch,
                old_mu_batch,
                current_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
            ) in self.batch_sampler(
                self.num_mini_batches
            ):
                # -----------------------------------------------------
                # 当前策略重新评估旧 rollout。
                # -----------------------------------------------------
                (
                    actions_log_prob_batch,
                    entropy_batch,
                ) = self.actor.evaluate(
                    actor_obs_batch,
                    actions_batch,
                )

                value_batch = (
                    self.critic.evaluate(
                        critic_obs_batch
                    )
                )

                mu_batch = (
                    self.actor.action_mean
                )

                sigma_batch = (
                    self.actor.distribution.std
                )

                # =====================================================
                # Adaptive KL learning rate：全部 CUDA
                # =====================================================
                if (
                    self.desired_kl is not None
                    and self.schedule
                    == "adaptive"
                ):
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(
                                sigma_batch
                                / old_sigma_batch
                                + 1.0e-5
                            )
                            + (
                                (
                                    old_sigma_batch.square()
                                    + (
                                        old_mu_batch
                                        - mu_batch
                                    ).square()
                                )
                                / (
                                    2.0
                                    * sigma_batch.square()
                                )
                            )
                            - 0.5,
                            dim=-1,
                        )

                        kl_mean = kl.mean()

                        current_lr = (
                            self._learning_rate_tensor
                        )

                        reduced_lr = torch.clamp(
                            current_lr / 1.2,
                            min=1.0e-5,
                        )

                        increased_lr = torch.clamp(
                            current_lr * 1.2,
                            max=1.0e-2,
                        )

                        increase_condition = (
                            (
                                kl_mean
                                < (
                                    self.desired_kl
                                    / 2.0
                                )
                            )
                            & (
                                kl_mean
                                > 0.0
                            )
                        )

                        new_lr = torch.where(
                            kl_mean
                            > (
                                self.desired_kl
                                * 2.0
                            ),
                            reduced_lr,
                            torch.where(
                                increase_condition,
                                increased_lr,
                                current_lr,
                            ),
                        )

                        # 原位修改 optimizer 正在引用的 CUDA LR。
                        current_lr.copy_(
                            new_lr
                        )

                # =====================================================
                # PPO clipped surrogate loss
                # =====================================================
                ratio = torch.exp(
                    actions_log_prob_batch
                    - old_actions_log_prob_batch.squeeze(
                        -1
                    )
                )

                advantages = (
                    advantages_batch.squeeze(
                        -1
                    )
                )

                surrogate = (
                    -advantages
                    * ratio
                )

                surrogate_clipped = (
                    -advantages
                    * torch.clamp(
                        ratio,
                        1.0
                        - self.clip_param,
                        1.0
                        + self.clip_param,
                    )
                )

                surrogate_loss = (
                    torch.maximum(
                        surrogate,
                        surrogate_clipped,
                    ).mean()
                )

                # =====================================================
                # Value loss
                # =====================================================
                if self.use_clipped_value_loss:
                    value_clipped = (
                        current_values_batch
                        + (
                            value_batch
                            - current_values_batch
                        ).clamp(
                            -self.clip_param,
                            self.clip_param,
                        )
                    )

                    value_losses = (
                        value_batch
                        - returns_batch
                    ).square()

                    value_losses_clipped = (
                        value_clipped
                        - returns_batch
                    ).square()

                    value_loss = (
                        torch.maximum(
                            value_losses,
                            value_losses_clipped,
                        ).mean()
                    )
                else:
                    value_loss = (
                        returns_batch
                        - value_batch
                    ).square().mean()

                # =====================================================
                # Total PPO loss
                # =====================================================
                entropy = (
                    entropy_batch.mean()
                )

                loss = (
                    surrogate_loss
                    + (
                        self.value_loss_coef
                        * value_loss
                    )
                    - (
                        self.entropy_coef
                        * entropy
                    )
                )

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    [
                        *self.actor.parameters(),
                        *self.critic.parameters(),
                    ],
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # -----------------------------------------------------
                # 只累计 CUDA scalar。
                # 不在 mini-batch 内 .item()。
                # -----------------------------------------------------
                if log_this_iteration:
                    mean_value_loss.add_(
                        value_loss.detach()
                    )

                    mean_surrogate_loss.add_(
                        surrogate_loss.detach()
                    )

                    mean_entropy.add_(
                        entropy.detach()
                    )

        if log_this_iteration:
            num_updates = (
                self.num_learning_epochs
                * self.num_mini_batches
            )

            # 每个 PPO iteration 最终输出时才读取三个 scalar。
            mean_value_loss_value = float(
                (
                    mean_value_loss
                    / num_updates
                ).item()
            )

            mean_surrogate_loss_value = float(
                (
                    mean_surrogate_loss
                    / num_updates
                ).item()
            )

            mean_entropy_value = float(
                (
                    mean_entropy
                    / num_updates
                ).item()
            )

            return (
                mean_value_loss_value,
                mean_surrogate_loss_value,
                mean_entropy_value,
            )

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy,
        )

    def check_exploding_gradient(
        self,
    ) -> bool:
        """保留 train_teacher.py 现有调用接口。

        不再进行逐参数 NaN 防御扫描，因此不触发 mini-batch CUDA 同步。
        """

        return False


__all__ = [
    "PPO",
]
