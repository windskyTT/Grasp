"""Teacher PPO rollout storage：全 CUDA。

当前训练：
    num_envs = 88（默认）
    rollout  = 70
    obs      = 119
    action   = 13

所有 rollout buffer 从创建开始就在 env.device=cuda:0。

GAE：
    dones 必须由 train_teacher.py 传入：
        terminated | truncated

这样 70-step timeout 会正确切断 return / advantage，
不会把 reset 后的新 episode value 接到旧 episode。
"""

from __future__ import annotations

import torch


class RolloutStorage:
    """PPO on-policy rollout buffer。"""

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        device,
    ):
        self.device = torch.device(
            device
        )

        self.num_transitions_per_env = (
            num_transitions_per_env
        )
        self.num_envs = num_envs

        shape = (
            num_transitions_per_env,
            num_envs,
        )

        # =============================================================
        # 所有 buffer 直接创建在 CUDA。
        # =============================================================
        self.critic_obs = torch.zeros(
            (
                *shape,
                *critic_obs_shape,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.actor_obs = torch.zeros(
            (
                *shape,
                *actor_obs_shape,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.rewards = torch.zeros(
            (
                *shape,
                1,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.actions = torch.zeros(
            (
                *shape,
                *actions_shape,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.dones = torch.zeros(
            (
                *shape,
                1,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        self.actions_log_prob = (
            torch.zeros(
                (
                    *shape,
                    1,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.values = torch.zeros(
            (
                *shape,
                1,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.returns = torch.zeros(
            (
                *shape,
                1,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.advantages = torch.zeros(
            (
                *shape,
                1,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.mu = torch.zeros(
            (
                *shape,
                *actions_shape,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.sigma = torch.zeros(
            (
                *shape,
                *actions_shape,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.step = 0

    @torch.no_grad()
    def add_transitions(
        self,
        actor_obs,
        critic_obs,
        actions,
        mu,
        sigma,
        rewards,
        dones,
        actions_log_prob,
    ) -> None:
        """把一个并行环境 step 写入 CUDA rollout buffer。

        输入均来自：
            env CUDA
            Actor CUDA
            Gaussian sampler CUDA
        """

        self.critic_obs[
            self.step
        ].copy_(
            critic_obs
        )

        self.actor_obs[
            self.step
        ].copy_(
            actor_obs
        )

        self.actions[
            self.step
        ].copy_(
            actions
        )

        self.mu[
            self.step
        ].copy_(
            mu
        )

        self.sigma[
            self.step
        ].copy_(
            sigma
            .detach()
            .expand_as(actions)
        )

        self.rewards[
            self.step
        ].copy_(
            rewards.reshape(
                -1,
                1,
            )
        )

        self.dones[
            self.step
        ].copy_(
            dones.reshape(
                -1,
                1,
            )
        )

        self.actions_log_prob[
            self.step
        ].copy_(
            actions_log_prob.reshape(
                -1,
                1,
            )
        )

        self.step += 1

    def clear(
        self,
    ) -> None:
        """PPO update 后从第 0 个 rollout slot 重新覆盖。"""

        self.step = 0

    @torch.no_grad()
    def compute_returns(
        self,
        last_values,
        critic,
        gamma,
        lam,
    ) -> None:
        """在 CUDA 上计算 value、GAE advantage 和 return。

        对第 t 步：

            delta_t
              =
            r_t
              +
            gamma * (1-done_t) * V_{t+1}
              -
            V_t

            A_t
              =
            delta_t
              +
            gamma * lambda * (1-done_t) * A_{t+1}

        done=True 时：
            bootstrap 被切断。

        train_teacher.py 当前必须传：
            done = terminated | truncated
        """

        # Linear 层可以直接处理：
        #     [T, N, obs_dim]
        # 最后一维作为 feature dimension。
        self.values.copy_(
            critic.predict(
                self.critic_obs
            )
        )

        advantage = torch.zeros_like(
            last_values
        )

        for step in reversed(
            range(
                self.num_transitions_per_env
            )
        ):
            if (
                step
                == self.num_transitions_per_env - 1
            ):
                next_values = (
                    last_values
                )
            else:
                next_values = (
                    self.values[
                        step + 1
                    ]
                )

            next_is_not_terminal = (
                1.0
                - self.dones[
                    step
                ].float()
            )

            delta = (
                self.rewards[step]
                + (
                    next_is_not_terminal
                    * gamma
                    * next_values
                )
                - self.values[step]
            )

            advantage = (
                delta
                + (
                    next_is_not_terminal
                    * gamma
                    * lam
                    * advantage
                )
            )

            self.returns[
                step
            ].copy_(
                advantage
                + self.values[step]
            )

        # Advantage normalization 全部在 CUDA。
        self.advantages.copy_(
            self.returns
            - self.values
        )

        self.advantages.sub_(
            self.advantages.mean()
        ).div_(
            self.advantages.std(
                correction=0
            )
            + 1.0e-8
        )

    def _flatten_rollout(
        self,
    ):
        """[T,N,...] -> [T*N,...]，只创建 view。"""

        return (
            self.actor_obs.flatten(
                0,
                1,
            ),
            self.critic_obs.flatten(
                0,
                1,
            ),
            self.actions.flatten(
                0,
                1,
            ),
            self.sigma.flatten(
                0,
                1,
            ),
            self.mu.flatten(
                0,
                1,
            ),
            self.values.flatten(
                0,
                1,
            ),
            self.advantages.flatten(
                0,
                1,
            ),
            self.returns.flatten(
                0,
                1,
            ),
            self.actions_log_prob.flatten(
                0,
                1,
            ),
        )

    def mini_batch_generator_shuffle(
        self,
        num_mini_batches,
    ):
        """GPU shuffled mini-batch generator。

        Teacher 当前配置 shuffle_batch=False，
        因此正式训练使用下面的 inorder generator。
        """

        batch_size = (
            self.num_envs
            * self.num_transitions_per_env
        )

        mini_batch_size = (
            batch_size
            // num_mini_batches
        )

        indices = torch.randperm(
            batch_size,
            device=self.device,
        )

        (
            actor_obs,
            critic_obs,
            actions,
            sigma,
            mu,
            values,
            advantages,
            returns,
            actions_log_prob,
        ) = self._flatten_rollout()

        for batch_id in range(
            num_mini_batches
        ):
            start = (
                batch_id
                * mini_batch_size
            )
            end = (
                start
                + mini_batch_size
            )

            batch_indices = indices[
                start:end
            ]

            yield (
                actor_obs[
                    batch_indices
                ],
                critic_obs[
                    batch_indices
                ],
                actions[
                    batch_indices
                ],
                sigma[
                    batch_indices
                ],
                mu[
                    batch_indices
                ],
                values[
                    batch_indices
                ],
                advantages[
                    batch_indices
                ],
                returns[
                    batch_indices
                ],
                actions_log_prob[
                    batch_indices
                ],
            )

    def mini_batch_generator_inorder(
        self,
        num_mini_batches,
    ):
        """RobustDexGrasp Teacher 使用的 ordered mini-batch。"""

        batch_size = (
            self.num_envs
            * self.num_transitions_per_env
        )

        mini_batch_size = (
            batch_size
            // num_mini_batches
        )

        (
            actor_obs,
            critic_obs,
            actions,
            sigma,
            mu,
            values,
            advantages,
            returns,
            actions_log_prob,
        ) = self._flatten_rollout()

        for batch_id in range(
            num_mini_batches
        ):
            start = (
                batch_id
                * mini_batch_size
            )
            end = (
                start
                + mini_batch_size
            )

            yield (
                actor_obs[
                    start:end
                ],
                critic_obs[
                    start:end
                ],
                actions[
                    start:end
                ],
                sigma[
                    start:end
                ],
                mu[
                    start:end
                ],
                values[
                    start:end
                ],
                advantages[
                    start:end
                ],
                returns[
                    start:end
                ],
                actions_log_prob[
                    start:end
                ],
            )


__all__ = [
    "RolloutStorage",
]
