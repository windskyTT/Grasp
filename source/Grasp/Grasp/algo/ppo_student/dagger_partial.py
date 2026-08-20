from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .storage import RolloutStorage


class Dagger:
    def __init__(
        self,
        expert_policy,
        actor_student,
        critic_student,
        prop_latent_encoder,
        tobeEncode_dim,
        prop_latent_dim,
        aff_vec_dim,
        total_obs_dim,
        teacher_obs_dim,
        t_steps,
        num_envs,
        num_transitions_per_env,
        num_learning_epochs,
        num_mini_batches,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        learning_rate=5e-4,
        max_grad_norm=0.5,
        learning_rate_schedule="adaptive",
        desired_kl=0.01,
        use_clipped_value_loss=True,
        log_dir="run",
        device="cpu",
        shuffle_batch=True,
        update_mlp=False,
        ppo_ratio=0.5,
    ):
        self.expert_policy = expert_policy
        self.actor_student = actor_student
        self.critic_student = critic_student
        self.prop_latent_encoder = prop_latent_encoder

        self.tobeEncode_dim = tobeEncode_dim
        self.prop_latent_dim = prop_latent_dim
        self.aff_vec_dim = aff_vec_dim
        self.total_obs_dim = total_obs_dim
        self.teacher_obs_dim = teacher_obs_dim
        self.T = t_steps
        self.history_flat_dim = self.tobeEncode_dim * self.T
        self.student_actor_obs_dim = (
            self.tobeEncode_dim
            + self.prop_latent_dim
            + self.aff_vec_dim
        )

        if actor_student.obs_shape != [self.student_actor_obs_dim]:
            raise ValueError(
                "Student actor observation mismatch: "
                f"actor={actor_student.obs_shape}, "
                f"expected={[self.student_actor_obs_dim]}"
            )
        if critic_student.obs_shape != [self.teacher_obs_dim]:
            raise ValueError(
                "Student critic observation mismatch: "
                f"critic={critic_student.obs_shape}, "
                f"expected={[self.teacher_obs_dim]}"
            )
        if self.total_obs_dim != self.history_flat_dim + self.teacher_obs_dim:
            raise ValueError(
                "Student total observation mismatch: "
                f"total={self.total_obs_dim}, "
                f"history+teacher="
                f"{self.history_flat_dim + self.teacher_obs_dim}"
            )

        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_student.obs_shape,
            critic_student.obs_shape,
            actor_student.action_shape,
            total_obs_dim,
            prop_latent_dim,
            device,
        )

        self.update_mlp = update_mlp
        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        if self.update_mlp:
            self.optimizer = optim.Adam(
                [
                    *self.prop_latent_encoder.parameters(),
                    *self.actor_student.parameters(),
                    *self.critic_student.parameters(),
                ],
                lr=learning_rate,
            )
            self.ppo_ratio = ppo_ratio
        else:
            self.optimizer = optim.Adam(
                self.prop_latent_encoder.parameters(),
                lr=learning_rate,
            )
            self.ppo_ratio = 0.0

        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.loss_fn = nn.MSELoss()

        self.log_dir = os.path.join(
            log_dir,
            datetime.now().strftime("%b%d_%H-%M-%S"),
        )
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule

        self.actions = None
        self.teacher_actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def update_ppo_ratio(self, ppo_ratio):
        self.ppo_ratio = ppo_ratio

    def _student_actor_obs_numpy(
        self,
        total_obs: np.ndarray,
        student_aff: np.ndarray,
    ) -> np.ndarray:
        history = total_obs[:, : self.history_flat_dim]
        latest_history = history[:, -self.tobeEncode_dim :]
        student_latent = (
            self.prop_latent_encoder(
                torch.from_numpy(history).to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )
        return np.concatenate(
            [latest_history, student_latent, student_aff],
            axis=1,
        ).astype(np.float32, copy=False)

    def _student_actor_obs_torch(
        self,
        total_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = total_obs[:, : self.history_flat_dim]
        latest_history = history[:, -self.tobeEncode_dim :]
        student_latent = self.prop_latent_encoder(history)
        student_aff = total_obs[:, -self.aff_vec_dim :]
        student_actor_obs = torch.cat(
            [latest_history, student_latent, student_aff],
            dim=1,
        )
        return student_actor_obs, student_latent

    def act(
        self,
        total_obs,
        student_driven_ratio,
        student_aff,
    ):
        teacher_obs = total_obs[:, -self.teacher_obs_dim :]
        student_mlp_obs = self._student_actor_obs_numpy(
            total_obs,
            student_aff,
        )

        self.actor_obs = student_mlp_obs
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor_student.sample(
                torch.from_numpy(student_mlp_obs).to(self.device)
            )
            self.teacher_actions = (
                self.expert_policy.architecture(
                    torch.from_numpy(teacher_obs).to(self.device)
                )
                .cpu()
                .numpy()
            )

        if torch.rand(1).item() < student_driven_ratio:
            return self.actions, student_mlp_obs
        return self.teacher_actions, student_mlp_obs

    def step(self, total_obs, rews, dones, value_obs):
        self.storage.add_transitions(
            self.actor_obs,
            value_obs,
            self.actions,
            self.actor_student.action_mean,
            self.actor_student.distribution.std_np,
            rews,
            dones,
            self.actions_log_prob,
            total_obs,
        )

    def update(self, value_obs):
        last_values = self.critic_student.predict(
            torch.from_numpy(value_obs).to(self.device)
        )
        self.storage.compute_returns(
            last_values.to(self.device),
            self.critic_student,
            self.gamma,
            self.lam,
        )
        prop_mse, action_mse = self._train_step()
        self.storage.clear()
        return prop_mse, action_mse

    def _train_step(self):
        prop_mse_sum = 0.0
        action_mse_sum = 0.0
        loss_counter = 0

        for _ in range(self.num_learning_epochs):
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
                total_obs_batch,
            ) in self.batch_sampler(self.num_mini_batches):
                actions_log_prob_batch, entropy_batch = (
                    self.actor_student.evaluate(
                        actor_obs_batch,
                        actions_batch,
                    )
                )
                value_batch = self.critic_student.evaluate(
                    critic_obs_batch
                )
                mu_batch = self.actor_student.action_mean
                sigma_batch = self.actor_student.distribution.std

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(
                                sigma_batch / old_sigma_batch + 1.0e-5
                            )
                            + (
                                torch.square(old_sigma_batch)
                                + torch.square(old_mu_batch - mu_batch)
                            )
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            dim=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(
                                1.0e-5,
                                self.learning_rate / 1.2,
                            )
                        elif (
                            kl_mean < self.desired_kl / 2.0
                            and kl_mean > 0.0
                        ):
                            self.learning_rate = min(
                                1.0e-2,
                                self.learning_rate * 1.2,
                            )
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                ratio = torch.exp(
                    actions_log_prob_batch
                    - torch.squeeze(old_actions_log_prob_batch)
                )
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(
                    advantages_batch
                ) * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
                surrogate_loss = torch.max(
                    surrogate,
                    surrogate_clipped,
                ).mean()

                if self.use_clipped_value_loss:
                    value_clipped = current_values_batch + (
                        value_batch - current_values_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (
                        value_batch - returns_batch
                    ).pow(2)
                    value_losses_clipped = (
                        value_clipped - returns_batch
                    ).pow(2)
                    value_loss = torch.max(
                        value_losses,
                        value_losses_clipped,
                    ).mean()
                else:
                    value_loss = (
                        returns_batch - value_batch
                    ).pow(2).mean()

                ppo_loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                )

                student_actor_obs, student_latent = (
                    self._student_actor_obs_torch(total_obs_batch)
                )
                latent_target = critic_obs_batch[
                    :,
                    self.tobeEncode_dim :
                    self.tobeEncode_dim + self.prop_latent_dim,
                ]
                loss_prop = self.loss_fn(
                    student_latent,
                    latent_target,
                )

                student_action = (
                    self.actor_student.architecture.architecture(
                        student_actor_obs
                    )
                )
                with torch.no_grad():
                    teacher_action = self.expert_policy.architecture(
                        critic_obs_batch
                    )
                loss_action = (
                    self.loss_fn(student_action, teacher_action) * 0.1
                )

                loss = (
                    ppo_loss * self.ppo_ratio
                    + loss_prop
                    + loss_action * (1.0 - self.ppo_ratio)
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.update_mlp:
                    nn.utils.clip_grad_norm_(
                        [
                            *self.prop_latent_encoder.parameters(),
                            *self.actor_student.parameters(),
                            *self.critic_student.parameters(),
                        ],
                        self.max_grad_norm,
                    )
                else:
                    nn.utils.clip_grad_norm_(
                        self.prop_latent_encoder.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                prop_mse_sum += loss_prop.item()
                action_mse_sum += loss_action.item()
                loss_counter += 1

        return (
            prop_mse_sum / loss_counter,
            action_mse_sum / loss_counter,
        )

    def check_exploding_gradient(self):
        parameters = [
            *self.prop_latent_encoder.parameters(),
            *self.actor_student.parameters(),
            *self.critic_student.parameters(),
        ]
        return any(
            not torch.isfinite(parameter).all()
            for parameter in parameters
        )