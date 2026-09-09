from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from .storage import RolloutStorage

class PPO:
    def __init__(self, actor, critic, num_envs, num_transitions_per_env, num_learning_epochs=4, num_mini_batches=4, gamma=0.996, lam=0.95, device="cuda:0", learning_rate=5e-4, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.0, shuffle_batch=False, desired_kl=0.01):
        self.actor, self.critic = actor, critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)
        self.num_envs, self.transitions = num_envs, num_transitions_per_env
        self.num_learning_epochs, self.num_mini_batches = num_learning_epochs, num_mini_batches
        self.gamma, self.lam, self.clip_param = gamma, lam, clip_param
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.shuffle_batch = shuffle_batch
        self.optimizer = optim.Adam([*actor.parameters(), *critic.parameters()], lr=learning_rate)
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = "adaptive"
        self.tot_timesteps = 0
        self.is_exploding_gradient = False
        self.actions = None; self.actions_log_prob = None; self.actor_obs = None

    def act(self, observations):
        self.actor_obs = observations
        with torch.no_grad(): self.actions, self.actions_log_prob = self.actor.sample(observations)
        return self.actions

    def step(self, value_obs, rewards, dones):
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, self.actor.action_mean.detach(), self.actor.distribution.std.detach(), rewards, dones, self.actions_log_prob)

    def update(self, actor_obs, value_obs):
        last_values = self.critic.predict(value_obs)
        self.storage.compute_returns(last_values, self.critic, self.gamma, self.lam)
        for _ in range(self.num_learning_epochs):
            for actor_obs_b, critic_obs_b, actions_b, old_sigma_b, old_mu_b, current_values_b, advantages_b, returns_b, old_log_prob_b in self.storage.batches(self.num_mini_batches, self.shuffle_batch):
                log_prob, entropy = self.actor.evaluate(actor_obs_b, actions_b)
                value = self.critic.evaluate(critic_obs_b)
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.no_grad():
                        sigma = self.actor.distribution.std
                        kl = torch.sum(
                            torch.log(sigma / old_sigma_b + 1.0e-5)
                            + (old_sigma_b.square() + (old_mu_b - self.actor.action_mean).square())
                            / (2.0 * sigma.square())
                            - 0.5,
                            dim=-1,
                        ).mean()
                        if kl > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.2)
                        elif kl < self.desired_kl / 2.0 and kl > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.2)
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.learning_rate
                ratio = torch.exp(log_prob - old_log_prob_b.squeeze(-1))
                surrogate = -advantages_b.squeeze(-1) * ratio
                clipped = -advantages_b.squeeze(-1) * ratio.clamp(1-self.clip_param, 1+self.clip_param)
                surrogate_loss = torch.maximum(surrogate, clipped).mean()
                value_clipped = current_values_b + (value - current_values_b).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((value - returns_b).square(), (value_clipped - returns_b).square()).mean()
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], 0.5)
                self.optimizer.step()
        self.tot_timesteps += self.transitions * self.num_envs
        self.storage.clear()

    def check_exploding_gradient(self): return self.is_exploding_gradient

__all__ = ["PPO"]
