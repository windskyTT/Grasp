from __future__ import annotations

import torch

class RolloutStorage:
    def __init__(self, num_envs, transitions, actor_shape, critic_shape, action_shape, device):
        self.device = torch.device(device)
        shape = (transitions, num_envs)
        self.actor_obs = torch.zeros(shape + tuple(actor_shape), device=device)
        self.critic_obs = torch.zeros(shape + tuple(critic_shape), device=device)
        self.actions = torch.zeros(shape + tuple(action_shape), device=device)
        self.rewards = torch.zeros(shape + (1,), device=device)
        self.dones = torch.zeros(shape + (1,), device=device)
        self.log_probs = torch.zeros(shape + (1,), device=device)
        self.mu = torch.zeros(shape + tuple(action_shape), device=device)
        self.sigma = torch.zeros(shape + tuple(action_shape), device=device)
        self.values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.transitions = transitions; self.num_envs = num_envs; self.step = 0

    def add_transitions(self, actor_obs, critic_obs, actions, mu, sigma, rewards, dones, log_probs):
        if self.step >= self.transitions: raise AssertionError("Rollout buffer overflow")
        i = self.step
        self.actor_obs[i] = actor_obs; self.critic_obs[i] = critic_obs; self.actions[i] = actions
        self.mu[i] = mu; self.sigma[i] = sigma; self.rewards[i, :, 0] = rewards; self.dones[i, :, 0] = dones
        self.log_probs[i, :, 0] = log_probs; self.step += 1

    def clear(self): self.step = 0

    def compute_returns(self, last_values, critic, gamma, lam):
        self.values = critic.predict(self.critic_obs.reshape(-1, self.critic_obs.shape[-1])).reshape_as(self.values)
        advantage = torch.zeros((self.num_envs, 1), device=self.device)
        for step in reversed(range(self.transitions)):
            next_values = last_values if step == self.transitions - 1 else self.values[step + 1]
            not_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + not_terminal * gamma * next_values - self.values[step]
            advantage = delta + not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def batches(self, count, shuffle):
        total = self.num_envs * self.transitions
        indices = torch.randperm(total, device=self.device) if shuffle else torch.arange(total, device=self.device)
        size = total // count
        flat = lambda value: value.reshape(total, *value.shape[2:])
        for batch in range(count):
            ids = indices[batch*size:(batch+1)*size]
            yield (flat(self.actor_obs)[ids], flat(self.critic_obs)[ids], flat(self.actions)[ids], flat(self.sigma)[ids], flat(self.mu)[ids], flat(self.values)[ids], flat(self.advantages)[ids], flat(self.returns)[ids], flat(self.log_probs)[ids])

__all__ = ["RolloutStorage"]
