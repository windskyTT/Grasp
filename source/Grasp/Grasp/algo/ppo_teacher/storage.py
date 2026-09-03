import torch


class RolloutStorage:
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        device,
    ):
        shape = (num_transitions_per_env, num_envs)
        self.critic_obs = torch.zeros(
            (*shape, *critic_obs_shape), device=device
        )
        self.actor_obs = torch.zeros(
            (*shape, *actor_obs_shape), device=device
        )
        self.rewards = torch.zeros((*shape, 1), device=device)
        self.actions = torch.zeros(
            (*shape, *actions_shape), device=device
        )
        self.dones = torch.zeros(
            (*shape, 1), dtype=torch.bool, device=device
        )
        self.actions_log_prob = torch.zeros((*shape, 1), device=device)
        self.values = torch.zeros((*shape, 1), device=device)
        self.returns = torch.zeros((*shape, 1), device=device)
        self.advantages = torch.zeros((*shape, 1), device=device)
        self.mu = torch.zeros((*shape, *actions_shape), device=device)
        self.sigma = torch.zeros((*shape, *actions_shape), device=device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.device = device
        self.step = 0

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
    ):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.critic_obs[self.step].copy_(critic_obs)
        self.actor_obs[self.step].copy_(actor_obs)
        self.actions[self.step].copy_(actions)
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma.detach().expand_as(actions))
        self.rewards[self.step].copy_(rewards.reshape(-1, 1))
        self.dones[self.step].copy_(dones.reshape(-1, 1))
        self.actions_log_prob[self.step].copy_(
            actions_log_prob.reshape(-1, 1)
        )
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, critic, gamma, lam):
        with torch.no_grad():
            self.values.copy_(critic.predict(self.critic_obs))

        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = (
                delta
                + next_is_not_terminal * gamma * lam * advantage
            )
            self.returns[step].copy_(advantage + self.values[step])

        self.advantages.copy_(self.returns - self.values)
        self.advantages.sub_(self.advantages.mean()).div_(
            self.advantages.std(correction=0) + 1e-8
        )

    def mini_batch_generator_shuffle(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size, device=self.device)

        actor_obs = self.actor_obs.flatten(0, 1)
        critic_obs = self.critic_obs.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        sigma = self.sigma.flatten(0, 1)
        mu = self.mu.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        actions_log_prob = self.actions_log_prob.flatten(0, 1)

        for batch_id in range(num_mini_batches):
            start = batch_id * mini_batch_size
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            yield (
                actor_obs[batch_indices],
                critic_obs[batch_indices],
                actions[batch_indices],
                sigma[batch_indices],
                mu[batch_indices],
                values[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
                actions_log_prob[batch_indices],
            )

    def mini_batch_generator_inorder(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        actor_obs = self.actor_obs.flatten(0, 1)
        critic_obs = self.critic_obs.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        sigma = self.sigma.flatten(0, 1)
        mu = self.mu.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        actions_log_prob = self.actions_log_prob.flatten(0, 1)

        for batch_id in range(num_mini_batches):
            start = batch_id * mini_batch_size
            end = start + mini_batch_size
            yield (
                actor_obs[start:end],
                critic_obs[start:end],
                actions[start:end],
                sigma[start:end],
                mu[start:end],
                values[start:end],
                advantages[start:end],
                returns[start:end],
                actions_log_prob[start:end],
            )
