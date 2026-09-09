from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class Actor:
    def __init__(self, architecture, distribution, device):
        self.architecture = architecture.to(device)
        self.distribution = distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs):
        self.action_mean = self.architecture.architecture(obs)
        return self.distribution.sample(self.action_mean)

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(obs)

    @property
    def obs_shape(self): return self.architecture.input_shape

    @property
    def action_shape(self): return self.architecture.output_shape

class Critic:
    def __init__(self, architecture, device): self.architecture = architecture.to(device)
    def predict(self, obs): return self.architecture.architecture(obs).detach()
    def evaluate(self, obs): return self.architecture.architecture(obs)
    def parameters(self): return [*self.architecture.parameters()]
    @property
    def obs_shape(self): return self.architecture.input_shape

class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size):
        super().__init__()
        modules = [nn.Linear(input_size, shape[0]), activation_fn()]
        for index in range(len(shape)-1): modules.extend((nn.Linear(shape[index], shape[index+1]), activation_fn()))
        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        self.input_shape = [input_size]
        self.output_shape = [output_size]
        self.init_weights()

    def init_weights(self):
        for layer in (item for item in self.architecture if isinstance(item, nn.Linear)):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super().__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))

    def sample(self, logits):
        distribution = Normal(logits, self.std)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions).sum(dim=1)

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std)
        return distribution.log_prob(outputs).sum(dim=1), distribution.entropy().sum(dim=1)

    def enforce_minimum_std(self, min_std):
        self.std.data = torch.maximum(self.std.detach(), min_std.detach())

__all__ = ["Actor", "Critic", "MLP", "MultivariateGaussianDiagonalCovariance"]
