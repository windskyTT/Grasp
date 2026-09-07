"""Teacher PPO network modules：CUDA-only 数值版本。

当前 Teacher：
    observation = 119
    action      = 13 = FR3 7 + Inspire active 6

GPU 约定：
    - 不使用 NumPy。
    - MLP Linear 参数直接创建在 CUDA。
    - Gaussian std Parameter 直接创建在 CUDA。
    - 不先创建 CPU 参数再 .to(cuda)。
    - rollout / evaluate 输入都应当已经是 CUDA Tensor。
    - TorchScript 导出不再把网络临时移动到 CPU。

Python float（例如 sqrt(2)、init_std）只是静态超参数；
真正的可训练参数从创建第一刻就在 GPU。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


TEACHER_POLICY_DEVICE = "cuda:0"


class Actor:
    """Teacher Actor：MLP + 对角高斯动作分布。"""

    def __init__(
        self,
        architecture,
        distribution,
        device: torch.device | str = TEACHER_POLICY_DEVICE,
    ):
        self.architecture = architecture
        self.distribution = distribution
        self.device = torch.device(device)

        # architecture / distribution 已经应当在 CUDA 上创建。
        # 这里的 .to(cuda) 只是统一 module device，不发生 CPU 初始化。
        self.architecture.to(self.device)
        self.distribution.to(self.device)

        # PPO.act() 后保存当前策略均值，供 storage.step() 写入旧策略参数。
        self.action_mean: torch.Tensor | None = None

    def sample(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据 CUDA observation 采样动作和 log-probability。"""

        self.action_mean = (
            self.architecture.architecture(obs)
        )
        return self.distribution.sample(
            self.action_mean
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO update 中重新计算新策略 log-probability / entropy。"""

        self.action_mean = (
            self.architecture.architecture(obs)
        )
        return self.distribution.evaluate(
            self.action_mean,
            actions,
        )

    def parameters(self):
        return [
            *self.architecture.parameters(),
            *self.distribution.parameters(),
        ]

    @torch.no_grad()
    def noiseless_action(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluation 使用策略均值，不添加高斯噪声。"""

        return self.architecture.architecture(
            obs
        )

    def save_deterministic_graph(
        self,
        file_name: str,
        example_input: torch.Tensor,
    ) -> None:
        """直接从当前 CUDA 网络导出 deterministic TorchScript。

        example_input 必须和 Actor 位于同一 CUDA device。
        不再执行：
            model -> CPU -> trace -> model -> GPU
        """

        traced_graph = torch.jit.trace(
            self.architecture.architecture,
            example_input,
        )
        torch.jit.save(
            traced_graph,
            file_name,
        )

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    """Teacher Critic：119D observation -> scalar value。"""

    def __init__(
        self,
        architecture,
        device: torch.device | str = TEACHER_POLICY_DEVICE,
    ):
        self.architecture = architecture
        self.device = torch.device(device)
        self.architecture.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Rollout / GAE bootstrap 使用，不构建 autograd graph。"""

        return self.architecture.architecture(
            obs
        )

    def evaluate(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """PPO value-loss 使用，需要梯度。"""

        return self.architecture.architecture(
            obs
        )

    def parameters(self):
        return [
            *self.architecture.parameters()
        ]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    """RobustDexGrasp 风格 MLP。

    Teacher 当前配置：
        Actor : 119 -> 128 -> 128 -> 13
        Critic: 119 -> 128 -> 128 -> 1

    所有 nn.Linear 在构造时直接位于 CUDA。
    """

    def __init__(
        self,
        shape,
        activation_fn,
        input_size,
        output_size,
        device: torch.device | str = TEACHER_POLICY_DEVICE,
    ):
        super().__init__()

        self.activation_fn = activation_fn
        self.device = torch.device(device)

        gain = math.sqrt(2.0)

        modules = [
            nn.Linear(
                input_size,
                shape[0],
                device=self.device,
            ),
            self.activation_fn(),
        ]
        scales = [gain]

        for index in range(
            len(shape) - 1
        ):
            modules.append(
                nn.Linear(
                    shape[index],
                    shape[index + 1],
                    device=self.device,
                )
            )
            modules.append(
                self.activation_fn()
            )
            scales.append(gain)

        modules.append(
            nn.Linear(
                shape[-1],
                output_size,
                device=self.device,
            )
        )
        scales.append(gain)

        self.architecture = nn.Sequential(
            *modules
        )

        self.init_weights(
            self.architecture,
            scales,
        )

        self.input_shape = [
            input_size
        ]
        self.output_shape = [
            output_size
        ]

    @staticmethod
    @torch.no_grad()
    def init_weights(
        sequential: nn.Sequential,
        scales: list[float],
    ) -> None:
        """直接在 CUDA Linear weight 上做 orthogonal initialization。"""

        linear_index = 0

        for module in sequential:
            if isinstance(
                module,
                nn.Linear,
            ):
                torch.nn.init.orthogonal_(
                    module.weight,
                    gain=scales[
                        linear_index
                    ],
                )
                linear_index += 1


class MultivariateGaussianDiagonalCovariance(
    nn.Module
):
    """13D 对角高斯策略的可训练标准差。"""

    def __init__(
        self,
        dim,
        init_std,
        fast_sampler,
        seed=0,
        device: torch.device | str = TEACHER_POLICY_DEVICE,
    ):
        super().__init__()

        self.dim = dim
        self.device = torch.device(
            device
        )

        # std Parameter 从创建第一刻就在 CUDA。
        self.std = nn.Parameter(
            torch.full(
                (dim,),
                init_std,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)

    def sample(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rollout action sampling，实际 RNG 在 sampler.py 的 CUDA generator。"""

        return self.fast_sampler.sample(
            logits,
            self.std,
        )

    def evaluate(
        self,
        logits: torch.Tensor,
        outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO update 中计算 Gaussian log-probability 与 entropy。"""

        std = self.std.reshape(
            self.dim
        )

        distribution = Normal(
            logits,
            std,
        )

        actions_log_prob = (
            distribution
            .log_prob(outputs)
            .sum(dim=1)
        )

        entropy = (
            distribution
            .entropy()
            .sum(dim=1)
        )

        return (
            actions_log_prob,
            entropy,
        )

    @torch.no_grad()
    def enforce_minimum_std(
        self,
        min_std: torch.Tensor,
    ) -> None:
        """在 CUDA 上限制策略探索标准差下界。

        原代码通过 self.std.data 重新绑定 storage。
        这里直接原位 copy_，Parameter 对象本身保持不变，
        Adam optimizer 继续引用同一个 Parameter。
        """

        self.std.copy_(
            torch.maximum(
                self.std,
                min_std,
            )
        )


__all__ = [
    "TEACHER_POLICY_DEVICE",
    "Actor",
    "Critic",
    "MLP",
    "MultivariateGaussianDiagonalCovariance",
]
