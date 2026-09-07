"""Teacher PPO Gaussian sampler：CUDA RNG。

采样链：
    Actor mean CUDA
        +
    std Parameter CUDA
        ↓
    torch.randn(..., device=mean.device)
        ↓
    action CUDA
    log-probability CUDA

Python seed 只是随机数生成器的静态初始化参数；
随机样本本身直接由 CUDA Generator 产生。
"""

from __future__ import annotations

import math

import torch


class TorchNormalSampler:
    """在策略所在 CUDA device 上采样 13D 对角高斯动作。"""

    def __init__(
        self,
        dim: int,
    ):
        self.dim = dim
        self.seed_value = 0

        # Generator 必须等第一次看到 mean.device 后，
        # 才能和策略使用完全相同的 CUDA device。
        self.generator: (
            torch.Generator
            | None
        ) = None

    def seed(
        self,
        seed: int,
    ) -> None:
        """保存 CUDA RNG seed；不生成任何数值 Tensor。"""

        self.seed_value = seed

    @torch.no_grad()
    def sample(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """直接在 mean.device 上生成 Gaussian action。

        输入：
            mean [B, 13]
            std  [13]

        输出：
            samples [B, 13]
            logprob [B]
        """

        if self.generator is None:
            self.generator = torch.Generator(
                device=mean.device
            )
            self.generator.manual_seed(
                self.seed_value
            )

        # 高斯噪声从生成第一刻就在 CUDA。
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=self.generator,
        )

        std_row = std.reshape(
            1,
            self.dim,
        )

        samples = (
            mean
            + noise * std_row
        )

        # 对角 Gaussian：
        #
        # log p(a | mu, sigma)
        #   =
        # -1/2 * sum(
        #     noise^2
        #     + 2 log(sigma)
        #     + log(2pi)
        # )
        #
        # 直接使用已经生成的 normalized noise，
        # 不需要构造 CPU/NumPy distribution。
        logprob = -0.5 * torch.sum(
            noise.square()
            + 2.0 * torch.log(
                std_row
            )
            + math.log(
                2.0 * math.pi
            ),
            dim=1,
        )

        return (
            samples,
            logprob,
        )


__all__ = [
    "TorchNormalSampler",
]
