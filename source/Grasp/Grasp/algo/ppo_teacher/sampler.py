import math

import torch


class TorchNormalSampler:
    """在策略设备上采样对角高斯动作。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.seed_value = 0
        self.generator = None

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def sample(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.generator is None:
            self.generator = torch.Generator(device=mean.device)
            self.generator.manual_seed(self.seed_value)

        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=self.generator,
        )
        std_row = std.reshape(1, self.dim)
        samples = mean + noise * std_row
        logprob = -0.5 * torch.sum(
            noise.square()
            + 2.0 * torch.log(std_row)
            + math.log(2.0 * math.pi),
            dim=1,
        )
        return samples, logprob
