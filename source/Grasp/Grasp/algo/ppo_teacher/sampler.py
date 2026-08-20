import numpy as np


class TorchNormalSampler:
    """与 RobustDexGrasp NormalSampler 保持调用接口一致的 NumPy 实现。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.rng = np.random.default_rng(0)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        samples: np.ndarray,
        logprob: np.ndarray,
    ) -> None:
        noise = self.rng.normal(
            loc=0.0,
            scale=1.0,
            size=mean.shape,
        ).astype(np.float32)

        std_row = std.reshape(1, self.dim)
        samples[:] = mean + noise * std_row

        normalized = (samples - mean) / std_row
        logprob[:] = -0.5 * np.sum(
            normalized**2
            + 2.0 * np.log(std_row)
            + np.log(2.0 * np.pi),
            axis=1,
        )