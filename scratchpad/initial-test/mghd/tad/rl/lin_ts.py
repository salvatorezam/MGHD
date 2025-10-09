"""Linear Thompson sampling bandit for contextual adaptation."""
from __future__ import annotations

import numpy as np


class LinTSBandit:
    """Simple linear Thompson sampling with Gaussian prior."""

    def __init__(self, d: int, prior_var: float = 1.0, noise_var: float = 1.0) -> None:
        self.d = int(d)
        self.A = np.eye(self.d) / float(prior_var)
        self.b = np.zeros(self.d)
        self.noise_var = float(noise_var)

    def sample_theta(self) -> np.ndarray:
        Sigma = np.linalg.inv(self.A)
        mu = Sigma @ self.b
        return np.random.multivariate_normal(mu, self.noise_var * Sigma)

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.A += np.outer(x, x)
        self.b += x * float(reward)


__all__ = ["LinTSBandit"]
