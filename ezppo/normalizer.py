"""
Running normalizer for observation preprocessing.
Uses online algorithm to track running mean and variance.
"""

import torch
import numpy as np


class RunningNormalizer:
    """
    Online running normalizer for observations.
    
    Uses Welford's algorithm for numerically stable running statistics.
    Clips normalized values to prevent extreme outliers.
    """

    def __init__(self, shape: int | list[int], epsilon: float = 1e-8, clip: float = 10.0):
        """
        Args:
            shape: Dimension of observations (int or list of ints).
                   Will be flattened to a 1D shape.
            epsilon: Small constant for numerical stability.
            clip: Maximum absolute value for normalized observations.
        """
        # Flatten shape if list or tuple
        if isinstance(shape, (list, tuple)):
            flat_shape = [int(np.prod(shape))]
        else:
            flat_shape = [int(shape)]

        self.mean = torch.zeros(flat_shape)
        self.var = torch.ones(flat_shape)
        self.count = epsilon
        self.epsilon = epsilon
        self.clip = clip

    def update(self, x: torch.Tensor) -> None:
        """
        Update running statistics with a batch of observations.
        
        Args:
            x: Tensor of shape [batch, ...] or [batch, obs_dim]
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean += delta * batch_count / total_count
        self.var = (
            (self.var * self.count + batch_var * batch_count +
             delta ** 2 * self.count * batch_count / total_count)
            / total_count
        )
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations.
        
        Args:
            x: Tensor of observations.
            
        Returns:
            Normalized tensor, clipped to [-clip, clip].
        """
        normed = (x - self.mean) / (torch.sqrt(self.var) + self.epsilon)
        return torch.clip(normed, -self.clip, self.clip)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize observations (inverse of normalize).
        
        Args:
            x: Normalized tensor.
            
        Returns:
            Original-scale tensor.
        """
        return (x * (torch.sqrt(self.var) + self.epsilon)) + self.mean

    def save(self, path: str) -> None:
        """Save statistics to a numpy file."""
        np.save(path, [self.mean.cpu(), self.var.cpu()])

    def load(self, path: str) -> None:
        """Load statistics from a numpy file."""
        data = np.load(path, allow_pickle=True)
        self.mean = torch.as_tensor(data[0]).to(self.mean.device)
        self.var = torch.as_tensor(data[1]).to(self.var.device)

    def to(self, device: torch.device | str) -> "RunningNormalizer":
        """Move statistics to device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def reset(self) -> None:
        """Reset statistics (useful for new training run)."""
        self.mean = torch.zeros_like(self.mean)
        self.var = torch.ones_like(self.var)
        self.count = self.epsilon
