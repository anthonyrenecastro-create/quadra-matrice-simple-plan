"""
Field Neural Network Components - Lightweight numpy-based field update operators.

Provides three composable operators used by SyntropyField:
- FieldUpdateNN: local smoothing + variance clipping
- SyntropyNN: mean-pulling toward a target value
- FeedbackNN: variance penalization via soft clipping
"""

import numpy as np


class FieldUpdateNN:
    """
    Applies a local Gaussian smoothing update with variance clipping.

    Parameters:
        variance_clip: Maximum allowed std-deviation magnitude per step.
    """

    def __init__(self, variance_clip: float = 0.2):
        self.variance_clip = variance_clip

    def update(self, field: np.ndarray) -> np.ndarray:
        """Smooth the field and clip its range to [mean ± variance_clip]."""
        # 3x3 box-blur approximation via row+col separable mean
        blurred = (
            np.roll(field, 1, axis=0) + field + np.roll(field, -1, axis=0)
        ) / 3.0
        blurred = (
            np.roll(blurred, 1, axis=1) + blurred + np.roll(blurred, -1, axis=1)
        ) / 3.0
        mean = blurred.mean()
        return np.clip(blurred, mean - self.variance_clip, mean + self.variance_clip)


class SyntropyNN:
    """
    Pulls the field mean toward a target value (syntropy pressure).

    Parameters:
        target_mean: Desired mean value for the field.
    """

    def __init__(self, target_mean: float = 0.5):
        self.target_mean = target_mean

    def pull_mean(self, field: np.ndarray) -> np.ndarray:
        """Shift the field so its mean moves 10 % of the way toward target_mean."""
        delta = self.target_mean - field.mean()
        return field + 0.1 * delta


class FeedbackNN:
    """
    Penalizes high-variance regions via soft-clipping toward local mean.
    """

    def penalize_variance(self, field: np.ndarray) -> np.ndarray:
        """Blend field toward its mean in proportion to local deviation."""
        mean = field.mean()
        std = field.std() + 1e-8
        # Soft-clip: dampen deviations beyond 1σ
        deviation = field - mean
        penalty = np.tanh(deviation / std)
        return mean + std * penalty
