"""
Chaos Controller - Lorenz-inspired variance damping.

Provides a scalar chaos signal derived from a simplified Lorenz attractor
used to modulate (damp) field variance each step.
"""

import numpy as np


class ChaosController:
    """
    Tracks a Lorenz-attractor state and uses it to damp scalar variance.

    Parameters:
        scale: Scaling factor applied to the Lorenz perturbation.
        sigma, rho, beta: Lorenz system parameters.
        dt: Integration time step.
    """

    def __init__(
        self,
        scale: float = 0.1,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
    ):
        self.scale = scale
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        # Start near the Lorenz strange attractor
        self._x, self._y, self._z = 0.1, 0.0, 0.0

    def _lorenz_step(self) -> float:
        """Advance Lorenz state one step and return the x-component."""
        dx = self.sigma * (self._y - self._x)
        dy = self._x * (self.rho - self._z) - self._y
        dz = self._x * self._y - self.beta * self._z
        self._x += dx * self.dt
        self._y += dy * self.dt
        self._z += dz * self.dt
        return self._x

    def damp_variance(self, var: float) -> float:
        """
        Apply Lorenz-derived damping to a scalar variance value.

        The Lorenz signal introduces structured perturbation while the
        overall effect is a slight reduction in variance magnitude.

        Returns:
            Damped variance (always >= 0).
        """
        lorenz_signal = self._lorenz_step()
        # Normalise to [-1, 1] range (x oscillates roughly in [-20, 20])
        normalised = np.tanh(lorenz_signal / 20.0)
        damped = var * (1.0 - self.scale * abs(normalised))
        return max(damped, 0.0)
