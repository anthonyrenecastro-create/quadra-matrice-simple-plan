"""
Field-level Governance - Waveform bridge for structured field modulation.

Distinct from the policy-level GovernanceService in quadra/core/governance/.
This module governs field dynamics via a sinusoidal waveform carrier that
gently anchors the field to a known oscillatory pattern.
"""

import numpy as np


class Governance:
    """
    Applies a spatially-uniform sinusoidal waveform modulation to a field.

    Parameters:
        amplitude: Peak amplitude of the waveform carrier.
        freq: Angular frequency (rad) of the waveform.
    """

    def __init__(self, amplitude: float = 0.1, freq: float = np.pi):
        self.amplitude = amplitude
        self.freq = freq
        self._t = 0  # internal step counter

    def waveform_bridge(self, field: np.ndarray) -> np.ndarray:
        """
        Add a waveform carrier to stabilise the field around its mean.

        The carrier value is uniform across the field (global phase), which
        gently nudges all cells toward a shared oscillatory rhythm without
        breaking spatial gradients.

        Returns:
            Modulated field array.
        """
        carrier = self.amplitude * np.sin(self.freq * self._t)
        self._t += 1
        return field + carrier
