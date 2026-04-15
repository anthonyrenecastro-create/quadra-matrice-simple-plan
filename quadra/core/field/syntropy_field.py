# Multi-field state with syntropy, memory patterns, and async option
import numpy as np
import asyncio
from collections import deque
from .field_nn import FieldUpdateNN, SyntropyNN, FeedbackNN
from .chaos import ChaosController
from .governance import Governance


class SyntropyField:
    """
    Multi-field state manager with syntropy pressure, memory patterns, and async support.

    Each ``step()`` applies a pipeline of operators:
    1. Local smoothing + variance clip  (FieldUpdateNN)
    2. Mean-pulling toward target       (SyntropyNN)
    3. Variance penalisation            (FeedbackNN)
    4. Waveform carrier injection       (Governance)
    5. Memory-based regularisation      (deque of downsampled snapshots)
    6. Lorenz-damped variance tracking  (ChaosController)

    Parameters:
        shape: 2-D field dimensions (H, W).
        seed: RNG seed for reproducible initialisation.
        target_mean: Desired steady-state field mean.
        variance_clip: Max deviation allowed by FieldUpdateNN.
        wave_amp: Waveform carrier amplitude (Governance).
        wave_freq: Waveform carrier frequency in radians (Governance).
        lorenz_scale: Lorenz damping scale (ChaosController).
        memory_size: Number of downsampled snapshots retained.
        reg_alpha: Memory regularisation blend weight (0 = no memory pull).
    """

    def __init__(
        self,
        shape=(32, 32),
        seed=42,
        target_mean=0.5,
        variance_clip=0.2,
        wave_amp=0.1,
        wave_freq=3.1415,
        lorenz_scale=0.1,
        memory_size=32,
        reg_alpha=0.1,
    ):
        self.reg_alpha = reg_alpha
        rng = np.random.RandomState(seed)
        self.field = rng.rand(*shape)
        self.update_nn = FieldUpdateNN(variance_clip=variance_clip)
        self.syntropy_nn = SyntropyNN(target_mean=target_mean)
        self.feedback_nn = FeedbackNN()
        self.governance = Governance(amplitude=wave_amp, freq=wave_freq)
        self.chaos = ChaosController(scale=lorenz_scale)
        self.history = []  # (mean, var)
        self.memory = deque(maxlen=memory_size)  # downsampled snapshots

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downsample(self, arr: np.ndarray, factor: int = 4) -> np.ndarray:
        """Average-pool *arr* by *factor* along each spatial axis."""
        h, w = arr.shape
        h2 = (h // factor) * factor
        w2 = (w // factor) * factor
        cropped = arr[:h2, :w2]
        reshaped = cropped.reshape(h2 // factor, factor, w2 // factor, factor)
        return reshaped.mean(axis=(1, 3))

    def _memory_mean(self):
        """Return element-wise mean of all retained memory snapshots, or None."""
        if len(self.memory) == 0:
            return None
        return np.mean(np.stack(list(self.memory), axis=0), axis=0)

    def _regularize_with_memory(self, field: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Blend *field* toward the upsampled memory mean.

        Uses nearest-neighbour upsampling to match the memory snapshot back
        to the full field resolution.
        """
        mem = self._memory_mean()
        if mem is None:
            return field
        mh, mw = mem.shape
        fh, fw = field.shape
        up = np.repeat(np.repeat(mem, fh // mh, axis=0), fw // mw, axis=1)
        return (1.0 - alpha) * field + alpha * up

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the field by one time step (synchronous)."""
        f = self.field
        f = self.update_nn.update(f)
        f = self.syntropy_nn.pull_mean(f)
        f = self.feedback_nn.penalize_variance(f)
        f = self.governance.waveform_bridge(f)
        # memory-based regularisation
        f = self._regularize_with_memory(f, alpha=self.reg_alpha)
        var = np.var(f)
        var = self.chaos.damp_variance(var)
        mean = np.mean(f)
        self.field = f
        self.history.append((mean, var))
        # push new downsampled pattern into memory
        self.memory.append(self._downsample(self.field))

    async def step_async(self) -> None:
        """Yield control, then advance one step (async-friendly)."""
        await asyncio.sleep(0)
        self.step()

    def metrics(self) -> dict:
        """Return the most recent ``{'mean': float, 'var': float}`` snapshot."""
        if len(self.history) == 0:
            return {'mean': 0.0, 'var': 0.0}
        m, v = self.history[-1]
        return {'mean': m, 'var': v}

    def get_history(self) -> list:
        """Return the full list of ``(mean, var)`` tuples."""
        return self.history

    def memory_size(self) -> int:
        """Return the number of snapshots currently stored in memory."""
        return len(self.memory)
