"""Helical Recursive Manifold (HRM) simulation stack.

This module implements a runnable approximation of the HRM blueprint:
- Nonlinear field evolution for phi
- Coupled state-vector dynamics
- Topology-aware routing over (channel, domain, layer)
- Macro-cycle evolution over theta
- Optimization over a multi-objective functional J(S, phi)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


@dataclass
class HRMState:
    """Unified HRM state container."""

    theta: float
    channel: int
    domain: int
    layer: int
    field: np.ndarray
    state_vector: np.ndarray
    guna: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    step_count: int = 0
    last_metrics: Dict[str, float] = field(default_factory=dict)


class TopologyEngine:
    """Manages genus-12 routing and domain/layer transitions."""

    def __init__(self, channels: int = 12, domains: int = 22, layers: int = 14):
        self.channels = channels
        self.domains = domains
        self.layers = layers

    def route(self, state: HRMState) -> HRMState:
        theta_norm = state.theta / (2.0 * np.pi)
        state.channel = int(theta_norm * self.channels) % self.channels + 1

        drift = int(np.sign(np.mean(state.field)))
        state.domain = ((state.domain - 1 + state.channel + drift) % self.domains) + 1

        salient = int(np.argmax(np.abs(state.state_vector[: self.layers])))
        state.layer = ((state.layer - 1 + salient + 1) % self.layers) + 1
        return state


class FieldEngine:
    """Solves the nonlinear HRM field equation for phi."""

    def __init__(
        self,
        shape: Tuple[int, int] = (12, 12),
        D_phi: float = 0.09,
        a: float = 0.04,
        b: float = 0.06,
        c: float = 0.025,
        lam: float = 0.08,
        g: float = 0.15,
        kernel_sigma: float = 2.0,
    ):
        self.shape = shape
        self.D_phi = D_phi
        self.a = a
        self.b = b
        self.c = c
        self.lam = lam
        self.g = g
        self.kernel_sigma = kernel_sigma

    @staticmethod
    def _laplacian(x: np.ndarray) -> np.ndarray:
        return (
            np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
            - 4.0 * x
        )

    def _biharmonic(self, x: np.ndarray) -> np.ndarray:
        return self._laplacian(self._laplacian(x))

    def _nonlocal_convolution(self, x: np.ndarray) -> np.ndarray:
        # Fast local proxy for K * phi: 4-neighbor averaging.
        return 0.25 * (
            np.roll(x, 1, axis=0)
            + np.roll(x, -1, axis=0)
            + np.roll(x, 1, axis=1)
            + np.roll(x, -1, axis=1)
        )

    def _source_term(self, state: HRMState) -> np.ndarray:
        s0, s1, s2 = state.state_vector[:3]
        return (
            s0 * np.sin(state.theta)
            + s1 * np.cos(state.theta)
            + 0.25 * s2 * np.tanh(np.mean(state.field))
        ) * np.ones_like(state.field)

    def solve(self, state: HRMState, dt: float = 0.02) -> np.ndarray:
        phi = state.field
        lap = self._laplacian(phi)
        biharm = self._biharmonic(phi)
        nonlocal_term = self._nonlocal_convolution(phi) - phi
        source = self._source_term(state)

        dphi = (
            self.D_phi * lap
            + self.a * phi
            - self.b * (np.abs(phi) ** 2) * phi
            - self.c * biharm
            + self.lam * nonlocal_term
            + self.g * source
        )

        state.field = phi + dt * dphi
        return state.field


class StateEngine:
    """Updates S_i using graph diffusion, logistic saturation and coupling."""

    def __init__(
        self,
        dim: int = 24,
        C: float = 2.0,
        kappa: float = 0.08,
        eta: float = 0.05,
        mu: float = 0.06,
        zeta: float = 0.09,
        psi: float = 0.02,
    ):
        self.dim = dim
        self.C = C
        self.kappa = kappa
        self.eta = eta
        self.mu = mu
        self.zeta = zeta
        self.psi = psi

        self.W = self._ring_coupling(dim)
        self.L = np.diag(np.sum(self.W, axis=1)) - self.W

    @staticmethod
    def _ring_coupling(dim: int) -> np.ndarray:
        w = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            w[i, (i - 1) % dim] = 1.0
            w[i, (i + 1) % dim] = 1.0
        w /= np.maximum(np.sum(w, axis=1, keepdims=True), 1e-8)
        return w

    def _state_target(self, phi: np.ndarray, theta: float) -> np.ndarray:
        flat = phi.reshape(-1)
        stats = np.array(
            [
                float(np.mean(flat)),
                float(np.std(flat)),
                float(np.max(flat)),
                float(np.min(flat)),
                float(np.mean(np.abs(flat))),
                float(np.var(flat)),
                float(np.sin(theta)),
                float(np.cos(theta)),
            ]
        )
        target = np.zeros(self.dim)
        target[: len(stats)] = stats
        return target

    def update(
        self,
        state: HRMState,
        phi: np.ndarray,
        cost_grad: np.ndarray,
        dt: float = 0.02,
    ) -> np.ndarray:
        s = state.state_vector

        graph_term = -self.L @ s
        logistic = self.eta * s * (1.0 - (np.linalg.norm(s) ** 2) / (self.C * self.C + 1e-8))
        coupled = self.mu * (self.W @ s - s)
        attractor = self.zeta * (self._state_target(phi, state.theta) - s)
        optimizer_drive = self.psi * cost_grad

        ds = self.kappa * graph_term + logistic + coupled + attractor + optimizer_drive
        s_new = s + dt * ds

        norm = np.linalg.norm(s_new)
        if norm > self.C:
            s_new = (s_new / (norm + 1e-8)) * self.C

        state.state_vector = s_new

        guna_raw = np.abs(s_new[:3]) + 1e-6
        guna = guna_raw / np.sum(guna_raw)
        state.guna = (float(guna[0]), float(guna[1]), float(guna[2]))
        return state.state_vector


class CycleEngine:
    """Computes the macro-cycle theta."""

    def __init__(self, omega: float = 2.0 * np.pi / 120.0):
        self.omega = omega

    def update(self, state: HRMState, dt: float = 1.0) -> float:
        state.theta = (state.theta + self.omega * dt) % (2.0 * np.pi)
        return state.theta


class OperatorEngine:
    """Applies channel-conditioned archetypal operators."""

    def apply(self, state: HRMState) -> HRMState:
        mode = state.channel % 3
        s = state.state_vector

        if mode == 0:
            s = 0.97 * s + 0.03 * np.roll(s, 1)
        elif mode == 1:
            s = 0.95 * s + 0.05 * np.roll(s, -1)
        else:
            s = np.tanh(1.1 * s)

        state.state_vector = s
        return state


class OptimizationEngine:
    """Evaluates J(S, phi) and adapts engine parameters."""

    def __init__(
        self,
        w_f: float = 0.25,
        w_p: float = 0.2,
        w_s: float = 0.25,
        w_c: float = 0.2,
        w_r: float = 0.1,
    ):
        self.w_f = w_f
        self.w_p = w_p
        self.w_s = w_s
        self.w_c = w_c
        self.w_r = w_r
        self.last_loss = 0.0

    @staticmethod
    def _fractal(phi: np.ndarray) -> float:
        gy, gx = np.gradient(phi)
        rough = np.mean(np.sqrt(gx * gx + gy * gy))
        return float(np.tanh(rough))

    @staticmethod
    def _precision(state_vector: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.var(state_vector[:8])))

    @staticmethod
    def _stability(state_vector: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.linalg.norm(np.diff(state_vector))))

    @staticmethod
    def _coherence(state_vector: np.ndarray) -> float:
        a = state_vector[:12]
        b = state_vector[12:24]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float((np.dot(a, b) / denom + 1.0) * 0.5)

    @staticmethod
    def _regularization(state_vector: np.ndarray, phi: np.ndarray) -> float:
        return float(0.1 * np.linalg.norm(state_vector) + 0.05 * np.mean(phi * phi))

    def evaluate(self, state: HRMState, phi: np.ndarray) -> Dict[str, float]:
        f_fractal = self._fractal(phi)
        f_precision = self._precision(state.state_vector)
        f_stability = self._stability(state.state_vector)
        f_coherence = self._coherence(state.state_vector)
        reg = self._regularization(state.state_vector, phi)

        objective = (
            self.w_f * f_fractal
            + self.w_p * f_precision
            + self.w_s * f_stability
            + self.w_c * f_coherence
            - self.w_r * reg
        )
        loss = -objective

        metrics = {
            "f_fractal": f_fractal,
            "f_precision": f_precision,
            "f_stability": f_stability,
            "f_coherence": f_coherence,
            "regularization": reg,
            "objective": objective,
            "loss": loss,
        }
        state.last_metrics = metrics
        self.last_loss = loss
        return metrics

    def step(
        self,
        state: HRMState,
        field_engine: FieldEngine,
        state_engine: StateEngine,
        metrics: Dict[str, float],
    ):
        # Keep dynamics stable by adjusting diffusion/coupling from objective terms.
        stability = metrics["f_stability"]
        coherence = metrics["f_coherence"]

        if stability < 0.45:
            field_engine.D_phi = min(field_engine.D_phi + 0.002, 0.2)
            state_engine.kappa = min(state_engine.kappa + 0.001, 0.2)
        else:
            field_engine.D_phi = max(field_engine.D_phi - 0.001, 0.03)

        if coherence < 0.45:
            field_engine.lam = min(field_engine.lam + 0.002, 0.2)
        else:
            field_engine.lam = max(field_engine.lam - 0.001, 0.02)

    def cost_gradient(self, state: HRMState) -> np.ndarray:
        # Simple surrogate gradient toward balanced state-vector energy.
        target = np.mean(np.abs(state.state_vector))
        return np.clip(target - state.state_vector, -0.2, 0.2)


class SimulationEngine:
    """Orchestrates the full HRM loop."""

    def __init__(self, field_shape: Tuple[int, int] = (12, 12), seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.topology = TopologyEngine()
        self.field_engine = FieldEngine(shape=field_shape)
        self.state_engine = StateEngine(dim=24)
        self.cycle_engine = CycleEngine()
        self.operator_engine = OperatorEngine()
        self.optimization_engine = OptimizationEngine()

    def initialize(self) -> HRMState:
        return HRMState(
            theta=0.0,
            channel=1,
            domain=1,
            layer=1,
            field=self.rng.normal(0.0, 0.05, self.field_engine.shape),
            state_vector=self.rng.normal(0.0, 0.1, 24),
        )

    def step(self, state: HRMState, dt: float = 0.02) -> Tuple[HRMState, Dict[str, float]]:
        self.cycle_engine.update(state, dt=1.0)
        self.operator_engine.apply(state)
        self.field_engine.solve(state, dt=dt)

        grad = self.optimization_engine.cost_gradient(state)
        self.state_engine.update(state, state.field, grad, dt=dt)
        self.topology.route(state)

        metrics = self.optimization_engine.evaluate(state, state.field)
        self.optimization_engine.step(state, self.field_engine, self.state_engine, metrics)

        state.step_count += 1
        return state, metrics

    def run(self, T: int = 100, dt: float = 0.02) -> Tuple[HRMState, Dict[str, float]]:
        state = self.initialize()
        metrics: Dict[str, float] = {}
        for _ in range(T):
            state, metrics = self.step(state, dt=dt)
        return state, metrics
