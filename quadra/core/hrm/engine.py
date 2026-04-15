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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


ENGINE_ROLE_MAP: Dict[str, str] = {
    "CycleEngine": "global modulator",
    "TopologyEngine": "constraint source",
    "OperatorEngine": "transform system",
    "FieldEngine": "local update system",
    "StateEngine": "local update system",
    "OptimizationEvaluator": "evaluator",
    "FeedbackShaper": "feedback shaper",
}

ENGINE_DEPENDENCY_MAP: Dict[str, str] = {
    "FieldEngine": "state, theta, topology, operators, phi_prev",
    "StateEngine": "state_prev, phi, theta, topology",
    "TopologyEngine": "state, phi, theta",
    "OptimizationEvaluator": "state, phi",
    "FeedbackShaper": "all simulation diagnostics; feeds back into field/state constraints",
}


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
    collapse_flag: bool = False
    collapse_memory: List[Dict[str, Any]] = field(default_factory=list)
    collapse_energy: float = 0.0
    attractor_anchor: Optional[np.ndarray] = None


class TopologyEngine:
    """Manages genus-12 routing and domain/layer transitions."""

    def __init__(self, channels: int = 12, domains: int = 22, layers: int = 14):
        self.channels = channels
        self.domains = domains
        self.layers = layers

    def route_explicit(
        self,
        state_vector: np.ndarray,
        phi: np.ndarray,
        theta: float,
        domain: int,
        layer: int,
    ) -> Tuple[int, int, int]:
        """Route using explicit dependencies: state, phi, theta."""
        theta_norm = theta / (2.0 * np.pi)
        channel = int(theta_norm * self.channels) % self.channels + 1

        drift = int(np.sign(np.mean(phi)))
        domain = ((domain - 1 + channel + drift) % self.domains) + 1

        salient = int(np.argmax(np.abs(state_vector[: self.layers])))
        layer = ((layer - 1 + salient + 1) % self.layers) + 1
        return channel, domain, layer

    def route(self, state: HRMState) -> HRMState:
        channel, domain, layer = self.route_explicit(
            state_vector=state.state_vector,
            phi=state.field,
            theta=state.theta,
            domain=state.domain,
            layer=state.layer,
        )
        state.channel = channel
        state.domain = domain
        state.layer = layer
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

    def solve(
        self,
        state: HRMState,
        dt: float = 0.02,
        topology: TopologyEngine | None = None,
        operator_gain: float = 1.0,
    ) -> np.ndarray:
        # Explicitly treat phi_prev as the local update source.
        phi = state.field.copy()
        lap = self._laplacian(phi)
        biharm = self._biharmonic(phi)
        nonlocal_term = self._nonlocal_convolution(phi) - phi
        source = self._source_term(state)

        # Topology-gated source amplification turns routing constraints into
        # in-loop field conditioning instead of post-hoc bookkeeping.
        topology_gain = 1.0
        if topology is not None:
            topology_gain += 0.02 * (state.channel / max(1, topology.channels))

        dphi = (
            self.D_phi * lap
            + self.a * phi
            - self.b * (np.abs(phi) ** 2) * phi
            - self.c * biharm
            + self.lam * nonlocal_term
            + self.g * topology_gain * operator_gain * source
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
        theta: float,
        topology: TopologyEngine,
        cost_grad: np.ndarray,
        dt: float = 0.02,
    ) -> np.ndarray:
        # Explicit state_prev dependency for local update dynamics.
        s = state.state_vector.copy()

        # Topology and macro-cycle phase inject in-loop constraints.
        domain_scale = 1.0 + 0.01 * (state.domain / max(1, topology.domains))
        layer_scale = 1.0 + 0.01 * (state.layer / max(1, topology.layers))
        theta_phase = 1.0 + 0.02 * np.sin(theta)

        graph_term = -self.L @ s
        logistic = self.eta * s * (1.0 - (np.linalg.norm(s) ** 2) / (self.C * self.C + 1e-8))
        coupled = self.mu * (self.W @ s - s)
        attractor = self.zeta * (self._state_target(phi, theta) - s)
        optimizer_drive = self.psi * cost_grad

        ds = (
            self.kappa * graph_term
            + logistic
            + coupled
            + attractor
            + optimizer_drive
        )
        ds *= domain_scale * layer_scale * theta_phase
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


class OptimizationEvaluator:
    """Computes the diagnostic objective J(S, phi) without shaping dynamics."""

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


class FeedbackShaper:
    """Injects optimization pressure directly into field/state updates."""

    def __init__(self):
        self.last_drive = 0.0
        self.last_objective = 0.0
        self.targets: Dict[str, float] = {
            "f_fractal": 0.62,
            "f_precision": 0.68,
            "f_stability": 0.70,
            "f_coherence": 0.72,
            "regularization": 0.34,
        }
        self.error_weights: Dict[str, float] = {
            "f_fractal": 0.22,
            "f_precision": 0.20,
            "f_stability": 0.24,
            "f_coherence": 0.24,
            "regularization": 0.10,
        }

    @staticmethod
    def _cost_gradient(state: HRMState, metric_errors: Dict[str, float]) -> np.ndarray:
        """Construct a structured gradient so pressure is directional, not generic."""
        s = state.state_vector
        grad = np.zeros_like(s)

        # Partitioned drives encourage different behavioral regimes across subspaces.
        grad[:8] = metric_errors["f_precision"] * np.sign(s[:8] + 1e-8)
        grad[8:16] = metric_errors["f_stability"] * (np.roll(s[8:16], 1) - s[8:16])
        grad[16:24] = metric_errors["f_coherence"] * np.tanh(1.2 * s[16:24])

        # Fractal demand pushes richer field-state exchange through phase coupling.
        phase = np.sin(state.theta)
        grad += 0.35 * metric_errors["f_fractal"] * phase

        # Regularization pressure damps runaway energy without flattening dynamics.
        energy = np.mean(np.abs(s))
        grad -= 0.18 * metric_errors["regularization"] * energy * np.sign(s + 1e-8)
        return np.clip(grad, -0.3, 0.3)

    def pressure(
        self,
        state: HRMState,
        topology: TopologyEngine,
    ) -> Dict[str, np.ndarray | float]:
        """Compute shaping signals before local updates run."""
        # Use prior diagnostics for temporal continuity and in-loop guidance.
        last_metrics = state.last_metrics
        metric_errors = {
            "f_fractal": self.targets["f_fractal"] - float(last_metrics.get("f_fractal", self.targets["f_fractal"])),
            "f_precision": self.targets["f_precision"] - float(last_metrics.get("f_precision", self.targets["f_precision"])),
            "f_stability": self.targets["f_stability"] - float(last_metrics.get("f_stability", self.targets["f_stability"])),
            "f_coherence": self.targets["f_coherence"] - float(last_metrics.get("f_coherence", self.targets["f_coherence"])),
            "regularization": self.targets["regularization"] - float(last_metrics.get("regularization", self.targets["regularization"])),
        }

        last_objective = float(last_metrics.get("objective", 0.0))
        objective_delta = last_objective - self.last_objective
        self.last_objective = last_objective

        weighted_error = float(
            sum(self.error_weights[k] * metric_errors[k] for k in metric_errors)
        )
        objective_pressure = float(np.tanh(1.8 * weighted_error + 0.7 * objective_delta))

        domain_bias = state.domain / max(1, topology.domains)
        layer_bias = state.layer / max(1, topology.layers)
        channel_bias = state.channel / max(1, topology.channels)
        topology_phase = np.sin(state.theta + 2.0 * np.pi * channel_bias)

        operator_gain = float(
            np.clip(
                1.0 + 0.07 * objective_pressure + 0.03 * topology_phase + 0.02 * (domain_bias - layer_bias),
                0.85,
                1.20,
            )
        )

        # Field and state constraints are shaped before dynamics are applied.
        dphi_shift = np.clip(
            0.006 * (0.65 * metric_errors["f_stability"] + 0.35 * metric_errors["f_fractal"]),
            -0.004,
            0.004,
        )
        lam_shift = np.clip(
            0.006 * (0.7 * metric_errors["f_coherence"] + 0.3 * metric_errors["f_fractal"]),
            -0.004,
            0.004,
        )
        kappa_shift = np.clip(
            0.003 * (0.6 * metric_errors["f_stability"] + 0.4 * metric_errors["f_precision"]),
            -0.002,
            0.002,
        )
        psi_shift = np.clip(
            0.003 * (0.65 * metric_errors["f_precision"] + 0.35 * objective_pressure),
            -0.002,
            0.002,
        )
        zeta_shift = np.clip(
            0.003 * (0.7 * metric_errors["f_coherence"] + 0.3 * objective_pressure),
            -0.002,
            0.002,
        )

        grad = self._cost_gradient(state, metric_errors)
        grad = grad * (1.0 + 0.08 * topology_phase)

        self.last_drive = float(np.mean(np.abs(grad)))
        return {
            "cost_grad": grad,
            "dphi_shift": float(dphi_shift),
            "lam_shift": float(lam_shift),
            "kappa_shift": float(kappa_shift),
            "psi_shift": float(psi_shift),
            "zeta_shift": float(zeta_shift),
            "operator_gain": float(operator_gain),
            "objective_pressure": objective_pressure,
            "feedback_drive": self.last_drive,
        }


class OptimizationEngine:
    """Backward-compatible facade exposing evaluator + shaper as one engine."""

    def __init__(
        self,
        w_f: float = 0.25,
        w_p: float = 0.2,
        w_s: float = 0.25,
        w_c: float = 0.2,
        w_r: float = 0.1,
    ):
        self.evaluator = OptimizationEvaluator(
            w_f=w_f,
            w_p=w_p,
            w_s=w_s,
            w_c=w_c,
            w_r=w_r,
        )
        self.shaper = FeedbackShaper()

    @property
    def last_loss(self) -> float:
        return self.evaluator.last_loss

    def evaluate(self, state: HRMState, phi: np.ndarray) -> Dict[str, float]:
        return self.evaluator.evaluate(state, phi)

    def pressure(self, state: HRMState, topology: TopologyEngine) -> Dict[str, np.ndarray | float]:
        return self.shaper.pressure(state, topology)

    def apply_constraints(
        self,
        field_engine: FieldEngine,
        state_engine: StateEngine,
        pressure: Dict[str, np.ndarray | float],
    ):
        field_engine.D_phi = float(np.clip(field_engine.D_phi + pressure["dphi_shift"], 0.03, 0.2))
        field_engine.lam = float(np.clip(field_engine.lam + pressure["lam_shift"], 0.02, 0.2))
        state_engine.kappa = float(np.clip(state_engine.kappa + pressure["kappa_shift"], 0.03, 0.2))
        state_engine.psi = float(np.clip(state_engine.psi + pressure["psi_shift"], 0.005, 0.08))
        state_engine.zeta = float(np.clip(state_engine.zeta + pressure["zeta_shift"], 0.03, 0.2))

    def step(
        self,
        state: HRMState,
        field_engine: FieldEngine,
        state_engine: StateEngine,
        metrics: Dict[str, float],
    ):
        # Kept for compatibility; shaping is now pre-update in SimulationEngine.
        stability = metrics["f_stability"]
        coherence = metrics["f_coherence"]

        if stability < 0.45:
            field_engine.D_phi = min(field_engine.D_phi + 0.001, 0.2)
            state_engine.kappa = min(state_engine.kappa + 0.0005, 0.2)

        if coherence < 0.45:
            field_engine.lam = min(field_engine.lam + 0.001, 0.2)

    def cost_gradient(self, state: HRMState) -> np.ndarray:
        return self.shaper._cost_gradient(
            state,
            {
                "f_fractal": 0.0,
                "f_precision": 0.0,
                "f_stability": 0.0,
                "f_coherence": 0.0,
                "regularization": 0.0,
            },
        )


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
        self.E_crit = 1.60
        self.anchor_update_rate = 0.08
        self.collapse_memory_limit = 128
        self._collapse_cooldown = 0
        self._collapse_cooldown_steps = 6
        self._baseline = {
            'D_phi': float(self.field_engine.D_phi),
            'b': float(self.field_engine.b),
            'g': float(self.field_engine.g),
            'kappa': float(self.state_engine.kappa),
            'mu': float(self.state_engine.mu),
            'psi': float(self.state_engine.psi),
        }

    @staticmethod
    def _estimate_coherence(state_vector: np.ndarray) -> float:
        a = state_vector[:12]
        b = state_vector[12:24]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float((np.dot(a, b) / denom + 1.0) * 0.5)

    @staticmethod
    def _gradient_norm(phi: np.ndarray) -> float:
        grads = np.gradient(phi)
        if isinstance(grads, list) or isinstance(grads, tuple):
            mag = np.zeros_like(phi, dtype=float)
            for g in grads:
                mag += g * g
            return float(np.mean(np.sqrt(mag)))
        return float(np.mean(np.abs(grads)))

    def _collapse_energy(
        self,
        state: HRMState,
        prev_state_vector: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        coherence = float(state.last_metrics.get('f_coherence', self._estimate_coherence(state.state_vector)))
        coherence_loss = float(np.clip(1.0 - coherence, 0.0, 1.0))

        grad_norm = self._gradient_norm(state.field)
        dstate = (state.state_vector - prev_state_vector) / max(dt, 1e-8)
        dstate_norm = float(np.linalg.norm(dstate))

        grad_term = float(np.tanh(grad_norm))
        velocity_term = float(np.tanh(dstate_norm))
        collapse_energy = grad_term + velocity_term + coherence_loss
        return {
            'collapse_energy': collapse_energy,
            'grad_term': grad_term,
            'velocity_term': velocity_term,
            'coherence_loss': coherence_loss,
        }

    def _update_anchor(self, state: HRMState, collapse_energy: float):
        coherence = float(state.last_metrics.get('f_coherence', self._estimate_coherence(state.state_vector)))
        if collapse_energy >= self.E_crit * 0.70 or coherence < 0.60:
            return

        if state.attractor_anchor is None:
            state.attractor_anchor = state.state_vector.copy()
            return

        state.attractor_anchor = (
            (1.0 - self.anchor_update_rate) * state.attractor_anchor
            + self.anchor_update_rate * state.state_vector
        )

    def _record_collapse_event(self, state: HRMState):
        state.collapse_memory.append(
            {
                'time': float(state.step_count),
                'state_snapshot': state.state_vector.copy(),
                'phi_snapshot': state.field.copy(),
            }
        )
        if len(state.collapse_memory) > self.collapse_memory_limit:
            state.collapse_memory = state.collapse_memory[-self.collapse_memory_limit :]

    def _apply_collapse_rebinding(self, state: HRMState, collapse_energy: float):
        if state.attractor_anchor is None:
            state.attractor_anchor = state.state_vector.copy()

        instability = max(0.0, collapse_energy - self.E_crit)
        alpha = float(np.clip(0.20 + 0.35 * instability, 0.20, 0.85))

        # Rebinding operator: S' = S + alpha * (S_anchor - S)
        state.state_vector = state.state_vector + alpha * (state.attractor_anchor - state.state_vector)

        # Collapse control: increase internal diffusion/coupling, reduce external forcing.
        damp_gain = float(np.clip(1.0 + 0.55 * instability, 1.0, 1.9))
        force_decay = float(np.clip(1.0 - 0.45 * instability, 0.35, 1.0))

        self.state_engine.kappa = float(np.clip(self._baseline['kappa'] * damp_gain, 0.03, 0.2))
        self.state_engine.mu = float(np.clip(self._baseline['mu'] * damp_gain, 0.02, 0.2))
        self.field_engine.D_phi = float(np.clip(self._baseline['D_phi'] * (1.0 + 0.35 * instability), 0.03, 0.2))
        self.field_engine.b = float(np.clip(self._baseline['b'] * damp_gain, 0.02, 0.4))
        self.field_engine.g = float(np.clip(self._baseline['g'] * force_decay, 0.01, 0.25))
        self.state_engine.psi = float(np.clip(self._baseline['psi'] * force_decay, 0.003, 0.08))

    def _relax_collapse_controls(self):
        relax = 0.85
        self.state_engine.kappa = float(np.clip(relax * self.state_engine.kappa + (1.0 - relax) * self._baseline['kappa'], 0.03, 0.2))
        self.state_engine.mu = float(np.clip(relax * self.state_engine.mu + (1.0 - relax) * self._baseline['mu'], 0.02, 0.2))
        self.field_engine.D_phi = float(np.clip(relax * self.field_engine.D_phi + (1.0 - relax) * self._baseline['D_phi'], 0.03, 0.2))
        self.field_engine.b = float(np.clip(relax * self.field_engine.b + (1.0 - relax) * self._baseline['b'], 0.02, 0.4))
        self.field_engine.g = float(np.clip(relax * self.field_engine.g + (1.0 - relax) * self._baseline['g'], 0.01, 0.25))
        self.state_engine.psi = float(np.clip(relax * self.state_engine.psi + (1.0 - relax) * self._baseline['psi'], 0.003, 0.08))

    @staticmethod
    def engine_roles() -> Dict[str, str]:
        """Classify each engine by manifold role."""
        return dict(ENGINE_ROLE_MAP)

    @staticmethod
    def dependency_map() -> Dict[str, str]:
        """Explicit dependency map for HRM loop analysis and tooling."""
        return dict(ENGINE_DEPENDENCY_MAP)

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
        prev_state_vector = state.state_vector.copy()

        # 1) Global modulation
        self.cycle_engine.update(state, dt=1.0)

        # 2) Constraint routing (topology is now an in-loop source)
        self.topology.route(state)

        # 3) Compute shaping pressure before local updates (in-loop conditioning)
        pressure = self.optimization_engine.pressure(state, self.topology)
        self.optimization_engine.apply_constraints(
            self.field_engine,
            self.state_engine,
            pressure,
        )

        # 4) Transform system
        self.operator_engine.apply(state)

        # 5) Local update systems with explicit dependencies
        self.field_engine.solve(
            state,
            dt=dt,
            topology=self.topology,
            operator_gain=float(pressure["operator_gain"]),
        )
        self.state_engine.update(
            state,
            state.field,
            state.theta,
            self.topology,
            np.asarray(pressure["cost_grad"]),
            dt=dt,
        )

        collapse_terms = self._collapse_energy(state, prev_state_vector, dt)
        state.collapse_energy = float(collapse_terms['collapse_energy'])
        collapse_triggered = state.collapse_energy >= self.E_crit

        if collapse_triggered:
            state.collapse_flag = True
            self._collapse_cooldown = self._collapse_cooldown_steps
            self._record_collapse_event(state)
            self._apply_collapse_rebinding(state, state.collapse_energy)
        else:
            if self._collapse_cooldown > 0:
                state.collapse_flag = True
                self._collapse_cooldown -= 1
                self._relax_collapse_controls()
            else:
                state.collapse_flag = False
                self._relax_collapse_controls()

        self._update_anchor(state, state.collapse_energy)

        # 6) Evaluator runs after shaped dynamics are integrated
        metrics = self.optimization_engine.evaluate(state, state.field)
        metrics["feedback_drive"] = float(pressure["feedback_drive"])
        metrics["objective_pressure"] = float(pressure["objective_pressure"])
        metrics["collapse_energy"] = float(state.collapse_energy)
        metrics["collapse_threshold"] = float(self.E_crit)
        metrics["collapse_event"] = float(1.0 if collapse_triggered else 0.0)
        metrics["collapse_flag"] = float(1.0 if state.collapse_flag else 0.0)
        metrics["collapse_memory_size"] = float(len(state.collapse_memory))
        metrics["coherence_loss"] = float(collapse_terms['coherence_loss'])
        metrics["grad_term"] = float(collapse_terms['grad_term'])
        metrics["velocity_term"] = float(collapse_terms['velocity_term'])

        state.step_count += 1
        return state, metrics

    def run(self, T: int = 100, dt: float = 0.02) -> Tuple[HRMState, Dict[str, float]]:
        state = self.initialize()
        metrics: Dict[str, float] = {}
        for _ in range(T):
            state, metrics = self.step(state, dt=dt)
        return state, metrics

    def extract_recursive_diagnostics(self, state: HRMState, latent_dim: int = 96) -> Dict[str, np.ndarray | float]:
        """Export compact, deterministic diagnostics for external recursive latent cores."""
        if latent_dim <= 0:
            latent_dim = 16

        features = np.array(
            [
                float(np.mean(state.field)),
                float(np.std(state.field)),
                float(np.var(state.field)),
                float(np.mean(np.abs(state.field))),
                float(np.linalg.norm(state.state_vector)),
                float(state.theta / (2.0 * np.pi)),
                float(state.channel / max(1, self.topology.channels)),
                float(state.domain / max(1, self.topology.domains)),
                float(state.layer / max(1, self.topology.layers)),
                float(state.last_metrics.get('objective', 0.0)),
                float(state.last_metrics.get('f_coherence', 0.0)),
                float(state.last_metrics.get('f_stability', 0.0)),
                float(state.last_metrics.get('f_precision', 0.0)),
                float(state.guna[0]),
                float(state.guna[1]),
                float(state.guna[2]),
            ],
            dtype=np.float32,
        )

        vec = np.array(state.state_vector, dtype=np.float32).flatten()
        raw = np.concatenate([features, vec], axis=0)
        if raw.size < latent_dim:
            raw = np.pad(raw, (0, latent_dim - raw.size), mode='constant')
        else:
            raw = raw[:latent_dim]

        norm = np.linalg.norm(raw) + 1e-8
        latent_anchor = raw / norm

        return {
            'latent_anchor': latent_anchor,
            'coherence': float(state.last_metrics.get('f_coherence', 0.0)),
            'stability': float(state.last_metrics.get('f_stability', 0.0)),
            'objective': float(state.last_metrics.get('objective', 0.0)),
        }
