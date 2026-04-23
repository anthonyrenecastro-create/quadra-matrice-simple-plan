import asyncio
import importlib.util
import logging
import random
import re
import sys
import time
import json
import numpy as np
import sympy as sp
from sympy.logic.boolalg import And, Or, Not, Implies
from sympy import symbols
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    class _FallbackLeaky(nn.Module):
        def __init__(self, beta: float = 0.95, spike_grad=None):
            super().__init__()
            self.beta = float(beta)

        def init_leaky(self):
            return None

        def forward(self, cur: torch.Tensor, mem):
            if mem is None:
                mem = torch.zeros_like(cur)
            mem = self.beta * mem + cur
            spk = (mem > 0.0).float()
            mem = mem * (1.0 - spk)
            return spk, mem

    class _FallbackSNN:
        Leaky = _FallbackLeaky

    class _FallbackSurrogate:
        @staticmethod
        def fast_sigmoid():
            return None

    snn = _FallbackSNN()
    surrogate = _FallbackSurrogate()
import networkx as nx
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp
from scipy.stats import variation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pattern Recognition Module ---
class PatternModule:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def extract_patterns(self, data: np.ndarray) -> np.ndarray:
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        if len(data) < self.kmeans.n_clusters:
            return np.zeros(self.kmeans.n_clusters)
        return self.kmeans.fit_predict(data)

# --- Neuroplasticity Manager ---
class NeuroplasticityManager:
    def __init__(self, oscillator, core_field, syntropy_engine):
        self.oscillator = oscillator
        self.core_field = core_field
        self.syntropy_engine = syntropy_engine
        self.data_queue = deque(maxlen=100)
        self.stability_threshold = 0.05
        self.update_counter = 0

    async def learn_async(self, streamed_text: str, train_every: int = 8):
        feature_vector = self.oscillator.process_streamed_data(streamed_text)
        self.data_queue.append(feature_vector)

        field_update = np.mean([v.cpu().numpy() for v in self.data_queue], axis=0)
        self.core_field.update_with_vibrational_mode(field_update)

        field_var = torch.var(self.core_field.get_state()).item()

        # Only do expensive inner training occasionally.
        self.update_counter += 1
        if self.update_counter % train_every == 0 and field_var > self.stability_threshold:
            synthetic_data = self.oscillator.generate_synthetic_data(num_samples=4)
            reward = self.oscillator.train(synthetic_data, streamed_text, epochs=1)
            self.oscillator.update_q_table(reward)

        if field_var > self.stability_threshold:
            await asyncio.sleep(0)

    def regulate_syntropy(self):
        for i in range(self.syntropy_engine.num_fields):
            field_mean = np.mean(self.syntropy_engine.field_data[i])
            if abs(field_mean - 0.5) > 0.1:
                self.syntropy_engine.field_data[i] += (0.5 - field_mean) * 0.1
                self.syntropy_engine.field_data[i] = np.clip(self.syntropy_engine.field_data[i], -1, 1)
        logger.info("Syntropy regulation applied.")

# --- Symbolic Interpreter Module ---
@dataclass
class SymbolicConfig:
    population_size: int = 12
    generations: int = 6
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.2

class SymbolicPredictiveInterpreter:
    def __init__(self, pattern_module, core_field, config: SymbolicConfig):
        self.pattern_module = pattern_module
        self.core_field = core_field
        self.config = config
        self.x, self.y = sp.symbols('x y')
        self.P, self.Q = sp.symbols('P Q', cls=sp.Function)
        self.z = sp.Symbol('z')
        self.knowledge_graph = nx.DiGraph()
        logger.info("Symbolic Predictive Interpreter initialized with knowledge graph.")

    def build_knowledge_graph(self, concepts: List[str], neural_output: torch.Tensor):
        """Build a knowledge graph from concepts and neural output."""
        self.knowledge_graph.clear()
        for concept in concepts:
            tokens = word_tokenize(concept.lower())
            for token in tokens:
                self.knowledge_graph.add_node(token)
                synsets = wordnet.synsets(token)
                for syn in synsets[:2]:
                    synonym = syn.lemmas()[0].name()
                    self.knowledge_graph.add_node(synonym)
                    self.knowledge_graph.add_edge(token, synonym, relation="synonym")
        neural_mean = torch.mean(neural_output).item()
        self.knowledge_graph.add_node("neural_state")
        self.knowledge_graph.add_edge("neural_state", concepts[0] if concepts else "default", relation="influences", weight=neural_mean)

    def query_knowledge_graph(self):
        """Query the knowledge graph for related concepts."""
        if not self.knowledge_graph.nodes:
            return "Empty knowledge graph."
        try:
            central_node = max(self.knowledge_graph.nodes, key=lambda n: self.knowledge_graph.degree(n))
            related = list(self.knowledge_graph.neighbors(central_node))
            return f"Central concept: {central_node}, related: {related}"
        except Exception as e:
            return f"Graph query failed: {e}"

    async def predict(self, concepts: List[str], neural_output: torch.Tensor) -> str:
        logger.info(f"Symbolic interpreter processing concepts: {concepts}")
        try:
            expr = self.x**2 - self.y**2
            factored = sp.factor(expr)
            proof_result = f"Proved: {expr} = {factored}"
            # Simplified logic representation without ForAll
            P_val = sp.Symbol('P_true')
            Q_val = sp.Symbol('Q_true')
            premise = Implies(P_val, Q_val)
            logic_result = f"Evaluated: P(True) => Q(True) = {premise}"
            semantic_result = []
            for concept in concepts:
                tokens = word_tokenize(concept.lower())
                synonyms = []
                for token in tokens:
                    synsets = wordnet.synsets(token)
                    if synsets:
                        synonyms.append(synsets[0].lemmas()[0].name())
                semantic_result.append(f"Concept '{concept}' mapped to synonyms: {synonyms[:3]}")
            self.build_knowledge_graph(concepts, neural_output)
            graph_result = self.query_knowledge_graph()
            neural_mean = torch.mean(neural_output).item()
            cluster_result = self.pattern_module.extract_patterns(neural_output.cpu().numpy().flatten())
            neural_integration = f"Neural input (mean={neural_mean:.4f}) clustered into {len(set(cluster_result))} groups."
            result = (f"Algebraic proof: {proof_result}. "
                     f"First-order logic: {logic_result}. "
                     f"Semantic reasoning: {'; '.join(semantic_result)}. "
                     f"Knowledge graph: {graph_result}. "
                     f"Neural integration: {neural_integration}")
        except Exception as e:
            result = f"Symbolic reasoning failed: {e}. Using {len(concepts)} concepts."
        await asyncio.sleep(0.1)
        return result

# --- Spiking Neural Network Components ---
class SpikingFieldUpdateNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec).mean(dim=0)

    def update_field(self, field: torch.Tensor, chaos_threshold: float = 0.1) -> torch.Tensor:
        update = self.forward(field)
        if torch.var(field) > chaos_threshold:
            update *= 0.5
        return field + update

class SpikingSyntropyNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec).mean(dim=0)

    def adjust_syntropy(self, field: torch.Tensor, target_mean: float = 0.5, speed_factor: float = 0.5) -> torch.Tensor:
        adjustment = (target_mean - torch.mean(field)) * speed_factor
        delta = self.forward(torch.full_like(field, torch.mean(field)))
        return field + delta * adjustment * 0.5

class SpikingFeedbackNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(hidden_size, 1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec).mean(dim=0)

    def penalize_variance(self, field: torch.Tensor) -> torch.Tensor:
        variance = torch.var(field)
        penalty = self.forward(torch.full((1, field.size(0)), variance.item())).item()
        return field - penalty * variance


class InteractionCoupler(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.55))
        self.beta = nn.Parameter(torch.tensor(0.35))
        self.gamma = nn.Parameter(torch.tensor(0.10))

        self.internal_proj = nn.Linear(4, hidden_dim)
        self.external_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, internal_stats: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        f_int = torch.tanh(self.internal_proj(internal_stats))
        f_ext = torch.tanh(self.external_proj(obs))
        interaction = self.alpha * f_int + self.beta * f_ext + self.gamma * (f_int * f_ext)
        return torch.tanh(self.out_proj(interaction))


def _load_hrm_simulation_engine_class():
    """Load HRM SimulationEngine directly to avoid package import side-effects."""
    module_name = "quadra_hrm_engine_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name].SimulationEngine

    engine_path = Path(__file__).resolve().parent / "quadra" / "core" / "hrm" / "engine.py"
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load HRM engine from {engine_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.SimulationEngine


class AtlanteanHRMElement:
    """Atlantean field solver coupled to HRM state for SNN modulation."""

    def __init__(self, field_size: int, device: torch.device):
        self.device = device
        self.field_size = int(field_size)
        self.grid_side = int(np.ceil(np.sqrt(self.field_size)))
        self.grid_size = self.grid_side * self.grid_side

        # Equation 1 field: excitability (pattern formation)
        self.phi1 = torch.randn(self.grid_size, device=self.device) * 0.05
        # Equation 2 field: entropy/resource memory
        self.phi5 = torch.ones(self.grid_size, device=self.device) * 0.1
        self.Phi = torch.zeros(self.grid_size, device=self.device)
        self.membrane = torch.zeros(self.grid_size, device=self.device)

        # Atlantean coefficients
        self.alpha = 0.18
        self.gamma1 = 0.07
        self.beta = 0.045
        self.zeta = 0.06
        self.omega = 0.05
        self.theta = 0.05
        self.gamma5 = 0.04
        self.base_threshold = 1.0
        self.base_leak = 0.12
        self.step_idx = 0
        self.interaction_coupler = InteractionCoupler(
            hidden_dim=self.grid_size,
            input_dim=self.grid_size,
        ).to(self.device)
        self.cached_stability = torch.tensor(0.0, device=self.device)
        self.cached_variance = torch.tensor(1.0, device=self.device)
        self.cached_phi1_mean = torch.mean(self.phi1)
        self.cached_phi5_mean = torch.mean(self.phi5)
        self.last_interaction_signal = torch.zeros(self.grid_size, device=self.device)
        self.last_run_full_fields = True

    def _prepare_signal(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        if x.numel() < self.grid_size:
            x = F.pad(x, (0, self.grid_size - x.numel()))
        else:
            x = x[: self.grid_size]
        return x

    def _derive_stability(self, variance: torch.Tensor, stability_hint: Optional[float]) -> torch.Tensor:
        derived = 1.0 / (1.0 + variance)
        if stability_hint is None:
            return derived
        hint = torch.tensor(float(stability_hint), device=self.device)
        return torch.clamp(0.5 * derived + 0.5 * hint, 0.0, 1.0)

    def _refresh_cached_stats(self, stability_hint: Optional[float]) -> None:
        field_variance = 0.5 * (torch.var(self.phi1) + torch.var(self.phi5))
        self.cached_variance = field_variance.detach()
        self.cached_stability = self._derive_stability(field_variance, stability_hint).detach()
        self.cached_phi1_mean = torch.mean(self.phi1).detach()
        self.cached_phi5_mean = torch.mean(self.phi5).detach()

    def _interaction_signal(self, obs: torch.Tensor) -> torch.Tensor:
        internal_stats = torch.stack(
            [
                self.cached_phi1_mean,
                self.cached_phi5_mean,
                self.cached_stability,
                self.cached_variance,
            ]
        ).unsqueeze(0)
        return self.interaction_coupler(internal_stats, obs.unsqueeze(0)).squeeze(0)

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        if x.numel() < self.grid_size:
            x = F.pad(x, (0, self.grid_size - x.numel()))
        else:
            x = x[: self.grid_size]
        return x.view(1, 1, self.grid_side, self.grid_side)

    def _from_grid(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        return flat[: self.grid_size]

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=self.device,
        ).view(1, 1, 3, 3)
        xg = self._to_grid(x)
        xg = F.pad(xg, (1, 1, 1, 1), mode='circular')
        return self._from_grid(F.conv2d(xg, k))

    def pde_step(self, hrm_force: float = 0.0, dt: float = 0.01):
        lap_phi1 = self.laplacian(self.phi1)
        bi_lap_phi1 = self.laplacian(lap_phi1)
        lap_phi5 = self.laplacian(self.phi5)
        lap_Phi = self.laplacian(self.Phi)

        reaction = -self.alpha * (torch.abs(self.phi1) ** 2) * self.phi1 + self.zeta * lap_Phi + 0.1 * hrm_force
        diffusion = -self.gamma1 * lap_phi1 + self.beta * bi_lap_phi1
        self.phi1 = torch.clamp(self.phi1 + dt * (diffusion + reaction), -4.0, 4.0)

        log_term = self.omega * self.phi5 * torch.log(torch.abs(self.phi5) + 1e-6)
        advection = self.theta * (lap_phi5 * self.Phi + self.phi5 * lap_Phi)
        self.phi5 = torch.clamp(self.phi5 + dt * (-self.gamma5 * lap_phi5 + log_term + advection), -1.0, 5.0)

    def dynamic_threshold(self, interaction_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        threshold = self.base_threshold - 0.5 * self.phi1
        if interaction_signal is None:
            return threshold
        return (
            threshold
            - 0.12 * interaction_signal
            + 0.08 * self.cached_variance
            - 0.08 * self.cached_stability
        )

    def effective_leak(self, interaction_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        leak = self.base_leak * (1.0 / (1.0 + torch.relu(self.phi5)))
        if interaction_signal is None:
            return leak
        return torch.clamp(
            leak
            + 0.06 * torch.sigmoid(interaction_signal)
            + 0.03 * self.cached_stability,
            0.70,
            0.99,
        )

    def step(
        self,
        input_signal: torch.Tensor,
        hrm_force: float = 0.0,
        dt: float = 0.01,
        stability_hint: Optional[float] = None,
    ) -> torch.Tensor:
        x = self._prepare_signal(input_signal)
        interaction_signal = self._interaction_signal(x)
        run_full_fields = (
            self.cached_stability.mean().item() < 0.82
            or interaction_signal.abs().mean().item() > 0.10
            or (self.step_idx % 4 == 0)
        )

        if run_full_fields:
            self.pde_step(hrm_force=hrm_force, dt=dt)
            self._refresh_cached_stats(stability_hint)

        threshold = self.dynamic_threshold(interaction_signal)
        leak = self.effective_leak(interaction_signal)

        self.membrane = self.membrane * (1.0 - leak) + x + 0.15 * interaction_signal
        spikes = (self.membrane > threshold).float()
        self.membrane = self.membrane * (1.0 - spikes)

        # Global interplay potential feedback.
        self.Phi = 0.9 * self.Phi + 0.1 * spikes
        self.last_interaction_signal = interaction_signal.detach()
        self.last_run_full_fields = run_full_fields
        self.step_idx += 1
        return torch.clamp(spikes + 0.10 * interaction_signal, 0.0, 1.0)[: self.field_size]

    def current_field(self) -> torch.Tensor:
        return self.phi1[: self.field_size].clone()

    def benchmark_gated_path(
        self,
        num_steps: int = 400,
        warmup_steps: int = 40,
        dt: float = 0.01,
        stability_hint: Optional[float] = 0.95,
    ) -> Dict[str, float]:
        """Micro-benchmark gated updates vs forced full PDE updates.

        Returns skipped-step frequency and wall-clock savings.
        """
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        def _snapshot_state() -> Dict[str, Any]:
            return {
                'phi1': self.phi1.clone(),
                'phi5': self.phi5.clone(),
                'Phi': self.Phi.clone(),
                'membrane': self.membrane.clone(),
                'cached_stability': self.cached_stability.clone(),
                'cached_variance': self.cached_variance.clone(),
                'cached_phi1_mean': self.cached_phi1_mean.clone(),
                'cached_phi5_mean': self.cached_phi5_mean.clone(),
                'last_interaction_signal': self.last_interaction_signal.clone(),
                'last_run_full_fields': bool(self.last_run_full_fields),
                'step_idx': int(self.step_idx),
            }

        def _restore_state(state: Dict[str, Any]) -> None:
            self.phi1 = state['phi1'].clone()
            self.phi5 = state['phi5'].clone()
            self.Phi = state['Phi'].clone()
            self.membrane = state['membrane'].clone()
            self.cached_stability = state['cached_stability'].clone()
            self.cached_variance = state['cached_variance'].clone()
            self.cached_phi1_mean = state['cached_phi1_mean'].clone()
            self.cached_phi5_mean = state['cached_phi5_mean'].clone()
            self.last_interaction_signal = state['last_interaction_signal'].clone()
            self.last_run_full_fields = bool(state['last_run_full_fields'])
            self.step_idx = int(state['step_idx'])

        def _run_timed_pass(
            force_full: bool = False,
            neutralize_interaction: bool = False,
        ) -> Dict[str, float]:
            skipped = 0
            full = 0

            coupler_backup = None
            if neutralize_interaction:
                coupler_backup = [p.detach().clone() for p in self.interaction_coupler.parameters()]
                with torch.no_grad():
                    for p in self.interaction_coupler.parameters():
                        p.zero_()

            try:
                for _ in range(warmup_steps):
                    if force_full:
                        self.cached_stability.fill_(0.0)
                    _ = self.step(signal, hrm_force=0.0, dt=dt, stability_hint=stability_hint)

                t_start = time.perf_counter()
                for _ in range(num_steps):
                    if force_full:
                        self.cached_stability.fill_(0.0)
                    _ = self.step(signal, hrm_force=0.0, dt=dt, stability_hint=stability_hint)
                    if self.last_run_full_fields:
                        full += 1
                    else:
                        skipped += 1
                elapsed = time.perf_counter() - t_start
            finally:
                if coupler_backup is not None:
                    with torch.no_grad():
                        for p, old in zip(self.interaction_coupler.parameters(), coupler_backup):
                            p.copy_(old)

            return {
                'elapsed_s': float(elapsed),
                'skipped_steps': float(skipped),
                'full_steps': float(full),
                'skipped_step_frequency': float(skipped / num_steps),
            }

        base_state = _snapshot_state()
        signal = torch.zeros(self.field_size, device=self.device)

        # Benchmark real gated behavior.
        _restore_state(base_state)
        gated = _run_timed_pass(force_full=False, neutralize_interaction=False)

        # Benchmark forced full updates by pinning stability below threshold.
        _restore_state(base_state)
        forced = _run_timed_pass(force_full=True, neutralize_interaction=False)

        # Benchmark best-case gating under temporarily neutralized interaction.
        _restore_state(base_state)
        low_interaction = _run_timed_pass(force_full=False, neutralize_interaction=True)

        _restore_state(base_state)

        gated_elapsed = gated['elapsed_s']
        forced_elapsed = forced['elapsed_s']
        low_elapsed = low_interaction['elapsed_s']

        savings_s = forced_elapsed - gated_elapsed
        savings_pct = (100.0 * savings_s / forced_elapsed) if forced_elapsed > 0 else 0.0
        best_case_savings_s = forced_elapsed - low_elapsed
        best_case_savings_pct = (100.0 * best_case_savings_s / forced_elapsed) if forced_elapsed > 0 else 0.0
        return {
            'num_steps': float(num_steps),
            'warmup_steps': float(warmup_steps),
            'skipped_steps': gated['skipped_steps'],
            'full_steps': gated['full_steps'],
            'skipped_step_frequency': gated['skipped_step_frequency'],
            'gated_total_ms': float(gated_elapsed * 1000.0),
            'forced_full_total_ms': float(forced_elapsed * 1000.0),
            'gated_step_ms': float((gated_elapsed / num_steps) * 1000.0),
            'forced_full_step_ms': float((forced_elapsed / num_steps) * 1000.0),
            'wall_clock_savings_ms': float(savings_s * 1000.0),
            'wall_clock_savings_pct': float(savings_pct),
            'low_interaction_skipped_steps': low_interaction['skipped_steps'],
            'low_interaction_full_steps': low_interaction['full_steps'],
            'low_interaction_skipped_step_frequency': low_interaction['skipped_step_frequency'],
            'low_interaction_total_ms': float(low_elapsed * 1000.0),
            'low_interaction_step_ms': float((low_elapsed / num_steps) * 1000.0),
            'best_case_wall_clock_savings_ms': float(best_case_savings_s * 1000.0),
            'best_case_wall_clock_savings_pct': float(best_case_savings_pct),
        }


class _OscillatorProxyNN(nn.Module):
    """Compatibility proxy so existing code can call nn1/nn2/nn3."""

    def __init__(self, parent: "OscillatorySynapseTheory", mode: str):
        super().__init__()
        self.parent = parent
        self.mode = mode
        self.gain = nn.Parameter(torch.tensor(1.0, device=parent.device))

    def forward(self, x: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        out = self.parent._route_through_atlantean_hrm(x, self.mode)
        return out * torch.sigmoid(self.gain)

    def update_field(self, field: torch.Tensor, chaos_threshold: float = 0.1) -> torch.Tensor:
        out = self.forward(field)
        if torch.var(out) > chaos_threshold:
            out = out * 0.75
        return out

    def adjust_syntropy(self, field: torch.Tensor, target_mean: float = 0.5, speed_factor: float = 0.5) -> torch.Tensor:
        out = self.forward(field)
        return out + (target_mean - torch.mean(out)) * speed_factor * 0.1

    def penalize_variance(self, field: torch.Tensor) -> torch.Tensor:
        out = self.forward(field)
        return out - 0.1 * torch.var(out)


def chaos_regularizer(field: torch.Tensor) -> torch.Tensor:
    """Penalize high-frequency chaos and broadband variance in the field."""
    x = field.flatten()
    if x.numel() < 8:
        return torch.tensor(0.0, device=field.device)

    fft = torch.fft.rfft(x)
    power = torch.abs(fft) ** 2

    # Penalize high-frequency energy
    cutoff = int(0.6 * power.numel())
    high_freq_penalty = power[cutoff:].mean() if cutoff < power.numel() else 0.0

    # Penalize broadband variance
    var_penalty = torch.var(field)

    return 0.1 * var_penalty + 0.05 * high_freq_penalty


# --- Free Energy Field Model (Thermodynamic Evolution) ---
class FreeEnergyFieldModel:
    """Evolves field according to gradient-flow PDE for phase field dynamics."""
    def __init__(self, nfields: int = 2, H: int = 32, W: int = 32, device: str = 'cpu'):
        self.device = torch.device(device)
        self.nfields = nfields
        self.H, self.W = H, W
        self.phi = torch.randn(nfields, H, W, device=self.device) * 0.05

        self.alpha = torch.tensor([0.02] * nfields, device=self.device)
        self.beta = torch.tensor([0.002] * nfields, device=self.device)
        self.r = torch.tensor([-0.35] * nfields, device=self.device)
        self.u = torch.tensor([1.0] * nfields, device=self.device)
        self.lam_shell = torch.tensor([4.0] * nfields, device=self.device)

        self.q_star = 3.5
        self.delta_shell = 0.45

    def laplacian(self, x):
        """Compute Laplacian with periodic boundary conditions."""
        return (
            torch.roll(x, 1, dims=-2) + torch.roll(x, -1, dims=-2) +
            torch.roll(x, 1, dims=-1) + torch.roll(x, -1, dims=-1) -
            4.0 * x
        )

    def biharmonic(self, x):
        """Compute ∇⁴ (4th order Laplacian)."""
        return self.laplacian(self.laplacian(x))

    def sixth_order(self, x):
        """Compute ∇⁶ (6th order Laplacian)."""
        return self.laplacian(self.biharmonic(x))

    def potential_prime(self, phi, k):
        """Derivative of potential: V'(ϕ) = r*ϕ + u*ϕ³."""
        return self.r[k] * phi + self.u[k] * phi ** 3

    def cubic_coupling(self):
        """Cubic coupling between fields."""
        total = torch.sum(self.phi, dim=0, keepdim=True)
        return 0.02 * self.phi * total

    def shell_penalty_real(self, phi):
        """Approximate shell operator penalty in real space."""
        lap = self.laplacian(phi)
        return lap * lap.mean()

    def functional_derivative(self):
        """Compute δF/δϕ (functional derivative of free energy)."""
        C = self.cubic_coupling()
        dF = torch.zeros_like(self.phi)

        for k in range(self.nfields):
            dF[k] = (
                -self.laplacian(self.phi[k])
                + self.potential_prime(self.phi[k], k)
                + 2.0 * self.alpha[k] * self.biharmonic(self.phi[k])
                - 2.0 * self.beta[k] * self.sixth_order(self.phi[k])
                + C[k]
                + self.lam_shell[k] * self.shell_penalty_real(self.phi[k])
            )
        return dF

    def step(self, dt: float = 1e-2, mobility: float = 1.0):
        """Gradient-flow evolution: ∂_t ϕ = -M * δF/δϕ."""
        dF = self.functional_derivative()
        self.phi = self.phi - dt * mobility * dF
        self.phi = torch.clamp(self.phi, -4.0, 4.0)
        return self.phi


def free_energy(phi_model: FreeEnergyFieldModel) -> torch.Tensor:
    """
    Compute free energy functional:
    F[ϕ] = ∫[ 0.5*|∇ϕ|² + 0.5*r*ϕ² + 0.25*u*ϕ⁴ + α*|∇²ϕ|² + β*|∇⁴ϕ|² ]
    """
    phi = phi_model.phi
    energy = torch.tensor(0.0, device=phi_model.device)

    for k in range(phi_model.nfields):
        # Gradient energy
        grad_x = torch.roll(phi[k], -1, dims=-1) - phi[k]
        grad_y = torch.roll(phi[k], -1, dims=-2) - phi[k]
        energy = energy + 0.5 * (grad_x ** 2 + grad_y ** 2).mean()

        # Potential energy
        energy = energy + 0.5 * phi_model.r[k] * (phi[k] ** 2).mean()
        energy = energy + 0.25 * phi_model.u[k] * (phi[k] ** 4).mean()

        # Higher-order regularization
        lap = phi_model.laplacian(phi[k])
        energy = energy + phi_model.alpha[k] * (lap ** 2).mean()
        energy = energy + phi_model.beta[k] * (phi_model.laplacian(lap) ** 2).mean()

    return energy


# --- Core Theoretical Engines ---
class OscillatorySynapseTheory:
    def __init__(self, field_size: int = 100, device: str = 'cpu'):
        self.device = torch.device(device)
        self.field_size = int(field_size)
        self.atlantean = AtlanteanHRMElement(self.field_size, self.device)
        self.field = self.atlantean.current_field()

        SimulationEngine = _load_hrm_simulation_engine_class()
        hrm_side = int(np.ceil(np.sqrt(max(36, self.field_size))))
        self.hrm_engine = SimulationEngine(field_shape=(hrm_side, hrm_side), seed=42)
        self.hrm_state = self.hrm_engine.initialize()
        self.last_hrm_metrics: Dict[str, float] = {}

        # Compatibility proxies: external modules still use nn1/nn2/nn3.
        self.nn1 = _OscillatorProxyNN(self, "update_field").to(self.device)
        self.nn2 = _OscillatorProxyNN(self, "adjust_syntropy").to(self.device)
        self.nn3 = _OscillatorProxyNN(self, "penalize_variance").to(self.device)
        self.optimizer1 = optim.Adam(self.nn1.parameters(), lr=0.001)
        self.optimizer2 = optim.Adam(self.nn2.parameters(), lr=0.001)
        self.optimizer3 = optim.Adam(self.nn3.parameters(), lr=0.001)
        self.optimizer_interaction = optim.Adam(self.atlantean.interaction_coupler.parameters(), lr=0.001)
        self.balanced = False
        self.message_queue = deque(maxlen=10)
        
        # Free energy field model for thermodynamic evolution
        field_grid_size = int(np.ceil(np.sqrt(self.field_size)))
        self.free_energy_model = FreeEnergyFieldModel(
            nfields=2, 
            H=field_grid_size, 
            W=field_grid_size, 
            device=device
        )
        self.prev_free_energy = free_energy(self.free_energy_model).detach()
        
        # Q-learning setup
        self.q_table = {}  # State-action value table
        self.actions = ['update_field', 'adjust_syntropy', 'penalize_variance']
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def _step_hrm(self, signal: torch.Tensor, dt: float = 0.02) -> float:
        vec = signal.detach().flatten().cpu().numpy()
        n = min(len(vec), self.hrm_state.state_vector.size)
        if n > 0:
            self.hrm_state.state_vector[:n] = 0.85 * self.hrm_state.state_vector[:n] + 0.15 * vec[:n]
        self.hrm_state, self.last_hrm_metrics = self.hrm_engine.step(self.hrm_state, dt=dt)
        return float(np.mean(self.hrm_state.field))

    def _route_through_atlantean_hrm(self, signal: torch.Tensor, mode: str) -> torch.Tensor:
        x = signal.to(self.device).flatten()
        if x.numel() < self.field_size:
            x = F.pad(x, (0, self.field_size - x.numel()))
        else:
            x = x[: self.field_size]

        hrm_force = self._step_hrm(x)
        stability_hint = float(self.last_hrm_metrics.get('f_stability', 0.0)) if self.last_hrm_metrics else None
        spikes = self.atlantean.step(x, hrm_force=hrm_force, dt=0.01, stability_hint=stability_hint)
        phi5 = self.atlantean.phi5[: self.field_size]
        interaction_signal = self.atlantean.last_interaction_signal[: self.field_size]

        if mode == "adjust_syntropy":
            out = 0.6 * spikes + 0.4 * torch.tanh(phi5) + 0.10 * interaction_signal
        elif mode == "penalize_variance":
            out = spikes - 0.1 * torch.var(spikes) + 0.10 * interaction_signal
        else:
            out = spikes + 0.10 * interaction_signal

        self.field = self.atlantean.current_field()
        return out

    def generate_synthetic_data(self, num_samples: int = 100) -> torch.Tensor:
        return torch.randn(num_samples, self.field.size(0), device=self.device) * 0.1

    def process_streamed_data(self, text: str) -> torch.Tensor:
        words = text.split()
        word_counts = Counter(words)
        feature_vector = np.array([word_counts.get(word, 0) for word in sorted(word_counts)[:self.field.size(0)]])
        if len(feature_vector) < self.field.size(0):
            feature_vector = np.pad(feature_vector, (0, self.field.size(0) - len(feature_vector)))
        return torch.tensor(feature_vector[:self.field.size(0)], dtype=torch.float32, device=self.device)

    def get_state_key(self, field: torch.Tensor) -> str:
        """Discretize field state for Q-learning."""
        return str(np.round(field.detach().cpu().numpy().mean(), 2))

    def choose_action(self, state: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table.get(state, {a: 0 for a in self.actions}).items(), key=lambda x: x[1])[0]

    def update_q_table(self, reward: float, old_state: str = None, action: str = None):
        if old_state and action:
            if old_state not in self.q_table:
                self.q_table[old_state] = {a: 0 for a in self.actions}
            new_state = self.get_state_key(self.field)
            if new_state not in self.q_table:
                self.q_table[new_state] = {a: 0 for a in self.actions}
            self.q_table[old_state][action] += self.alpha * (
                reward + self.gamma * max(self.q_table[new_state].values()) - self.q_table[old_state][action]
            )

    def train(self, synthetic_data: torch.Tensor, streamed_text: Optional[str] = None, epochs: int = 10) -> float:
        logger.info("Starting Atlantean-HRM routed training with free energy descent...")
        total_reward = 0
        for epoch in range(epochs):
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()
            self.optimizer_interaction.zero_grad()
            
            # Store free energy before evolution step
            prev_F = free_energy(self.free_energy_model).detach()
            
            total_loss = 0
            task_loss = 0
            state = self.get_state_key(self.field)
            epoch_reward = 0.0
            
            for data in synthetic_data:
                action = self.choose_action(state)
                if action == 'update_field':
                    field = self.nn1.update_field(data)
                elif action == 'adjust_syntropy':
                    field = self.nn2.adjust_syntropy(data)
                else:
                    field = self.nn3.penalize_variance(data)
                
                # Task loss: variance reduction + mean centering + chaos regularizer
                task_loss_step = (
                    torch.var(field)
                    + torch.abs(torch.mean(field) - 0.5)
                    + chaos_regularizer(field)
                )
                task_loss += task_loss_step
                
                # Energy descent reward signal
                state = self.get_state_key(field)
            
            if streamed_text:
                streamed_data = self.process_streamed_data(streamed_text)
                action = self.choose_action(state)
                if action == 'update_field':
                    field = self.nn1.update_field(streamed_data)
                elif action == 'adjust_syntropy':
                    field = self.nn2.adjust_syntropy(streamed_data)
                else:
                    field = self.nn3.penalize_variance(streamed_data)
                
                task_loss_step = (
                    torch.var(field)
                    + torch.abs(torch.mean(field) - 0.5)
                    + chaos_regularizer(field)
                )
                task_loss += task_loss_step
                state = self.get_state_key(field)
            
            received_state = self.receive_state()
            if received_state is not None:
                action = self.choose_action(state)
                if action == 'update_field':
                    field = self.nn1.update_field(received_state)
                elif action == 'adjust_syntropy':
                    field = self.nn2.adjust_syntropy(received_state)
                else:
                    field = self.nn3.penalize_variance(received_state)
                
                task_loss_step = (
                    torch.var(field)
                    + torch.abs(torch.mean(field) - 0.5)
                    + chaos_regularizer(field)
                )
                task_loss += task_loss_step
                state = self.get_state_key(field)
            
            # Evolve field according to free energy gradient flow
            self.free_energy_model.step(dt=0.01, mobility=1.0)
            
            # Compute new free energy and descent penalty
            new_F = free_energy(self.free_energy_model)
            descent_penalty = torch.relu(new_F - prev_F)
            
            # Compute total loss: task + free energy + descent penalty
            denom = max(1, len(synthetic_data) + (1 if streamed_text else 0) + (1 if received_state is not None else 0))
            task_loss = task_loss / denom
            
            total_loss = task_loss + 0.1 * new_F + 1.0 * descent_penalty
            
            # Compute reward from energy descent
            energy_descent = prev_F.item() - new_F.item()
            epoch_reward = 1.0 / (1.0 + new_F.item()) + 0.5 * max(0, energy_descent)
            total_reward += epoch_reward
            
            # Q-table update: reward based on achieving lower energy
            state_key = self.get_state_key(self.field)
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0.0 for action in self.actions}
            for action in self.actions:
                if action not in self.q_table[state_key]:
                    self.q_table[state_key][action] = 0.0
                self.q_table[state_key][action] += self.alpha * (epoch_reward - self.q_table[state_key][action])
            
            # Lightweight regularization
            reg = 1e-4 * (
                self.nn1.gain.pow(2) + self.nn2.gain.pow(2) + self.nn3.gain.pow(2)
            )
            (total_loss + reg).backward()
            self.optimizer1.step()
            self.optimizer2.step()
            self.optimizer3.step()
            self.optimizer_interaction.step()
            
            # Store energy for next epoch
            self.prev_free_energy = new_F.detach()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Task Loss: {task_loss.item():.4f}, Free Energy: {new_F.item():.4f}, Energy Descent: {energy_descent:.4f}, Reward: {epoch_reward:.4f}")
        return total_reward / max(1, epochs)

    def share_state(self):
        self.message_queue.append(self.field.clone())
        logger.info("Shared spiking neural network field state.")

    def receive_state(self) -> Optional[torch.Tensor]:
        if self.message_queue:
            return self.message_queue.popleft()
        return None

    def save_weights(self, path: str = "nn_weights.pth"):
        torch.save({
            'nn1': self.nn1.state_dict(),
            'nn2': self.nn2.state_dict(),
            'nn3': self.nn3.state_dict(),
            'interaction_coupler': self.atlantean.interaction_coupler.state_dict(),
            'field': self.field,
            'q_table': self.q_table,
            'phi1': self.atlantean.phi1,
            'phi5': self.atlantean.phi5,
            'Phi': self.atlantean.Phi,
            'membrane': self.atlantean.membrane,
            'cached_stability': self.atlantean.cached_stability,
            'cached_variance': self.atlantean.cached_variance,
            'cached_phi1_mean': self.atlantean.cached_phi1_mean,
            'cached_phi5_mean': self.atlantean.cached_phi5_mean,
            'last_interaction_signal': self.atlantean.last_interaction_signal,
            'last_run_full_fields': self.atlantean.last_run_full_fields,
            'step_idx': self.atlantean.step_idx,
            'hrm_state_vector': torch.tensor(self.hrm_state.state_vector, dtype=torch.float32),
        }, path)
        logger.info(f"Saved neural network weights to {path}")
    
    def load_weights(self, path: str = "nn_weights.pth"):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only.
            checkpoint = torch.load(path, map_location=self.device)
        if 'nn1' in checkpoint:
            self.nn1.load_state_dict(checkpoint['nn1'], strict=False)
        if 'nn2' in checkpoint:
            self.nn2.load_state_dict(checkpoint['nn2'], strict=False)
        if 'nn3' in checkpoint:
            self.nn3.load_state_dict(checkpoint['nn3'], strict=False)
        if 'interaction_coupler' in checkpoint:
            self.atlantean.interaction_coupler.load_state_dict(checkpoint['interaction_coupler'], strict=False)
        self.field = checkpoint.get('field', self.field)
        self.q_table = checkpoint.get('q_table', {})
        if 'phi1' in checkpoint:
            self.atlantean.phi1 = checkpoint['phi1'].to(self.device)
        if 'phi5' in checkpoint:
            self.atlantean.phi5 = checkpoint['phi5'].to(self.device)
        if 'Phi' in checkpoint:
            self.atlantean.Phi = checkpoint['Phi'].to(self.device)
        if 'membrane' in checkpoint:
            self.atlantean.membrane = checkpoint['membrane'].to(self.device)
        if 'cached_stability' in checkpoint:
            self.atlantean.cached_stability = checkpoint['cached_stability'].to(self.device)
        if 'cached_variance' in checkpoint:
            self.atlantean.cached_variance = checkpoint['cached_variance'].to(self.device)
        if 'cached_phi1_mean' in checkpoint:
            self.atlantean.cached_phi1_mean = checkpoint['cached_phi1_mean'].to(self.device)
        if 'cached_phi5_mean' in checkpoint:
            self.atlantean.cached_phi5_mean = checkpoint['cached_phi5_mean'].to(self.device)
        if 'last_interaction_signal' in checkpoint:
            self.atlantean.last_interaction_signal = checkpoint['last_interaction_signal'].to(self.device)
        if 'last_run_full_fields' in checkpoint:
            self.atlantean.last_run_full_fields = bool(checkpoint['last_run_full_fields'])
        if 'step_idx' in checkpoint:
            self.atlantean.step_idx = int(checkpoint['step_idx'])
        if 'hrm_state_vector' in checkpoint:
            vec = checkpoint['hrm_state_vector'].detach().cpu().numpy()
            n = min(len(vec), self.hrm_state.state_vector.size)
            self.hrm_state.state_vector[:n] = vec[:n]
        self.field = self.atlantean.current_field()
        logger.info(f"Loaded neural network weights from {path}")

    def benchmark_gated_path(
        self,
        num_steps: int = 400,
        warmup_steps: int = 40,
        dt: float = 0.01,
        stability_hint: Optional[float] = 0.95,
    ) -> Dict[str, float]:
        """Engine-level wrapper for gated-path micro-benchmark."""
        return self.atlantean.benchmark_gated_path(
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            dt=dt,
            stability_hint=stability_hint,
        )