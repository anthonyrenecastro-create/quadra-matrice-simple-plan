#!/usr/bin/env python3
"""
CognitionSim - GPU Optimized Version
High-performance neural field processing with CUDA acceleration and multi-core support.

Features:
- Automatic GPU detection and allocation
- Multi-GPU support with data parallelism
- Mixed precision training (FP16)
- Asynchronous data transfers
- Memory-efficient batching
- CPU fallback for compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import logging
import importlib.util
import sys
from collections import deque
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import os
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    module_name = "quadra_hrm_engine_runtime_gpu"
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


class AtlanteanHRMGPUElement(nn.Module):
    """Atlantean field solver routed through HRM dynamics for GPU/CPU execution."""

    def __init__(self, field_size: int, device: torch.device):
        super().__init__()
        self.field_size = int(field_size)
        self.device = device
        self.grid_side = int(np.ceil(np.sqrt(self.field_size)))
        self.grid_size = self.grid_side * self.grid_side

        self.phi1 = torch.randn(self.grid_size, device=self.device) * 0.05
        self.phi5 = torch.ones(self.grid_size, device=self.device) * 0.1
        self.Phi = torch.zeros(self.grid_size, device=self.device)
        self.membrane = torch.zeros(self.grid_size, device=self.device)

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
        # ── Gate / mode config ────────────────────────────────────────────────
        # GPU benefits from a longer interval to amortise kernel launches
        self.gate_interval: int = 16
        self._cached_gate: bool = True
        # Mode: "CALM" | "RECOVERY" | "CHAOTIC"
        self.gate_mode: str = "CALM"
        self._CALM_RESONANCE: float = 0.15
        self._CHAOS_RESONANCE: float = 0.80
        self._INTER_FULL: float = 0.10
        # Hysteresis
        self._GATE_CLOSE_N: int = 4   # fewer steps needed on GPU (longer intervals)
        self._close_streak: int = 0
        self._variance_history: deque = deque(maxlen=16)
        # Resonance damping
        self.resonance_decay: float = 0.97
        self._PHI_MAX_NORM: float = 3.5
        self.resonance_loss: float = 0.0
        self._RESONANCE_LAMBDA: float = 0.05
        # Recovery duty: PDE fires if gate_epoch % _recovery_period < _recovery_active
        self._recovery_period: int = 2
        self._recovery_active: int = 1

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
        return x.reshape(-1)[: self.grid_size]

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

        # Norm clipping — prevents PDE blow-up that causes latency spikes
        phi1_norm = self.phi1.norm()
        if phi1_norm > self._PHI_MAX_NORM:
            self.phi1 = self.phi1 * (self._PHI_MAX_NORM / phi1_norm)
        # Resonance damping
        self.phi1 = self.phi1 * self.resonance_decay
        self.phi5 = self.phi5 * self.resonance_decay

    def step(
        self,
        input_signal: torch.Tensor,
        hrm_force: float = 0.0,
        dt: float = 0.01,
        stability_hint: Optional[float] = None,
    ) -> torch.Tensor:
        x = self._prepare_signal(input_signal)

        # ── Mode-based gate: once per gate_interval steps ──────────────────
        is_gate_step = (self.step_idx % self.gate_interval == 0)
        if is_gate_step:
            resonance: float = self.phi1.abs().mean().item()  # one sync per interval

            var_now = float(self.cached_variance.item())
            self._variance_history.append(var_now)
            variance_slope = 0.0
            if len(self._variance_history) >= 4:
                h = list(self._variance_history)
                half = len(h) // 2
                variance_slope = sum(h[half:]) / (len(h) - half) - sum(h[:half]) / half

            if resonance > self._CHAOS_RESONANCE:
                candidate = "CHAOTIC"
            elif resonance < self._CALM_RESONANCE and variance_slope < 0.0:
                candidate = "CALM"
            else:
                candidate = "RECOVERY"

            if candidate == "CALM":
                self._close_streak += 1
                if self._close_streak < self._GATE_CLOSE_N:
                    candidate = self.gate_mode if self.gate_mode == "CALM" else "RECOVERY"
            else:
                self._close_streak = 0

            self.gate_mode = candidate

            if self.gate_mode == "CALM":
                run_full_fields = False
                interaction_signal = self.last_interaction_signal
            elif self.gate_mode == "CHAOTIC":
                run_full_fields = True
                with torch.no_grad():
                    interaction_signal = self._interaction_signal(x)
            else:  # RECOVERY: partial — alternate PDE on/off per gate epoch
                with torch.no_grad():
                    interaction_signal = self._interaction_signal(x)
                gate_epoch = self.step_idx // self.gate_interval
                run_full_fields = (gate_epoch % self._recovery_period < self._recovery_active)

            self._cached_gate = run_full_fields
            self.last_interaction_signal = interaction_signal.detach()
        else:
            run_full_fields = self._cached_gate
            interaction_signal = self.last_interaction_signal
        # ─────────────────────────────────────────────────────────────────────

        if run_full_fields:
            self.pde_step(hrm_force=hrm_force, dt=dt)
            self._refresh_cached_stats(stability_hint)

        # Residual resonance penalty for training integration
        self.resonance_loss = float(self._RESONANCE_LAMBDA * self.phi1.abs().mean().item())

        threshold = (
            self.base_threshold
            - 0.5 * self.phi1
            - 0.12 * interaction_signal
            + 0.08 * self.cached_variance
            - 0.08 * self.cached_stability
        )
        leak = torch.clamp(
            self.base_leak * (1.0 / (1.0 + torch.relu(self.phi5)))
            + 0.06 * torch.sigmoid(interaction_signal)
            + 0.03 * self.cached_stability,
            0.70,
            0.99,
        )

        self.membrane = self.membrane * (1.0 - leak) + x + 0.15 * interaction_signal
        spikes = (self.membrane > threshold).float()
        self.membrane = self.membrane * (1.0 - spikes)
        self.Phi = 0.9 * self.Phi + 0.1 * spikes
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
        """Micro-benchmark gated updates vs forced full PDE updates."""
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        def _sync_if_cuda() -> None:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

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
                'gate_mode': str(self.gate_mode),
                '_cached_gate': bool(self._cached_gate),
                '_close_streak': int(self._close_streak),
                '_variance_history': list(self._variance_history),
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
            self.gate_mode = str(state['gate_mode'])
            self._cached_gate = bool(state['_cached_gate'])
            self._close_streak = int(state['_close_streak'])
            self._variance_history = deque(state['_variance_history'], maxlen=16)

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

                _sync_if_cuda()
                t_start = time.perf_counter()
                for _ in range(num_steps):
                    if force_full:
                        self.cached_stability.fill_(0.0)
                    _ = self.step(signal, hrm_force=0.0, dt=dt, stability_hint=stability_hint)
                    if self.last_run_full_fields:
                        full += 1
                    else:
                        skipped += 1
                _sync_if_cuda()
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

        import time

        base_state = _snapshot_state()
        signal = torch.zeros(self.field_size, device=self.device)

        _restore_state(base_state)
        gated = _run_timed_pass(force_full=False, neutralize_interaction=False)

        _restore_state(base_state)
        forced = _run_timed_pass(force_full=True, neutralize_interaction=False)

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


@dataclass
class GPUConfig:
    """GPU optimization configuration"""
    use_cuda: bool = True
    mixed_precision: bool = True
    num_gpus: int = 0  # 0 = auto-detect
    batch_size: int = 32
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    def __post_init__(self):
        if self.use_cuda and torch.cuda.is_available():
            if self.num_gpus == 0:
                self.num_gpus = torch.cuda.device_count()
            logger.info(f"🚀 GPU acceleration enabled: {self.num_gpus} GPU(s) detected")
            for i in range(self.num_gpus):
                logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        else:
            self.use_cuda = False
            self.mixed_precision = False
            logger.warning("⚠️  CUDA not available. Running on CPU.")


class GPUOptimizedField(nn.Module):
    """GPU-optimized neural field with automatic device management"""
    
    def __init__(self, field_size: int = 100, hidden_size: int = 256, config: Optional[GPUConfig] = None):
        super().__init__()
        self.config = config or GPUConfig()
        self.field_size = field_size
        self.hidden_size = hidden_size
        
        # Determine device
        if self.config.use_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        # Build network with GPU-optimized layers
        self.encoder = nn.Sequential(
            nn.Linear(field_size, hidden_size),
            nn.LayerNorm(hidden_size),  # More stable than BatchNorm on GPU
            nn.GELU(),  # GPU-optimized activation
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
        ).to(self.device)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, field_size),
        ).to(self.device)
        
        # Field state on GPU
        self.field = torch.randn(field_size, device=self.device) * 0.1
        
        # Multi-GPU support
        if self.config.num_gpus > 1:
            self.encoder = DataParallel(self.encoder)
            self.decoder = DataParallel(self.decoder)
            logger.info(f"✨ Multi-GPU mode: Using {self.config.num_gpus} GPUs")
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        logger.info(f"🧠 GPU-Optimized Field initialized on {self.device}")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic mixed precision"""
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.config.mixed_precision and self.device.type == 'cuda',
        ):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        return decoded
    
    def update_field(self, input_data: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
        """GPU-accelerated field update"""
        with torch.no_grad():
            if input_data.device != self.device:
                input_data = input_data.to(self.device, non_blocking=True)
            
            # Compute update
            update = self.forward(input_data.unsqueeze(0) if input_data.dim() == 1 else input_data)
            
            # Apply update with momentum
            self.field = self.field + learning_rate * (update.squeeze() - self.field)
            
        return self.field.clone()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {"device": "cpu"}
        
        return {
            "device": str(self.device),
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1e6,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1e6,
        }


class GPUSpikingNetwork(nn.Module):
    """GPU-optimized spiking neural network with batch processing"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_steps: int = 10, 
                 config: Optional[GPUConfig] = None):
        super().__init__()
        self.config = config or GPUConfig()
        self.num_steps = num_steps
        
        if self.config.use_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        # Spike-compatible layers
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.fc3 = nn.Linear(hidden_size, input_size).to(self.device)
        
        # Learnable threshold and decay
        self.threshold = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.decay = nn.Parameter(torch.tensor(0.95, device=self.device))
        
        if self.config.num_gpus > 1:
            self.fc1 = DataParallel(self.fc1)
            self.fc2 = DataParallel(self.fc2)
            self.fc3 = DataParallel(self.fc3)
    
    def spike_activation(self, membrane: torch.Tensor) -> tuple:
        """GPU-optimized spike generation"""
        spikes = (membrane > self.threshold).float()
        membrane = membrane * (1 - spikes) * self.decay
        return spikes, membrane
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process spikes across time steps on GPU"""
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        
        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Initialize membrane potentials on GPU
        mem1 = torch.zeros(batch_size, self.fc1.out_features if not isinstance(self.fc1, DataParallel) 
                          else self.fc1.module.out_features, device=self.device)
        mem2 = torch.zeros(batch_size, self.fc2.out_features if not isinstance(self.fc2, DataParallel)
                          else self.fc2.module.out_features, device=self.device)
        mem3 = torch.zeros(batch_size, self.fc3.out_features if not isinstance(self.fc3, DataParallel)
                          else self.fc3.module.out_features, device=self.device)
        
        spike_history = []
        
        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.config.mixed_precision and self.device.type == 'cuda',
        ):
            # Vectorized spike processing on GPU
            for _ in range(self.num_steps):
                # Layer 1
                mem1 = mem1 + self.fc1(x)
                spk1, mem1 = self.spike_activation(mem1)

                # Layer 2
                mem2 = mem2 + self.fc2(spk1)
                spk2, mem2 = self.spike_activation(mem2)

                # Layer 3
                mem3 = mem3 + self.fc3(spk2)
                spk3, mem3 = self.spike_activation(mem3)

                spike_history.append(spk3)

            # Aggregate spikes efficiently on GPU
            output = torch.stack(spike_history).mean(dim=0)
        
        return output.squeeze() if batch_size == 1 else output


class GPUOscillatorEngine:
    """High-performance oscillator engine with GPU acceleration"""
    
    def __init__(self, field_size: int = 100, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self.field_size = field_size
        
        # GPU-optimized components
        self.field_network = GPUOptimizedField(field_size, config=self.config)
        self.spike_network = GPUSpikingNetwork(field_size, config=self.config)
        self.device = self.field_network.device

        # New routed core: Atlantean field dynamics + HRM manifold.
        self.atlantean = AtlanteanHRMGPUElement(field_size, self.device)
        SimulationEngine = _load_hrm_simulation_engine_class()
        hrm_side = int(np.ceil(np.sqrt(max(36, field_size))))
        self.hrm_engine = SimulationEngine(field_shape=(hrm_side, hrm_side), seed=42)
        self.hrm_state = self.hrm_engine.initialize()
        self.last_hrm_metrics: Dict[str, float] = {}
        
        # Optimizers with GPU-friendly settings
        self.optimizer_field = optim.AdamW(
            self.field_network.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            foreach=True  # GPU optimization
        )
        
        self.optimizer_spike = optim.AdamW(
            self.spike_network.parameters(),
            lr=0.001,
            foreach=True
        )
        self.optimizer_interaction = optim.AdamW(
            self.atlantean.interaction_coupler.parameters(),
            lr=0.001,
            foreach=True
        )
        
        # Mixed precision scalers
        self.scaler_field = GradScaler() if self.config.mixed_precision else None
        self.scaler_spike = GradScaler() if self.config.mixed_precision else None
        
        logger.info("⚡ GPU Oscillator Engine ready")

    def _step_hrm(self, signal: torch.Tensor, dt: float = 0.02) -> float:
        vec = signal.detach().flatten().cpu().numpy()
        n = min(len(vec), self.hrm_state.state_vector.size)
        if n > 0:
            self.hrm_state.state_vector[:n] = 0.85 * self.hrm_state.state_vector[:n] + 0.15 * vec[:n]
        self.hrm_state, self.last_hrm_metrics = self.hrm_engine.step(self.hrm_state, dt=dt)
        return float(np.mean(self.hrm_state.field))

    def _route_batch_through_atlantean_hrm(self, batch: torch.Tensor) -> torch.Tensor:
        outputs = []
        for row in batch:
            hrm_force = self._step_hrm(row)
            stability_hint = float(self.last_hrm_metrics.get('f_stability', 0.0)) if self.last_hrm_metrics else None
            routed = self.atlantean.step(row, hrm_force=hrm_force, dt=0.01, stability_hint=stability_hint)
            outputs.append(routed)
        return torch.stack(outputs, dim=0)
    
    def process_batch(self, batch: torch.Tensor, return_spikes: bool = False) -> Dict[str, torch.Tensor]:
        """Process a batch of data on GPU"""
        device = self.device
        
        if batch.device != device:
            batch = batch.to(device, non_blocking=True)
        
        # Route everything through the Atlantean-HRM core.
        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.config.mixed_precision and device.type == 'cuda',
        ):
            spike_output = self._route_batch_through_atlantean_hrm(batch)
            field_output = spike_output

            reconstruction_loss = nn.functional.mse_loss(field_output, batch)
            spike_loss = torch.var(spike_output) + torch.abs(spike_output.mean() - 0.5)
            total_loss = reconstruction_loss + 0.1 * spike_loss
        
        result = {
            'field_output': field_output.detach(),
            'loss': total_loss.detach(),
        }
        
        if return_spikes:
            result['spike_output'] = spike_output.detach()
        
        return result
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step with gradient accumulation"""
        device = self.device
        
        if batch.device != device:
            batch = batch.to(device, non_blocking=True)
        
        # Forward pass with automatic mixed precision
        with torch.amp.autocast(
            device_type="cuda",
            enabled=self.config.mixed_precision and device.type == 'cuda',
        ):
            spike_output = self._route_batch_through_atlantean_hrm(batch)
            field_output = spike_output

            reconstruction_loss = nn.functional.mse_loss(field_output, batch)
            spike_loss = torch.var(spike_output) + torch.abs(spike_output.mean() - 0.5)
            # Keep legacy optimizers active with tiny regularization drive.
            reg = 1e-6 * (
                sum((p ** 2).mean() for p in self.field_network.parameters())
                + sum((p ** 2).mean() for p in self.spike_network.parameters())
            )
            total_loss = reconstruction_loss + 0.1 * spike_loss + reg
        
        # Backward pass with gradient scaling
        self.optimizer_field.zero_grad(set_to_none=True)  # More efficient on GPU
        self.optimizer_spike.zero_grad(set_to_none=True)
        self.optimizer_interaction.zero_grad(set_to_none=True)
        
        if self.scaler_field:
            self.scaler_field.scale(total_loss).backward()
            self.scaler_field.step(self.optimizer_field)
            self.scaler_field.step(self.optimizer_spike)
            self.scaler_field.step(self.optimizer_interaction)
            self.scaler_field.update()
        else:
            total_loss.backward()
            self.optimizer_field.step()
            self.optimizer_spike.step()
            self.optimizer_interaction.step()
        
        return total_loss.item()
    
    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark GPU performance"""
        import time
        
        device = self.field_network.device
        batch = torch.randn(self.config.batch_size, self.field_size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = self.process_batch(batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            _ = self.train_step(batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        return {
            'iterations': num_iterations,
            'total_time_s': elapsed,
            'time_per_iteration_ms': (elapsed / num_iterations) * 1000,
            'throughput_samples_per_sec': (num_iterations * self.config.batch_size) / elapsed,
            'device': str(device),
        }

    def benchmark_gated_path(
        self,
        num_steps: int = 400,
        warmup_steps: int = 40,
        dt: float = 0.01,
        stability_hint: Optional[float] = 0.95,
    ) -> Dict[str, float]:
        """Engine-level wrapper for gated-path micro-benchmark."""
        metrics = self.atlantean.benchmark_gated_path(
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            dt=dt,
            stability_hint=stability_hint,
        )
        metrics['device'] = str(self.device)
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        config_payload = {
            'use_cuda': self.config.use_cuda,
            'mixed_precision': self.config.mixed_precision,
            'num_gpus': self.config.num_gpus,
            'batch_size': self.config.batch_size,
            'pin_memory': self.config.pin_memory,
            'num_workers': self.config.num_workers,
            'prefetch_factor': self.config.prefetch_factor,
            'persistent_workers': self.config.persistent_workers,
        }
        torch.save({
            'field_network': self.field_network.state_dict(),
            'spike_network': self.spike_network.state_dict(),
            'interaction_coupler': self.atlantean.interaction_coupler.state_dict(),
            'optimizer_field': self.optimizer_field.state_dict(),
            'optimizer_spike': self.optimizer_spike.state_dict(),
            'optimizer_interaction': self.optimizer_interaction.state_dict(),
            'atlantean_phi1': self.atlantean.phi1,
            'atlantean_phi5': self.atlantean.phi5,
            'atlantean_Phi': self.atlantean.Phi,
            'atlantean_membrane': self.atlantean.membrane,
            'atlantean_cached_stability': self.atlantean.cached_stability,
            'atlantean_cached_variance': self.atlantean.cached_variance,
            'atlantean_cached_phi1_mean': self.atlantean.cached_phi1_mean,
            'atlantean_cached_phi5_mean': self.atlantean.cached_phi5_mean,
            'atlantean_last_interaction_signal': self.atlantean.last_interaction_signal,
            'atlantean_last_run_full_fields': self.atlantean.last_run_full_fields,
            'atlantean_step_idx': self.atlantean.step_idx,
            'hrm_state_vector': torch.tensor(self.hrm_state.state_vector, dtype=torch.float32),
            'config': config_payload,
        }, path)
        logger.info(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except pickle.UnpicklingError:
            # Legacy checkpoints may include custom classes (for example GPUConfig).
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only.
            checkpoint = torch.load(path, map_location=self.device)
        self.field_network.load_state_dict(checkpoint['field_network'])
        self.spike_network.load_state_dict(checkpoint['spike_network'])
        if 'interaction_coupler' in checkpoint:
            self.atlantean.interaction_coupler.load_state_dict(checkpoint['interaction_coupler'])
        self.optimizer_field.load_state_dict(checkpoint['optimizer_field'])
        self.optimizer_spike.load_state_dict(checkpoint['optimizer_spike'])
        if 'optimizer_interaction' in checkpoint:
            self.optimizer_interaction.load_state_dict(checkpoint['optimizer_interaction'])
        if 'atlantean_phi1' in checkpoint:
            self.atlantean.phi1 = checkpoint['atlantean_phi1'].to(self.device)
        if 'atlantean_phi5' in checkpoint:
            self.atlantean.phi5 = checkpoint['atlantean_phi5'].to(self.device)
        if 'atlantean_Phi' in checkpoint:
            self.atlantean.Phi = checkpoint['atlantean_Phi'].to(self.device)
        if 'atlantean_membrane' in checkpoint:
            self.atlantean.membrane = checkpoint['atlantean_membrane'].to(self.device)
        if 'atlantean_cached_stability' in checkpoint:
            self.atlantean.cached_stability = checkpoint['atlantean_cached_stability'].to(self.device)
        if 'atlantean_cached_variance' in checkpoint:
            self.atlantean.cached_variance = checkpoint['atlantean_cached_variance'].to(self.device)
        if 'atlantean_cached_phi1_mean' in checkpoint:
            self.atlantean.cached_phi1_mean = checkpoint['atlantean_cached_phi1_mean'].to(self.device)
        if 'atlantean_cached_phi5_mean' in checkpoint:
            self.atlantean.cached_phi5_mean = checkpoint['atlantean_cached_phi5_mean'].to(self.device)
        if 'atlantean_last_interaction_signal' in checkpoint:
            self.atlantean.last_interaction_signal = checkpoint['atlantean_last_interaction_signal'].to(self.device)
        if 'atlantean_last_run_full_fields' in checkpoint:
            self.atlantean.last_run_full_fields = bool(checkpoint['atlantean_last_run_full_fields'])
        if 'atlantean_step_idx' in checkpoint:
            self.atlantean.step_idx = int(checkpoint['atlantean_step_idx'])
        if 'hrm_state_vector' in checkpoint:
            vec = checkpoint['hrm_state_vector'].detach().cpu().numpy()
            n = min(len(vec), self.hrm_state.state_vector.size)
            self.hrm_state.state_vector[:n] = vec[:n]
        logger.info(f"📂 Checkpoint loaded: {path}")


def get_optimal_config() -> GPUConfig:
    """Auto-configure based on available hardware"""
    config = GPUConfig()
    
    if torch.cuda.is_available():
        # Adjust batch size based on GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory_gb >= 16:  # High-end GPU
            config.batch_size = 64
            config.num_workers = 8
        elif gpu_memory_gb >= 8:  # Mid-range GPU
            config.batch_size = 32
            config.num_workers = 4
        else:  # Entry-level GPU
            config.batch_size = 16
            config.num_workers = 2
        
        logger.info(f"📊 Auto-configured for {gpu_memory_gb:.1f}GB GPU: batch_size={config.batch_size}")
    
    return config


if __name__ == "__main__":
    # Demo GPU capabilities
    print("\n" + "="*80)
    print("🚀 QUADRA MATRIX GPU OPTIMIZATION DEMO")
    print("="*80 + "\n")
    
    config = get_optimal_config()
    engine = GPUOscillatorEngine(field_size=100, config=config)
    
    print("\n📈 Running benchmark...")
    stats = engine.benchmark(num_iterations=100)
    
    print(f"\n🎯 Performance Results:")
    print(f"   Device: {stats['device']}")
    print(f"   Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"   Time per iteration: {stats['time_per_iteration_ms']:.2f} ms")
    
    mem_stats = engine.field_network.get_memory_stats()
    if 'allocated_mb' in mem_stats:
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {mem_stats['allocated_mb']:.1f} MB")
        print(f"   Reserved: {mem_stats['reserved_mb']:.1f} MB")
    
    print("\n✨ GPU optimization complete!\n")
