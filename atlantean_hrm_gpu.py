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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    def step(self, input_signal: torch.Tensor, hrm_force: float = 0.0, dt: float = 0.01) -> torch.Tensor:
        x = input_signal.flatten()
        if x.numel() < self.grid_size:
            x = F.pad(x, (0, self.grid_size - x.numel()))
        else:
            x = x[: self.grid_size]

        self.pde_step(hrm_force=hrm_force, dt=dt)
        threshold = self.base_threshold - 0.5 * self.phi1
        leak = self.base_leak * (1.0 / (1.0 + torch.relu(self.phi5)))

        self.membrane = self.membrane * (1.0 - leak) + x
        spikes = (self.membrane > threshold).float()
        self.membrane = self.membrane * (1.0 - spikes)
        self.Phi = 0.9 * self.Phi + 0.1 * spikes
        return spikes[: self.field_size]

    def current_field(self) -> torch.Tensor:
        return self.phi1[: self.field_size].clone()


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
    
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic mixed precision"""
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        
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
    
    @torch.cuda.amp.autocast(enabled=True)
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
            routed = self.atlantean.step(row, hrm_force=hrm_force, dt=0.01)
            outputs.append(routed)
        return torch.stack(outputs, dim=0)
    
    def process_batch(self, batch: torch.Tensor, return_spikes: bool = False) -> Dict[str, torch.Tensor]:
        """Process a batch of data on GPU"""
        device = self.device
        
        if batch.device != device:
            batch = batch.to(device, non_blocking=True)
        
        # Route everything through the Atlantean-HRM core.
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
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
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
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
        
        if self.scaler_field:
            self.scaler_field.scale(total_loss).backward()
            self.scaler_field.step(self.optimizer_field)
            self.scaler_field.step(self.optimizer_spike)
            self.scaler_field.update()
        else:
            total_loss.backward()
            self.optimizer_field.step()
            self.optimizer_spike.step()
        
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
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'field_network': self.field_network.state_dict(),
            'spike_network': self.spike_network.state_dict(),
            'optimizer_field': self.optimizer_field.state_dict(),
            'optimizer_spike': self.optimizer_spike.state_dict(),
            'atlantean_phi1': self.atlantean.phi1,
            'atlantean_phi5': self.atlantean.phi5,
            'atlantean_Phi': self.atlantean.Phi,
            'hrm_state_vector': torch.tensor(self.hrm_state.state_vector, dtype=torch.float32),
            'config': self.config,
        }, path)
        logger.info(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.field_network.load_state_dict(checkpoint['field_network'])
        self.spike_network.load_state_dict(checkpoint['spike_network'])
        self.optimizer_field.load_state_dict(checkpoint['optimizer_field'])
        self.optimizer_spike.load_state_dict(checkpoint['optimizer_spike'])
        if 'atlantean_phi1' in checkpoint:
            self.atlantean.phi1 = checkpoint['atlantean_phi1'].to(self.device)
        if 'atlantean_phi5' in checkpoint:
            self.atlantean.phi5 = checkpoint['atlantean_phi5'].to(self.device)
        if 'atlantean_Phi' in checkpoint:
            self.atlantean.Phi = checkpoint['atlantean_Phi'].to(self.device)
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
