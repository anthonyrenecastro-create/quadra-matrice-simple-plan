#!/usr/bin/env python3
"""
CognitionSim Training Script - Optimized for Exponential Speed Growth
Leverages oscillatory dynamics, adaptive K-clustering, neuroplasticity acceleration,
symbolic prediction, and SNN temporal processing for ultra-fast convergence.

Enhanced with Tails-inspired encrypted memory system for persistent state management.
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from atlantean_hrm_spi import (
    PatternModule,
    NeuroplasticityManager,
    SymbolicPredictiveInterpreter,
    SymbolicConfig,
    OscillatorySynapseTheory
)
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json

# Import Tails memory system
try:
    from tails_memory import TailsMemoryManager, MemoryTier, MemoryEnabledTrainer
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("WARNING: Tails memory module not available")

# Import noise injection module
try:
    from utils.noise_injection import NoiseInjector, NoiseType, create_robust_training_noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("WARNING: Noise injection module not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreField:
    """Quantum-inspired field state manager with vibrational modes"""
    def __init__(self, size: int = 100):
        self.state = torch.randn(size) * 0.1
        self.history = deque(maxlen=50)
        self.vibrational_frequency = 1.0
        
    def update_with_vibrational_mode(self, update: np.ndarray):
        """Update field with oscillatory modulation"""
        update_tensor = torch.tensor(update, dtype=torch.float32)
        if update_tensor.shape[0] != self.state.shape[0]:
            update_tensor = torch.nn.functional.pad(
                update_tensor, 
                (0, max(0, self.state.shape[0] - update_tensor.shape[0]))
            )[:self.state.shape[0]]
        
        # Apply vibrational modulation for faster convergence
        phase = len(self.history) * 0.1
        modulation = np.sin(phase * self.vibrational_frequency)
        self.state = self.state * 0.9 + update_tensor * 0.1 * (1 + modulation * 0.2)
        self.history.append(self.state.clone())
        
    def get_state(self) -> torch.Tensor:
        return self.state
    
    def get_resonance_metric(self) -> float:
        """Measure field coherence for adaptive clustering"""
        if len(self.history) < 2:
            return 0.0
        recent = torch.stack(list(self.history)[-10:])
        return 1.0 / (1.0 + torch.std(recent).item())


class SyntropyEngine:
    """Manages multiple coupled fields with syntropy (order emergence)"""
    def __init__(self, num_fields: int = 5, field_size: int = 100):
        self.num_fields = num_fields
        self.field_data = [np.random.randn(field_size) * 0.1 for _ in range(num_fields)]
        self.coupling_strength = 0.1
        
    def couple_fields(self):
        """Enable information flow between fields"""
        for i in range(self.num_fields - 1):
            coupling = (self.field_data[i] + self.field_data[i+1]) * self.coupling_strength
            self.field_data[i] += coupling
            self.field_data[i+1] += coupling


class AdaptiveKClusterOptimizer:
    """Dynamically adjusts K-means clusters based on oscillatory patterns and field resonance"""
    def __init__(self, initial_k: int = 3, max_k: int = 20):
        self.current_k = initial_k
        self.max_k = max_k
        self.min_k = 2
        self.performance_history = deque(maxlen=20)
        self.adaptation_rate = 0.15
        
    def update_k(self, resonance: float, pattern_complexity: float):
        """Adjust K based on field resonance and pattern complexity"""
        # Higher resonance + complexity = need more clusters
        target_k = int(self.min_k + (self.max_k - self.min_k) * resonance * pattern_complexity)
        
        # Smooth transition using oscillatory damping
        delta = target_k - self.current_k
        self.current_k += int(delta * self.adaptation_rate)
        self.current_k = max(self.min_k, min(self.max_k, self.current_k))
        
        return self.current_k
    
    def get_pattern_complexity(self, data: np.ndarray) -> float:
        """Measure data complexity for adaptive clustering"""
        if len(data) < 2:
            return 0.5
        variance = np.var(data)
        entropy = -np.sum(np.abs(data) * np.log(np.abs(data) + 1e-10))
        return np.clip(variance * entropy * 0.01, 0.0, 1.0)


class NeuroplasticityAccelerator:
    """Exponentially increases learning speed through adaptive plasticity"""
    def __init__(self):
        self.base_learning_rate = 0.001
        self.current_learning_rate = self.base_learning_rate
        self.acceleration_factor = 1.0
        self.success_streak = 0
        self.plasticity_threshold = 0.7
        
    def accelerate(self, reward: float, field_stability: float):
        """Exponentially increase learning based on success and stability"""
        if reward > self.plasticity_threshold and field_stability > 0.6:
            self.success_streak += 1
            # Exponential growth with safety bounds
            self.acceleration_factor *= (1.0 + 0.1 * np.sqrt(self.success_streak))
            self.acceleration_factor = min(self.acceleration_factor, 10.0)
        else:
            self.success_streak = max(0, self.success_streak - 1)
            self.acceleration_factor *= 0.95
            self.acceleration_factor = max(self.acceleration_factor, 1.0)
        
        self.current_learning_rate = self.base_learning_rate * self.acceleration_factor
        return self.current_learning_rate
    
    def get_speedup_factor(self) -> float:
        return self.acceleration_factor


class SymbolicSpeedOptimizer:
    """Uses symbolic prediction to skip redundant computations"""
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.prediction_cache = {}
        self.cache_hits = 0
        self.total_queries = 0

    async def predict_and_cache(self, concepts: List[str], neural_output: torch.Tensor, force: bool = False) -> str:
        """Cache symbolic predictions for recurring patterns"""
        self.total_queries += 1
        cache_key = tuple(sorted(concepts[:3]))

        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[cache_key]

        if not force and torch.var(neural_output).item() < 0.01:
            return "symbolic-skipped-stable-state"

        result = await self.interpreter.predict(concepts, neural_output)
        self.prediction_cache[cache_key] = result

        if len(self.prediction_cache) > 100:
            self.prediction_cache.pop(next(iter(self.prediction_cache)))

        return result
    
    def get_cache_efficiency(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


class CognitionSimTrainer:
    """Main training orchestrator with exponential optimization"""
    def __init__(self, field_size: int = 100, device: str = 'cpu', 
                 enable_noise: bool = True, noise_intensity: float = 0.15):
        self.device = device
        self.field_size = field_size
        
        # Initialize noise injection
        self.enable_noise = enable_noise
        self.noise_intensity = noise_intensity
        if NOISE_AVAILABLE and enable_noise:
            self.noise_injector = NoiseInjector(enabled=True, intensity=noise_intensity)
            logger.info(f"🔊 Noise injection ENABLED (intensity={noise_intensity})")
        else:
            self.noise_injector = None
            logger.info("🔇 Noise injection DISABLED")
        
        # Initialize core components
        self.core_field = CoreField(field_size)
        self.syntropy_engine = SyntropyEngine(num_fields=5, field_size=field_size)
        self.oscillator = OscillatorySynapseTheory(field_size=field_size, device=device)
        
        # Adaptive components
        self.k_optimizer = AdaptiveKClusterOptimizer(initial_k=3, max_k=20)
        self.pattern_module = PatternModule(n_clusters=self.k_optimizer.current_k)
        
        # Neuroplasticity and symbolic optimization
        self.plasticity_accelerator = NeuroplasticityAccelerator()
        self.neuroplasticity_mgr = NeuroplasticityManager(
            self.oscillator, self.core_field, self.syntropy_engine
        )
        
        symbolic_config = SymbolicConfig(
            population_size=12, generations=6, alpha=1.0, beta=0.1, gamma=0.2
        )
        self.symbolic_interpreter = SymbolicPredictiveInterpreter(
            self.pattern_module, self.core_field, symbolic_config
        )
        self.symbolic_optimizer = SymbolicSpeedOptimizer(self.symbolic_interpreter)
        
        # Metrics tracking
        self.metrics = {
            'training_time': [],
            'rewards': [],
            'learning_rates': [],
            'speedup_factors': [],
            'k_values': [],
            'cache_efficiency': [],
            'field_resonance': []
        }
        
    async def train_on_text_batch(self, texts: List[str], batch_id: int) -> Dict:
        """Train on a batch of text with full optimization stack"""
        batch_start = time.time()
        
        # Process texts through oscillatory system
        batch_concepts = []
        for text in texts:
            # Neuroplasticity-driven learning
            await self.neuroplasticity_mgr.learn_async(text)
            
            # Extract concepts for symbolic processing
            words = text.split()[:10]  # Top 10 words as concepts
            batch_concepts.extend(words)
        
        # Get field state metrics
        resonance = self.core_field.get_resonance_metric()
        field_state = self.core_field.get_state()
        
        # 🔊 INJECT NOISE INTO FIELD STATE for robustness
        if self.noise_injector is not None:
            noisy_field_state = self.noise_injector.inject(
                field_state, 
                NoiseType.GAUSSIAN, 
                self.noise_intensity
            )
            # Apply additional field perturbation
            noisy_field_state = self.noise_injector.inject(
                noisy_field_state,
                NoiseType.FIELD_PERTURBATION,
                self.noise_intensity * 0.5
            )
            field_state_for_processing = noisy_field_state
            logger.debug(f"Noise injected into field state (batch {batch_id})")
        else:
            field_state_for_processing = field_state
        
        field_stability = 1.0 / (1.0 + torch.var(field_state_for_processing).item())
        
        # Adaptive K-clustering optimization
        pattern_complexity = self.k_optimizer.get_pattern_complexity(
            field_state_for_processing.cpu().numpy()
        )
        new_k = self.k_optimizer.update_k(resonance, pattern_complexity)
        
        if new_k != self.pattern_module.kmeans.n_clusters:
            self.pattern_module = PatternModule(n_clusters=new_k)
            logger.info(f"📊 Adapted K-clusters: {new_k} (resonance={resonance:.3f}, complexity={pattern_complexity:.3f})")
        
        # Symbolic prediction with caching for speed
        unique_concepts = list(set(batch_concepts))[:20]  # Limit to top 20 unique
        symbolic_result = await self.symbolic_optimizer.predict_and_cache(
            unique_concepts, field_state_for_processing
        )
        
        # Get reward and accelerate learning
        synthetic_data = self.oscillator.generate_synthetic_data(num_samples=20)
        
        # 🔊 INJECT NOISE INTO SYNTHETIC DATA
        if self.noise_injector is not None:
            noisy_synthetic_data = []
            for data_point in synthetic_data:
                noisy_point = self.noise_injector.inject(
                    data_point,
                    NoiseType.DROPOUT,
                    self.noise_intensity * 0.3
                )
                noisy_synthetic_data.append(noisy_point)
            synthetic_data = noisy_synthetic_data
        
        reward = self.oscillator.train(synthetic_data, texts[0] if texts else None, epochs=3)
        
        new_lr = self.plasticity_accelerator.accelerate(reward, field_stability)
        speedup = self.plasticity_accelerator.get_speedup_factor()
        
        # Update optimizer learning rates with accelerated values
        for param_group in self.oscillator.optimizer1.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.oscillator.optimizer2.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.oscillator.optimizer3.param_groups:
            param_group['lr'] = new_lr
        
        # Couple fields for syntropy emergence
        self.syntropy_engine.couple_fields()
        self.neuroplasticity_mgr.regulate_syntropy()
        
        batch_time = time.time() - batch_start
        
        # Track metrics
        self.metrics['training_time'].append(batch_time)
        self.metrics['rewards'].append(reward)
        self.metrics['learning_rates'].append(new_lr)
        self.metrics['speedup_factors'].append(speedup)
        self.metrics['k_values'].append(new_k)
        self.metrics['cache_efficiency'].append(self.symbolic_optimizer.get_cache_efficiency())
        self.metrics['field_resonance'].append(resonance)
        
        return {
            'batch_id': batch_id,
            'batch_time': batch_time,
            'reward': reward,
            'learning_rate': new_lr,
            'speedup': speedup,
            'k_clusters': new_k,
            'cache_efficiency': self.symbolic_optimizer.get_cache_efficiency(),
            'resonance': resonance,
            'concepts_processed': len(unique_concepts)
        }
    
    async def train(self, dataset_name: str = "HuggingFaceFW/fineweb-edu", dataset_config: str = "CC-MAIN-2013-20", 
                   num_batches: int = 100, batch_size: int = 10, use_streaming: bool = True):
        """Main training loop with exponential optimization"""
        logger.info(f"🚀 Starting CognitionSim Training on {dataset_name}" + (f"/{dataset_config}" if dataset_config else ""))
        logger.info(f"📈 Exponential optimization enabled: Oscillatory K-clustering + Neuroplasticity + Symbolic caching + SNN")
        
        # Load dataset
        logger.info("📦 Loading dataset from HuggingFace...")
        if "fineweb" in dataset_name.lower():
            logger.info("⚠️  Note: fineweb-edu requires authentication. Login with: huggingface-cli login")
        
        try:
            if use_streaming:
                logger.info("🌊 Using streaming mode to save disk space...")
                if dataset_config:
                    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split="train", streaming=True)
                logger.info("✅ Dataset loaded in streaming mode")
            else:
                if dataset_config:
                    dataset = load_dataset(dataset_name, dataset_config, split="train")
                else:
                    dataset = load_dataset(dataset_name, split="train")
                logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info("Falling back to wikitext dataset...")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            use_streaming = False
        
        overall_start = time.time()
        
        # Training loop with streaming
        if use_streaming:
            logger.info(f"🔄 Starting streaming training for {num_batches} batches...")
            dataset_iter = iter(dataset)
            
            for batch_id in range(num_batches):
                texts = []
                
                # Collect batch from stream
                for _ in range(batch_size):
                    try:
                        sample = next(dataset_iter)
                        # Extract text
                        if 'text' in sample:
                            text = sample['text']
                        elif 'content' in sample:
                            text = sample['content']
                        else:
                            text = str(sample)[:500]
                        
                        # Filter valid texts
                        if text and len(text.strip()) > 10:
                            texts.append(text)
                    except StopIteration:
                        logger.warning("Reached end of dataset, restarting iterator")
                        dataset_iter = iter(dataset)
                        break
                
                if not texts:
                    continue
                
                # Train on batch
                batch_metrics = await self.train_on_text_batch(texts, batch_id)
                
                # Log progress
                if batch_id % 10 == 0:
                    avg_time = np.mean(self.metrics['training_time'][-10:]) if len(self.metrics['training_time']) >= 10 else 0
                    avg_reward = np.mean(self.metrics['rewards'][-10:]) if len(self.metrics['rewards']) >= 10 else 0
                    current_speedup = self.metrics['speedup_factors'][-1] if self.metrics['speedup_factors'] else 1.0
                    
                    logger.info(f"📊 Batch {batch_id}/{num_batches} | "
                              f"Time: {avg_time:.3f}s | "
                              f"Reward: {avg_reward:.4f} | "
                              f"Speedup: {current_speedup:.2f}x | "
                              f"K: {batch_metrics['k_clusters']} | "
                              f"Texts: {len(texts)}")
        else:
            # Non-streaming mode (original logic)
            for batch_id in range(num_batches):
                # Sample batch
                indices = np.random.choice(len(dataset), size=min(batch_size, len(dataset)), replace=False)
                
                # Extract text from various possible fields
                texts = []
                for i in indices:
                    sample = dataset[int(i)]
                    # Try different common text fields
                    if 'text' in sample:
                        texts.append(sample['text'])
                    elif 'content' in sample:
                        texts.append(sample['content'])
                    elif 'description' in sample:
                        texts.append(sample['description'])
                    else:
                        # For structured data, create a string representation
                        texts.append(str(sample)[:500])  # Limit length
                
                # Filter out empty texts
                texts = [t for t in texts if t and len(t.strip()) > 10]
                if not texts:
                    continue
                
                # Train on batch
                batch_metrics = await self.train_on_text_batch(texts, batch_id)
                
                # Log progress
                if batch_id % 10 == 0:
                    avg_time = np.mean(self.metrics['training_time'][-10:])
                    avg_reward = np.mean(self.metrics['rewards'][-10:])
                    current_speedup = self.metrics['speedup_factors'][-1]
                    
                    logger.info(f"📊 Batch {batch_id}/{num_batches} | "
                              f"Time: {avg_time:.3f}s | "
                              f"Reward: {avg_reward:.4f} | "
                              f"Speedup: {current_speedup:.2f}x | "
                              f"K: {batch_metrics['k_clusters']} | "
                              f"Cache: {batch_metrics['cache_efficiency']:.2%} | "
                              f"LR: {batch_metrics['learning_rate']:.6f}")
        
        overall_time = time.time() - overall_start
        
        # Final statistics
        logger.info("=" * 80)
        logger.info("🎯 TRAINING COMPLETE - EXPONENTIAL OPTIMIZATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total time: {overall_time:.2f}s ({num_batches} batches)")
        logger.info(f"📈 Average batch time: {np.mean(self.metrics['training_time']):.3f}s")
        logger.info(f"🚀 Peak speedup factor: {max(self.metrics['speedup_factors']):.2f}x")
        logger.info(f"🎓 Final learning rate: {self.metrics['learning_rates'][-1]:.6f}")
        logger.info(f"🔢 K-clusters range: {min(self.metrics['k_values'])} → {max(self.metrics['k_values'])}")
        logger.info(f"💾 Symbolic cache efficiency: {self.metrics['cache_efficiency'][-1]:.2%}")
        logger.info(f"🌊 Field resonance: {self.metrics['field_resonance'][-1]:.4f}")
        logger.info(f"🏆 Average reward: {np.mean(self.metrics['rewards']):.4f}")
        
        # Noise injection statistics
        if self.noise_injector is not None:
            noise_stats = self.noise_injector.get_stats()
            logger.info(f"🔊 Noise injections: {noise_stats['total_injections']}")
            logger.info(f"🔊 Noise by type: {noise_stats['by_type']}")
        
        # Calculate exponential speedup
        if len(self.metrics['training_time']) > 20:
            early_avg = np.mean(self.metrics['training_time'][:10])
            late_avg = np.mean(self.metrics['training_time'][-10:])
            speedup_achieved = early_avg / late_avg if late_avg > 0 else 1.0
            logger.info(f"⚡ Empirical speedup (early vs late): {speedup_achieved:.2f}x")
        
        # Save results
        self.save_results()
        self.plot_metrics()
        
        return self.metrics
    
    def save_results(self):
        """Save training metrics and model weights"""
        # Save metrics
        with open('training_metrics.json', 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.metrics.items()}, f, indent=2)
        logger.info("💾 Metrics saved to training_metrics.json")
        
        # Save neural network weights
        self.oscillator.save_weights("atlantean_hrm_weights.pth")
        logger.info("💾 Model weights saved to atlantean_hrm_weights.pth")
    
    def plot_metrics(self):
        """Generate visualization of training metrics"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('CognitionSim Training - Exponential Optimization Metrics', fontsize=16)
        
        # Training time
        axes[0, 0].plot(self.metrics['training_time'], label='Batch Time', color='blue', alpha=0.6)
        axes[0, 0].set_title('Training Time per Batch')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Speedup factor
        axes[0, 1].plot(self.metrics['speedup_factors'], label='Speedup Factor', color='green')
        axes[0, 1].set_title('Neuroplasticity Speedup Factor')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Speedup (x)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rewards
        axes[1, 0].plot(self.metrics['rewards'], label='Reward', color='purple', alpha=0.7)
        axes[1, 0].set_title('Training Rewards')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # K-clusters adaptation
        axes[1, 1].plot(self.metrics['k_values'], label='K-clusters', color='orange', marker='o', markersize=2)
        axes[1, 1].set_title('Adaptive K-Clustering')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Number of Clusters')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Cache efficiency
        axes[2, 0].plot(self.metrics['cache_efficiency'], label='Cache Hit Rate', color='red')
        axes[2, 0].set_title('Symbolic Prediction Cache Efficiency')
        axes[2, 0].set_xlabel('Batch')
        axes[2, 0].set_ylabel('Efficiency (%)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Field resonance
        axes[2, 1].plot(self.metrics['field_resonance'], label='Field Resonance', color='cyan')
        axes[2, 1].set_title('Oscillatory Field Resonance')
        axes[2, 1].set_xlabel('Batch')
        axes[2, 1].set_ylabel('Resonance')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
        logger.info("📊 Metrics plot saved to training_metrics.png")
        plt.close()


async def main():
    """Main entry point"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CognitionSim Training with Noise Injection')
    parser.add_argument('--dataset', type=str, default='HuggingFaceFW/fineweb-edu',
                       help='HuggingFace dataset name (default: HuggingFaceFW/fineweb-edu)')
    parser.add_argument('--dataset-config', type=str, default='CC-MAIN-2013-20',
                       help='Dataset configuration (default: CC-MAIN-2013-20, options: default, CC-MAIN-2013-20, CC-MAIN-2013-48)')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming mode to save disk space (default: True)')
    parser.add_argument('--enable-noise', action='store_true', default=True,
                       help='Enable noise injection (default: True)')
    parser.add_argument('--noise-intensity', type=float, default=0.15,
                       help='Noise injection intensity 0.0-1.0 (default: 0.15)')
    parser.add_argument('--num-batches', type=int, default=100,
                       help='Number of training batches (default: 100)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size (default: 10)')
    parser.add_argument('--field-size', type=int, default=100,
                       help='Field size (default: 100)')
    args = parser.parse_args()
    
    logger.info("🌟 CognitionSim - Exponential Training System with NOISE INJECTION")
    logger.info("=" * 80)
    logger.info(f"� Dataset: {args.dataset}" + (f" ({args.dataset_config})" if args.dataset_config else ""))
    logger.info(f"🔊 Noise injection: {'ENABLED' if args.enable_noise else 'DISABLED'}")
    logger.info(f"🔊 Noise intensity: {args.noise_intensity}")
    logger.info(f"📦 Training batches: {args.num_batches}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Device: {device}")
    
    # Initialize trainer with noise injection
    trainer = CognitionSimTrainer(
        field_size=args.field_size, 
        device=device,
        enable_noise=args.enable_noise,
        noise_intensity=args.noise_intensity
    )
    
    # Train on HuggingFace dataset
    await trainer.train(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        use_streaming=args.streaming
    )
    
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
