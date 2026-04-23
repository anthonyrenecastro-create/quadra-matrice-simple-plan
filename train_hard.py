#!/usr/bin/env python3
"""
CognitionSim - Hard Training Run (10-15 minutes)

Dataset  : WikiText-103 (100x larger than WikiText-2, long-form Wikipedia articles)
Field    : 256 neurons  (vs 150 baseline)
Batches  : 150          (vs 50 baseline)
Batch sz : 15 texts     (vs 10 baseline)
Concepts : adaptive 10-20 per batch by stability
SNN epochs: adaptive 1-3 per batch by stability
Noise    : 0.25         (vs 0.15 baseline)
HRM      : field_shape (24, 24)  (vs default 12x12)

Run: python train_hard.py
"""

import asyncio
import logging
import sys
import json
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FIELD_SIZE   = 256
NUM_BATCHES  = 150
BATCH_SIZE   = 15
NOISE_LEVEL  = 0.25
DATASET      = "wikitext"
DATASET_CFG  = "wikitext-103-raw-v1"
WEIGHTS_OUT  = "atlantean_hrm_hard.pth"
METRICS_OUT  = "training_metrics_hard.json"


# ──────────────────────────────────────────────
# PATCH: inject harder hyper-params into the
# CognitionSimTrainer before it runs, without
# duplicating the whole training loop.
# ──────────────────────────────────────────────
def _patch_trainer(trainer):
    """Wire adaptive hard-mode hyper-parameters into an existing trainer instance."""
    import torch

    # Larger symbolic config
    from atlantean_hrm_spi import SymbolicConfig, SymbolicPredictiveInterpreter
    from train_atlantean_hrm import SymbolicSpeedOptimizer

    hard_cfg = SymbolicConfig(
        population_size=20,   # more symbolic candidates
        generations=10,       # deeper search
        alpha=1.2,
        beta=0.15,
        gamma=0.25,
    )
    trainer.symbolic_interpreter = SymbolicPredictiveInterpreter(
        trainer.pattern_module, trainer.core_field, hard_cfg
    )
    trainer.symbolic_optimizer = SymbolicSpeedOptimizer(trainer.symbolic_interpreter)

    # Deeper K-clustering ceiling
    trainer.k_optimizer.max_k = 30

    async def hard_batch(self, texts, batch_id):
        from utils.noise_injection import NoiseType
        import torch

        batch_start = time.time()
        batch_concepts = []

        for text in texts:
            await self.neuroplasticity_mgr.learn_async(text, train_every=8)
            words = text.split()[:12]
            batch_concepts.extend(words)

        resonance = self.core_field.get_resonance_metric()
        field_state = self.core_field.get_state()

        if self.noise_injector is not None:
            field_state = self.noise_injector.inject(field_state, NoiseType.GAUSSIAN, self.noise_intensity * 0.6)
            field_state = self.noise_injector.inject(field_state, NoiseType.FIELD_PERTURBATION, self.noise_intensity * 0.3)

        field_var = torch.var(field_state).item()
        field_stability = 1.0 / (1.0 + field_var)

        # Adaptive budget
        if field_stability > 0.92:
            synthetic_samples = 6
            snn_epochs = 1
            concept_cap = 10
            run_symbolic = (batch_id % 8 == 0)
        elif field_stability > 0.80:
            synthetic_samples = 10
            snn_epochs = 2
            concept_cap = 15
            run_symbolic = (batch_id % 4 == 0)
        else:
            synthetic_samples = 16
            snn_epochs = 3
            concept_cap = 20
            run_symbolic = True

        pattern_complexity = self.k_optimizer.get_pattern_complexity(field_state.cpu().numpy())
        new_k = self.k_optimizer.update_k(resonance, pattern_complexity)

        from atlantean_hrm_spi import PatternModule
        if new_k != self.pattern_module.kmeans.n_clusters and batch_id % 5 == 0:
            self.pattern_module = PatternModule(n_clusters=new_k)

        unique_concepts = list(set(batch_concepts))[:concept_cap]

        if run_symbolic:
            await self.symbolic_optimizer.predict_and_cache(unique_concepts, field_state)

        synthetic_data = self.oscillator.generate_synthetic_data(num_samples=synthetic_samples)
        reward = self.oscillator.train(
            synthetic_data,
            texts[0] if texts else None,
            epochs=snn_epochs,
        )

        new_lr = self.plasticity_accelerator.accelerate(reward, field_stability)
        speedup = self.plasticity_accelerator.get_speedup_factor()

        for opt in [self.oscillator.optimizer1, self.oscillator.optimizer2, self.oscillator.optimizer3]:
            for pg in opt.param_groups:
                pg['lr'] = new_lr

        self.syntropy_engine.couple_fields()
        self.neuroplasticity_mgr.regulate_syntropy()

        batch_time = time.time() - batch_start
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
            'concepts_processed': len(unique_concepts),
        }

    import types
    trainer.train_on_text_batch = types.MethodType(hard_batch, trainer)
    return trainer


async def main():
    print("\n" + "=" * 80)
    print("🔥 COGNITIONSIM — HARD TRAINING RUN")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"  • Dataset   : {DATASET}/{DATASET_CFG}  (WikiText-103, ~500MB)")
    print(f"  • Field size: {FIELD_SIZE} neurons  (vs 150 baseline)")
    print(f"  • Batches   : {NUM_BATCHES}           (vs 50 baseline)")
    print(f"  • Batch sz  : {BATCH_SIZE} texts     (vs 10 baseline)")
    print("  • SNN epochs: adaptive (1-3) by field stability")
    print("  • Concepts  : adaptive cap (10-20) by field stability")
    print(f"  • Noise     : {NOISE_LEVEL}          (vs 0.15 baseline)")
    print(f"\n⏱️  Estimated time: 10-15 minutes")
    print(f"🎯 Goal: Push coherence, fractal complexity, and Guna balance\n")
    print("=" * 80 + "\n")

    try:
        from train_atlantean_hrm import CognitionSimTrainer

        logger.info(f"Initializing trainer — field_size={FIELD_SIZE}, noise={NOISE_LEVEL}...")
        trainer = CognitionSimTrainer(
            field_size=FIELD_SIZE,
            device='cpu',
            enable_noise=True,
            noise_intensity=NOISE_LEVEL,
        )

        logger.info("Applying hard hyper-parameters...")
        trainer = _patch_trainer(trainer)

        logger.info(f"Loading {DATASET_CFG}...")
        metrics = await trainer.train(
            dataset_name=DATASET,
            dataset_config=DATASET_CFG,
            num_batches=NUM_BATCHES,
            batch_size=BATCH_SIZE,
            use_streaming=True,
        )

        # ── Results ──────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("✅  HARD TRAINING COMPLETE")
        print("=" * 80)
        print(f"\n📈 Final Metrics:")
        print(f"  • Peak Speedup      : {max(metrics['speedup_factors']):.2f}x")
        print(f"  • Final Reward      : {metrics['rewards'][-1]:.4f}")
        print(f"  • Avg Reward        : {sum(metrics['rewards'])/len(metrics['rewards']):.4f}")
        print(f"  • Cache Efficiency  : {metrics['cache_efficiency'][-1]:.2%}")
        print(f"  • Field Resonance   : {metrics['field_resonance'][-1]:.4f}")
        print(f"  • K-Clusters (final): {int(metrics['k_values'][-1])}")
        print(f"  • Total Batches     : {len(metrics['rewards'])}")

        # Save weights
        logger.info(f"Saving weights to {WEIGHTS_OUT}...")
        trainer.oscillator.save_weights(WEIGHTS_OUT)
        print(f"\n💾 Weights: {WEIGHTS_OUT}")

        # Save metrics
        serializable = {k: [float(x) for x in v] for k, v in metrics.items()}
        with open(METRICS_OUT, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"📊 Metrics: {METRICS_OUT}")

        print("\n" + "=" * 80)
        print("🔥 Hard training session finished.")
        print("=" * 80 + "\n")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
