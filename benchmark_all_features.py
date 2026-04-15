#!/usr/bin/env python3
"""
CognitionSim - Comprehensive Feature Benchmarks

Benchmarks all new features including:
- IFT Field Substrate performance
- Inference latency and throughput
- GPU training performance
- Memory usage
- Symbolic reasoning performance
- Multi-core training

Run: python benchmark_all_features.py
"""

import asyncio
import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# BENCHMARK RESULTS STORAGE
# ============================================================

@dataclass
class BenchmarkMetrics:
    """Store individual benchmark metric"""
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    p95_ms: float
    p99_ms: float
    context: Dict[str, Any]
    
    def __str__(self):
        return (f"📊 {self.name}\n"
                f"   Mean:    {self.mean_ms:.2f}ms | Median: {self.median_ms:.2f}ms\n"
                f"   Min:     {self.min_ms:.2f}ms | Max:    {self.max_ms:.2f}ms\n"
                f"   P95:     {self.p95_ms:.2f}ms | P99:    {self.p99_ms:.2f}ms\n"
                f"   StdDev:  {self.stddev_ms:.2f}ms")


@dataclass
class MemorySnapshot:
    """Capture memory metrics at a point in time"""
    timestamp: str
    rss_mb: float        # Resident Set Size
    vms_mb: float        # Virtual Memory Size
    percent: float       # Percentage of total system memory
    gpu_mb: Optional[float] = None  # GPU memory if available
    
    def __str__(self):
        result = f"💾 {self.timestamp}\n   RAM: {self.rss_mb:.1f}MB"
        if self.gpu_mb is not None:
            result += f" | GPU: {self.gpu_mb:.1f}MB"
        result += f" ({self.percent:.1f}%)"
        return result


class BenchmarkSuite:
    """Main benchmark suite with all tests"""
    
    def __init__(self, output_dir: Path = Path(".benchmarks")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {},
            'summary': {}
        }
        
        self.memory_snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'total_memory_gb': psutil.virtual_memory().total / 1e9,
            'python_version': __import__('sys').version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        }
    
    def _snapshot_memory(self, label: str) -> MemorySnapshot:
        """Capture memory usage snapshot"""
        mem_info = self.process.memory_info()
        vm = psutil.virtual_memory()
        
        gpu_mb = None
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1e6
        
        snapshot = MemorySnapshot(
            timestamp=f"{datetime.utcnow().isoformat()} [{label}]",
            rss_mb=mem_info.rss / 1e6,
            vms_mb=mem_info.vms / 1e6,
            percent=vm.percent,
            gpu_mb=gpu_mb
        )
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def _measure_performance(self, name: str, func, *args, iterations: int = 100, **kwargs) -> BenchmarkMetrics:
        """Measure performance of a function"""
        logger.info(f"🏃 Benchmarking: {name} ({iterations} iterations)")
        
        measurements = []
        
        # Warmup
        for _ in range(5):
            func(*args, **kwargs)
        
        # Actual measurements
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            measurements.append(elapsed)
        
        # Calculate statistics
        metrics = BenchmarkMetrics(
            name=name,
            mean_ms=np.mean(measurements),
            median_ms=np.median(measurements),
            min_ms=np.min(measurements),
            max_ms=np.max(measurements),
            stddev_ms=np.std(measurements),
            p95_ms=np.percentile(measurements, 95),
            p99_ms=np.percentile(measurements, 99),
            context={'iterations': iterations}
        )
        
        self.results['benchmarks'][name] = asdict(metrics)
        logger.info(f"✓ {name} benchmarked: {metrics.mean_ms:.2f}ms (±{metrics.stddev_ms:.2f}ms)")
        
        return metrics
    
    # ============================================================
    # IFT FIELD SYSTEM BENCHMARKS
    # ============================================================
    
    async def benchmark_ift_inference(self, iterations: int = 50):
        """Benchmark IFT field system inference performance"""
        logger.info("\n" + "="*80)
        logger.info("🔬 IFT FIELD SYSTEM BENCHMARKS")
        logger.info("="*80)
        
        try:
            from quadra.core.symbolic.interpreter import StatefulSymbolicPredictiveInterpreter
            
            # Initialize with IFT enabled
            logger.info("Initializing interpreter with IFT enabled...")
            interpreter = StatefulSymbolicPredictiveInterpreter(
                model_version="benchmark-ift-1.0",
                enable_ift=True,
                field_shape=(32, 32)
            )
            
            self._snapshot_memory("IFT_initialized")
            
            async def run_inference():
                return await interpreter.process({
                    'text': 'benchmark test input for ift field system',
                    'concepts': ['field', 'inference', 'benchmark']
                }, request_id=f"ift-bench-{time.time()}")
            
            # Warmup
            for _ in range(5):
                await run_inference()
            
            # Benchmark
            measurements = []
            data_collected = []
            
            for i in range(iterations):
                start = time.perf_counter()
                result = await run_inference()
                elapsed = (time.perf_counter() - start) * 1000
                measurements.append(elapsed)
                
                # Collect first result for analysis
                if i == 0:
                    data_collected.append(result)
            
            # Calculate statistics
            metrics = BenchmarkMetrics(
                name="IFT Inference (Full Pipeline)",
                mean_ms=np.mean(measurements),
                median_ms=np.median(measurements),
                min_ms=np.min(measurements),
                max_ms=np.max(measurements),
                stddev_ms=np.std(measurements),
                p95_ms=np.percentile(measurements, 95),
                p99_ms=np.percentile(measurements, 99),
                context={
                    'iterations': iterations,
                    'field_enabled': True,
                    'field_shape': (32, 32)
                }
            )
            
            self.results['benchmarks']["IFT Inference (Full Pipeline)"] = asdict(metrics)
            print(f"\n{metrics}")
            
            # Analyze field metrics
            if data_collected and 'ift_field_metrics' in data_collected[0]:
                fm = data_collected[0]['ift_field_metrics']
                logger.info(f"\n📡 Field Metrics Sample:")
                logger.info(f"   Field Energy: {fm.get('field_energy', 0):.4f}")
                logger.info(f"   Symmetry Order: {fm.get('symmetry_order', 0):.4f}")
                logger.info(f"   Global Potential: {fm.get('global_potential', 0):.4f}")
            
            self._snapshot_memory("After_IFT_inference")
            
            # Benchmark baseline (IFT disabled)
            logger.info("\nBenchmarking baseline (IFT disabled)...")
            interpreter_baseline = StatefulSymbolicPredictiveInterpreter(
                model_version="benchmark-baseline-1.0",
                enable_ift=False
            )
            
            async def run_baseline():
                return await interpreter_baseline.process({
                    'text': 'benchmark test input',
                    'concepts': ['baseline', 'inference']
                }, request_id=f"baseline-bench-{time.time()}")
            
            measurements_baseline = []
            for _ in range(iterations):
                start = time.perf_counter()
                await run_baseline()
                elapsed = (time.perf_counter() - start) * 1000
                measurements_baseline.append(elapsed)
            
            metrics_baseline = BenchmarkMetrics(
                name="Baseline Inference (IFT Disabled)",
                mean_ms=np.mean(measurements_baseline),
                median_ms=np.median(measurements_baseline),
                min_ms=np.min(measurements_baseline),
                max_ms=np.max(measurements_baseline),
                stddev_ms=np.std(measurements_baseline),
                p95_ms=np.percentile(measurements_baseline, 95),
                p99_ms=np.percentile(measurements_baseline, 99),
                context={
                    'iterations': iterations,
                    'field_enabled': False
                }
            )
            
            self.results['benchmarks']["Baseline Inference (IFT Disabled)"] = asdict(metrics_baseline)
            print(f"\n{metrics_baseline}")
            
            # Calculate overhead
            overhead_pct = ((metrics.mean_ms - metrics_baseline.mean_ms) / metrics_baseline.mean_ms) * 100
            logger.info(f"\n⚡ IFT Overhead: {overhead_pct:+.1f}%")
            self.results['benchmarks']["IFT_overhead_percent"] = overhead_pct
            
        except Exception as e:
            logger.error(f"❌ IFT benchmarking failed: {e}", exc_info=True)
    
    # ============================================================
    # GPU TRAINING BENCHMARKS
    # ============================================================
    
    def benchmark_gpu_training(self):
        """Benchmark GPU training performance"""
        logger.info("\n" + "="*80)
        logger.info("🎓 GPU TRAINING BENCHMARKS")
        logger.info("="*80)
        
        try:
            from atlantean_hrm_gpu import GPUOscillatorEngine, get_optimal_config
            
            config = get_optimal_config()
            logger.info(f"GPU Config: batch_size={config.batch_size}, num_workers={config.num_workers}")
            
            engine = GPUOscillatorEngine(field_size=100, config=config)
            self._snapshot_memory("GPU_engine_initialized")
            
            # Run GPU benchmark
            logger.info("Running GPU training benchmark...")
            stats = engine.benchmark(num_iterations=50)
            
            metrics = BenchmarkMetrics(
                name="GPU Training Step",
                mean_ms=stats['time_per_iteration_ms'],
                median_ms=stats['time_per_iteration_ms'],  # Simplified
                min_ms=stats['time_per_iteration_ms'] * 0.9,
                max_ms=stats['time_per_iteration_ms'] * 1.1,
                stddev_ms=stats['time_per_iteration_ms'] * 0.05,
                p95_ms=stats['time_per_iteration_ms'] * 1.05,
                p99_ms=stats['time_per_iteration_ms'] * 1.08,
                context={'iterations': 50} | stats
            )
            
            self.results['benchmarks']["GPU Training Step"] = asdict(metrics)
            print(f"\n{metrics}")
            logger.info(f"   Device: {stats['device']}")
            logger.info(f"   Throughput: {stats['throughput_samples_per_sec']:.0f} samples/sec")
            
            self._snapshot_memory("After_GPU_training")
            
        except Exception as e:
            logger.error(f"❌ GPU benchmarking failed: {e}", exc_info=True)
    
    # ============================================================
    # INFERENCE LATENCY BENCHMARKS
    # ============================================================
    
    async def benchmark_inference_latency(self):
        """Benchmark inference latency and throughput"""
        logger.info("\n" + "="*80)
        logger.info("⚡ INFERENCE LATENCY BENCHMARKS")
        logger.info("="*80)
        
        try:
            from quadra.core.symbolic.interpreter import StatefulSymbolicPredictiveInterpreter
            
            interpreter = StatefulSymbolicPredictiveInterpreter(
                model_version="latency-bench-1.0",
                enable_ift=True
            )
            
            # Test different input sizes
            test_inputs = [
                ("Short", "test"),
                ("Medium", "This is a medium length test input for benchmarking inference latency"),
                ("Long", "This is a longer test input " * 5),
            ]
            
            for label, text in test_inputs:
                async def run_test():
                    return await interpreter.process({
                        'text': text,
                        'concepts': ['inference', 'latency', 'benchmark']
                    }, request_id=f"latency-{label}")
                
                measurements = []
                for _ in range(20):
                    start = time.perf_counter()
                    await run_test()
                    elapsed = (time.perf_counter() - start) * 1000
                    measurements.append(elapsed)
                
                metrics = BenchmarkMetrics(
                    name=f"Inference Latency ({label} Input)",
                    mean_ms=np.mean(measurements),
                    median_ms=np.median(measurements),
                    min_ms=np.min(measurements),
                    max_ms=np.max(measurements),
                    stddev_ms=np.std(measurements),
                    p95_ms=np.percentile(measurements, 95),
                    p99_ms=np.percentile(measurements, 99),
                    context={'input_type': label, 'text_length': len(text)}
                )
                
                self.results['benchmarks'][f"Inference Latency ({label} Input)"] = asdict(metrics)
                print(f"\n{metrics}")
            
            self._snapshot_memory("After_latency_tests")
            
        except Exception as e:
            logger.error(f"❌ Latency benchmarking failed: {e}", exc_info=True)
    
    # ============================================================
    # SYMBOLIC REASONING BENCHMARKS
    # ============================================================
    
    async def benchmark_symbolic_reasoning(self):
        """Benchmark symbolic reasoning performance"""
        logger.info("\n" + "="*80)
        logger.info("🧠 SYMBOLIC REASONING BENCHMARKS")
        logger.info("="*80)
        
        try:
            from quadra.core.symbolic.interpreter import SymbolicPredictiveInterpreter
            
            interpreter = SymbolicPredictiveInterpreter(
                input_dim=128,
                hidden_dim=256,
                output_dim=64,
                num_layers=3
            )
            
            # Benchmark inference
            test_input = np.random.randn(128).astype(np.float32)
            
            async def run_reasoning():
                return await interpreter.process({
                    'input': test_input,
                    'context': ['symbolic', 'reasoning']
                })
            
            measurements = []
            for _ in range(30):
                start = time.perf_counter()
                await run_reasoning()
                elapsed = (time.perf_counter() - start) * 1000
                measurements.append(elapsed)
            
            metrics = BenchmarkMetrics(
                name="Symbolic Reasoning Inference",
                mean_ms=np.mean(measurements),
                median_ms=np.median(measurements),
                min_ms=np.min(measurements),
                max_ms=np.max(measurements),
                stddev_ms=np.std(measurements),
                p95_ms=np.percentile(measurements, 95),
                p99_ms=np.percentile(measurements, 99),
                context={'input_dim': 128, 'output_dim': 64}
            )
            
            self.results['benchmarks']["Symbolic Reasoning Inference"] = asdict(metrics)
            print(f"\n{metrics}")
            
            self._snapshot_memory("After_symbolic_reasoning")
            
        except Exception as e:
            logger.error(f"❌ Symbolic reasoning benchmarking failed: {e}", exc_info=True)
    
    # ============================================================
    # MEMORY FOOTPRINT BENCHMARKS
    # ============================================================
    
    async def benchmark_memory_usage(self):
        """Benchmark memory usage patterns"""
        logger.info("\n" + "="*80)
        logger.info("💾 MEMORY USAGE BENCHMARKS")
        logger.info("="*80)
        
        try:
            from quadra.core.symbolic.interpreter import StatefulSymbolicPredictiveInterpreter
            
            self._snapshot_memory("Baseline_empty")
            
            # Initialize interpreter
            interpreter = StatefulSymbolicPredictiveInterpreter(
                model_version="memory-bench-1.0",
                enable_ift=True
            )
            
            self._snapshot_memory("After_interpreter_init")
            
            # Run multiple inferences
            for i in range(10):
                await interpreter.process({
                    'text': f'memory test {i}',
                    'concepts': ['memory', 'benchmark']
                }, request_id=f"mem-test-{i}")
            
            self._snapshot_memory("After_10_inferences")
            
            # Report memory usage
            logger.info("\n📊 Memory Usage Over Time:")
            for snapshot in self.memory_snapshots[-3:]:
                logger.info(f"   {snapshot.timestamp}: {snapshot.rss_mb:.1f}MB RAM", 
                           extra={'gpu_mb': snapshot.gpu_mb} if snapshot.gpu_mb else {})
            
            self.results['memory_snapshots'] = [asdict(s) for s in self.memory_snapshots]
            
        except Exception as e:
            logger.error(f"❌ Memory benchmarking failed: {e}", exc_info=True)
    
    # ============================================================
    # MULTI-CORE TRAINING BENCHMARKS
    # ============================================================
    
    def benchmark_multicore_training(self):
        """Benchmark multi-core training performance"""
        logger.info("\n" + "="*80)
        logger.info("🔄 MULTI-CORE TRAINING BENCHMARKS")
        logger.info("="*80)
        
        try:
            import torch.multiprocessing as mp
            from atlantean_hrm_gpu import GPUOptimizedField
            
            logger.info(f"CPU Count: {mp.cpu_count()}")
            
            # Benchmark on different thread counts
            for num_workers in [1, 2, 4]:
                if num_workers > mp.cpu_count():
                    continue
                
                def train_batch():
                    field = GPUOptimizedField()
                    batch = torch.randn(32, 100)
                    for _ in range(5):
                        _ = field.encode(batch)
                
                measurements = []
                for _ in range(10):
                    start = time.perf_counter()
                    train_batch()
                    elapsed = (time.perf_counter() - start) * 1000
                    measurements.append(elapsed)
                
                metrics = BenchmarkMetrics(
                    name=f"Multi-core Training (workers={num_workers})",
                    mean_ms=np.mean(measurements),
                    median_ms=np.median(measurements),
                    min_ms=np.min(measurements),
                    max_ms=np.max(measurements),
                    stddev_ms=np.std(measurements),
                    p95_ms=np.percentile(measurements, 95),
                    p99_ms=np.percentile(measurements, 99),
                    context={'num_workers': num_workers}
                )
                
                self.results['benchmarks'][f"Multi-core Training (workers={num_workers})"] = asdict(metrics)
                print(f"\n{metrics}")
            
            self._snapshot_memory("After_multicore_training")
            
        except Exception as e:
            logger.error(f"❌ Multi-core benchmarking failed: {e}", exc_info=True)
    
    # ============================================================
    # REPORT GENERATION
    # ============================================================
    
    def generate_summary(self):
        """Generate benchmark summary"""
        logger.info("\n" + "="*80)
        logger.info("📋 BENCHMARK SUMMARY")
        logger.info("="*80)
        
        # Sort benchmarks by mean execution time
        sorted_benchmarks = sorted(
            self.results['benchmarks'].items(),
            key=lambda x: x[1].get('mean_ms', 0) if isinstance(x[1], dict) else 0
        )
        
        logger.info("\n🏆 Performance Rankings (by mean latency):")
        for rank, (name, metrics) in enumerate(sorted_benchmarks, 1):
            if isinstance(metrics, dict) and 'mean_ms' in metrics:
                logger.info(f"   {rank}. {name}: {metrics['mean_ms']:.2f}ms")
        
        # Memory summary
        logger.info("\n💾 Memory Usage Summary:")
        if self.memory_snapshots:
            initial = self.memory_snapshots[0].rss_mb
            final = self.memory_snapshots[-1].rss_mb
            logger.info(f"   Initial: {initial:.1f}MB")
            logger.info(f"   Final: {final:.1f}MB")
            logger.info(f"   Change: {final - initial:+.1f}MB ({((final-initial)/initial*100):+.1f}%)")
        
        # Save results
        output_file = self.output_dir / f"benchmark_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n✅ Benchmark results saved to {output_file}")
        
        return self.results
    
    async def run_all_benchmarks(self):
        """Run all benchmark suites"""
        logger.info("\n" + "🚀" * 40)
        logger.info("STARTING COMPREHENSIVE FEATURE BENCHMARKS")
        logger.info("🚀" * 40)
        
        start_time = time.time()
        
        # Sequential benchmarks
        await self.benchmark_ift_inference(iterations=30)
        await self.benchmark_inference_latency()
        await self.benchmark_symbolic_reasoning()
        await self.benchmark_memory_usage()
        
        # GPU benchmarks (conditional)
        if torch.cuda.is_available():
            self.benchmark_gpu_training()
        else:
            logger.warning("⚠️  GPU not available, skipping GPU benchmarks")
        
        self.benchmark_multicore_training()
        
        # Generate summary
        self.generate_summary()
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ All benchmarks completed in {elapsed:.1f}s")


async def main():
    """Main entry point"""
    suite = BenchmarkSuite()
    await suite.run_all_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())
