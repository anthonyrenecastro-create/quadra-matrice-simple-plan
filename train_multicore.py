#!/usr/bin/env python3
"""
CognitionSim Multi-Core Training
Parallel training with GPU acceleration and multi-threading support.

Features:
- Multi-GPU distributed training
- Multi-core CPU parallelization
- Asynchronous data loading
- Real-time performance monitoring
- Automatic checkpointing
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import logging
import time
import os
from typing import Optional, Dict, List
from dataclasses import dataclass
import threading
from queue import Queue
import json

from atlantean_hrm_gpu import GPUOscillatorEngine, GPUConfig, get_optimal_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Efficient dataset for text processing"""
    
    def __init__(self, num_samples: int = 1000, field_size: int = 100):
        self.num_samples = num_samples
        self.field_size = field_size
        
        # Pre-generate data for speed
        self.data = torch.randn(num_samples, field_size) * 0.1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class MetricsCollector:
    """Thread-safe metrics collection"""
    
    def __init__(self):
        self.metrics = {
            'loss': [],
            'throughput': [],
            'gpu_memory': [],
            'cpu_usage': [],
        }
        self.lock = threading.Lock()
    
    def add_metric(self, name: str, value: float):
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)
    
    def get_average(self, name: str, last_n: int = 100) -> float:
        with self.lock:
            if name in self.metrics and len(self.metrics[name]) > 0:
                return np.mean(self.metrics[name][-last_n:])
            return 0.0
    
    def save(self, path: str):
        with self.lock:
            with open(path, 'w') as f:
                json.dump({k: v[-1000:] for k, v in self.metrics.items()}, f)
        logger.info(f"📊 Metrics saved to {path}")


def train_single_gpu(rank: int, world_size: int, config: GPUConfig, 
                     num_epochs: int = 10, dataset_size: int = 10000):
    """Training function for single GPU in distributed setting"""
    
    # Setup distributed training
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    logger.info(f"🚀 GPU {rank}/{world_size} starting training...")
    
    # Create dataset and dataloader
    dataset = TextDataset(num_samples=dataset_size, field_size=100)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        )
    
    # Create engine
    engine = GPUOscillatorEngine(field_size=100, config=config)
    
    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        engine.field_network = DDP(engine.field_network, device_ids=[rank])
        engine.spike_network = DDP(engine.spike_network, device_ids=[rank])
    
    # Metrics collector
    metrics = MetricsCollector()
    
    # Training loop
    start_time = time.time()
    total_samples = 0
    
    for epoch in range(num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Train step
            loss = engine.train_step(batch)
            
            epoch_loss += loss
            num_batches += 1
            total_samples += batch.size(0)
            
            # Log progress
            if batch_idx % 50 == 0 and rank == 0:
                elapsed = time.time() - start_time
                throughput = total_samples / elapsed if elapsed > 0 else 0
                
                metrics.add_metric('loss', loss)
                metrics.add_metric('throughput', throughput)
                
                mem_stats = engine.field_network.get_memory_stats()
                if 'allocated_mb' in mem_stats:
                    metrics.add_metric('gpu_memory', mem_stats['allocated_mb'])
                
                logger.info(
                    f"GPU {rank} | Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss:.4f} | "
                    f"Throughput: {throughput:.1f} samples/s"
                )
        
        # Epoch summary
        if rank == 0:
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / num_batches
            
            logger.info(
                f"\n{'='*80}\n"
                f"Epoch {epoch+1} Complete:\n"
                f"  Average Loss: {avg_loss:.4f}\n"
                f"  Time: {epoch_time:.2f}s\n"
                f"  Samples/sec: {len(dataset)/epoch_time:.1f}\n"
                f"{'='*80}\n"
            )
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}_gpu_{rank}.pth"
                engine.save_checkpoint(checkpoint_path)
    
    # Final stats
    if rank == 0:
        total_time = time.time() - start_time
        logger.info(
            f"\n🎉 Training Complete!\n"
            f"   Total time: {total_time/60:.2f} minutes\n"
            f"   Total samples: {total_samples:,}\n"
            f"   Overall throughput: {total_samples/total_time:.1f} samples/s\n"
        )
        
        # Save metrics
        metrics.save('training_metrics_multicore.json')
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def train_multi_core_cpu(num_processes: int = 2, num_epochs: int = 10):
    """Multi-core CPU training using process parallelism"""
    
    logger.info(f"🔧 Multi-core CPU training with {num_processes} processes")
    
    # Use CPU config
    config = GPUConfig(use_cuda=False)
    config.num_workers = 0  # Use main process for data loading
    config.batch_size = 16
    
    def worker_fn(worker_id: int, queue: Queue):
        """Worker process function"""
        engine = GPUOscillatorEngine(field_size=100, config=config)
        dataset = TextDataset(num_samples=10000 // num_processes, field_size=100)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        worker_metrics = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                loss = engine.train_step(batch)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            worker_metrics.append({'epoch': epoch, 'loss': avg_loss, 'worker': worker_id})
            
            logger.info(f"Worker {worker_id} | Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        queue.put(worker_metrics)
    
    # Start worker processes
    import multiprocessing
    result_queue = multiprocessing.Queue()
    processes = []
    
    start_time = time.time()
    
    for i in range(num_processes):
        p = multiprocessing.Process(target=worker_fn, args=(i, result_queue))
        p.start()
        processes.append(p)
    
    # Wait for completion
    for p in processes:
        p.join()
    
    # Collect results
    all_metrics = []
    while not result_queue.empty():
        all_metrics.extend(result_queue.get())
    
    total_time = time.time() - start_time
    
    logger.info(
        f"\n🎉 Multi-core CPU training complete!\n"
        f"   Processes: {num_processes}\n"
        f"   Total time: {total_time/60:.2f} minutes\n"
    )


def train_multi_gpu():
    """Launch multi-GPU distributed training"""
    
    config = get_optimal_config()
    
    if config.num_gpus > 1:
        logger.info(f"🚀 Launching distributed training on {config.num_gpus} GPUs")
        
        # Spawn processes for each GPU
        mp.spawn(
            train_single_gpu,
            args=(config.num_gpus, config, 10, 10000),
            nprocs=config.num_gpus,
            join=True
        )
    else:
        logger.info("🔧 Single GPU/CPU training")
        train_single_gpu(0, 1, config, num_epochs=10, dataset_size=10000)


def benchmark_configurations():
    """Benchmark different configurations"""
    
    print("\n" + "="*80)
    print("⚡ QUADRA MATRIX MULTI-CORE BENCHMARK")
    print("="*80 + "\n")
    
    configs = [
        ("Single GPU", 1),
        ("Dual GPU", 2) if torch.cuda.device_count() >= 2 else None,
        ("Quad GPU", 4) if torch.cuda.device_count() >= 4 else None,
    ]
    
    results = {}
    
    for config_name, num_gpus in filter(None, configs):
        if num_gpus > torch.cuda.device_count():
            continue
        
        print(f"\n📊 Testing: {config_name}")
        print("-" * 80)
        
        config = GPUConfig(num_gpus=num_gpus)
        engine = GPUOscillatorEngine(field_size=100, config=config)
        
        stats = engine.benchmark(num_iterations=100)
        results[config_name] = stats
        
        print(f"✨ {config_name}:")
        print(f"   Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"   Time/iteration: {stats['time_per_iteration_ms']:.2f} ms")
    
    # Print comparison
    print("\n" + "="*80)
    print("📈 PERFORMANCE COMPARISON")
    print("="*80)
    
    for name, stats in results.items():
        print(f"{name:20s}: {stats['throughput_samples_per_sec']:8.1f} samples/sec")
    
    print("\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CognitionSim Multi-Core Training")
    parser.add_argument('--mode', choices=['single', 'multi-gpu', 'multi-cpu', 'benchmark'],
                       default='single', help='Training mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--processes', type=int, default=2, help='Number of CPU processes')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        config = get_optimal_config()
        train_single_gpu(0, 1, config, num_epochs=args.epochs, dataset_size=10000)
    elif args.mode == 'multi-gpu':
        train_multi_gpu()
    elif args.mode == 'multi-cpu':
        train_multi_core_cpu(num_processes=args.processes, num_epochs=args.epochs)
    elif args.mode == 'benchmark':
        benchmark_configurations()
