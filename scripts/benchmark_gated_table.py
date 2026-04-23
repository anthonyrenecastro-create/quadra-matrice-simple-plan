#!/usr/bin/env python3
"""Compact CLI benchmark for gated field-update modes.

Runs live, low-interaction, and forced-full benchmark modes for SPI and/or GPU
engines and prints a compact table in one command.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Add repository root to import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlantean_hrm_spi import OscillatorySynapseTheory
from atlantean_hrm_gpu import GPUConfig, GPUOscillatorEngine


def _fmt(value: float, width: int = 10, decimals: int = 3) -> str:
    return f"{value:>{width}.{decimals}f}"


def _build_rows(label: str, stats: Dict[str, float]) -> List[Tuple[str, str, str, str, str, str, str]]:
    live_total = float(stats["gated_total_ms"])
    live_step = float(stats["gated_step_ms"])
    live_skip = float(stats["skipped_step_frequency"]) * 100.0

    low_total = float(stats["low_interaction_total_ms"])
    low_step = float(stats["low_interaction_step_ms"])
    low_skip = float(stats["low_interaction_skipped_step_frequency"]) * 100.0

    forced_total = float(stats["forced_full_total_ms"])
    forced_step = float(stats["forced_full_step_ms"])

    # Positive value means mode is faster than forced-full baseline.
    live_savings = 100.0 * (forced_total - live_total) / forced_total if forced_total > 0 else 0.0
    low_savings = 100.0 * (forced_total - low_total) / forced_total if forced_total > 0 else 0.0

    return [
        (label, "live", _fmt(live_skip, 8, 2), _fmt(live_step), _fmt(live_total), _fmt(live_savings, 9, 2), "gated"),
        (label, "low-int", _fmt(low_skip, 8, 2), _fmt(low_step), _fmt(low_total), _fmt(low_savings, 9, 2), "gated"),
        (label, "forced", _fmt(0.0, 8, 2), _fmt(forced_step), _fmt(forced_total), _fmt(0.0, 9, 2), "baseline"),
    ]


def _print_table(rows: List[Tuple[str, str, str, str, str, str, str]]) -> None:
    headers = ("engine", "mode", "skip%", "step_ms", "total_ms", "save_vs_forced%", "note")
    widths = (8, 8, 8, 10, 10, 15, 9)

    def line(values: Tuple[str, ...]) -> str:
        return " | ".join(v.rjust(w) for v, w in zip(values, widths))

    print(line(headers))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        print(line(r))


def _run_spi(args: argparse.Namespace) -> Dict[str, float]:
    spi = OscillatorySynapseTheory(field_size=args.field_size, device="cpu")
    return spi.benchmark_gated_path(
        num_steps=args.steps,
        warmup_steps=args.warmup,
        dt=args.dt,
        stability_hint=args.stability_hint,
    )


def _run_gpu(args: argparse.Namespace) -> Dict[str, float]:
    config = GPUConfig(
        use_cuda=args.use_cuda,
        mixed_precision=args.mixed_precision,
        batch_size=args.batch_size,
    )
    gpu = GPUOscillatorEngine(field_size=args.field_size, config=config)
    return gpu.benchmark_gated_path(
        num_steps=args.steps,
        warmup_steps=args.warmup,
        dt=args.dt,
        stability_hint=args.stability_hint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact gated-path benchmark table")
    parser.add_argument("--engine", choices=["spi", "gpu", "both"], default="both")
    parser.add_argument("--field-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--stability-hint", type=float, default=0.95)
    parser.add_argument("--use-cuda", action="store_true", help="Enable CUDA path for GPU engine")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision in GPU config")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rows: List[Tuple[str, str, str, str, str, str, str]] = []

    if args.engine in ("spi", "both"):
        spi_stats = _run_spi(args)
        rows.extend(_build_rows("spi", spi_stats))

    if args.engine in ("gpu", "both"):
        gpu_stats = _run_gpu(args)
        rows.extend(_build_rows("gpu", gpu_stats))

    print("\nGated Path Micro-Benchmark")
    print(f"steps={args.steps} warmup={args.warmup} field={args.field_size} dt={args.dt} stability_hint={args.stability_hint}")
    _print_table(rows)


if __name__ == "__main__":
    main()
