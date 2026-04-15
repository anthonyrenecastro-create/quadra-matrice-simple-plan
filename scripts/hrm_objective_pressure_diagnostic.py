#!/usr/bin/env python3
"""Tiny diagnostic for HRM objective-pressure coupling.

Runs a short HRM simulation and records:
- objective
- objective_pressure
- f_coherence

Outputs:
- CSV trace file
- PNG plot (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadra.core.hrm import SimulationEngine


def run_trace(steps: int, dt: float) -> list[dict[str, float]]:
    engine = SimulationEngine()
    state = engine.initialize()
    rows: list[dict[str, float]] = []

    for i in range(steps):
        state, metrics = engine.step(state, dt=dt)
        rows.append(
            {
                "step": float(i + 1),
                "objective": float(metrics.get("objective", 0.0)),
                "objective_pressure": float(metrics.get("objective_pressure", 0.0)),
                "coherence": float(metrics.get("f_coherence", 0.0)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "objective", "objective_pressure", "coherence"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float]], plot_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    steps = [int(r["step"]) for r in rows]
    objective = [r["objective"] for r in rows]
    pressure = [r["objective_pressure"] for r in rows]
    coherence = [r["coherence"] for r in rows]

    def rolling_variance(series: list[float], window: int) -> list[float]:
        out: list[float] = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            chunk = series[start : i + 1]
            mu = sum(chunk) / max(1, len(chunk))
            var = sum((x - mu) ** 2 for x in chunk) / max(1, len(chunk))
            out.append(var)
        return out

    win = max(10, len(steps) // 20)
    objective_var = rolling_variance(objective, win)
    pressure_var = rolling_variance(pressure, win)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(steps, objective, label="objective", linewidth=1.8)
    ax.plot(steps, pressure, label="objective_pressure", linewidth=1.8)
    ax.plot(steps, coherence, label="coherence", linewidth=1.8)
    ax.set_title("HRM Coupling Diagnostic")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(steps, objective_var, label=f"rolling var(objective), w={win}", linewidth=1.6)
    ax2.plot(steps, pressure_var, label=f"rolling var(objective_pressure), w={win}", linewidth=1.6)
    ax2.set_xlabel("step")
    ax2.set_ylabel("rolling variance")
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="HRM objective-pressure diagnostic")
    parser.add_argument("--steps", type=int, default=400, help="simulation steps")
    parser.add_argument("--dt", type=float, default=0.02, help="simulation delta-t")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("diagnostics"),
        help="output directory for CSV/PNG",
    )
    args = parser.parse_args()

    rows = run_trace(steps=args.steps, dt=args.dt)

    csv_path = args.out_dir / "hrm_objective_pressure_trace.csv"
    png_path = args.out_dir / "hrm_objective_pressure_trace.png"

    write_csv(rows, csv_path)
    has_plot = write_plot(rows, png_path)

    print(f"trace csv: {csv_path}")
    if has_plot:
        print(f"trace plot: {png_path}")
    else:
        print("trace plot: skipped (matplotlib not installed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
