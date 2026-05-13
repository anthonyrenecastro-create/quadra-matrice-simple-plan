"""
ablation_grid.py — 3^5 = 243 combination sweep over AtlanteanHRMElement hyperparameters.

Grid axes:
  damping       : resonance_decay       = [0.95, 0.97, 0.99]
  close_N       : _GATE_CLOSE_N         = [3, 6, 9]
  phi_norm      : _PHI_MAX_NORM         = [2.5, 3.5, 5.0]
  recovery_duty : (_recovery_period,    = [(4,1)=25%, (2,1)=50%, (4,3)=75%]
                   _recovery_active)
  lam           : _RESONANCE_LAMBDA     = [0.01, 0.05, 0.10]

Metrics (lower is better for all):
  quality_retained   — mean |resonance| in final SIMPLE region after recovery
  calm_steps         — steps after first chaos onset until gate_mode=="CALM"
  lat_p95, lat_p99   — latency percentiles (ms) across all steps
  rl_integral        — sum of resonance_loss during RECOVERY regions

Composite score = mean of per-metric normalized ranks (0=best, 1=worst).
"""

import time
import itertools
import statistics
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
from atlantean_hrm_spi import AtlanteanHRMElement   # SPI engine only

# ---------------------------------------------------------------------------
# Adversarial region definitions  (mirrors benchmark_adversarial_regions.py)
# ---------------------------------------------------------------------------
REGIONS = [
    ("SIMPLE-1",   "A", 80),
    ("CHAOTIC-1",  "C", 60),
    ("RECOVERY-1", "R", 80),
    ("SIMPLE-2",   "A", 60),
    ("CHAOTIC-2",  "C", 50),
    ("RECOVERY-2", "R", 60),
]
TOTAL_STEPS = sum(r[2] for r in REGIONS)

GRID = {
    "damping":       [0.95, 0.97, 0.99],
    "close_N":       [3, 6, 9],
    "phi_norm":      [2.5, 3.5, 5.0],
    "recovery_duty": [("25%", 4, 1), ("50%", 2, 1), ("75%", 4, 3)],
    "lam":           [0.01, 0.05, 0.10],
}

FIELD_SIZE = 64   # keep small for speed during ablation
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _simple_signal(step: int, size: int) -> torch.Tensor:
    t = torch.linspace(0, 1, size)
    return (torch.sin(2 * 3.14159 * 3 * t + step * 0.07)
            + 0.5 * torch.cos(2 * 3.14159 * 7 * t)).unsqueeze(0)


def _chaotic_signal(step: int, size: int) -> torch.Tensor:
    base = _simple_signal(step, size)
    noise = torch.randn_like(base) * 1.5
    flip = torch.sign(torch.randn(1)) * base
    return (flip + noise) * (1 + 0.3 * torch.sin(torch.tensor(step * 0.41)))


def _recovery_signal(step: int, size: int) -> torch.Tensor:
    # gradual return: blend simple←→chaotic
    alpha = min(step / 30.0, 1.0)
    return alpha * _simple_signal(step, size) + (1 - alpha) * _chaotic_signal(step, size) * 0.3


# ---------------------------------------------------------------------------
# Run one configuration
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    quality_retained: float = 0.0   # mean |resonance| in final SIMPLE region
    calm_steps: int = 9999           # steps from first chaos until CALM
    lat_p95: float = 0.0
    lat_p99: float = 0.0
    rl_integral: float = 0.0         # sum resonance_loss during RECOVERY regions
    composite: float = 0.0           # filled after ranking


def run_config(
    damping: float,
    close_N: int,
    phi_norm: float,
    recovery_duty: Tuple[str, int, int],
    lam: float,
) -> RunMetrics:
    duty_label, rp, ra = recovery_duty
    hrm = AtlanteanHRMElement(field_size=FIELD_SIZE, device=DEVICE)
    hrm.resonance_decay      = damping
    hrm._GATE_CLOSE_N        = close_N
    hrm._PHI_MAX_NORM        = phi_norm
    hrm._RESONANCE_LAMBDA    = lam
    hrm._recovery_period     = rp
    hrm._recovery_active     = ra

    latencies: List[float] = []
    resonances_final_simple: List[float] = []
    rl_integral: float = 0.0

    # Timeline: track first chaos onset and first CALM recovery
    first_chaos_step: int = -1
    first_calm_after_chaos: int = -1

    global_step = 0
    for region_name, region_type, region_len in REGIONS:
        is_chaos   = region_type == "C"
        is_recovery = region_type == "R"
        is_simple  = region_type == "A"

        # Inject noise at start of chaotic region
        if is_chaos:
            with torch.no_grad():
                hrm.phi1 = hrm.phi1 + torch.randn_like(hrm.phi1) * 1.8
                hrm.phi5 = hrm.phi5 + torch.randn_like(hrm.phi5) * 0.9
            if hasattr(hrm, "_refresh_cached_stats"):
                hrm._refresh_cached_stats(None)
            if first_chaos_step < 0:
                first_chaos_step = global_step

        if is_simple or is_recovery:
            with torch.no_grad():
                hrm.phi1 = hrm.phi1 * 0.7
                hrm.phi5 = hrm.phi5 * 0.7
            if hasattr(hrm, "_refresh_cached_stats"):
                hrm._refresh_cached_stats(None)

        for local_step in range(region_len):
            if is_chaos:
                sig = _chaotic_signal(local_step, FIELD_SIZE)
                # periodic noise pulses
                if local_step % 12 == 6:
                    with torch.no_grad():
                        hrm.phi1 = hrm.phi1 + torch.randn_like(hrm.phi1) * 0.9
            elif is_recovery:
                sig = _recovery_signal(local_step, FIELD_SIZE)
            else:
                sig = _simple_signal(local_step, FIELD_SIZE)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = hrm.step(sig)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            res = float(hrm.phi1.norm().item()) / max(hrm.phi1.numel() ** 0.5, 1)

            if is_recovery:
                rl_integral += hrm.resonance_loss

            if is_simple and region_name.startswith("SIMPLE-2"):
                resonances_final_simple.append(res)

            # Track return to calm
            if first_chaos_step >= 0 and first_calm_after_chaos < 0:
                if hrm.gate_mode == "CALM":
                    first_calm_after_chaos = global_step

            global_step += 1

    # Compute metrics
    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    p95 = lat_sorted[int(0.95 * n)]
    p99 = lat_sorted[int(0.99 * n)]

    quality = (statistics.mean(resonances_final_simple)
               if resonances_final_simple else 999.0)

    calm_steps = (first_calm_after_chaos - first_chaos_step
                  if first_calm_after_chaos >= 0 and first_chaos_step >= 0
                  else 9999)

    return RunMetrics(
        quality_retained=quality,
        calm_steps=calm_steps,
        lat_p95=p95,
        lat_p99=p99,
        rl_integral=rl_integral,
    )


# ---------------------------------------------------------------------------
# Main: run all 243 combos, rank, report
# ---------------------------------------------------------------------------

def main() -> None:
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Running ablation: {len(combos)} combinations × ~{TOTAL_STEPS} steps each")
    print(f"Estimated time: ~{len(combos) * TOTAL_STEPS * 0.6 / 1000:.0f}s")
    print()

    records = []
    for i, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        m = run_config(
            damping=cfg["damping"],
            close_N=cfg["close_N"],
            phi_norm=cfg["phi_norm"],
            recovery_duty=cfg["recovery_duty"],
            lam=cfg["lam"],
        )
        records.append((cfg, m))
        if i % 25 == 0 or i == len(combos):
            print(f"  [{i:3d}/{len(combos)}] last: "
                  f"q={m.quality_retained:.4f}  calm={m.calm_steps:4d}  "
                  f"p95={m.lat_p95:.2f}ms  rl={m.rl_integral:.3f}")

    # Rank and compute composite (normalized rank 0=best, 1=worst)
    n = len(records)

    def rank_by(key_fn, descending=False):
        vals = [key_fn(r) for r in records]
        order = sorted(range(n), key=lambda i: vals[i], reverse=descending)
        ranks = [0.0] * n
        for pos, idx in enumerate(order):
            ranks[idx] = pos / max(n - 1, 1)
        return ranks

    r_quality  = rank_by(lambda r: r[1].quality_retained)
    r_calm     = rank_by(lambda r: r[1].calm_steps)
    r_p95      = rank_by(lambda r: r[1].lat_p95)
    r_p99      = rank_by(lambda r: r[1].lat_p99)
    r_rl       = rank_by(lambda r: r[1].rl_integral)

    for i, (cfg, m) in enumerate(records):
        m.composite = (r_quality[i] + r_calm[i] + r_p95[i] + r_p99[i] + r_rl[i]) / 5.0

    # Sort by composite
    records.sort(key=lambda r: r[1].composite)

    # Pretty-print top-20 table
    header = (f"{'#':>3}  {'damping':>7}  {'close_N':>7}  {'phi_norm':>8}  "
              f"{'duty':>5}  {'lam':>5}  "
              f"{'quality':>7}  {'calm':>6}  {'p95ms':>6}  {'p99ms':>6}  "
              f"{'rl_int':>7}  {'score':>6}")
    sep = "-" * len(header)
    print(f"\n{'=== TOP 20 CONFIGURATIONS (composite rank) ':=<80}")
    print(header)
    print(sep)
    for rank, (cfg, m) in enumerate(records[:20], 1):
        duty_label = cfg["recovery_duty"][0]
        print(f"{rank:>3}  {cfg['damping']:>7.2f}  {cfg['close_N']:>7d}  "
              f"{cfg['phi_norm']:>8.1f}  {duty_label:>5}  {cfg['lam']:>5.2f}  "
              f"{m.quality_retained:>7.4f}  {m.calm_steps:>6d}  "
              f"{m.lat_p95:>6.2f}  {m.lat_p99:>6.2f}  "
              f"{m.rl_integral:>7.3f}  {m.composite:>6.4f}")

    # Per-objective winners
    def winner(key_fn, label):
        best_i = min(range(n), key=lambda i: key_fn(records[i]))
        cfg, m = records[best_i]
        duty_label = cfg["recovery_duty"][0]
        print(f"  {label:<28} → damping={cfg['damping']}  close_N={cfg['close_N']}  "
              f"phi_norm={cfg['phi_norm']}  duty={duty_label}  lam={cfg['lam']}  "
              f"val={key_fn(records[best_i]):.4f}")

    print(f"\n{'=== PER-OBJECTIVE WINNERS ':=<80}")
    winner(lambda r: r[1].quality_retained, "quality_retained (lowest)")
    winner(lambda r: r[1].calm_steps,       "return_to_calm (fewest steps)")
    winner(lambda r: r[1].lat_p95,          "latency p95 (lowest ms)")
    winner(lambda r: r[1].lat_p99,          "latency p99 (lowest ms)")
    winner(lambda r: r[1].rl_integral,      "resonance_loss integral (lowest)")

    # Recommended config (composite #1)
    best_cfg, best_m = records[0]
    print(f"\n{'=== RECOMMENDED CONFIGURATION ':=<80}")
    print(f"  resonance_decay   = {best_cfg['damping']}")
    print(f"  _GATE_CLOSE_N     = {best_cfg['close_N']}")
    print(f"  _PHI_MAX_NORM     = {best_cfg['phi_norm']}")
    print(f"  recovery_duty     = {best_cfg['recovery_duty'][0]}  "
          f"(_recovery_period={best_cfg['recovery_duty'][1]}, "
          f"_recovery_active={best_cfg['recovery_duty'][2]})")
    print(f"  _RESONANCE_LAMBDA = {best_cfg['lam']}")
    print(f"\n  composite score   = {best_m.composite:.4f}  "
          f"(quality={best_m.quality_retained:.4f}  "
          f"calm={best_m.calm_steps}  "
          f"p95={best_m.lat_p95:.2f}ms  "
          f"p99={best_m.lat_p99:.2f}ms  "
          f"rl={best_m.rl_integral:.3f})")


if __name__ == "__main__":
    main()
