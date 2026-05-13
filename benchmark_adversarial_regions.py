#!/usr/bin/env python3
"""
benchmark_adversarial_regions.py
─────────────────────────────────────────────────────────────────────────────
Stress-test the gated PDE engine across alternating signal regions:

  SIMPLE   – repetitive, low-entropy  (AAAAAAA…)
  CHAOTIC  – high-entropy burst       (8391XZQ! logic-reversal, nested deps)
  RECOVERY – return to simple text    (the quick brown fox…)

Tracks per-step:
  • skip_flag      – was the PDE gated (skipped) this step?
  • resonance      – mean |phi1| as field resonance proxy
  • k_clusters     – number of distinct active membrane clusters
  • latency_ms     – wall-clock time for one step

Outputs a rich ASCII table, percentile summary, and a JSON trace file.
"""

import json
import time
import numpy as np
import torch
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))
from atlantean_hrm_spi import AtlanteanHRMElement

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

FIELD_SIZE = 64


def _text_to_signal(text: str, size: int, device: torch.device) -> torch.Tensor:
    """Encode a string as a float tensor via ordinal values, normalised to [-1, 1]."""
    codes = [ord(c) / 127.0 - 1.0 for c in text[:size]]
    if len(codes) < size:
        codes += [0.0] * (size - len(codes))
    return torch.tensor(codes[:size], dtype=torch.float32, device=device)


@dataclass
class Region:
    name: str
    label: str       # A=simple, C=chaotic, R=recovery
    steps: int
    signal_text: str


REGIONS: List[Region] = [
    Region("SIMPLE-1",   "A", 80,
           "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    Region("CHAOTIC",    "C", 60,
           "8391XZQ! ~logic_reversal~ ∀x¬(P(x)∧Q(x)) nested(dep(lambda:x**2)) "
           "symbolic::constraint[!@#$%^&*] entropy_burst 9182 NULL VOID BREAK"),
    Region("RECOVERY-1", "R", 80,
           "the quick brown fox jumps over the lazy dog and then walks back again"),
    Region("SIMPLE-2",   "A", 60,
           "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    Region("CHAOTIC-2",  "C", 50,
           "8391XZQ! 7T!%2 reversal(¬A→B) constraint[x∈∅] NULLNULL nested nested "
           "BREAK BREAK #@! dependency::cycle(a→b→c→a) runtime_error exception"),
    Region("RECOVERY-2", "R", 60,
           "we are going back to normal text now everything should calm down again"),
]

# Noise magnitude injected into phi1/phi5 at the START of a chaotic region and
# as periodic pulses within it.  Simple/recovery regions get a small damping push.
_CHAOS_NOISE    = 1.8   # large spike — tanks stability → gate opens wide
_CHAOS_PULSE_N  = 12    # re-inject every N steps inside the chaotic region
_CHAOS_PULSE    = 0.9   # reduced follow-up pulse magnitude
_SIMPLE_DAMP    = 0.05  # gentle drift toward zero keeps field calm

REGION_COLORS = {"A": "[ SIMPLE ]", "C": "[CHAOTIC ]", "R": "[RECOVER]"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP-LEVEL TRACKING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    region: str
    label: str
    skipped: bool
    gate_mode: str      # CALM | RECOVERY | CHAOTIC
    resonance: float
    resonance_loss: float
    k_clusters: int
    latency_ms: float


def _compute_resonance(element: AtlanteanHRMElement) -> float:
    """Mean absolute phi1 as a resonance proxy (stays on CPU; phi1 is small)."""
    return float(element.phi1.abs().mean().item())


def _compute_clusters(element: AtlanteanHRMElement, k: int = 4) -> int:
    """
    K-means on the membrane tensor to count distinct activation clusters.
    Uses the already-on-CPU field; k is capped to available samples.
    """
    data = element.membrane.detach().cpu().numpy().reshape(-1, 1)
    k_actual = min(k, max(1, len(np.unique(data.round(2)))))
    if k_actual < 2:
        return 1
    km = KMeans(n_clusters=k_actual, random_state=0, n_init="auto")
    km.fit(data)
    # Return number of clusters that have meaningful membership (>1 sample)
    counts = np.bincount(km.labels_)
    return int(np.sum(counts > 1))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(field_size: int = FIELD_SIZE) -> List[StepRecord]:
    device = torch.device("cpu")
    element = AtlanteanHRMElement(field_size, device)
    # Warm up — run 20 steps on simple signal before recording
    warmup_sig = _text_to_signal("A" * field_size, field_size, device)
    for _ in range(20):
        element.step(warmup_sig, hrm_force=0.0, dt=0.01, stability_hint=0.95)

    records: List[StepRecord] = []
    global_step = 0

    for region in REGIONS:
        sig = _text_to_signal(region.signal_text, field_size, device)
        hint = 0.28 if region.label == "C" else 0.92

        # ── Field-state injection at region boundary ──────────────────────
        if region.label == "C":
            # Destabilise: add large noise to both PDE fields so
            # cached_stability drops below _STAB_FULL → gate is forced open.
            # We also force a stats refresh so the cache picks up the new variance.
            with torch.no_grad():
                element.phi1.add_(torch.randn_like(element.phi1) * _CHAOS_NOISE)
                element.phi5.add_(torch.randn_like(element.phi5) * _CHAOS_NOISE * 0.5)
            element._refresh_cached_stats(stability_hint=hint)
            element._cached_gate = True   # mark gate as needing a decision next step
        elif region.label in ("A", "R"):
            # Gentle damping pull — moves fields toward zero, variance falls,
            # stability climbs.  Does NOT force a refresh; the gate makes that call.
            with torch.no_grad():
                element.phi1.mul_(0.7)
                element.phi5.mul_(0.7)
            element._refresh_cached_stats(stability_hint=hint)
        # ─────────────────────────────────────────────────────────────────

        for local_step in range(region.steps):
            # Periodic chaos pulses keep the field unstable throughout the region
            if region.label == "C" and local_step > 0 and local_step % _CHAOS_PULSE_N == 0:
                with torch.no_grad():
                    element.phi1.add_(torch.randn_like(element.phi1) * _CHAOS_PULSE)
                    element.phi5.add_(torch.randn_like(element.phi5) * _CHAOS_PULSE * 0.4)
                element._refresh_cached_stats(stability_hint=hint)
                element._cached_gate = True

            t0 = time.perf_counter()
            element.step(sig, hrm_force=0.0, dt=0.01, stability_hint=hint)
            latency = (time.perf_counter() - t0) * 1000.0

            skipped = not element.last_run_full_fields
            resonance = _compute_resonance(element)
            k = _compute_clusters(element)

            records.append(StepRecord(
                step=global_step,
                region=region.name,
                label=region.label,
                skipped=skipped,
                gate_mode=element.gate_mode,
                resonance=resonance,
                resonance_loss=element.resonance_loss,
                k_clusters=k,
                latency_ms=latency,
            ))
            global_step += 1

    return records


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(values: List[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def print_report(records: List[StepRecord]) -> None:
    from collections import defaultdict, Counter
    print()
    print("=" * 100)
    print("  ADVERSARIAL REGION BENCHMARK  -- mode / skip / resonance / clusters / latency")
    print("=" * 100)

    by_region: dict = defaultdict(list)
    for r in records:
        by_region[r.region].append(r)

    header = (
        f"{'Region':<14} {'Type':<10} {'Steps':>6}  "
        f"{'Mode dist':>20}  {'Skip%':>7}  {'Resonance':>10}  "
        f"{'RLoss':>7}  {'Clusters':>9}  {'Lat p50':>8}  {'Lat p95':>8}  {'Lat max':>8}"
    )
    print(header)
    print("-" * 100)

    prev_clusters = None
    for region in REGIONS:
        rs = by_region[region.name]
        skip_pct = 100.0 * sum(r.skipped for r in rs) / len(rs)
        resonances = [r.resonance for r in rs]
        rl = [r.resonance_loss for r in rs]
        lats = [r.latency_ms for r in rs]
        ks = [r.k_clusters for r in rs]
        k_mode = int(np.median(ks))
        cluster_change = ""
        if prev_clusters is not None and k_mode != prev_clusters:
            cluster_change = f" <- was {prev_clusters}"
        prev_clusters = k_mode

        mode_counts = Counter(r.gate_mode for r in rs)
        total = len(rs)
        mode_dist = " ".join(
            f"{m[0]}:{100 * c // total}%" for m, c in sorted(mode_counts.items())
        )

        flag = REGION_COLORS[region.label]
        print(
            f"{region.name:<14} {flag:<10} {len(rs):>6}  "
            f"{mode_dist:>20}  "
            f"{skip_pct:>6.1f}%  "
            f"{np.mean(resonances):>10.4f}  "
            f"{np.mean(rl):>7.5f}  "
            f"{k_mode:>6}{cluster_change:<5}  "
            f"{_percentile(lats, 50):>7.3f}  "
            f"{_percentile(lats, 95):>7.3f}  "
            f"{max(lats):>7.3f}"
        )

    # Gate mode transitions
    print()
    print("-" * 100)
    print("  Gate mode transitions:")
    prev_mode = None
    for r in records:
        if r.gate_mode != prev_mode:
            print(f"    step {r.step:>4}  {r.region:<14} "
                  f"{prev_mode or '---':>10} -> {r.gate_mode:<10}  "
                  f"resonance={r.resonance:.4f}  resonance_loss={r.resonance_loss:.5f}")
            prev_mode = r.gate_mode

    # Latency spikes
    total_lats = [r.latency_ms for r in records]
    spike_threshold = _percentile(total_lats, 95)
    spikes = [r for r in records if r.latency_ms >= spike_threshold]

    print()
    print("-" * 100)
    print(f"  Latency spike threshold (P95): {spike_threshold:.3f} ms  -- "
          f"{len(spikes)} spikes across {len(records)} total steps")
    print()
    print(f"  {'Step':>5}  {'Region':<14} {'Type':<10}  {'Mode':<10}  {'Skipped':>7}  {'Latency':>9}")
    for r in spikes[:10]:
        print(f"  {r.step:>5}  {r.region:<14} {REGION_COLORS[r.label]:<10}  "
              f"{r.gate_mode:<10}  {'yes' if r.skipped else 'no':>7}  {r.latency_ms:>8.3f} ms")
    if len(spikes) > 10:
        print(f"   ... and {len(spikes) - 10} more")

    # Resonance distribution
    print()
    print("-" * 100)
    print("  Resonance distribution by region type")
    print()
    for label, typename in [("A", "SIMPLE"), ("C", "CHAOTIC"), ("R", "RECOVER")]:
        rs = [r for r in records if r.label == label]
        if not rs:
            continue
        res = [r.resonance for r in rs]
        p25 = _percentile(res, 25)
        p50 = _percentile(res, 50)
        p75 = _percentile(res, 75)
        skip_pct = 100.0 * sum(r.skipped for r in rs) / len(rs)
        mode_counts = Counter(r.gate_mode for r in rs)
        mode_str = "  ".join(f"{m}={c}" for m, c in sorted(mode_counts.items()))
        print(
            f"  {typename:<8}  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f}  "
            f"skip={skip_pct:.1f}%   modes: {mode_str}"
        )

    # Resonance decay bar chart
    print()
    print("-" * 100)
    print("  Resonance over time (smoothed per-10-step mean):")
    chunk = 10
    chunks = [records[i:i + chunk] for i in range(0, len(records), chunk)]
    bar_max = max(np.mean([r.resonance for r in c]) for c in chunks if c)
    for c in chunks:
        val = np.mean([r.resonance for r in c])
        bar_w = int(40 * val / bar_max) if bar_max > 0 else 0
        print(f"  steps {c[0].step:>3}-{c[-1].step:<3} [{c[0].label}]  "
              f"{'#' * bar_w:<40}  {val:.4f}")

    # Global stats
    print()
    print("-" * 100)
    total_skipped = sum(r.skipped for r in records)
    print(f"  Total steps: {len(records)}  |  Skipped: {total_skipped} "
          f"({100.0 * total_skipped / len(records):.1f}%)  |  Full PDE: {len(records) - total_skipped}")
    print(f"  Overall latency: mean={np.mean(total_lats):.3f}ms  "
          f"p50={_percentile(total_lats, 50):.3f}ms  "
          f"p95={spike_threshold:.3f}ms  "
          f"max={max(total_lats):.3f}ms")
    avg_rl = np.mean([r.resonance_loss for r in records])
    print(f"  Avg resonance_loss: {avg_rl:.6f}  (lambda=0.05)")
    print("=" * 100)
    print()


def save_trace(records: List[StepRecord], path: Path) -> None:
    data = [asdict(r) for r in records]
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Trace saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-size", type=int, default=FIELD_SIZE,
                        help="Field size (default 64)")
    parser.add_argument("--out", type=str, default=".benchmarks/adversarial_trace.json",
                        help="Output JSON trace path")
    args = parser.parse_args()

    print(f"\n  Field size: {args.field_size}  |  Regions: {len(REGIONS)}  |  "
          f"Total steps: {sum(r.steps for r in REGIONS)}")
    print("  Running…")

    records = run_benchmark(args.field_size)
    print_report(records)
    save_trace(records, Path(args.out))
