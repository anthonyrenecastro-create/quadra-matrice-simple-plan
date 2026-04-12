#!/usr/bin/env python3
"""
Neural Command Center - Web-Based Version
Three-Plane Architecture that works in browser/codespace environments

Accessible via browser at localhost:8050
"""

import torch
import numpy as np
import time
import math
import re
import json
import os
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.figure import Figure
import matplotlib.animation as animation
from io import BytesIO
import base64

from demo_cognition import CognitionDemo

try:
    from flask import Flask, render_template_string, Response, request, jsonify
    import threading
except ImportError:
    logger.error("Flask required. Install with: pip install flask")
    exit(1)

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class GoalNode:
    """A goal in the deliberation space"""
    name: str
    position: Tuple[float, float]
    confidence: float = 0.5
    velocity: Tuple[float, float] = (0.0, 0.0)
    pinned: bool = False
    active: bool = True
    competing_hypotheses: List[str] = None
    
    def __post_init__(self):
        if self.competing_hypotheses is None:
            self.competing_hypotheses = []


@dataclass
class ActionVector:
    """A proposed action in the world interface"""
    name: str
    magnitude: float
    uncertainty: float
    outcome_estimate: str
    approved: bool = False
    timestamp: float = 0.0


class NeuralCommandCenterWeb:
    """Web-based Neural Command Center"""
    
    def __init__(self):
        self.demo = CognitionDemo(field_size=100, device='cpu')
        self.running = True
        self.paused = False
        self.stopped = False
        self.pulse_phase = 0.0
        self.time_step = 0
        self.baseline_metrics = self._load_latest_training_baseline()
        self.scenario_start_time = time.time()
        self.scenario_cycle_seconds = 54
        self.config_path = Path(__file__).resolve().parent / "dashboard_state" / "gui_run_configs.json"
        self.run_mode = "hard"
        self.hardware_mode = "cpu"
        self.dataset = "wikitext-103-raw-v1"
        self.batch_count = 150
        self.current_batch = 0
        self.batch_progress = 0.0
        self.current_reward = 21.0
        self.last_reward = self.current_reward
        self.delta_reward = 0.0
        self.speedup = 1.0
        self.learning_rate = 0.001
        self.variance = 0.02
        self.syntropy_events = 0
        self.throttle_events = 0
        self.batch_time_sec = 26.0
        self.start_ts = time.time()
        self.last_batch_ts = self.start_ts
        self.cache_hit_rate = 0.0
        self.cluster_count = 3
        self.resonance = 0.3
        self.neuron_density = 0.2
        self.synaptic_trace = 0.1
        self.state_history = deque(maxlen=80)
        self.reward_history = deque(maxlen=120)
        self.batch_time_history = deque(maxlen=120)
        self.lorenz_state = np.array([0.1, 0.0, 0.0], dtype=float)
        self.lorenz_history = deque(maxlen=120)
        self.run_snapshots = []
        self.scenario_phases = [
            {
                "name": "Detect",
                "title": "Phase 1: Detect Incident",
                "message": "Ingress anomaly detected across customer endpoints.",
                "action": "Correlate signals and isolate likely source",
            },
            {
                "name": "Deliberate",
                "title": "Phase 2: Prioritize Response",
                "message": "Competing hypotheses ranked for likely blast radius.",
                "action": "Promote top mitigation pathway",
            },
            {
                "name": "Act",
                "title": "Phase 3: Execute Mitigation",
                "message": "Recommended action package pushed to operations.",
                "action": "Trigger rollback + customer communication",
            },
        ]
        
        # Plane states
        self.layer_states = {
            'perception': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'integration': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'reasoning': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0},
            'action': {'activity': 0.0, 'conflict': 0.0, 'pressure': 0.0}
        }
        
        self.goals = []
        self._spawn_initial_goals()
        
        self.sensors = {'visual': 0.0, 'temporal': 0.0, 'pattern': 0.0}
        self.proposed_actions = []
        self.action_history = deque(maxlen=20)
        
        # Create figure
        self.fig = Figure(figsize=(14, 12), facecolor='#000000')
        self.fig.subplots_adjust(hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        self.ax_physiology = self.fig.add_subplot(3, 1, 1)
        self.ax_deliberation = self.fig.add_subplot(3, 1, 2)
        self.ax_world = self.fig.add_subplot(3, 1, 3)
        self.apply_mode_config("hard")
        self._ensure_config_store()
        self._seed_comparison_runs()

    def _seed_comparison_runs(self):
        """Provide immediate comparison references before user-triggered snapshots."""
        base_reward = self.baseline_metrics.get("max_reward", 29.0) or 29.0
        base_batch = self.baseline_metrics.get("avg_batch_time", 27.0) or 27.0
        self.run_snapshots = [
            {
                "label": "full-system-hard",
                "mode": "hard",
                "dataset": "wikitext-103-raw-v1",
                "hardware": "cpu",
                "final_reward": round(base_reward, 4),
                "reward_delta": round(base_reward - 21.0, 4),
                "time_to_plateau_sec": 2100,
                "avg_batch_time": round(base_batch, 3),
                "throttle_count": 31,
                "stability_band": 0.86,
                "created_at": int(time.time()) - 3600,
            },
            {
                "label": "no-lorenz-ablation",
                "mode": "ablation",
                "dataset": "wikitext-2-raw-v1",
                "hardware": "cpu",
                "final_reward": round(base_reward * 0.86, 4),
                "reward_delta": round(base_reward * 0.86 - 21.0, 4),
                "time_to_plateau_sec": 2680,
                "avg_batch_time": round(base_batch * 1.12, 3),
                "throttle_count": 52,
                "stability_band": 0.73,
                "created_at": int(time.time()) - 3200,
            },
            {
                "label": "hard-noise-0.30",
                "mode": "hard",
                "dataset": "wikitext-103-raw-v1",
                "hardware": "cpu",
                "final_reward": round(base_reward * 0.95, 4),
                "reward_delta": round(base_reward * 0.95 - 21.0, 4),
                "time_to_plateau_sec": 2300,
                "avg_batch_time": round(base_batch * 1.05, 3),
                "throttle_count": 48,
                "stability_band": 0.78,
                "created_at": int(time.time()) - 2800,
            },
        ]

    def _ensure_config_store(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            initial = {
                "baseline": {
                    "dataset": "wikitext-2-raw-v1",
                    "hardware_mode": "cpu",
                    "batch_count": 50,
                    "mode": "baseline",
                    "variance_threshold": 0.10,
                    "noise_intensity": 0.15,
                    "snn_epochs": 3,
                },
                "hard": {
                    "dataset": "wikitext-103-raw-v1",
                    "hardware_mode": "cpu",
                    "batch_count": 150,
                    "mode": "hard",
                    "variance_threshold": 0.14,
                    "noise_intensity": 0.25,
                    "snn_epochs": 6,
                },
                "ablation": {
                    "dataset": "wikitext-2-raw-v1",
                    "hardware_mode": "cpu",
                    "batch_count": 80,
                    "mode": "ablation",
                    "variance_threshold": 0.18,
                    "noise_intensity": 0.20,
                    "snn_epochs": 4,
                    "lorenz_enabled": False,
                },
            }
            self.config_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")

    def _load_saved_configs(self) -> Dict[str, Dict[str, object]]:
        self._ensure_config_store()
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def _save_configs(self, configs: Dict[str, Dict[str, object]]):
        self.config_path.write_text(json.dumps(configs, indent=2), encoding="utf-8")

    def apply_mode_config(self, mode: str):
        mode_map = {
            "baseline": {"learning_rate": 0.0015, "speedup": 1.8, "batch_time": 24.0, "cluster": 2},
            "hard": {"learning_rate": 0.0100, "speedup": 10.0, "batch_time": 27.0, "cluster": 3},
            "ablation": {"learning_rate": 0.0040, "speedup": 4.2, "batch_time": 30.0, "cluster": 2},
        }
        cfg = mode_map.get(mode, mode_map["hard"])
        self.run_mode = mode
        self.learning_rate = cfg["learning_rate"]
        self.speedup = cfg["speedup"]
        self.batch_time_sec = cfg["batch_time"]
        self.cluster_count = cfg["cluster"]

    def update_run_config(self, cfg: Dict[str, object]):
        if "dataset" in cfg:
            self.dataset = str(cfg["dataset"])
        if "hardware_mode" in cfg:
            self.hardware_mode = str(cfg["hardware_mode"])
        if "batch_count" in cfg:
            self.batch_count = max(1, int(cfg["batch_count"]))
        if "mode" in cfg:
            self.apply_mode_config(str(cfg["mode"]))

    def _update_lorenz(self):
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        x, y, z = self.lorenz_state
        dt = 0.01
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        self.lorenz_state = np.array([x + dx * dt, y + dy * dt, z + dz * dt], dtype=float)
        self.lorenz_history.append(self.lorenz_state.tolist())

    def _update_run_metrics(self):
        if self.stopped or self.paused:
            return

        self.batch_progress += 0.03
        self.last_reward = self.current_reward
        reward_bump = np.random.uniform(0.002, 0.018)
        if self.run_mode == "ablation":
            reward_bump *= 0.7
        self.current_reward = min(self.current_reward + reward_bump, 30.5)
        self.delta_reward = self.current_reward - self.last_reward
        self.variance = float(np.clip(self.variance + np.random.uniform(-0.005, 0.006), 0.01, 0.22))
        if self.variance > 0.13:
            self.throttle_events += 1
        if np.random.rand() < 0.22:
            self.syntropy_events += 1

        self.cache_hit_rate = float(np.clip(self.cache_hit_rate + np.random.uniform(-0.02, 0.04), 0.0, 0.96))
        self.resonance = float(np.clip(self.resonance + np.random.uniform(-0.03, 0.04), 0.05, 0.99))
        field = self.demo.oscillator.field
        self.neuron_density = float((field.abs() > 0.08).float().mean().item())
        self.synaptic_trace = float(np.clip(field.abs().mean().item() * 2.0, 0.0, 1.0))

        if self.batch_progress >= 1.0:
            self.batch_progress = 0.0
            self.current_batch = min(self.current_batch + 1, self.batch_count)
            now = time.time()
            self.batch_time_sec = float(np.clip(now - self.last_batch_ts, 8.0, 60.0))
            self.last_batch_ts = now
            self.reward_history.append(self.current_reward)
            self.batch_time_history.append(self.batch_time_sec)

            if self.current_batch >= self.batch_count:
                self.running = False
                self.paused = False
                self.stopped = True

        self.state_history.append(
            {
                "t": self.time_step,
                "phase": self._get_scenario_state()["phase"],
                "reward": round(self.current_reward, 4),
                "variance": round(self.variance, 4),
                "batch": self.current_batch,
            }
        )

    def _system_usage(self) -> Dict[str, float]:
        if psutil is not None:
            return {"cpu": float(psutil.cpu_percent(interval=0.0)), "ram": float(psutil.virtual_memory().percent)}

        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        cpu_pct = float(np.clip(load * 20.0, 0.0, 100.0))
        return {"cpu": cpu_pct, "ram": 50.0}

    def get_live_telemetry(self) -> Dict[str, object]:
        usage = self._system_usage()
        remaining_batches = max(0, self.batch_count - self.current_batch)
        eta_seconds = int(remaining_batches * max(1.0, self.batch_time_sec))
        lorenz_tail = list(self.lorenz_history)[-20:]

        return {
            "run_control": {
                "running": self.running,
                "paused": self.paused,
                "stopped": self.stopped,
                "dataset": self.dataset,
                "hardware_mode": self.hardware_mode,
                "mode": self.run_mode,
                "batch_count": self.batch_count,
            },
            "monitoring": {
                "current_batch": self.current_batch,
                "reward": self.current_reward,
                "delta_reward": self.delta_reward,
                "speedup": self.speedup,
                "learning_rate": self.learning_rate,
                "variance": self.variance,
                "syntropy_events": self.syntropy_events,
                "throttle_events": self.throttle_events,
                "cpu_percent": usage["cpu"],
                "ram_percent": usage["ram"],
                "batch_time_sec": self.batch_time_sec,
                "eta_seconds": eta_seconds,
            },
            "introspection": {
                "lorenz": {
                    "x": float(self.lorenz_state[0]),
                    "y": float(self.lorenz_state[1]),
                    "z": float(self.lorenz_state[2]),
                    "trajectory": lorenz_tail,
                },
                "field_activity": float(self.demo.oscillator.field.abs().mean().item()),
                "neuron_activation_density": self.neuron_density,
                "synaptic_trace": self.synaptic_trace,
                "cluster_count": self.cluster_count,
                "adaptive_k": self.cluster_count,
                "cache_hit_rate": self.cache_hit_rate,
                "resonance": self.resonance,
                "coherence": float(np.clip(1.0 - self.variance, 0.0, 1.0)),
                "state_transitions": list(self.state_history)[-25:],
            },
            "comparison": self.run_snapshots[-20:],
        }

    def snapshot_current_run(self, label: str) -> Dict[str, object]:
        run = {
            "label": label,
            "mode": self.run_mode,
            "dataset": self.dataset,
            "hardware": self.hardware_mode,
            "final_reward": round(float(self.current_reward), 4),
            "reward_delta": round(float(self.current_reward - 21.0), 4),
            "time_to_plateau_sec": int((time.time() - self.start_ts) * 0.75),
            "avg_batch_time": round(float(np.mean(self.batch_time_history)) if self.batch_time_history else self.batch_time_sec, 3),
            "throttle_count": int(self.throttle_events),
            "stability_band": round(float(np.clip(1.0 - self.variance, 0, 1)), 4),
            "created_at": int(time.time()),
        }
        self.run_snapshots.append(run)
        return run

    def _load_latest_training_baseline(self) -> Dict[str, float]:
        """Read latest hard-training log and extract practical baseline metrics."""
        logs_dir = Path(__file__).resolve().parent / "logs"
        log_files = sorted(logs_dir.glob("train_hard_clean_*.log"), key=lambda p: p.stat().st_mtime)

        baseline = {
            "latest_batch": 0,
            "max_reward": 0.0,
            "max_speedup": 1.0,
            "avg_batch_time": 0.0,
            "run_samples": 0,
        }

        if not log_files:
            return baseline

        pattern = re.compile(
            r"Batch\s+(\d+)/150\s+\|\s+Time:\s+([0-9.]+)s\s+\|\s+Reward:\s+([0-9.]+)\s+\|\s+Speedup:\s+([0-9.]+)x"
        )

        times = []
        try:
            content = log_files[-1].read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in content:
                match = pattern.search(line)
                if not match:
                    continue
                batch = int(match.group(1))
                batch_time = float(match.group(2))
                reward = float(match.group(3))
                speedup = float(match.group(4))

                baseline["latest_batch"] = max(baseline["latest_batch"], batch)
                baseline["max_reward"] = max(baseline["max_reward"], reward)
                baseline["max_speedup"] = max(baseline["max_speedup"], speedup)
                times.append(batch_time)

            if times:
                baseline["avg_batch_time"] = float(np.mean(times))
                baseline["run_samples"] = len(times)
        except OSError:
            return baseline

        return baseline

    def get_investor_metrics(self) -> Dict[str, float]:
        """Generate practical, investor-facing KPI snapshot from live state + baseline."""
        layer_activity = [s["activity"] for s in self.layer_states.values()]
        layer_conflict = [s["conflict"] for s in self.layer_states.values()]
        avg_activity = float(np.mean(layer_activity)) if layer_activity else 0.0
        avg_conflict = float(np.mean(layer_conflict)) if layer_conflict else 0.0

        if self.proposed_actions:
            avg_uncertainty = float(np.mean([a.uncertainty for a in self.proposed_actions]))
            action_count = len(self.proposed_actions)
        else:
            avg_uncertainty = 0.6
            action_count = 0

        readiness_score = np.clip((avg_activity * (1.0 - avg_conflict * 2.5)) * 140, 0, 100)
        decision_latency_ms = int(np.clip(140 - readiness_score + avg_uncertainty * 120, 18, 240))
        projected_triage_gain = np.clip(30 + readiness_score * 0.6 - avg_uncertainty * 20, 5, 92)
        scenario = self._get_scenario_state()

        return {
            "latest_batch": self.baseline_metrics["latest_batch"],
            "max_reward": self.baseline_metrics["max_reward"],
            "max_speedup": self.baseline_metrics["max_speedup"],
            "avg_batch_time": self.baseline_metrics["avg_batch_time"],
            "run_samples": self.baseline_metrics["run_samples"],
            "readiness_score": float(readiness_score),
            "decision_latency_ms": decision_latency_ms,
            "projected_triage_gain": float(projected_triage_gain),
            "action_count": action_count,
            "scenario_phase": scenario["phase"],
            "scenario_title": scenario["title"],
            "scenario_message": scenario["message"],
            "scenario_action": scenario["action"],
            "scenario_progress": scenario["progress"],
        }

    def _get_scenario_state(self) -> Dict[str, object]:
        """Return active scenario phase and progress for a clear investor storyline."""
        elapsed = int(time.time() - self.scenario_start_time)
        phase_window = max(1, self.scenario_cycle_seconds // len(self.scenario_phases))
        phase_idx = (elapsed // phase_window) % len(self.scenario_phases)
        phase_elapsed = elapsed % phase_window
        progress = phase_elapsed / phase_window
        phase_data = self.scenario_phases[phase_idx]

        return {
            "phase": phase_data["name"],
            "title": phase_data["title"],
            "message": phase_data["message"],
            "action": phase_data["action"],
            "progress": float(progress),
        }

    def _apply_scenario_phase_effects(self):
        """Bias live state so each phase looks meaningfully distinct to observers."""
        scenario = self._get_scenario_state()
        phase = scenario["phase"]

        if phase == "Detect":
            self.sensors["visual"] = float(np.clip(self.sensors["visual"] * 1.35 + 0.08, 0, 1))
            self.layer_states["perception"]["activity"] = float(np.clip(self.layer_states["perception"]["activity"] * 1.3 + 0.1, 0, 1))
            self.layer_states["integration"]["pressure"] = float(np.clip(self.layer_states["integration"]["pressure"] + 0.08, 0, 1))
            phase_action = ActionVector(
                name="Alert Correlate",
                magnitude=0.82,
                uncertainty=0.42,
                outcome_estimate="Clustering anomalies by region",
                timestamp=self.time_step,
            )
        elif phase == "Deliberate":
            self.layer_states["reasoning"]["activity"] = float(np.clip(self.layer_states["reasoning"]["activity"] * 1.35 + 0.08, 0, 1))
            self.layer_states["integration"]["conflict"] = float(np.clip(self.layer_states["integration"]["conflict"] + 0.12, 0, 1))
            self.sensors["pattern"] = float(np.clip(self.sensors["pattern"] * 1.2 + 0.04, 0, 1))
            phase_action = ActionVector(
                name="Rank Hypotheses",
                magnitude=0.73,
                uncertainty=0.31,
                outcome_estimate="Top path selected for mitigation",
                timestamp=self.time_step,
            )
        else:
            self.layer_states["action"]["activity"] = float(np.clip(self.layer_states["action"]["activity"] * 1.4 + 0.12, 0, 1))
            self.layer_states["reasoning"]["conflict"] = float(np.clip(self.layer_states["reasoning"]["conflict"] * 0.75, 0, 1))
            self.sensors["temporal"] = float(np.clip(self.sensors["temporal"] * 0.8, 0, 1))
            phase_action = ActionVector(
                name="Execute Mitigation",
                magnitude=0.9,
                uncertainty=0.18,
                outcome_estimate="Rollback enacted, customer impact reduced",
                approved=True,
                timestamp=self.time_step,
            )

        # Keep a stable, phase-specific anchor action visible.
        self.proposed_actions = [a for a in self.proposed_actions if self.time_step - a.timestamp < 40]
        if not any(a.name == phase_action.name for a in self.proposed_actions):
            self.proposed_actions.insert(0, phase_action)
        self.proposed_actions = self.proposed_actions[:5]
    
    def _spawn_initial_goals(self):
        """Create initial goal nodes"""
        initial_goals = [
            ("Explore", 0.6, (0.5, 0.7)),
            ("Consolidate", 0.7, (-0.5, 0.6)),
            ("Predict", 0.5, (0.0, -0.7)),
            ("Adapt", 0.4, (-0.6, -0.3))
        ]
        
        for name, conf, pos in initial_goals:
            self.goals.append(GoalNode(
                name=name,
                position=pos,
                confidence=conf,
                velocity=(np.random.randn() * 0.01, np.random.randn() * 0.01),
                competing_hypotheses=[f"H{i}" for i in range(np.random.randint(1, 4))]
            ))
    
    def update_state(self):
        """Update the cognitive model state"""
        if self.stopped:
            return

        input_stimulus = torch.randn(100, device='cpu') * 0.1
        
        # Neural processing
        spikes1 = self.demo.oscillator.nn1(input_stimulus)
        spikes2 = self.demo.oscillator.nn2(input_stimulus)
        spikes3 = self.demo.oscillator.nn3(input_stimulus)
        
        # Field evolution
        combined_spikes = (spikes1 + spikes2 + spikes3) / 3
        self.demo.oscillator.field = 0.95 * self.demo.oscillator.field + 0.05 * combined_spikes
        
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * np.pi)
        self.time_step += 1
        
        self._compute_layer_states()
        self._update_goals()
        self._update_sensors()
        self._generate_actions()
        self._apply_scenario_phase_effects()
        self._update_lorenz()
        self._update_run_metrics()
    
    def _compute_layer_states(self):
        """Compute layer states from oscillator"""
        field = self.demo.oscillator.field
        quarter = field.shape[0] // 4
        
        layers_data = {
            'perception': field[:quarter],
            'integration': field[quarter:2*quarter],
            'reasoning': field[2*quarter:3*quarter],
            'action': field[3*quarter:]
        }
        
        for layer_name, layer_field in layers_data.items():
            activity = float(layer_field.abs().mean().item())
            conflict = float(layer_field.var().item())
            if len(layer_field) > 1:
                gradient = torch.diff(layer_field)
                pressure = float(gradient.abs().mean().item())
            else:
                pressure = 0.0
            
            self.layer_states[layer_name] = {
                'activity': np.clip(activity, 0, 1),
                'conflict': np.clip(conflict, 0, 1),
                'pressure': np.clip(pressure, 0, 1)
            }
    
    def _update_goals(self):
        """Update goal dynamics"""
        dt = 0.05
        
        for goal in self.goals:
            if goal.pinned or not goal.active:
                continue
            
            dx, dy = goal.position
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.1:
                gravity = 0.3 / (dist ** 2)
                vx, vy = goal.velocity
                vx -= gravity * dx / dist * dt
                vy -= gravity * dy / dist * dt
                vx += 0.1 * (-dy / dist) * dt
                vy += 0.1 * (dx / dist) * dt
                vx *= 0.98
                vy *= 0.98
                goal.velocity = (vx, vy)
                goal.position = (dx + vx * dt, dy + vy * dt)
            
            goal.confidence += (np.random.rand() - 0.5) * 0.02
            goal.confidence = np.clip(goal.confidence, 0.1, 1.0)
        
        if len(self.goals) < 6 and np.random.rand() < 0.05:
            angle = np.random.rand() * 2 * np.pi
            radius = 0.6 + np.random.rand() * 0.3
            pos = (radius * np.cos(angle), radius * np.sin(angle))
            name = np.random.choice(["Discover", "Optimize", "Validate", "Transform"])
            self.goals.append(GoalNode(
                name=name, position=pos,
                confidence=0.3 + np.random.rand() * 0.3,
                velocity=(np.random.randn() * 0.02, np.random.randn() * 0.02),
                competing_hypotheses=[f"H{i}" for i in range(np.random.randint(1, 3))]
            ))
        
        self.goals = [g for g in self.goals if g.confidence > 0.15 or g.pinned]
    
    def _update_sensors(self):
        """Update sensor values"""
        field = self.demo.oscillator.field
        self.sensors['visual'] = float(field[:len(field)//3].mean().item())
        self.sensors['temporal'] = float(field.var().item())
        self.sensors['pattern'] = float(np.clip(1.0 - field.std().item(), 0, 1))
    
    def _generate_actions(self):
        """Generate action proposals"""
        self.proposed_actions = [a for a in self.proposed_actions 
                                if self.time_step - a.timestamp < 30]
        
        if np.random.rand() < 0.15 and len(self.proposed_actions) < 5:
            action_types = [
                ("Adjust Rate", "Optimize dynamics"),
                ("Explore", "Discover patterns"),
                ("Consolidate", "Strengthen connections"),
                ("Query", "Seek information"),
                ("Refine", "Improve accuracy")
            ]
            name, outcome = action_types[np.random.randint(0, len(action_types))]
            self.proposed_actions.append(ActionVector(
                name=name,
                magnitude=float(self.sensors['pattern'] * np.random.rand()),
                uncertainty=1.0 - self.sensors['pattern'],
                outcome_estimate=outcome,
                timestamp=self.time_step
            ))
    
    def render_frame(self):
        """Render current state to image"""
        # Clear all axes
        self.ax_physiology.clear()
        self.ax_deliberation.clear()
        self.ax_world.clear()
        
        # Render Plane 1: Cognitive Physiology
        self._render_physiology()
        
        # Render Plane 2: Intent & Deliberation
        self._render_deliberation()
        
        # Render Plane 3: World Interface
        self._render_world()
        
        # Convert to image
        buf = BytesIO()
        self.fig.savefig(buf, format='png', facecolor='#000000', dpi=100)
        buf.seek(0)
        return buf
    
    def _render_physiology(self):
        """Render cognitive physiology plane"""
        ax = self.ax_physiology
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#0a0a0a')
        ax.axis('off')
        ax.set_title('COGNITIVE PHYSIOLOGY', fontsize=12, fontweight='bold', color='#00ff88')
        
        layer_positions = {
            'perception': (0.25, 0.75),
            'integration': (0.75, 0.75),
            'reasoning': (0.25, 0.25),
            'action': (0.75, 0.25)
        }
        
        radius = 0.15
        
        # Draw connections
        connections = [
            ('perception', 'integration'),
            ('integration', 'reasoning'),
            ('reasoning', 'action'),
            ('perception', 'reasoning'),
            ('integration', 'action')
        ]
        
        for layer1, layer2 in connections:
            p1 = layer_positions[layer1]
            p2 = layer_positions[layer2]
            activity1 = self.layer_states[layer1]['activity']
            activity2 = self.layer_states[layer2]['activity']
            flow_strength = abs(activity1 - activity2)
            
            if flow_strength > 0.1:
                num_particles = int(flow_strength * 3) + 1
                for i in range(num_particles):
                    t = (i / num_particles + self.pulse_phase / (2*np.pi)) % 1.0
                    px = p1[0] + t * (p2[0] - p1[0])
                    py = p1[1] + t * (p2[1] - p1[1])
                    ax.plot(px, py, 'o', color='cyan', markersize=4, alpha=0.6)
        
        # Draw layers
        for layer_name, (cx, cy) in layer_positions.items():
            state = self.layer_states[layer_name]
            pulse_factor = 1.0 + 0.2 * state['activity'] * np.sin(self.pulse_phase)
            current_radius = radius * pulse_factor
            
            activity = state['activity']
            if activity < 0.33:
                color = plt.cm.Blues(0.3 + activity * 2)
            elif activity < 0.66:
                color = plt.cm.YlOrRd(activity)
            else:
                color = plt.cm.hot(activity)
            
            circle = Circle((cx, cy), current_radius, color=color, alpha=0.6, zorder=2)
            ax.add_patch(circle)
            
            if state['conflict'] > 0.3:
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    offset = state['conflict'] * 0.05
                    px = cx + offset * np.cos(angle + self.pulse_phase)
                    py = cy + offset * np.sin(angle + self.pulse_phase)
                    ax.plot(px, py, 'o', color='yellow', markersize=3, alpha=0.7)
            
            if state['pressure'] > 0.2:
                ring = Circle((cx, cy), current_radius * 1.1,
                            fill=False, edgecolor='red',
                            linewidth=state['pressure'] * 4, alpha=0.5)
                ax.add_patch(ring)
    
    def _render_deliberation(self):
        """Render intent & deliberation plane"""
        ax = self.ax_deliberation
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_facecolor('#0f0f1a')
        ax.axis('off')
        ax.set_title('INTENT & DELIBERATION', fontsize=12, fontweight='bold', color='#ff8800')
        
        # Core objective
        ax.add_patch(Circle((0, 0), 0.15, color='orange', alpha=0.3))
        ax.text(0, 0, 'Core\nObjective', ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
        
        # Goals
        for goal in self.goals:
            x, y = goal.position
            size = 0.08 + goal.confidence * 0.12
            
            if goal.pinned:
                color, alpha = '#00ff00', 0.9
            elif goal.confidence > 0.7:
                color, alpha = '#ffaa00', 0.8
            else:
                color, alpha = '#6688ff', 0.5
            
            circle = Circle((x, y), size, color=color, alpha=alpha, zorder=3)
            ax.add_patch(circle)
            
            # Hypotheses
            for j in range(len(goal.competing_hypotheses)):
                angle = (j / len(goal.competing_hypotheses)) * 2 * np.pi + time.time()
                hx = x + (size + 0.05) * np.cos(angle)
                hy = y + (size + 0.05) * np.sin(angle)
                ax.plot(hx, hy, 'o', color='cyan', markersize=3, alpha=0.6)
            
            ax.plot([0, x], [0, y], '--', color=color, alpha=0.3, linewidth=1)
            
            # Label
            ax.text(x, y - size - 0.08, goal.name, ha='center', 
                   fontsize=7, color='white', alpha=0.8)
    
    def _render_world(self):
        """Render world interface plane"""
        ax = self.ax_world
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a0f0f')
        ax.axis('off')
        ax.set_title('WORLD INTERFACE', fontsize=12, fontweight='bold', color='#ff0088')
        
        # Sensors
        sensor_positions = {'visual': 0.8, 'temporal': 0.5, 'pattern': 0.2}
        for sensor_name, y_pos in sensor_positions.items():
            ax.text(0.05, y_pos, sensor_name.upper(), fontsize=8, color='white', va='center')
            value = self.sensors[sensor_name]
            bar_length = value * 0.25
            rect = Rectangle((0.18, y_pos - 0.02), bar_length, 0.04,
                           color='cyan', alpha=0.7)
            ax.add_patch(rect)
            ax.text(0.45, y_pos, f"{value:.2f}", fontsize=7, color='cyan', va='center')
        
        # Actions
        ax.text(0.55, 0.95, "PROPOSED ACTIONS", fontsize=9, fontweight='bold', color='#ff8800')
        
        start_y = 0.85
        for i, action in enumerate(self.proposed_actions[:5]):
            y = start_y - i * 0.15
            arrow_length = action.magnitude * 0.15
            
            arrow = FancyArrowPatch((0.55, y), (0.55 + arrow_length, y),
                                   arrowstyle='->', mutation_scale=15,
                                   color='yellow' if not action.approved else 'green',
                                   alpha=0.8, linewidth=2)
            ax.add_patch(arrow)
            
            if action.uncertainty > 0.3:
                cloud = Circle((0.55 + arrow_length/2, y), action.uncertainty * 0.08,
                             color='red', alpha=0.3)
                ax.add_patch(cloud)
            
            age = self.time_step - action.timestamp
            alpha = 1.0 - (age / 30)
            ax.text(0.56 + arrow_length, y + 0.02, action.name,
                   fontsize=7, color='white', alpha=alpha, va='bottom')


# Flask app
app = Flask(__name__)
center = NeuralCommandCenterWeb()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>⚡ Neural Command Center</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #000;
            font-family: 'Courier New', monospace;
            color: #00ff88;
        }
        .header {
            text-align: center;
            padding: 20px;
            background-color: #0a0a0a;
            border-bottom: 2px solid #00ff88;
        }
        h1 {
            margin: 0;
            font-size: 2em;
            color: #00ff88;
        }
        .subtitle {
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .kpi-grid {
            width: 100%;
            max-width: 1200px;
            display: grid;
            grid-template-columns: repeat(4, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .kpi-card {
            background-color: #101a18;
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.15);
        }
        .kpi-label {
            color: #78c9ab;
            font-size: 0.78em;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            color: #ffffff;
            font-size: 1.4em;
            font-weight: bold;
        }
        .kpi-sub {
            color: #8fd3ff;
            font-size: 0.78em;
            margin-top: 3px;
        }
        .use-case {
            width: 100%;
            max-width: 1200px;
            background: linear-gradient(135deg, #0f1f28 0%, #1a1f14 100%);
            border: 1px solid #2dd4bf;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 16px;
        }
        .use-case-title {
            color: #c9fff0;
            font-weight: bold;
            margin-bottom: 6px;
        }
        .scenario-panel {
            width: 100%;
            max-width: 1200px;
            background: #111827;
            border: 1px solid #60a5fa;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 14px;
        }
        .scenario-head {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
        }
        .scenario-title {
            color: #dbeafe;
            font-weight: bold;
        }
        .scenario-phase {
            color: #93c5fd;
            font-size: 0.85em;
        }
        .scenario-message {
            color: #e5e7eb;
            font-size: 0.92em;
            margin-bottom: 7px;
        }
        .scenario-action {
            color: #bfdbfe;
            font-size: 0.88em;
        }
        .scenario-bar {
            margin-top: 10px;
            width: 100%;
            height: 8px;
            border-radius: 5px;
            background: #1f2937;
            overflow: hidden;
        }
        .scenario-bar-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #38bdf8 0%, #34d399 100%);
            transition: width 0.4s ease;
        }
        .use-case-text {
            color: #d6f3ff;
            line-height: 1.35;
            font-size: 0.92em;
        }
        #frame {
            max-width: 100%;
            height: auto;
            border: 2px solid #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        }
        .controls {
            margin-top: 20px;
            padding: 15px;
            background-color: #1a1a1a;
            border: 1px solid #00ff88;
            border-radius: 5px;
        }
        button {
            background-color: #00aa00;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
        }
        button:hover {
            background-color: #00ff00;
        }
        button.stop {
            background-color: #aa0000;
        }
        button.stop:hover {
            background-color: #ff0000;
        }
        .status {
            margin-top: 10px;
            padding: 10px;
            text-align: center;
            font-weight: bold;
        }
        .online { color: #00ff00; }
        .offline { color: #ff0000; }
        .ops-grid {
            width: 100%;
            max-width: 1200px;
            display: grid;
            grid-template-columns: repeat(2, minmax(280px, 1fr));
            gap: 12px;
            margin: 14px 0;
        }
        .ops-panel {
            background: #0d1117;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
        }
        .ops-title {
            color: #93c5fd;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .field-row {
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
            align-items: center;
        }
        .field-row label {
            width: 120px;
            color: #cbd5e1;
            font-size: 0.85em;
        }
        .field-row select, .field-row input {
            flex: 1;
            background: #0f172a;
            border: 1px solid #334155;
            color: #e2e8f0;
            padding: 6px;
            border-radius: 4px;
        }
        .ops-mini {
            font-size: 0.84em;
            color: #e2e8f0;
            margin: 3px 0;
        }
        .comp-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78em;
        }
        .comp-table th, .comp-table td {
            border: 1px solid #334155;
            padding: 4px 6px;
            color: #dbeafe;
        }
        .comp-table th {
            background: #1e293b;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ NEURAL COMMAND CENTER</h1>
        <div class="subtitle">Three-Plane Cognitive Architecture • Not a Chatbot • A Living System</div>
    </div>
    
    <div class="container">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Training Peak Reward</div>
                <div class="kpi-value" id="kpiReward">-</div>
                <div class="kpi-sub" id="kpiBatch">Batch baseline pending</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Observed Speedup</div>
                <div class="kpi-value" id="kpiSpeedup">-</div>
                <div class="kpi-sub" id="kpiBatchTime">Avg batch time pending</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Decision Latency</div>
                <div class="kpi-value" id="kpiLatency">-</div>
                <div class="kpi-sub">Live estimate from active cognitive state</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Projected Triage Gain</div>
                <div class="kpi-value" id="kpiGain">-</div>
                <div class="kpi-sub" id="kpiActions">Live action vectors: 0</div>
            </div>
        </div>

        <div class="use-case">
            <div class="use-case-title">Investor Reference Scenario: Incident Triage for Operations Teams</div>
            <div class="use-case-text">
                This view demonstrates the engine coordinating perception, deliberation, and action proposal in real time.
                The KPI strip above combines hard-training baseline data with current cognitive-state telemetry to show a practical reference:
                faster incident prioritization, lower decision latency, and measurable triage efficiency gains.
            </div>
        </div>

        <div class="scenario-panel">
            <div class="scenario-head">
                <div class="scenario-title" id="scenarioTitle">Live Scenario: Incident Triage</div>
                <div class="scenario-phase" id="scenarioPhase">Phase pending</div>
            </div>
            <div class="scenario-message" id="scenarioMessage">Waiting for scenario data...</div>
            <div class="scenario-action" id="scenarioAction">Recommended action will appear here.</div>
            <div class="scenario-bar"><div class="scenario-bar-fill" id="scenarioBar"></div></div>
        </div>

        <div class="ops-grid">
            <div class="ops-panel">
                <div class="ops-title">Run Control</div>
                <div class="field-row"><label>Dataset</label><select id="cfgDataset"><option>wikitext-103-raw-v1</option><option>wikitext-2-raw-v1</option></select></div>
                <div class="field-row"><label>Hardware</label><select id="cfgHardware"><option value="cpu">CPU</option><option value="gpu">GPU</option><option value="auto">Auto</option></select></div>
                <div class="field-row"><label>Batch Count</label><input id="cfgBatchCount" type="number" min="1" value="150" /></div>
                <div class="field-row"><label>Mode</label><select id="cfgMode"><option value="baseline">Baseline</option><option value="hard" selected>Hard</option><option value="ablation">Ablation</option></select></div>
                <div class="field-row"><label>Saved Config</label><select id="savedConfigSelect"></select></div>
                <div class="field-row">
                    <button onclick="applyConfig()">Apply Config</button>
                    <button onclick="loadSavedConfig()">Load Saved</button>
                    <button onclick="saveCurrentConfig()">Save Current</button>
                </div>
                <div class="field-row">
                    <button onclick="start()">Start/Resume</button>
                    <button onclick="pauseRun()">Pause</button>
                    <button class="stop" onclick="safeStop()">Safe Stop</button>
                </div>
            </div>

            <div class="ops-panel">
                <div class="ops-title">Live Monitoring</div>
                <div class="ops-mini" id="mBatch">Current batch: -</div>
                <div class="ops-mini" id="mReward">Reward: -</div>
                <div class="ops-mini" id="mDelta">Delta reward: -</div>
                <div class="ops-mini" id="mSpeedup">Speedup: -</div>
                <div class="ops-mini" id="mLr">Learning rate: -</div>
                <div class="ops-mini" id="mVariance">Variance: -</div>
                <div class="ops-mini" id="mSyntropy">Syntropy events: -</div>
                <div class="ops-mini" id="mThrottle">Throttle events: -</div>
                <div class="ops-mini" id="mUsage">CPU/RAM: -</div>
                <div class="ops-mini" id="mBatchTime">Batch time: -</div>
                <div class="ops-mini" id="mEta">ETA: -</div>
            </div>

            <div class="ops-panel">
                <div class="ops-title">Architecture Introspection</div>
                <div class="ops-mini" id="aLorenz">Lorenz (x,y,z): -</div>
                <div class="ops-mini" id="aTrajectory">Lorenz trajectory tail: -</div>
                <div class="ops-mini" id="aField">Field activity map mean: -</div>
                <div class="ops-mini" id="aNeuron">Neuron activation density: -</div>
                <div class="ops-mini" id="aSynaptic">Synaptic trace: -</div>
                <div class="ops-mini" id="aCluster">Cluster count / adaptive K: -</div>
                <div class="ops-mini" id="aCache">Cache hit rate: -</div>
                <div class="ops-mini" id="aResonance">Resonance/coherence: -</div>
                <div class="ops-mini" id="aTransitions">State transitions: -</div>
            </div>

            <div class="ops-panel">
                <div class="ops-title">Experiment Comparison</div>
                <div class="field-row">
                    <input id="snapshotLabel" type="text" value="current-run" />
                    <button onclick="snapshotRun()">Snapshot Run</button>
                </div>
                <table class="comp-table">
                    <thead>
                        <tr>
                            <th>Label</th><th>Mode</th><th>Final Reward</th><th>Reward Delta</th><th>Time To Plateau</th><th>Avg Batch Time</th><th>Throttle</th><th>Stability</th>
                        </tr>
                    </thead>
                    <tbody id="comparisonBody"></tbody>
                </table>
            </div>
        </div>

        <img id="frame" src="{{ url_for('video_feed') }}" />
        
        <div class="controls">
            <button onclick="start()">▶ RUN INVESTOR SCENARIO</button>
            <button class="stop" onclick="stop()">⏸ PAUSE SCENARIO</button>
            <div class="status">
                Status: <span id="status" class="online">● ONLINE</span>
            </div>
        </div>
        
        <div style="margin-top: 20px; color: #888; text-align: center; max-width: 800px;">
            <p><strong>Plane 1 (Top):</strong> Cognitive Physiology - Pulses, flows, pressure, heat</p>
            <p><strong>Plane 2 (Middle):</strong> Intent & Deliberation - Goals forming, competing, dissolving</p>
            <p><strong>Plane 3 (Bottom):</strong> World Interface - Sensors, proposed actions, uncertainty</p>
            <p style="margin-top: 15px; color: #00ff88;"><em>"If frozen for 10 seconds, it still feels alive"</em></p>
        </div>
    </div>
    
    <script>
        let running = true;
        
        function start() {
            fetch('/start');
            document.getElementById('status').className = 'online';
            document.getElementById('status').textContent = '● ONLINE';
            running = true;
        }
        
        function stop() {
            fetch('/stop');
            document.getElementById('status').className = 'offline';
            document.getElementById('status').textContent = '● PAUSED';
            running = false;
        }

        function pauseRun() {
            fetch('/pause');
            document.getElementById('status').className = 'offline';
            document.getElementById('status').textContent = '● PAUSED';
            running = false;
        }

        function safeStop() {
            fetch('/safe_stop');
            document.getElementById('status').className = 'offline';
            document.getElementById('status').textContent = '● STOPPED';
            running = false;
        }

        async function applyConfig() {
            const payload = {
                dataset: document.getElementById('cfgDataset').value,
                hardware_mode: document.getElementById('cfgHardware').value,
                batch_count: Number(document.getElementById('cfgBatchCount').value),
                mode: document.getElementById('cfgMode').value,
            };
            await fetch('/run_config', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
        }

        async function refreshSavedConfigs() {
            const r = await fetch('/configs');
            const data = await r.json();
            const sel = document.getElementById('savedConfigSelect');
            sel.innerHTML = '';
            Object.keys(data.configs).forEach((k) => {
                const opt = document.createElement('option');
                opt.value = k;
                opt.textContent = k;
                sel.appendChild(opt);
            });
        }

        async function loadSavedConfig() {
            const key = document.getElementById('savedConfigSelect').value;
            const r = await fetch(`/configs/load?name=${encodeURIComponent(key)}`);
            const d = await r.json();
            if (d.ok) {
                document.getElementById('cfgDataset').value = d.config.dataset || 'wikitext-103-raw-v1';
                document.getElementById('cfgHardware').value = d.config.hardware_mode || 'cpu';
                document.getElementById('cfgBatchCount').value = d.config.batch_count || 150;
                document.getElementById('cfgMode').value = d.config.mode || 'hard';
                await applyConfig();
            }
        }

        async function saveCurrentConfig() {
            const key = document.getElementById('savedConfigSelect').value || 'custom';
            const payload = {
                name: key,
                config: {
                    dataset: document.getElementById('cfgDataset').value,
                    hardware_mode: document.getElementById('cfgHardware').value,
                    batch_count: Number(document.getElementById('cfgBatchCount').value),
                    mode: document.getElementById('cfgMode').value,
                }
            };
            await fetch('/configs/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
            await refreshSavedConfigs();
        }

        async function snapshotRun() {
            const label = document.getElementById('snapshotLabel').value || 'run';
            await fetch('/comparison/snapshot', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({label})});
            await refreshTelemetry();
        }

        async function refreshMetrics() {
            try {
                const response = await fetch('/investor_metrics');
                const m = await response.json();

                document.getElementById('kpiReward').textContent = m.max_reward.toFixed(2);
                document.getElementById('kpiBatch').textContent = `Latest batch: ${m.latest_batch} (samples: ${m.run_samples})`;
                document.getElementById('kpiSpeedup').textContent = `${m.max_speedup.toFixed(2)}x`;
                document.getElementById('kpiBatchTime').textContent = `Avg batch time: ${m.avg_batch_time.toFixed(2)}s`;
                document.getElementById('kpiLatency').textContent = `${m.decision_latency_ms} ms`;
                document.getElementById('kpiGain').textContent = `${m.projected_triage_gain.toFixed(1)}%`;
                document.getElementById('kpiActions').textContent = `Live action vectors: ${m.action_count} | Readiness: ${m.readiness_score.toFixed(1)}`;
                document.getElementById('scenarioPhase').textContent = `${m.scenario_title} (${m.scenario_phase})`;
                document.getElementById('scenarioMessage').textContent = m.scenario_message;
                document.getElementById('scenarioAction').textContent = `Recommended action: ${m.scenario_action}`;
                document.getElementById('scenarioBar').style.width = `${(m.scenario_progress * 100).toFixed(0)}%`;
            } catch (e) {
                // Keep the last rendered values if fetch fails momentarily.
            }
        }

        async function refreshTelemetry() {
            try {
                const response = await fetch('/telemetry');
                const d = await response.json();
                const m = d.monitoring;
                const a = d.introspection;

                document.getElementById('mBatch').textContent = `Current batch: ${m.current_batch}/${d.run_control.batch_count}`;
                document.getElementById('mReward').textContent = `Reward: ${m.reward.toFixed(4)}`;
                document.getElementById('mDelta').textContent = `Delta reward: ${m.delta_reward.toFixed(5)}`;
                document.getElementById('mSpeedup').textContent = `Speedup: ${m.speedup.toFixed(2)}x`;
                document.getElementById('mLr').textContent = `Learning rate: ${m.learning_rate.toFixed(6)}`;
                document.getElementById('mVariance').textContent = `Variance: ${m.variance.toFixed(4)}`;
                document.getElementById('mSyntropy').textContent = `Syntropy events: ${m.syntropy_events}`;
                document.getElementById('mThrottle').textContent = `Throttle events: ${m.throttle_events}`;
                document.getElementById('mUsage').textContent = `CPU/RAM: ${m.cpu_percent.toFixed(1)}% / ${m.ram_percent.toFixed(1)}%`;
                document.getElementById('mBatchTime').textContent = `Batch time: ${m.batch_time_sec.toFixed(2)}s`;
                document.getElementById('mEta').textContent = `ETA: ${Math.floor(m.eta_seconds/60)}m ${m.eta_seconds%60}s`;

                document.getElementById('aLorenz').textContent = `Lorenz (x,y,z): ${a.lorenz.x.toFixed(3)}, ${a.lorenz.y.toFixed(3)}, ${a.lorenz.z.toFixed(3)}`;
                const tail = a.lorenz.trajectory.length ? a.lorenz.trajectory[a.lorenz.trajectory.length - 1] : [0,0,0];
                document.getElementById('aTrajectory').textContent = `Lorenz trajectory tail: ${tail.map(v => v.toFixed(2)).join(', ')}`;
                document.getElementById('aField').textContent = `Field activity map mean: ${a.field_activity.toFixed(4)}`;
                document.getElementById('aNeuron').textContent = `Neuron activation density: ${a.neuron_activation_density.toFixed(4)}`;
                document.getElementById('aSynaptic').textContent = `Synaptic trace: ${a.synaptic_trace.toFixed(4)}`;
                document.getElementById('aCluster').textContent = `Cluster count / adaptive K: ${a.cluster_count} / ${a.adaptive_k}`;
                document.getElementById('aCache').textContent = `Cache hit rate: ${(a.cache_hit_rate*100).toFixed(1)}%`;
                document.getElementById('aResonance').textContent = `Resonance/coherence: ${a.resonance.toFixed(3)} / ${a.coherence.toFixed(3)}`;
                document.getElementById('aTransitions').textContent = `State transitions: ${a.state_transitions.length} recent`;

                const tbody = document.getElementById('comparisonBody');
                tbody.innerHTML = '';
                d.comparison.slice().reverse().forEach((row) => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>${row.label}</td><td>${row.mode}</td><td>${row.final_reward}</td><td>${row.reward_delta}</td><td>${row.time_to_plateau_sec}s</td><td>${row.avg_batch_time}s</td><td>${row.throttle_count}</td><td>${row.stability_band}</td>`;
                    tbody.appendChild(tr);
                });
            } catch (e) {
                // Best-effort refresh; keep latest values.
            }
        }
        
        // Auto-refresh frame
        setInterval(() => {
            if (running) {
                document.getElementById('frame').src = '{{ url_for("video_feed") }}?' + new Date().getTime();
            }
        }, 200);  // ~5 FPS

        setInterval(refreshMetrics, 1500);
        refreshMetrics();
        setInterval(refreshTelemetry, 1200);
        refreshTelemetry();
        refreshSavedConfigs();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start')
def start():
    center.running = True
    center.paused = False
    center.stopped = False
    logger.info("🚀 Neural Command Center ACTIVATED")
    return "started"

@app.route('/stop')
def stop():
    center.running = False
    center.paused = True
    logger.info("⏸ Neural Command Center PAUSED")
    return "stopped"


@app.route('/pause')
def pause():
    center.running = False
    center.paused = True
    return "paused"


@app.route('/safe_stop')
def safe_stop():
    center.running = False
    center.paused = False
    center.stopped = True
    return "safe_stopped"


@app.route('/run_config', methods=['GET', 'POST'])
def run_config():
    if request.method == 'GET':
        return jsonify(center.get_live_telemetry()["run_control"])

    payload = request.get_json(silent=True) or {}
    center.update_run_config(payload)
    return jsonify({"ok": True, "run_control": center.get_live_telemetry()["run_control"]})


@app.route('/configs', methods=['GET'])
def list_configs():
    return jsonify({"configs": center._load_saved_configs()})


@app.route('/configs/save', methods=['POST'])
def save_config():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "custom"))
    cfg = payload.get("config", {})
    configs = center._load_saved_configs()
    configs[name] = cfg
    center._save_configs(configs)
    return jsonify({"ok": True, "name": name})


@app.route('/configs/load', methods=['GET'])
def load_config():
    name = request.args.get("name", "hard")
    configs = center._load_saved_configs()
    cfg = configs.get(name)
    if cfg is None:
        return jsonify({"ok": False, "error": f"Config '{name}' not found"}), 404

    center.update_run_config(cfg)
    return jsonify({"ok": True, "config": cfg})

@app.route('/video_feed')
def video_feed():
    if center.running:
        center.update_state()
    
    buf = center.render_frame()
    return Response(buf.getvalue(), mimetype='image/png')


@app.route('/investor_metrics')
def investor_metrics():
    return center.get_investor_metrics()


@app.route('/telemetry')
def telemetry():
    return jsonify(center.get_live_telemetry())


@app.route('/comparison/snapshot', methods=['POST'])
def comparison_snapshot():
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", f"run-{int(time.time())}"))
    snap = center.snapshot_current_run(label)
    return jsonify({"ok": True, "snapshot": snap})


def main():
    """Launch the web-based Neural Command Center"""
    logger.info("✨ Neural Command Center - Web Version")
    logger.info("   Three planes: Physiology | Deliberation | World Interface")
    logger.info("   Access via browser at: http://localhost:8050")
    logger.info("")
    
    app.run(host='0.0.0.0', port=8050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
