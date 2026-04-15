"""Focused tests for HRM collapse detection and rebinding."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_hrm_module():
    """Load HRM engine directly to avoid package-level side effects in quadra.__init__."""
    engine_path = Path(__file__).resolve().parents[1] / "quadra" / "core" / "hrm" / "engine.py"
    spec = importlib.util.spec_from_file_location("hrm_engine_test_module", engine_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collapse_trigger_sets_flag_grows_memory_and_rebinds(monkeypatch):
    """Force E(t) >= E_crit and verify collapse state + rebinding behavior."""
    hrm_module = _load_hrm_module()
    engine = hrm_module.SimulationEngine(field_shape=(6, 6), seed=11)
    state = engine.initialize()

    # Set a stable anchor and start far from it so rebinding has measurable effect.
    state.attractor_anchor = np.zeros_like(state.state_vector)
    state.state_vector = np.ones_like(state.state_vector) * 3.0

    # Keep the update path deterministic so rebinding dominates the state change.
    monkeypatch.setattr(engine.operator_engine, "apply", lambda s: s)
    monkeypatch.setattr(engine.field_engine, "solve", lambda *args, **kwargs: state.field)
    monkeypatch.setattr(engine.state_engine, "update", lambda *args, **kwargs: state.state_vector)

    # Force collapse by construction: E(t) > E_crit every step.
    monkeypatch.setattr(
        engine,
        "_collapse_energy",
        lambda *_args, **_kwargs: {
            "collapse_energy": engine.E_crit + 0.5,
            "grad_term": 0.6,
            "velocity_term": 0.6,
            "coherence_loss": 0.3,
        },
    )

    pre_distance = float(np.linalg.norm(state.state_vector - state.attractor_anchor))
    pre_memory_size = len(state.collapse_memory)

    state, metrics = engine.step(state, dt=0.02)

    post_distance = float(np.linalg.norm(state.state_vector - state.attractor_anchor))
    assert metrics["collapse_energy"] >= metrics["collapse_threshold"]
    assert state.collapse_flag is True
    assert len(state.collapse_memory) == pre_memory_size + 1
    assert post_distance < pre_distance

    # Snapshot is recorded before rebinding, so it should differ from final state.
    first_snapshot = state.collapse_memory[-1]["state_snapshot"]
    assert not np.allclose(first_snapshot, state.state_vector)

    # A second forced collapse should grow memory again.
    state, _ = engine.step(state, dt=0.02)
    assert len(state.collapse_memory) == pre_memory_size + 2
