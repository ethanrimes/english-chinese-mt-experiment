"""Tests for src/ecmt/training/timings.py — registry + ETA extrapolation."""

from __future__ import annotations

import json
from pathlib import Path

from ecmt.training import timings as T


def _write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def test_estimate_no_history(tmp_path, monkeypatch):
    fake = tmp_path / "timings.jsonl"
    monkeypatch.setattr(T, "REGISTRY_PATH", fake)
    out = T.estimate_duration_seconds(pairs=10_000, epochs=10, effective_batch_size=64)
    assert out["seconds"] is None
    assert out["expected_steps"] > 0


def test_estimate_with_history(tmp_path, monkeypatch):
    fake = tmp_path / "timings.jsonl"
    monkeypatch.setattr(T, "REGISTRY_PATH", fake)
    _write_events(fake, [
        {"event": "run_start", "run_id": "r1", "hardware": "A100-80GB", "scale": 10000},
        {"event": "step_progress", "run_id": "r1", "step": 1000, "elapsed_seconds": 300.0, "steps_per_sec": 3.33, "scale": 10000},
        {"event": "run_end", "run_id": "r1", "elapsed_seconds": 930.0, "n_steps_completed": 3100, "steps_per_sec_final": 3.33, "status": "success", "scale": 10000},
        {"event": "run_start", "run_id": "r2", "hardware": "A100-80GB", "scale": 50000},
        {"event": "run_end", "run_id": "r2", "elapsed_seconds": 2820.0, "n_steps_completed": 9400, "steps_per_sec_final": 3.33, "status": "success", "scale": 50000},
    ])
    out = T.estimate_duration_seconds(pairs=100_000, epochs=5, effective_batch_size=64)
    assert out["seconds"] is not None
    assert out["sps_used"] > 0
    assert out["expected_steps"] == (100_000 * 2 * 5) // 64


def test_estimate_override(tmp_path, monkeypatch):
    fake = tmp_path / "timings.jsonl"
    monkeypatch.setattr(T, "REGISTRY_PATH", fake)
    out = T.estimate_duration_seconds(pairs=1_000_000, epochs=2, effective_batch_size=64, sps_override=3.0)
    assert out["sps_source"] == "override"
    assert out["seconds"] == round((1_000_000 * 2 * 2) / 64 / 3.0, 1)
