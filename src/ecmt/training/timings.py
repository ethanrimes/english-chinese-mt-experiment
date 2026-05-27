"""Append-only training-time registry + ETA extrapolator.

Every training run appends events to `models/runs/timings.jsonl`:

  - run_start    — scale, epochs, expected steps, hardware, git sha
  - step_progress — step, elapsed_seconds, steps_per_sec (every N steps)
  - run_end      — final elapsed, n_steps_completed, status

The estimator reads this log, computes median steps/sec across past runs
(grouped by hardware where possible), and predicts wallclock for a requested
scale at a requested epoch count. This is the safety rail that lets you say:
"the 1M run will take ~6 hours based on past experience — worth it?"
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from loguru import logger

REGISTRY_PATH = Path("models/runs/timings.jsonl")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _hardware_label() -> str:
    """Best-effort one-line GPU label, e.g. 'A100-80GB-SXM4'."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace(" ", "-")
    except Exception:
        pass
    return platform.machine() or "unknown"


def _append(event: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class RunTimer:
    """Use as: with RunTimer(run_id=..., scale=...) as t: ... t.log_step(step, ...)"""

    run_id: str
    scale: int
    n_examples: int
    epochs: int
    expected_steps: int
    effective_batch_size: int
    extra: dict[str, Any] = field(default_factory=dict)
    _start_wall: float = 0.0
    _last_step_logged: int = 0

    def __enter__(self) -> "RunTimer":
        self._start_wall = time.monotonic()
        _append({
            "event": "run_start",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "scale": self.scale,
            "n_examples": self.n_examples,
            "epochs": self.epochs,
            "expected_steps": self.expected_steps,
            "effective_batch_size": self.effective_batch_size,
            "hardware": _hardware_label(),
            "git_sha": _git_sha(),
            **self.extra,
        })
        return self

    def log_step(self, step: int) -> None:
        elapsed = time.monotonic() - self._start_wall
        sps = step / elapsed if elapsed > 0 else 0.0
        _append({
            "event": "step_progress",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "step": step,
            "elapsed_seconds": round(elapsed, 2),
            "steps_per_sec": round(sps, 4),
            "scale": self.scale,
            "hardware": _hardware_label(),
        })
        self._last_step_logged = step

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.monotonic() - self._start_wall
        status = "failed" if exc_type is not None else "success"
        sps = self._last_step_logged / elapsed if elapsed > 0 else 0.0
        _append({
            "event": "run_end",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "elapsed_seconds": round(elapsed, 2),
            "n_steps_completed": self._last_step_logged,
            "steps_per_sec_final": round(sps, 4),
            "status": status,
            "scale": self.scale,
            "hardware": _hardware_label(),
            "exc_type": exc_type.__name__ if exc_type else None,
        })
        return False  # don't swallow exceptions


def load_events(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def median_steps_per_sec(
    events: list[dict[str, Any]] | None = None,
    *,
    hardware: str | None = None,
    min_steps_logged: int = 100,
) -> float | None:
    """Median observed steps/sec across past runs.

    Uses each run's *final* steps_per_sec_final from run_end if available,
    else the last step_progress event with step >= min_steps_logged.
    """
    if events is None:
        events = load_events()
    # bucket by run
    runs: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        rid = e.get("run_id")
        if not rid:
            continue
        runs.setdefault(rid, []).append(e)

    sps_values: list[float] = []
    for rid, evts in runs.items():
        hw = None
        for e in evts:
            hw = e.get("hardware") or hw
        if hardware and hw != hardware:
            continue
        # prefer run_end
        end = next((e for e in evts if e.get("event") == "run_end"), None)
        if end is not None and end.get("steps_per_sec_final"):
            sps_values.append(float(end["steps_per_sec_final"]))
            continue
        # fall back to last step_progress with enough steps
        progs = [e for e in evts if e.get("event") == "step_progress" and e.get("step", 0) >= min_steps_logged]
        if progs:
            sps_values.append(float(progs[-1].get("steps_per_sec") or 0.0))

    sps_values = [v for v in sps_values if v > 0]
    if not sps_values:
        return None
    return median(sps_values)


def estimate_duration_seconds(
    pairs: int,
    epochs: int,
    *,
    effective_batch_size: int = 64,
    bidirectional: bool = True,
    sps_override: float | None = None,
    hardware: str | None = None,
) -> dict[str, Any]:
    """Predict wallclock seconds for a scale, given history.

    Returns {expected_steps, sps_used, sps_source, seconds, hours, confidence}.
    """
    mult = 2 if bidirectional else 1
    examples = pairs * mult * epochs
    steps = max(1, examples // effective_batch_size)
    if sps_override is not None:
        sps = sps_override
        src = "override"
    else:
        sps = median_steps_per_sec(hardware=hardware)
        src = "history-median" if sps else "none"
    if sps is None or sps <= 0:
        return {
            "expected_steps": steps,
            "sps_used": None,
            "sps_source": src,
            "seconds": None,
            "hours": None,
            "confidence": "none",
            "note": "no prior runs — set --sps-override or run a small scale first to seed the estimator",
        }
    seconds = steps / sps
    events = load_events()
    n_completed = sum(1 for e in events if e.get("event") == "run_end" and e.get("status") == "success")
    conf = "low" if n_completed < 2 else ("medium" if n_completed < 5 else "high")
    return {
        "expected_steps": steps,
        "sps_used": round(sps, 3),
        "sps_source": src,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600.0, 2),
        "confidence": conf,
        "n_prior_completed_runs": n_completed,
    }
