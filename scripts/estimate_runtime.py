"""Estimate training wallclock for one or more data scales.

Reads `models/runs/timings.jsonl` and extrapolates from past completed runs.
With no priors, requires --sps-override.

Usage:
    python scripts/estimate_runtime.py                          # all sweep scales
    python scripts/estimate_runtime.py --scales 10000,50000     # specific scales
    python scripts/estimate_runtime.py --sps-override 3.0       # assume 3 step/sec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.training.timings import estimate_duration_seconds, load_events  # noqa: E402
from ecmt.utils.config import load_config  # noqa: E402

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _format_hours(h: float | None) -> str:
    if h is None:
        return "—"
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 10:
        return f"{h:.1f} h"
    return f"{h:.0f} h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-config", default="configs/sweep_data_scale.yaml")
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument(
        "--scales",
        default=None,
        help="comma-separated scales to estimate (default: all from sweep config)",
    )
    ap.add_argument("--sps-override", type=float, default=None, help="assume this many steps/sec")
    ap.add_argument("--hardware", default=None, help="filter prior runs to this GPU label")
    ap.add_argument("--effective-batch-size", type=int, default=None, help="override effective batch")
    args = ap.parse_args()

    sweep = load_config(args.sweep_config)
    tcfg = load_config(args.training_config)
    bsz = args.effective_batch_size or (
        int(tcfg.trainer.per_device_train_batch_size) * int(tcfg.trainer.gradient_accumulation_steps)
    )

    if args.scales:
        wanted = [int(x) for x in args.scales.split(",")]
        # synthesize a run record for each
        records = []
        for n in wanted:
            sweep_entry = next((r for r in sweep.runs if int(r.scale) == n), None)
            epochs = int(sweep_entry.overrides["trainer.num_train_epochs"]) if sweep_entry else int(tcfg.trainer.num_train_epochs)
            records.append({"name": f"scale-{n}", "scale": n, "epochs": epochs})
    else:
        records = []
        for r in sweep.runs:
            epochs = int(r.overrides["trainer.num_train_epochs"]) if "overrides" in r else int(tcfg.trainer.num_train_epochs)
            records.append({"name": str(r.name), "scale": int(r.scale), "epochs": epochs})

    rows = []
    cumulative_seconds = 0.0
    for rec in records:
        est = estimate_duration_seconds(
            pairs=rec["scale"],
            epochs=rec["epochs"],
            effective_batch_size=bsz,
            sps_override=args.sps_override,
            hardware=args.hardware,
        )
        if est.get("seconds") is not None:
            cumulative_seconds += est["seconds"]
        rows.append({**rec, **est, "cumulative_hours": round(cumulative_seconds / 3600.0, 2) if est.get("seconds") else None})

    events = load_events()
    n_prior = sum(1 for e in events if e.get("event") == "run_end")

    if _RICH:
        console = Console()
        tbl = Table(title=f"Runtime estimates (prior completed runs: {n_prior})", show_lines=False)
        tbl.add_column("Run")
        tbl.add_column("Pairs", justify="right")
        tbl.add_column("Epochs", justify="right")
        tbl.add_column("Steps", justify="right")
        tbl.add_column("sps used", justify="right")
        tbl.add_column("ETA", justify="right")
        tbl.add_column("Cumulative", justify="right")
        tbl.add_column("Confidence")
        for r in rows:
            tbl.add_row(
                r["name"],
                f"{r['scale']:,}",
                str(r["epochs"]),
                f"{r['expected_steps']:,}",
                f"{r['sps_used']}" if r["sps_used"] else "—",
                _format_hours(r.get("hours")),
                _format_hours(r.get("cumulative_hours")),
                r.get("confidence", "—"),
            )
        console.print(tbl)
        if est.get("sps_source") == "none":
            console.print(
                "[yellow]No prior runs found.[/yellow] Pass --sps-override (e.g. --sps-override 3.0 "
                "for a single A100 with a 360M model) or run the smallest scale first to seed the estimator."
            )
    else:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
