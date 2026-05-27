"""Train one or more SmolLM2-360M-ext models on the data-scale subsets.

You can train at a single scale or queue several scales sequentially. Before
each run, the ETA is printed based on prior runs in models/runs/timings.jsonl
so you can decide whether to bail.

Examples:
    # one run
    python scripts/06_train.py --scale 10000

    # several runs, in order
    python scripts/06_train.py --scales 10000,50000,100000

    # all sweep scales
    python scripts/06_train.py --all-sweep-scales

    # resume a specific checkpoint
    python scripts/06_train.py --scale 50000 --resume-from models/runs/scale-50000-XXX/checkpoint-1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.training.timings import estimate_duration_seconds  # noqa: E402
from ecmt.training.trainer import build_and_train  # noqa: E402
from ecmt.utils.config import apply_overrides, load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402


def _resolve_scales(args: argparse.Namespace, sweep_path: str) -> list[dict]:
    """Return a list of {scale, overrides} dicts to run, in order."""
    sweep = load_config(sweep_path)
    sweep_by_scale = {int(r.scale): r for r in sweep.runs}

    if args.all_sweep_scales:
        return [
            {"scale": int(r.scale), "overrides": dict(r.get("overrides", {}))}
            for r in sweep.runs
        ]
    if args.scales:
        wanted = [int(x) for x in args.scales.split(",")]
    elif args.scale is not None:
        wanted = [args.scale]
    else:
        raise SystemExit("must pass --scale, --scales, or --all-sweep-scales")
    out = []
    for n in wanted:
        if n in sweep_by_scale:
            out.append({"scale": n, "overrides": dict(sweep_by_scale[n].get("overrides", {}))})
        else:
            out.append({"scale": n, "overrides": {}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--sweep-config", default="configs/sweep_data_scale.yaml")

    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--scale", type=int, default=None, help="single data scale to train")
    sel.add_argument("--scales", default=None, help="comma-separated list of scales (run in order)")
    sel.add_argument("--all-sweep-scales", action="store_true", help="run every scale from sweep config")

    ap.add_argument("--resume-from", default=None, help="resume from a checkpoint dir (single-scale only)")
    ap.add_argument("--yes", action="store_true", help="skip the ETA confirmation prompt")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotted-key=value override applied to training config (repeatable)",
    )
    args = ap.parse_args()

    setup_logging()
    tcfg_base = load_config(args.training_config)
    mcfg = load_config(args.model_config)

    # CLI overrides apply to every scale below.
    cli_overrides = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in args.override}

    runs = _resolve_scales(args, args.sweep_config)

    # Compute total ETA and confirm.
    total_seconds = 0.0
    for r in runs:
        merged = apply_overrides(tcfg_base, {**cli_overrides, **r["overrides"]})
        epochs = int(round(float(merged.trainer.num_train_epochs)))
        bsz = int(merged.trainer.per_device_train_batch_size) * int(merged.trainer.gradient_accumulation_steps)
        eta = estimate_duration_seconds(pairs=r["scale"], epochs=epochs, effective_batch_size=bsz)
        r["_eta"] = eta
        if eta.get("seconds") is not None:
            total_seconds += eta["seconds"]
        logger.info(
            f"plan: scale={r['scale']:,}  epochs={epochs}  expected_steps={eta['expected_steps']:,}  "
            f"ETA={eta.get('hours')}h  conf={eta.get('confidence')}"
        )

    logger.info(f"total ETA across {len(runs)} run(s): ~{total_seconds / 3600.0:.1f} hours")
    if not args.yes and total_seconds and total_seconds > 3600:
        try:
            ans = input("continue? [y/N] ")
        except EOFError:
            ans = "n"
        if ans.strip().lower() not in ("y", "yes"):
            logger.warning("aborted by user")
            return 130

    # Execute.
    summaries = []
    for r in runs:
        merged = apply_overrides(tcfg_base, {**cli_overrides, **r["overrides"]})
        merged_d = OmegaConf.to_container(merged, resolve=True)
        mcfg_d = OmegaConf.to_container(mcfg, resolve=True)
        summary = build_and_train(
            scale=r["scale"],
            training_cfg=merged_d,
            model_cfg=mcfg_d,
            resume_from=args.resume_from if len(runs) == 1 else None,
        )
        summaries.append(summary)

    logger.info(f"completed {len(summaries)} run(s): {[s['run_id'] for s in summaries]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
