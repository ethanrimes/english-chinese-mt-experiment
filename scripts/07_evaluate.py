"""Evaluate a checkpoint (or every checkpoint in a run dir) on FLORES-200.

Usage:
    # Evaluate a single checkpoint
    python scripts/07_evaluate.py --checkpoint models/runs/scale-10000-XXX/final

    # Evaluate every snapshot in a run dir (for the per-scale learning curve)
    python scripts/07_evaluate.py --run-dir models/runs/scale-10000-XXX

    # Final report on FLORES-200 devtest (not dev)
    python scripts/07_evaluate.py --checkpoint <ckpt> --on devtest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.training.eval import evaluate_checkpoint  # noqa: E402
from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402


def _checkpoints_in(run_dir: Path) -> list[Path]:
    out = sorted(
        [p for p in run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if (run_dir / "final").exists():
        out.append(run_dir / "final")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--checkpoint", help="path to a single checkpoint dir")
    sel.add_argument("--run-dir", help="run dir; evaluate every checkpoint and 'final'")
    ap.add_argument("--on", choices=["dev", "devtest"], default="dev")
    ap.add_argument("--report-name", default=None, help="filename under reports/")
    args = ap.parse_args()

    setup_logging()
    tcfg = load_config(args.training_config)
    dcfg = load_config(args.data_config)

    splits_root = Path(dcfg.splits_root)
    flores_dev = splits_root / "dev_flores_dev.parquet"
    flores_test = splits_root / "test_flores_devtest.parquet"

    template = tcfg.prompt.template
    direction_tokens = dict(tcfg.prompt.direction_tokens)
    metrics = list(tcfg.eval.metrics)
    comet_id = str(tcfg.eval.comet_model)

    targets: list[Path]
    if args.checkpoint:
        targets = [Path(args.checkpoint)]
        report_label = Path(args.checkpoint).name
    else:
        rd = Path(args.run_dir)
        targets = _checkpoints_in(rd)
        report_label = rd.name
        if not targets:
            logger.error(f"no checkpoints found under {rd}")
            return 1
        logger.info(f"evaluating {len(targets)} checkpoint(s) under {rd}")

    all_results = []
    for ckpt in targets:
        try:
            res = evaluate_checkpoint(
                checkpoint_dir=ckpt,
                flores_dev_path=flores_dev,
                flores_test_path=flores_test,
                template=str(template),
                direction_tokens=direction_tokens,
                max_new_tokens=int(tcfg.eval.max_new_tokens),
                num_beams=int(tcfg.eval.num_beams),
                metrics=metrics,
                comet_model_id=comet_id,
                on_split=args.on,
            )
            all_results.append(res)
        except Exception as e:
            logger.exception(f"  {ckpt}: eval failed: {e!r}")

    reports_dir = REPO_ROOT / "reports" / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = args.report_name or f"{report_label}__{args.on}.json"
    out_path = reports_dir / report_name
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, ensure_ascii=False)
    logger.info(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
