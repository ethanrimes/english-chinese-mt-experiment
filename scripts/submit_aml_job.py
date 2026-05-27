"""Submit one or more training jobs to Azure ML.

Wraps `az ml job create` for each requested scale. Before submission, prints
the ETA total (from prior-run timings.jsonl) and asks for confirmation if the
predicted total wallclock exceeds a threshold.

Requires:
  - az CLI logged in (`az login`)
  - environment vars AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
  - data + extended model already uploaded to a datastore (set --splits-uri / --model-uri)

Usage:
    python scripts/submit_aml_job.py --scales 10000,50000
    python scripts/submit_aml_job.py --all
    python scripts/submit_aml_job.py --scales 5000000 --splits-uri azureml://... --model-uri azureml://...
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.training.timings import estimate_duration_seconds  # noqa: E402
from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402

TEMPLATE_PATH = REPO_ROOT / "azure" / "job_template.yaml"


def _render_template(text: str, vars: dict[str, str]) -> str:
    for k, v in vars.items():
        text = text.replace("{{ " + k + " }}", v)
    return text


def _render_overrides_cli(overrides: dict) -> str:
    return " ".join(f"--override {k}={v}" for k, v in overrides.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--sweep-config", default="configs/sweep_data_scale.yaml")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--scales", default=None, help="comma-separated scales to submit")
    sel.add_argument("--all", action="store_true", help="submit every sweep scale")
    ap.add_argument("--splits-uri", default=os.environ.get("ECMT_SPLITS_URI", ""), help="AML uri_folder for data/splits")
    ap.add_argument("--model-uri", default=os.environ.get("ECMT_MODEL_URI", ""), help="AML uri_folder for models/extended")
    ap.add_argument(
        "--workspace", default=os.environ.get("AZURE_ML_WORKSPACE", ""),
    )
    ap.add_argument(
        "--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP", ""),
    )
    ap.add_argument("--dry-run", action="store_true", help="render YAMLs but don't submit")
    ap.add_argument("--yes", action="store_true", help="skip ETA confirmation")
    args = ap.parse_args()

    setup_logging()
    tcfg = load_config(args.training_config)
    sweep = load_config(args.sweep_config)

    if args.all:
        wanted = [int(r.scale) for r in sweep.runs]
    else:
        wanted = [int(s) for s in args.scales.split(",")]

    sweep_by_scale = {int(r.scale): r for r in sweep.runs}
    plans = []
    total_seconds = 0.0
    bsz = int(tcfg.trainer.per_device_train_batch_size) * int(tcfg.trainer.gradient_accumulation_steps)

    for n in wanted:
        entry = sweep_by_scale.get(n)
        overrides = dict(entry.overrides) if entry else {}
        epochs = int(round(float(overrides.get("trainer.num_train_epochs", tcfg.trainer.num_train_epochs))))
        eta = estimate_duration_seconds(pairs=n, epochs=epochs, effective_batch_size=bsz)
        run_name = (entry.name if entry else f"scale-{n}")
        tag = (entry.experiment_tag if entry else f"scale_{n}")
        plans.append({
            "scale": n,
            "run_name": str(run_name),
            "experiment_tag": str(tag),
            "epochs": epochs,
            "overrides": overrides,
            "eta": eta,
        })
        if eta.get("seconds") is not None:
            total_seconds += eta["seconds"]

    logger.info("plan:")
    for p in plans:
        eta = p["eta"]
        logger.info(
            f"  scale={p['scale']:,}  epochs={p['epochs']}  "
            f"ETA={eta.get('hours')}h  conf={eta.get('confidence')}"
        )
    logger.info(f"total ETA ≈ {total_seconds / 3600.0:.1f} hours  (parallel: bounded by max single-job time)")

    if not args.yes and total_seconds > 4 * 3600:
        try:
            ans = input("submit all? [y/N] ")
        except EOFError:
            ans = "n"
        if ans.strip().lower() not in ("y", "yes"):
            logger.warning("aborted")
            return 130

    if not args.workspace or not args.resource_group:
        logger.error("workspace/resource group required (set AZURE_ML_WORKSPACE + AZURE_RESOURCE_GROUP)")
        return 2
    if not args.splits_uri or not args.model_uri:
        logger.error("--splits-uri and --model-uri are required (upload data/splits + models/extended first)")
        return 2

    template_text = TEMPLATE_PATH.read_text(encoding="utf-8")

    job_ids = []
    for p in plans:
        rendered = _render_template(template_text, {
            "scale": str(p["scale"]),
            "run_name": p["run_name"],
            "experiment_tag": p["experiment_tag"],
            "experiment_name": str(tcfg.experiment_name),
            "splits_data_uri": args.splits_uri,
            "extended_model_uri": args.model_uri,
            "overrides_cli": _render_overrides_cli(p["overrides"]),
        })
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
            fh.write(rendered)
            tmp_path = fh.name
        cmd = [
            "az", "ml", "job", "create",
            "-f", tmp_path,
            "-w", args.workspace,
            "-g", args.resource_group,
        ]
        logger.info(f"submitting: {p['run_name']}")
        if args.dry_run:
            logger.info(f"  DRY RUN: {tmp_path}\n{rendered}")
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(out.stdout.strip())
            job_ids.append(p["run_name"])
        except subprocess.CalledProcessError as e:
            logger.error(f"  az failed: {e.stderr}")
            return e.returncode
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    logger.info(f"submitted {len(job_ids)} job(s): {job_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
