"""Sync completed AML training-run artifacts back to the local repo.

For each finished job (or one specified by --job-name), downloads:
  - The full job log bundle (logs + outputs/runs)
  - Lands them under models/runs/<run-id>/ to match local-run layout

Usage:
    python azure/sync_runs.py                     # all completed jobs in experiment
    python azure/sync_runs.py --job-name <name>   # one specific
    python azure/sync_runs.py --since 2026-05-28  # only after a date
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

REPO = Path(__file__).resolve().parent.parent
SUBSCRIPTION = os.environ.get("AZURE_SUBSCRIPTION_ID", "9a04b64b-af19-4519-be50-56ec2acbd855")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "ecmt-rg")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE", "ecmt-ws")
EXPERIMENT_NAME = "ecmt-sft-v1"
RUNS_LOCAL = REPO / "models" / "runs"


def _client() -> MLClient:
    return MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def _download_one(client: MLClient, job_name: str) -> Path | None:
    print(f"[sync] downloading job: {job_name}")
    dest = RUNS_LOCAL / job_name
    dest.mkdir(parents=True, exist_ok=True)
    try:
        client.jobs.download(
            name=job_name,
            download_path=str(dest),
            output_name="runs",
            all=False,
        )
        # Also pull logs alongside
        client.jobs.download(
            name=job_name,
            download_path=str(dest),
            all=True,
        )
    except Exception as e:
        print(f"[sync]   ERROR: {e}")
        return None
    print(f"[sync]   landed at {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--since", default=None, help="ISO date; only jobs created after")
    args = ap.parse_args()

    client = _client()

    if args.job_name:
        _download_one(client, args.job_name)
        return 0

    print(f"[sync] listing jobs in experiment {EXPERIMENT_NAME}")
    cutoff = None
    if args.since:
        cutoff = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    n_done = 0
    for job in client.jobs.list():
        if getattr(job, "experiment_name", None) != EXPERIMENT_NAME:
            continue
        if getattr(job, "status", "") not in ("Completed", "Failed"):
            continue
        if cutoff and getattr(job, "creation_context", None) and job.creation_context.created_at < cutoff:
            continue
        if _download_one(client, job.name):
            n_done += 1
    print(f"[sync] done. {n_done} job(s) synced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
