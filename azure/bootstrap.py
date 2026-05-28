"""Provision Azure resources for the M60 training run using azure-ai-ml SDK.

This bypasses the `az ml` CLI extension (which fails to install on this box)
and drives everything directly via the Python SDK.

Stages (idempotent — each checks if the resource already exists):
  1. Resource group           (via az CLI fallback for the RG, which works)
  2. AML workspace            (azure-ai-ml SDK)
  3. NV12 compute cluster     (azure-ai-ml SDK)
  4. Docker environment       (azure-ai-ml SDK, builds in cloud)
  5. Upload data + extended model to default datastore
  6. Submit training job

Run sub-stages individually:
    python azure/bootstrap.py rg
    python azure/bootstrap.py workspace
    python azure/bootstrap.py compute
    python azure/bootstrap.py env
    python azure/bootstrap.py upload
    python azure/bootstrap.py submit --scale 10000
    python azure/bootstrap.py all --scale 10000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    AmlCompute,
    BuildContext,
    Environment,
    Workspace,
)
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import AzureCliCredential

REPO = Path(__file__).resolve().parent.parent

SUBSCRIPTION = os.environ.get("AZURE_SUBSCRIPTION_ID", "9a04b64b-af19-4519-be50-56ec2acbd855")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "ecmt-rg")
LOCATION = os.environ.get("AZURE_LOCATION", "eastus")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE", "ecmt-ws")
COMPUTE_NAME = os.environ.get("ECMT_COMPUTE_NAME", "gpu-t4-16g")
COMPUTE_SIZE = os.environ.get("ECMT_COMPUTE_SIZE", "Standard_NC8as_T4_v3")  # 1x T4 16GB (~$0.53/hr) — needs quota. Fallback options below in module docstring.
ENV_NAME = "ecmt-pytorch-m60"
EXPERIMENT_NAME = "ecmt-sft-v1"


def _log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def _ml_client() -> MLClient:
    return MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def stage_rg() -> None:
    _log(f"ensuring resource group: {RESOURCE_GROUP} ({LOCATION})")
    out = subprocess.run(
        f'az group show -n {RESOURCE_GROUP}',
        capture_output=True, text=True, shell=True,
    )
    if out.returncode == 0:
        _log("  exists")
        return
    subprocess.run(
        f'az group create -n {RESOURCE_GROUP} -l {LOCATION} --subscription {SUBSCRIPTION}',
        check=True, shell=True,
    )


def stage_workspace() -> None:
    _log(f"ensuring AML workspace: {WORKSPACE_NAME}")
    client = MLClient(
        credential=AzureCliCredential(),
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
    )
    try:
        ws = client.workspaces.get(WORKSPACE_NAME)
        _log(f"  exists: {ws.id}")
        return
    except (ResourceNotFoundError, HttpResponseError):
        pass
    _log("  creating (~3-5 min)...")
    ws = Workspace(
        name=WORKSPACE_NAME,
        location=LOCATION,
        display_name="English-Chinese MT scaling study",
        description="SmolLM2-360M data-scaling sweep on M60.",
        tags={"project": "ecmt"},
    )
    op = client.workspaces.begin_create(workspace=ws)
    op.wait()
    _log(f"  created: {op.result().id}")


def stage_compute() -> None:
    _log(f"ensuring compute cluster: {COMPUTE_NAME} ({COMPUTE_SIZE})")
    client = _ml_client()
    try:
        c = client.compute.get(COMPUTE_NAME)
        _log(f"  exists: size={c.size} min={c.min_instances} max={c.max_instances}")
        return
    except (ResourceNotFoundError, HttpResponseError):
        pass
    c = AmlCompute(
        name=COMPUTE_NAME,
        size=COMPUTE_SIZE,
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=600,
        tier="dedicated",
        description="2x M60 (16 GB) for ECMT training",
    )
    op = client.compute.begin_create_or_update(c)
    op.wait()
    _log("  ready")


def stage_env() -> None:
    _log(f"ensuring environment: {ENV_NAME}")
    client = _ml_client()
    env = Environment(
        name=ENV_NAME,
        description="PyTorch 2.5 + transformers/trl/deepspeed for SmolLM2 SFT on M60 (sm_50).",
        build=BuildContext(path=str(REPO / "azure")),
    )
    res = client.environments.create_or_update(env)
    _log(f"  submitted: name={res.name} version={res.version}")
    _log("  Docker build runs in cloud — proceeds asynchronously.")


def stage_upload() -> None:
    """Upload splits + extended model to the workspace's default datastore."""
    _log("uploading splits + extended model to default datastore")
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    client = _ml_client()
    splits_src = REPO / "data" / "splits"
    ext_src = REPO / "models" / "extended" / "SmolLM2-360M-ext"
    if not ext_src.exists():
        _log(f"  ERROR: {ext_src} not found — run scripts/04_extend_model.py first")
        sys.exit(2)

    splits_asset = Data(
        path=str(splits_src),
        type=AssetTypes.URI_FOLDER,
        name="ecmt-splits",
        description="Data-scale parquet splits for ECMT.",
    )
    res = client.data.create_or_update(splits_asset)
    _log(f"  splits asset: {res.name} v{res.version} -> {res.path}")

    ext_asset = Data(
        path=str(ext_src),
        type=AssetTypes.URI_FOLDER,
        name="ecmt-extended-model",
        description="SmolLM2-360M with extended Chinese vocab.",
    )
    res2 = client.data.create_or_update(ext_asset)
    _log(f"  model asset: {res2.name} v{res2.version} -> {res2.path}")

    state_file = REPO / "azure" / ".uploaded.json"
    state_file.write_text(json.dumps({
        "splits": f"azureml:{res.name}:{res.version}",
        "extended": f"azureml:{res2.name}:{res2.version}",
    }, indent=2), encoding="utf-8")
    _log(f"  wrote {state_file}")


def stage_submit(scale: int) -> None:
    _log(f"submitting training job: scale={scale}")
    client = _ml_client()
    state_file = REPO / "azure" / ".uploaded.json"
    if not state_file.exists():
        _log("  ERROR: no .uploaded.json — run stage 'upload' first")
        sys.exit(2)
    uris = json.loads(state_file.read_text(encoding="utf-8"))

    # Sweep overrides per scale (smaller scales -> more epochs).
    sweep = {
        10000:    {"epochs": 10, "eval_steps": 100,  "save_steps": 100,  "logging_steps": 20},
        50000:    {"epochs": 6,  "eval_steps": 200,  "save_steps": 200,  "logging_steps": 50},
        100000:   {"epochs": 5,  "eval_steps": 300,  "save_steps": 300,  "logging_steps": 50},
        500000:   {"epochs": 3,  "eval_steps": 500,  "save_steps": 500,  "logging_steps": 50},
        1000000:  {"epochs": 2,  "eval_steps": 1000, "save_steps": 1000, "logging_steps": 50},
        5000000:  {"epochs": 1,  "eval_steps": 2000, "save_steps": 2000, "logging_steps": 50},
    }
    s = sweep.get(scale, {"epochs": 3, "eval_steps": 500, "save_steps": 500, "logging_steps": 50})

    # Hardware-family training overrides. Selected by COMPUTE_SIZE.
    #   - T4 (Turing sm_75): fp16, single-GPU, comfortable batch on 16GB
    #   - A10/A100 (Ampere sm_80+): bf16, larger batch
    #   - M60 (Maxwell sm_52): fp16, multi-GPU via DeepSpeed ZeRO-2, tiny batch
    family = _detect_family(COMPUTE_SIZE)
    hw_overrides, launcher_prefix = _hw_overrides(family)

    sweep_overrides = [
        "--override", f"trainer.num_train_epochs={s['epochs']}",
        "--override", f"trainer.eval_steps={s['eval_steps']}",
        "--override", f"trainer.save_steps={s['save_steps']}",
        "--override", f"trainer.logging_steps={s['logging_steps']}",
    ]
    all_overrides = hw_overrides + sweep_overrides

    run_name = f"scale-{scale}-{family}-{time.strftime('%Y%m%dT%H%M%S')}"

    cmd_str = " ".join([
        "set -eux",
        "&&", "pip install -e .",
        "&&", "mkdir -p data",
        "&&", "ln -sfn ${{inputs.splits}} data/splits",
        "&&", "mkdir -p models",
        "&&", "ln -sfn ${{inputs.extended_model}} models/extended",
        "&&", "mkdir -p models/runs",
        "&&", "ln -sfn ${{outputs.runs}} models/runs_out",
        "&&", "export ECMT_OUTPUT_ROOT=models/runs_out",
        "&&", launcher_prefix, "scripts/06_train.py",
        "--scale", str(scale), "--yes",
        *all_overrides,
    ])

    job = command(
        code=str(REPO),
        command=cmd_str,
        environment=f"azureml:{ENV_NAME}@latest",
        compute=COMPUTE_NAME,
        display_name=run_name,
        experiment_name=EXPERIMENT_NAME,
        inputs={
            "splits": Input(type="uri_folder", path=uris["splits"], mode="ro_mount"),
            "extended_model": Input(type="uri_folder", path=uris["extended"], mode="ro_mount"),
        },
        outputs={
            "runs": Output(type="uri_folder", mode="rw_mount"),
        },
        environment_variables={
            "HF_HOME": "/tmp/.hf",
            "TRANSFORMERS_CACHE": "/tmp/.hf/transformers",
            "HF_DATASETS_CACHE": "/tmp/.hf/datasets",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
        },
        tags={"project": "ecmt", "scale": str(scale), "hardware": family},
    )
    submitted = client.jobs.create_or_update(job)
    _log(f"  submitted: {submitted.name} status={submitted.status}")
    _log(f"  studio URL: {submitted.studio_url}")
    state = REPO / "azure" / ".jobs.jsonl"
    with state.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "name": submitted.name,
            "scale": scale,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "studio_url": submitted.studio_url,
        }) + "\n")


def _detect_family(size: str) -> str:
    s = size.lower()
    if "_t4_" in s or "t4" in s:
        return "T4"
    if "a100" in s:
        return "A100"
    if "a10v5" in s or "a10_v5" in s or "_a10_" in s:
        return "A10"
    if "h100" in s:
        return "H100"
    if "_nv12" == s.lower().split("standard_")[-1] or s.endswith("_nv6") or s.endswith("_nv24"):
        return "M60"
    return "GPU"


def _hw_overrides(family: str) -> tuple[list[str], str]:
    """Return (config overrides, command-line launcher prefix)."""
    if family in ("A100", "H100", "A10"):
        # bf16, ample VRAM
        return ([
            "--override", "trainer.bf16=true",
            "--override", "trainer.fp16=false",
            "--override", "trainer.per_device_train_batch_size=16",
            "--override", "trainer.per_device_eval_batch_size=32",
            "--override", "trainer.gradient_accumulation_steps=4",
            "--override", "trainer.max_seq_length=1024",
        ], "python")
    if family == "T4":
        # fp16, 16GB
        return ([
            "--override", "trainer.bf16=false",
            "--override", "trainer.fp16=true",
            "--override", "trainer.per_device_train_batch_size=8",
            "--override", "trainer.per_device_eval_batch_size=16",
            "--override", "trainer.gradient_accumulation_steps=8",
            "--override", "trainer.max_seq_length=1024",
        ], "python")
    if family == "M60":
        # fp16, 2x8GB via DeepSpeed ZeRO-2 (if you can resurrect the SKU somewhere)
        return ([
            "--override", "trainer.bf16=false",
            "--override", "trainer.fp16=true",
            "--override", "trainer.per_device_train_batch_size=2",
            "--override", "trainer.per_device_eval_batch_size=4",
            "--override", "trainer.gradient_accumulation_steps=16",
            "--override", "trainer.max_seq_length=768",
        ], "deepspeed --num_gpus 2")
    # default: conservative
    return ([
        "--override", "trainer.bf16=false",
        "--override", "trainer.fp16=true",
        "--override", "trainer.per_device_train_batch_size=4",
        "--override", "trainer.gradient_accumulation_steps=16",
    ], "python")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rg", "workspace", "compute", "env", "upload", "submit", "all"])
    ap.add_argument("--scale", type=int, default=10000)
    args = ap.parse_args()

    if args.stage in ("rg", "all"):
        stage_rg()
    if args.stage in ("workspace", "all"):
        stage_workspace()
    if args.stage in ("compute", "all"):
        stage_compute()
    if args.stage in ("env", "all"):
        stage_env()
    if args.stage in ("upload", "all"):
        stage_upload()
    if args.stage in ("submit", "all"):
        stage_submit(args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
