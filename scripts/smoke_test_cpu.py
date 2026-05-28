"""CPU smoke test: validates the full training pipeline against the extended model.

Runs a tiny ~20-step training pass at scale=10000 with 0.01 epoch on CPU just
to exercise every callback (timings, retention, forgetting probe), prove the
extended model loads correctly, and write real artifacts to models/runs/smoke/
that you can inspect.

Does NOT train to convergence — purely a pipeline validation.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ecmt.training.trainer import build_and_train  # noqa: E402
from ecmt.utils.config import apply_overrides, load_config  # noqa: E402

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Wipe any prior smoke run
SMOKE_DIR = REPO / "models" / "runs" / "smoke-cpu"
if SMOKE_DIR.exists():
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)

tcfg = load_config(str(REPO / "configs" / "training.yaml"))
mcfg = load_config(str(REPO / "configs" / "model.yaml"))

overrides = {
    "trainer.bf16": "false",
    "trainer.fp16": "false",                       # CPU: keep fp32
    "trainer.per_device_train_batch_size": "1",
    "trainer.per_device_eval_batch_size": "1",
    "trainer.gradient_accumulation_steps": "1",
    "trainer.num_train_epochs": "0.005",            # ~10 steps on scale=10000
    "trainer.eval_steps": "5",
    "trainer.save_steps": "5",
    "trainer.logging_steps": "1",
    "trainer.max_seq_length": "256",
    "trainer.gradient_checkpointing": "false",      # speed up tiny run
    "trainer.optim": "adamw_torch",
    # disable wandb / mlflow / tensorboard for the smoke test
    "tracking.wandb.enabled": "false",
    "tracking.mlflow.enabled": "false",
    "tracking.tensorboard.enabled": "false",
}
merged = apply_overrides(tcfg, overrides)
from omegaconf import OmegaConf  # noqa: E402
training_cfg = OmegaConf.to_container(merged, resolve=True)
training_cfg["output_root"] = str(REPO / "models" / "runs")
model_cfg = OmegaConf.to_container(mcfg, resolve=True)

t0 = time.time()
summary = build_and_train(
    scale=10000,
    training_cfg=training_cfg,
    model_cfg=model_cfg,
    run_name="smoke-cpu",
)
dt = time.time() - t0
print(f"\n[smoke] run finished in {dt:.1f}s")
print(f"[smoke] summary: {summary}")
print("[smoke] artifacts:")
for p in sorted((REPO / "models" / "runs" / "smoke-cpu").rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(REPO)}  ({p.stat().st_size} bytes)")
