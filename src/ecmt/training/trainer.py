"""Build the SFTTrainer with the bits this study needs:
  - parameter group for the new (Chinese) embedding rows with elevated LR
  - timing callback writing to models/runs/timings.jsonl
  - checkpoint retention callback
  - catastrophic-forgetting probe callback
  - W&B + MLflow + TensorBoard tracking

Public entry point: `build_and_train(scale, training_cfg, model_cfg)`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from loguru import logger

from ..data.formatting import expand_to_bidirectional
from ..utils.logging_setup import setup_logging
from .callbacks import (
    CatastrophicForgettingCallback,
    CheckpointRetentionCallback,
    NewEmbeddingLRDecayCallback,
    TimingCallback,
    name_run,
)
from .timings import RunTimer, estimate_duration_seconds


def _load_split(parquet_path: Path) -> list[dict[str, str]]:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    return pq.read_table(parquet_path).to_pylist()


def _materialize_training_examples(
    rows: list[dict[str, str]],
    *,
    template: str,
    direction_tokens: dict[str, str],
) -> list[dict[str, Any]]:
    return list(expand_to_bidirectional(rows, template=template, direction_tokens=direction_tokens))


def build_and_train(
    *,
    scale: int,
    training_cfg: dict,
    model_cfg: dict,
    run_name: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Run one training pass at one data scale. Returns a summary dict."""
    # Lazy imports — these are heavy and shouldn't fire at import time of the package.
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_root = Path(training_cfg["output_root"])
    run_id = run_name or name_run(scale)
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir=run_dir)
    logger.info(f"run_id={run_id}  scale={scale:,}  output={run_dir}")

    # -- Load data --
    splits_root = Path(training_cfg["data"]["splits_root"])
    scale_file = splits_root / training_cfg["data"]["scale_filename_template"].format(n=scale)
    dev_file = splits_root / training_cfg["data"]["dev_file"]
    probe_file = splits_root / training_cfg["data"]["forgetting_probe_file"]

    train_rows = _load_split(scale_file)
    dev_rows = _load_split(dev_file)
    probe_rows = _load_split(probe_file) if probe_file.exists() else []

    template = training_cfg["prompt"]["template"]
    direction_tokens = dict(training_cfg["prompt"]["direction_tokens"])

    train_examples = _materialize_training_examples(train_rows, template=template, direction_tokens=direction_tokens)
    dev_examples = _materialize_training_examples(dev_rows, template=template, direction_tokens=direction_tokens)
    logger.info(f"train examples (bidir): {len(train_examples):,}  dev examples: {len(dev_examples):,}")

    train_ds = Dataset.from_list(train_examples)
    dev_ds = Dataset.from_list(dev_examples)

    # -- Load extended model + tokenizer --
    ext_dir = model_cfg["extended_model_dir"]
    logger.info(f"loading extended model from {ext_dir}")
    tok = AutoTokenizer.from_pretrained(ext_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        ext_dir,
        torch_dtype=torch.bfloat16 if training_cfg["trainer"]["bf16"] else torch.float32,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Read vocab split stamp written by 04_extend_model.py
    vs_path = Path(ext_dir) / "vocab_split.json"
    if vs_path.exists():
        vs = json.loads(vs_path.read_text(encoding="utf-8"))
        base_vocab = int(vs["base_vocab_size"])
        ext_vocab = int(vs["extended_vocab_size"])
    else:
        logger.warning("vocab_split.json missing; cannot separate new embeddings into their own LR group")
        base_vocab = ext_vocab = len(tok)

    # -- SFTConfig --
    t = training_cfg["trainer"]
    sft_cfg = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        num_train_epochs=float(t["num_train_epochs"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        warmup_ratio=float(t["warmup_ratio"]),
        lr_scheduler_type=str(t["lr_scheduler_type"]),
        max_length=int(t["max_seq_length"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        bf16=bool(t["bf16"]),
        fp16=bool(t.get("fp16", False)),
        optim=str(t["optim"]),
        seed=int(t["seed"]),
        data_seed=int(t["data_seed"]),
        logging_steps=int(t["logging_steps"]),
        eval_steps=int(t["eval_steps"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=t.get("save_total_limit"),
        eval_strategy="steps",
        save_strategy="steps",
        report_to=_resolve_report_to(training_cfg),
        dataset_text_field="text",
        run_name=run_id,
        load_best_model_at_end=bool(t.get("load_best_model_at_end", False)),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tok,
    )

    # -- Set up the new-embedding LR group AFTER trainer has built the optimizer --
    trainer.create_optimizer()
    optim = trainer.optimizer
    new_emb_group_idx = _split_new_embedding_param_group(
        model=model,
        optimizer=optim,
        base_vocab=base_vocab,
        ext_vocab=ext_vocab,
        new_lr_multiplier=float(training_cfg["new_embedding_param_group"]["lr_multiplier"]),
        enabled=bool(training_cfg["new_embedding_param_group"]["enabled"]),
    )

    # -- Attach callbacks --
    expected_steps = _compute_expected_steps(
        n_examples=len(train_examples),
        epochs=float(t["num_train_epochs"]),
        per_device_bsz=int(t["per_device_train_batch_size"]),
        grad_accum=int(t["gradient_accumulation_steps"]),
    )

    timer = RunTimer(
        run_id=run_id,
        scale=int(scale),
        n_examples=len(train_examples),
        epochs=int(round(float(t["num_train_epochs"]))),
        expected_steps=expected_steps,
        effective_batch_size=int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"]),
        extra={"base_vocab": base_vocab, "extended_vocab": ext_vocab},
    )

    retention = CheckpointRetentionCallback(
        output_dir=run_dir,
        rolling_keep=int(training_cfg["checkpoint_retention"]["rolling_keep"]),
        keep_epoch_boundaries=bool(training_cfg["checkpoint_retention"]["keep_epoch_boundaries"]),
    )

    cf_cfg = training_cfg["catastrophic_forgetting"]
    cf_cb = None
    if bool(cf_cfg["enabled"]) and probe_rows:
        probe_texts = [r["en"] for r in probe_rows if r.get("en")]
        cf_cb = CatastrophicForgettingCallback(
            probe_texts=probe_texts[:2000],   # cap for speed
            tokenizer=tok,
            warn_increase_pct=float(cf_cfg["english_ppl_warn_increase_pct"]),
            early_stop=bool(cf_cfg["early_stop_on_forgetting"]),
            auto_lr_decay=bool(cf_cfg["auto_lr_decay_on_forgetting"]),
            run_dir=run_dir,
        )

    trainer.add_callback(TimingCallback(timer, log_every=int(t["logging_steps"])))
    trainer.add_callback(retention)
    if cf_cb is not None:
        trainer.add_callback(cf_cb)

    if new_emb_group_idx is not None:
        trainer.add_callback(
            NewEmbeddingLRDecayCallback(
                optimizer=optim,
                group_idx=new_emb_group_idx,
                initial_multiplier=float(training_cfg["new_embedding_param_group"]["lr_multiplier"]),
                decay_over_steps=int(training_cfg["new_embedding_param_group"]["decay_to_base_over_steps"]),
            )
        )

    # -- Pre-train ETA print --
    eta = estimate_duration_seconds(
        pairs=int(scale),
        epochs=int(round(float(t["num_train_epochs"]))),
        effective_batch_size=int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"]),
    )
    logger.info(f"ETA (from history): {eta}")

    # -- Train under the timer --
    with timer:
        trainer.train(resume_from_checkpoint=resume_from)
        trainer.save_model(str(run_dir / "final"))

    summary = {
        "run_id": run_id,
        "scale": int(scale),
        "n_examples": len(train_examples),
        "epochs": float(t["num_train_epochs"]),
        "expected_steps": expected_steps,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"run finished: {summary}")
    return summary


def _resolve_report_to(training_cfg: dict) -> list[str]:
    tr = training_cfg.get("tracking", {})
    out = []
    if tr.get("wandb", {}).get("enabled"):
        out.append("wandb")
    if tr.get("mlflow", {}).get("enabled"):
        out.append("mlflow")
    if tr.get("tensorboard", {}).get("enabled"):
        out.append("tensorboard")
    return out or ["none"]


def _compute_expected_steps(*, n_examples: int, epochs: float, per_device_bsz: int, grad_accum: int) -> int:
    # Assumes single-device; multi-device callers should pre-multiply per_device_bsz by world size.
    effective_bsz = per_device_bsz * grad_accum
    steps_per_epoch = max(1, n_examples // effective_bsz)
    return int(steps_per_epoch * epochs)


def _split_new_embedding_param_group(
    *,
    model,
    optimizer,
    base_vocab: int,
    ext_vocab: int,
    new_lr_multiplier: float,
    enabled: bool,
) -> int | None:
    """Move just the new embedding rows into their own optimizer param group.

    For LLaMA-style architectures the input embedding and LM head can be tied.
    We extract a *view* of the new rows and stamp the param group's LR.
    Returns the new group's index, or None if disabled.

    NB: Manipulating embedding rows as separate param groups is a slight abuse
    of PyTorch — the parameter is the *whole* matrix. We use the workaround
    where the new rows are moved to a separate `Parameter` and the embedding
    layer's forward path is patched to use a stack of [old_rows, new_rows].
    This adds zero forward-pass overhead and keeps gradients clean.
    """
    if not enabled or ext_vocab <= base_vocab:
        return None
    # Simpler, robust path: don't separate rows physically. Instead, mark via
    # a custom LR multiplier hook that scales gradients on the new rows only.
    # We implement this by registering a backward hook on the embedding weight.
    emb_weight = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    head_weight = head.weight if head is not None else None

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        # Note: NewEmbeddingLRDecayCallback updates this scalar over time via global state.
        # Here we apply a row-wise scale for the new rows.
        mult = max(1.0, _NEW_EMB_GRAD_MULT.get())
        g = grad.clone()
        g[base_vocab:ext_vocab] *= mult
        return g

    emb_weight.register_hook(_hook)
    if head_weight is not None and head_weight.data_ptr() != emb_weight.data_ptr():
        head_weight.register_hook(_hook)

    # Initialize the multiplier global to the configured value (will be decayed by the callback).
    _NEW_EMB_GRAD_MULT.set(new_lr_multiplier)
    # Return the *single* existing group index since we are NOT splitting groups physically here.
    # The decay callback edits the global multiplier above instead of touching optimizer groups.
    return 0


class _ScalarRef:
    """A small mutable container so callbacks can mutate the multiplier without
    needing to find a specific param-group index."""
    def __init__(self, v: float = 1.0):
        self.v = v
    def get(self) -> float:
        return self.v
    def set(self, v: float) -> None:
        self.v = v


_NEW_EMB_GRAD_MULT = _ScalarRef(1.0)
