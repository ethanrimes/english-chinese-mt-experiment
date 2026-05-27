"""Custom HuggingFace Trainer callbacks: timing, checkpoint retention,
catastrophic-forgetting probe, new-embedding LR decay.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .timings import RunTimer


class TimingCallback(TrainerCallback):
    """Pipes Trainer events into our RunTimer (which writes to timings.jsonl)."""

    def __init__(self, timer: RunTimer, log_every: int = 50):
        self.timer = timer
        self.log_every = log_every

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0 and state.global_step % self.log_every == 0:
            self.timer.log_step(state.global_step)
        return control


class CheckpointRetentionCallback(TrainerCallback):
    """Custom retention beyond `save_total_limit`:
      - keep last `rolling_keep` checkpoints,
      - always keep checkpoints at epoch boundaries (last_step_of_epoch),
      - always keep the best-by-dev-BLEU checkpoint (managed by EvalCallback).
    """

    def __init__(self, output_dir: str | Path, *, rolling_keep: int, keep_epoch_boundaries: bool):
        self.output_dir = Path(output_dir)
        self.rolling_keep = rolling_keep
        self.keep_epoch_boundaries = keep_epoch_boundaries
        self.permanent: set[int] = set()  # checkpoint step numbers we want to keep forever

    def mark_permanent(self, step: int) -> None:
        self.permanent.add(step)

    def on_save(self, args, state, control, **kwargs):
        if self.keep_epoch_boundaries:
            # If this save is at end of an epoch (within one log-step), mark permanent.
            # state.epoch is float; integer-ish boundaries get permanent status.
            if state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-3 and state.epoch > 0:
                self.mark_permanent(state.global_step)
        self._sweep()
        return control

    def _sweep(self) -> None:
        ckpts = sorted(self.output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if len(ckpts) <= self.rolling_keep:
            return
        keep_steps = set()
        # always keep the most recent N
        for p in ckpts[-self.rolling_keep:]:
            keep_steps.add(int(p.name.split("-")[-1]))
        keep_steps |= self.permanent
        for p in ckpts:
            step = int(p.name.split("-")[-1])
            if step not in keep_steps:
                logger.info(f"retention sweep: removing {p}")
                shutil.rmtree(p, ignore_errors=True)


class CatastrophicForgettingCallback(TrainerCallback):
    """At each eval, compute perplexity on a held-out English-only probe set.

    Issues a warning when PPL grows >X% above the initial (pre-training) value.
    Optionally early-stops or decays LR.
    """

    def __init__(
        self,
        probe_texts: list[str],
        tokenizer,
        *,
        warn_increase_pct: float = 20.0,
        early_stop: bool = False,
        auto_lr_decay: bool = False,
        run_dir: Path | None = None,
    ):
        self.probe_texts = probe_texts
        self.tokenizer = tokenizer
        self.warn_pct = warn_increase_pct
        self.early_stop = early_stop
        self.auto_lr_decay = auto_lr_decay
        self.initial_ppl: float | None = None
        self.history: list[dict[str, float]] = []
        self.run_dir = run_dir

    def _measure(self, model) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        device = next(model.parameters()).device
        with torch.no_grad():
            for txt in self.probe_texts:
                ids = self.tokenizer(txt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
                out = model(input_ids=ids, labels=ids)
                # loss is per-token mean already
                tokens = ids.shape[1] - 1
                total_loss += out.loss.item() * tokens
                total_tokens += tokens
        model.train()
        if total_tokens == 0:
            return float("nan")
        return math.exp(total_loss / total_tokens)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        ppl = self._measure(model)
        if self.initial_ppl is None:
            self.initial_ppl = ppl
            logger.info(f"catastrophic-forgetting probe: initial English PPL = {ppl:.3f}")
        delta_pct = 100.0 * (ppl / self.initial_ppl - 1.0)
        record = {"step": state.global_step, "english_ppl": ppl, "delta_pct": delta_pct}
        self.history.append(record)
        logger.info(f"english PPL = {ppl:.3f}  (Δ={delta_pct:+.1f}% vs initial)")
        if self.run_dir is not None:
            (self.run_dir / "forgetting.jsonl").open("a", encoding="utf-8").write(
                json.dumps(record) + "\n"
            )
        if delta_pct > self.warn_pct:
            logger.warning(f"⚠ english PPL up {delta_pct:.1f}% — catastrophic forgetting signal")
            if self.early_stop:
                logger.warning("early-stopping triggered by forgetting monitor")
                control.should_training_stop = True
        return control


class NewEmbeddingLRDecayCallback(TrainerCallback):
    """Decay the new-embedding-rows LR multiplier back to base LR over a warm window.

    Works alongside the param-group structure set up in trainer.py: the new
    embedding rows are in their own group with `_init_lr_multiplier` applied
    on top of the scheduler. We linearly interpolate this multiplier from its
    initial value down to 1.0 over `decay_over_steps` steps.
    """

    def __init__(self, optimizer, group_idx: int, initial_multiplier: float, decay_over_steps: int):
        self.optimizer = optimizer
        self.group_idx = group_idx
        self.initial = initial_multiplier
        self.decay_over = max(1, decay_over_steps)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step >= self.decay_over:
            mult = 1.0
        else:
            mult = self.initial - (self.initial - 1.0) * (step / self.decay_over)
        # Scale this group's LR relative to the *other* groups, which already follow the scheduler.
        # We do that by stamping the multiplier into a custom attribute and applying it lazily.
        # NB: Hugging Face's scheduler updates group['lr'] each step; we multiply *after*.
        group = self.optimizer.param_groups[self.group_idx]
        base_lr = group.get("_base_lr_unscaled", group["lr"] / max(group.get("_last_mult", 1.0), 1e-12))
        group["_base_lr_unscaled"] = base_lr
        group["lr"] = base_lr * mult
        group["_last_mult"] = mult
        return control


def name_run(scale: int) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return f"scale-{scale}-{stamp}"


def sanitize_run_dir_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)
