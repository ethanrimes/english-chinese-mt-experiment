"""Base-model loading + embedding resize for the extended tokenizer.

The training script never loads the bare SmolLM2-360M directly — it always
loads the *extended* model produced by scripts/04_extend_model.py, where the
embedding matrix and LM head already match the extended tokenizer's size.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_base(base_id: str, cache_dir: str | Path | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Fetch the base SmolLM2 checkpoint + tokenizer from HF (or cache)."""
    cache = str(cache_dir) if cache_dir else None
    tok = AutoTokenizer.from_pretrained(base_id, cache_dir=cache, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_id, cache_dir=cache)
    return model, tok


def extend_embeddings(
    model: PreTrainedModel,
    new_vocab_size: int,
    *,
    init_mean: float = 0.0,
    init_std: float = 0.02,
) -> PreTrainedModel:
    """Resize input embeddings + LM head to match a new (larger) vocab.

    Original rows are preserved verbatim; new rows are initialized N(mean, std).
    HuggingFace's `resize_token_embeddings` already copies old rows over and
    initializes the new tail. We override the new-rows initialization here for
    reproducibility (HF's default differs across versions).
    """
    old_size = model.get_input_embeddings().weight.shape[0]
    if new_vocab_size < old_size:
        raise ValueError(f"new_vocab_size={new_vocab_size} < old={old_size}; this function only grows")

    model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[old_size:].normal_(mean=init_mean, std=init_std)
        # LM head is tied for most LLaMA-style models; if untied, init those new rows too.
        head = model.get_output_embeddings()
        if head is not None and head.weight.data_ptr() != emb.data_ptr():
            head.weight[old_size:].normal_(mean=init_mean, std=init_std)

    logger.info(
        f"resized embeddings: {old_size} -> {new_vocab_size}  "
        f"(new rows initialized N({init_mean}, {init_std}))"
    )
    return model


def new_embedding_param_indices(old_vocab_size: int, new_vocab_size: int) -> slice:
    """Index slice locating the new embedding rows, for use by the LR-group callback."""
    return slice(old_vocab_size, new_vocab_size)
