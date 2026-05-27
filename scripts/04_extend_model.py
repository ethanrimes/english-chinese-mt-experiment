"""Load base SmolLM2-360M, resize embeddings to match the extended tokenizer,
and save the extended model to models/extended/.

This is the *single starting checkpoint* shared across all 4 data scales —
training only forks from here. That makes runs strictly comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.model.load import extend_embeddings, load_base  # noqa: E402
from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="configs/model.yaml")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.model_config)

    merged_tok_dir = Path(cfg.tokenizer_extension.merged_tokenizer_dir)
    if not merged_tok_dir.exists():
        logger.error(f"{merged_tok_dir} not found — run 03_train_tokenizer_extension.py first")
        return 1

    tok = AutoTokenizer.from_pretrained(str(merged_tok_dir), use_fast=True)
    target_vocab = len(tok)
    logger.info(f"merged tokenizer vocab size = {target_vocab}")

    logger.info(f"loading base model {cfg.base_model.hf_id}")
    model, _ = load_base(str(cfg.base_model.hf_id), cache_dir=str(cfg.base_model.local_cache_dir))
    old_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"base vocab = {old_size}; resizing to {target_vocab}")

    model = extend_embeddings(
        model,
        new_vocab_size=target_vocab,
        init_mean=float(cfg.tokenizer_extension.new_row_init_mean),
        init_std=float(cfg.tokenizer_extension.new_row_init_std),
    )

    out_dir = Path(cfg.extended_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # Stamp a tiny metadata file so the trainer's LR-group code knows which rows are "new".
    (out_dir / "vocab_split.json").write_text(
        f'{{"base_vocab_size": {old_size}, "extended_vocab_size": {target_vocab}}}\n',
        encoding="utf-8",
    )
    logger.info(f"extended model + tokenizer saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
