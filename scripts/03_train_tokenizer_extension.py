"""Train a Chinese SentencePiece BPE and merge it into the base SmolLM2 tokenizer.

Pulls the Chinese side of every kept training pair from
`data/processed/all_pairs.parquet`, writes a temporary text file, trains SP,
then merges Han-only pieces into the base tokenizer and saves the result to
`tokenizers/built/merged/`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.model.tokenizer_extension import extend_tokenizer, train_sentencepiece_on_chinese  # noqa: E402
from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--vocab-size", type=int, default=None, help="override model.yaml vocab size")
    ap.add_argument(
        "--source-parquet",
        default=None,
        help="parquet with a 'zh' column to train SP on (defaults to processed/all_pairs.parquet)",
    )
    args = ap.parse_args()

    setup_logging()
    data_cfg = load_config(args.data_config)
    mcfg = load_config(args.model_config).tokenizer_extension
    if args.vocab_size:
        mcfg.vocab_size = args.vocab_size

    src = Path(args.source_parquet) if args.source_parquet else Path(data_cfg.processed_root) / "all_pairs.parquet"
    if not src.exists():
        logger.error(f"{src} not found — run 02_prepare_data.py first")
        return 1

    table = pq.read_table(src, columns=["zh"])
    zh_lines = table.column("zh").to_pylist()
    logger.info(f"loaded {len(zh_lines):,} Chinese sentences from {src}")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for s in zh_lines:
            if s and s.strip():
                tmp.write(s.replace("\n", " ").strip() + "\n")
    logger.info(f"sp training corpus -> {tmp_path}")

    sp_out_dir = Path(mcfg.sp_model_dir)
    sp_model_path = train_sentencepiece_on_chinese(
        chinese_text_file=tmp_path,
        out_dir=sp_out_dir,
        vocab_size=int(mcfg.vocab_size),
        character_coverage=float(mcfg.character_coverage),
        model_type=str(mcfg.model_type),
        byte_fallback=bool(mcfg.byte_fallback),
    )
    logger.info(f"sp model -> {sp_model_path}")

    model_cfg_top = load_config(args.model_config)
    n_zh, n_specials = extend_tokenizer(
        base_tokenizer_id=str(model_cfg_top.base_model.hf_id),
        sp_model_path=sp_model_path,
        out_dir=Path(mcfg.merged_tokenizer_dir),
        special_tokens=list(mcfg.special_tokens),
    )
    logger.info(f"added {n_zh:,} Chinese tokens + {n_specials} special tokens")

    try:
        tmp_path.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
