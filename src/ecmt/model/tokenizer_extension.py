"""Extend a base HuggingFace tokenizer with new Chinese BPE pieces.

Strategy:
  1. Train a SentencePiece BPE model on the Chinese side of all training data.
  2. Read the SP vocabulary. Filter to *only* pieces that contain at least one
     Han character — we want to keep the English vocabulary of the base model
     untouched, so we don't add any new Latin/ASCII tokens that might collide
     with the base tokenizer's merges.
  3. Add the filtered pieces to the base HF tokenizer as new tokens.
  4. Add a small set of special control tokens (direction markers, src/tgt).
  5. Save the extended tokenizer.

Returns the count of new tokens added so the caller can size the new
embedding rows accordingly.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

_HAN_RE = re.compile(r"[一-鿿㐀-䶿]")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def train_sentencepiece_on_chinese(
    chinese_text_file: Path,
    out_dir: Path,
    *,
    vocab_size: int = 16000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    byte_fallback: bool = True,
) -> Path:
    """Train a SentencePiece BPE on Chinese-only text. Returns the .model path."""
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = out_dir / "zh_sp"
    spm.SentencePieceTrainer.train(
        input=str(chinese_text_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        byte_fallback=byte_fallback,
        # We do not want sentence-piece to add any of its own special tokens to our vocab;
        # we will merge into the HF tokenizer's existing specials.
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        unk_piece="<unk>",
        normalization_rule_name="nmt_nfkc",
        train_extremely_large_corpus=False,
    )
    return model_prefix.with_suffix(".model")


def _read_sp_pieces(sp_model_path: Path) -> list[str]:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_path))
    return [sp.id_to_piece(i) for i in range(sp.GetPieceSize())]


def _keep_piece(piece: str) -> bool:
    """Keep a SentencePiece piece only if it contains Han and no ASCII letters."""
    # SP uses '▁' as the word-start marker; strip it for content check.
    content = piece.lstrip("▁")
    if not content:
        return False
    if not _HAN_RE.search(content):
        return False
    if _ALPHA_RE.search(content):
        return False
    return True


def extend_tokenizer(
    base_tokenizer_id: str,
    sp_model_path: Path,
    out_dir: Path,
    *,
    special_tokens: list[str],
) -> tuple[int, int]:
    """Append CJK pieces from the SP model to the base HF tokenizer; save extended tokenizer.

    Returns: (n_new_chinese_tokens, n_new_special_tokens)
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_tokenizer_id, use_fast=True)
    base_vocab_size = len(tok)
    base_vocab = set(tok.get_vocab().keys())

    pieces = _read_sp_pieces(sp_model_path)
    candidate = [p for p in pieces if _keep_piece(p) and p not in base_vocab]
    logger.info(
        f"sp pieces total={len(pieces)} kept-after-filter={len(candidate)} "
        f"base_vocab={base_vocab_size}"
    )

    # HF strips SP's '▁' boundary marker before storing — but we DO want byte-fallback-style
    # pieces stored verbatim. Use add_tokens, which dedupes if a piece already exists.
    n_added_chinese = tok.add_tokens(candidate, special_tokens=False)

    # Now add control tokens as *special* tokens so they're not split.
    n_added_special = tok.add_special_tokens(
        {"additional_special_tokens": list(special_tokens)}
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out_dir))
    logger.info(
        f"extended tokenizer saved to {out_dir}  "
        f"(base={base_vocab_size}, +chinese={n_added_chinese}, +specials={n_added_special}, "
        f"final={len(tok)})"
    )
    return n_added_chinese, n_added_special
