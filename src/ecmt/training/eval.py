"""FLORES-200 evaluation: BLEU, chrF++, COMET in both directions.

Designed to be callable both:
  - as a standalone script (scripts/07_evaluate.py) over a checkpoint dir, and
  - from inside the training loop as a TrainerCallback (planned: future work).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from loguru import logger
from tqdm import tqdm


def _build_prompt(src_text: str, direction: str, *, template: str, direction_tokens: dict[str, str]) -> str:
    """Build a *prompt* (no target) for generation. Mirrors formatting.py's training-time format."""
    # We render the template with an empty target so the model generates after "<|tgt|> ".
    return template.format(direction=direction_tokens[direction], src=src_text, tgt="").rstrip()


def _decode_after_tgt_token(text: str, tgt_marker: str = "<|tgt|>") -> str:
    """Strip everything up to and including the <|tgt|> marker from a generated sequence."""
    if tgt_marker in text:
        return text.split(tgt_marker, 1)[1].strip()
    return text.strip()


def generate_translations(
    *,
    model,
    tokenizer,
    rows: list[dict[str, str]],
    direction: str,
    template: str,
    direction_tokens: dict[str, str],
    max_new_tokens: int = 256,
    num_beams: int = 4,
    batch_size: int = 8,
    device: str | None = None,
) -> list[str]:
    """Run generation on a list of {en, zh, ...} rows; return list of hypothesis strings."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    src_key = "en" if direction == "en2zh" else "zh"
    prompts = [_build_prompt(r[src_key], direction, template=template, direction_tokens=direction_tokens) for r in rows]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    hyps: list[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"gen[{direction}]"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=False)
        for d, p in zip(decoded, batch):
            # strip the prompt prefix (left-padded) then everything before <|tgt|>
            d = d[d.find(p) + len(p) :] if p in d else d
            # also handle the case where special tokens got stripped — fall back to last line
            hyps.append(_decode_after_tgt_token(d).split("\n", 1)[0].strip())
    return hyps


def score_translations(
    *,
    refs: list[str],
    hyps: list[str],
    src_lang: str,
    tgt_lang: str,
    sources: list[str] | None = None,
    metrics: list[str],
    comet_model_id: str | None = None,
) -> dict[str, Any]:
    """Score hyps against refs. `sources` (for COMET) are the source-language sentences."""
    import sacrebleu

    result: dict[str, Any] = {"n": len(hyps)}

    if "bleu" in metrics:
        bleu_tok = "zh" if tgt_lang == "zh" else "13a"
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tok)
        result["bleu"] = bleu.score
        result["bleu_signature"] = bleu.get_signature().__str__()

    if "chrf" in metrics:
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
        result["chrf"] = chrf.score
        result["chrf_signature"] = chrf.get_signature().__str__()

    if "comet" in metrics:
        if not comet_model_id:
            logger.warning("comet requested but no model id; skipping")
        elif sources is None:
            logger.warning("comet requested but no source sentences; skipping")
        else:
            try:
                from comet import download_model, load_from_checkpoint  # type: ignore
                path = download_model(comet_model_id)
                cm = load_from_checkpoint(path)
                data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
                pred = cm.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0, progress_bar=False)
                result["comet"] = float(pred.system_score)
            except Exception as e:
                logger.exception(f"comet eval failed: {e!r}")
                result["comet"] = None

    return result


def evaluate_checkpoint(
    *,
    checkpoint_dir: Path,
    flores_dev_path: Path,
    flores_test_path: Path | None,
    template: str,
    direction_tokens: dict[str, str],
    max_new_tokens: int,
    num_beams: int,
    metrics: list[str],
    comet_model_id: str | None,
    on_split: str = "dev",
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src_path = flores_dev_path if on_split == "dev" else flores_test_path
    if src_path is None or not src_path.exists():
        raise FileNotFoundError(f"FLORES {on_split} parquet missing: {src_path}")

    logger.info(f"loading checkpoint {checkpoint_dir}")
    tok = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir), torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    rows = pq.read_table(src_path).to_pylist()
    # FLORES rows are {en, zh, source}; we know this from the loader.
    en_refs = [r["en"] for r in rows]
    zh_refs = [r["zh"] for r in rows]

    summary: dict[str, Any] = {"checkpoint": str(checkpoint_dir), "split": on_split, "n": len(rows)}

    # zh→en
    hyps_en = generate_translations(
        model=model, tokenizer=tok, rows=rows, direction="zh2en",
        template=template, direction_tokens=direction_tokens,
        max_new_tokens=max_new_tokens, num_beams=num_beams,
    )
    summary["zh2en"] = score_translations(
        refs=en_refs, hyps=hyps_en, sources=zh_refs, src_lang="zh", tgt_lang="en",
        metrics=metrics, comet_model_id=comet_model_id,
    )

    # en→zh
    hyps_zh = generate_translations(
        model=model, tokenizer=tok, rows=rows, direction="en2zh",
        template=template, direction_tokens=direction_tokens,
        max_new_tokens=max_new_tokens, num_beams=num_beams,
    )
    summary["en2zh"] = score_translations(
        refs=zh_refs, hyps=hyps_zh, sources=en_refs, src_lang="en", tgt_lang="zh",
        metrics=metrics, comet_model_id=comet_model_id,
    )

    logger.info(f"eval summary: {json.dumps({k: v for k, v in summary.items() if k != 'checkpoint'}, ensure_ascii=False, indent=2)}")
    return summary
