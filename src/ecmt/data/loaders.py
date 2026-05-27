"""Source-specific loaders.

Each loader reads a raw download (whatever format the source ships in) and
yields uniform `(en, zh, source_id)` rows that the rest of the pipeline can
consume.

Adding a new source:
  1. Add a `download_*` function (in scripts/01_download_data.py) that fetches the bytes.
  2. Add a `load_*` function here that decodes them into the unified schema.
  3. Register both in `LOADERS` below.
"""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

UNIFIED_SCHEMA = ("en", "zh", "source")


def _strip_control(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s).strip()


def load_news_commentary_v18(raw_dir: Path) -> Iterator[dict[str, str]]:
    """News-Commentary v18: TSV.gz with columns en\tzh."""
    src_path = raw_dir / "news_commentary_v18" / "news-commentary-v18.1.en-zh.tsv.gz"
    with gzip.open(src_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            en, zh = parts
            en = _strip_control(en)
            zh = _strip_control(zh)
            if not en or not zh:
                continue
            yield {"en": en, "zh": zh, "source": "news_commentary_v18"}


def load_moses_zip(
    raw_dir: Path,
    source_id: str,
    expected_en: str,
    expected_zh: str,
    sample_lines: int | None = None,
) -> Iterator[dict[str, str]]:
    """Generic Moses-format loader: a .zip with parallel .en and .zh text files."""
    zip_path = next((raw_dir / source_id).glob("*.zip"), None)
    if zip_path is None:
        logger.warning(f"{source_id}: no .zip found in {raw_dir / source_id}; skipping")
        return
    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        if expected_en not in names or expected_zh not in names:
            logger.warning(
                f"{source_id}: expected files {expected_en}/{expected_zh} not in zip "
                f"(have {sorted(names)}); skipping"
            )
            return
        with zf.open(names[expected_en]) as efh, zf.open(names[expected_zh]) as zfh:
            etxt = io.TextIOWrapper(efh, encoding="utf-8", errors="replace")
            ztxt = io.TextIOWrapper(zfh, encoding="utf-8", errors="replace")
            for i, (en, zh) in enumerate(zip(etxt, ztxt)):
                if sample_lines is not None and i >= sample_lines:
                    break
                en = _strip_control(en)
                zh = _strip_control(zh)
                if not en or not zh:
                    continue
                yield {"en": en, "zh": zh, "source": source_id}


def load_ted2020_opus100(raw_dir: Path) -> Iterator[dict[str, str]]:
    """opus-100 en-zh split (HF dataset, saved to parquet by the downloader)."""
    import pyarrow.parquet as pq

    pq_path = raw_dir / "ted2020" / "train.parquet"
    if not pq_path.exists():
        logger.warning(f"ted2020: {pq_path} not found; skipping")
        return
    table = pq.read_table(pq_path)
    # opus-100 stores rows as {"translation": {"en": ..., "zh": ...}} so we
    # normalize both possible layouts.
    if "translation" in table.column_names:
        for row in table.to_pylist():
            tr = row["translation"]
            en = _strip_control(tr.get("en", ""))
            zh = _strip_control(tr.get("zh", ""))
            if en and zh:
                yield {"en": en, "zh": zh, "source": "ted2020"}
    else:
        for row in table.to_pylist():
            en = _strip_control(row.get("en", ""))
            zh = _strip_control(row.get("zh", ""))
            if en and zh:
                yield {"en": en, "zh": zh, "source": "ted2020"}


def load_flores200(raw_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Load FLORES-200 dev + devtest. Returns {split: [rows]} — for eval only."""
    import pyarrow.parquet as pq

    out: dict[str, list[dict[str, str]]] = {}
    for split in ("dev", "devtest"):
        pq_path = raw_dir / "flores200" / f"{split}.parquet"
        if not pq_path.exists():
            logger.warning(f"flores200: {pq_path} not found; skipping {split}")
            continue
        tbl = pq.read_table(pq_path)
        rows = tbl.to_pylist()
        # FLORES rows shape varies by HF dataset version; normalize.
        normed: list[dict[str, str]] = []
        for r in rows:
            en = r.get("eng_Latn") or r.get("en") or r.get("sentence_eng_Latn")
            zh = r.get("zho_Hans") or r.get("zh") or r.get("sentence_zho_Hans")
            if en and zh:
                normed.append({"en": _strip_control(en), "zh": _strip_control(zh), "source": "flores200"})
        out[split] = normed
        logger.info(f"flores200/{split}: loaded {len(normed)} rows")
    return out


# Map source_id -> loader callable taking (raw_dir, source_cfg) and yielding rows.
LOADERS: dict[str, Any] = {
    "news_commentary_v18": lambda raw_dir, cfg: load_news_commentary_v18(raw_dir),
    "ted2020": lambda raw_dir, cfg: load_ted2020_opus100(raw_dir),
    "wikimatrix": lambda raw_dir, cfg: load_moses_zip(
        raw_dir, "wikimatrix",
        cfg["expected_pair_files"]["en"],
        cfg["expected_pair_files"]["zh"],
        cfg.get("sample_lines"),
    ),
    "open_subtitles": lambda raw_dir, cfg: load_moses_zip(
        raw_dir, "open_subtitles",
        cfg["expected_pair_files"]["en"],
        cfg["expected_pair_files"]["zh"],
        cfg.get("sample_lines"),
    ),
    "un_pc": lambda raw_dir, cfg: load_moses_zip(
        raw_dir, "un_pc",
        cfg["expected_pair_files"]["en"],
        cfg["expected_pair_files"]["zh"],
        cfg.get("sample_lines"),
    ),
}
