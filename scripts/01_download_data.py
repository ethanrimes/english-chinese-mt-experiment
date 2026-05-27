"""Download all parallel corpora declared in configs/data.yaml.

Each source has its own download mechanism (HF dataset vs HTTP archive). This
script is deliberately not parallel — we want clean error messages per source
and most users will run it once.

Usage:
    python scripts/01_download_data.py --profile smoke
    python scripts/01_download_data.py --profile full
    python scripts/01_download_data.py --only news_commentary_v18,flores200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
from loguru import logger

# Make `src/` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402

CHUNK = 1024 * 1024  # 1 MiB


def http_download(url: str, dest: Path, *, retries: int = 3) -> None:
    """Streamed HTTP download with a tiny retry loop."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  exists: {dest.name} ({dest.stat().st_size/1e6:.1f} MB) — skipping")
        return
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                logger.info(f"  GET {url}  ({total/1e6:.1f} MB)" if total else f"  GET {url}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:
            last_err = e
            logger.warning(f"  attempt {attempt}/{retries} failed: {e!r}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"download failed: {url}") from last_err


def download_news_commentary_v18(cfg: dict, raw_root: Path) -> None:
    dest = raw_root / "news_commentary_v18" / "news-commentary-v18.1.en-zh.tsv.gz"
    http_download(cfg["url"], dest)


def download_moses_zip(source_id: str, cfg: dict, raw_root: Path) -> None:
    url = cfg["url"]
    fname = url.rsplit("/", 1)[-1]
    dest = raw_root / source_id / fname
    http_download(url, dest)


def download_flores200(cfg: dict, raw_root: Path) -> None:
    """FLORES-200 via HF. We save dev + devtest to parquet for fast reload."""
    from datasets import load_dataset

    out_dir = raw_root / "flores200"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in cfg["splits"]:
        out_path = out_dir / f"{split}.parquet"
        if out_path.exists():
            logger.info(f"  flores200/{split}: parquet exists — skipping")
            continue
        # FLORES+ on HF has each lang as a separate config; we load and zip the two we need.
        logger.info(f"  flores200/{split}: loading eng_Latn + zho_Hans from HF")
        try:
            en_ds = load_dataset(cfg["huggingface"], "eng_Latn", split=split)
            zh_ds = load_dataset(cfg["huggingface"], "zho_Hans", split=split)
        except Exception as e:
            logger.error(f"  flores200/{split}: HF load failed: {e!r}")
            raise
        assert len(en_ds) == len(zh_ds), "flores200 en/zh row counts must match"
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            {"eng_Latn": en_row["text"], "zho_Hans": zh_row["text"], "id": en_row.get("id", i)}
            for i, (en_row, zh_row) in enumerate(zip(en_ds, zh_ds))
        ]
        pq.write_table(pa.Table.from_pylist(rows), out_path)
        logger.info(f"  flores200/{split}: wrote {len(rows)} rows to {out_path}")


def download_ted2020_opus100(cfg: dict, raw_root: Path) -> None:
    """We pull opus-100 zh-en train+validation and save to parquet."""
    from datasets import load_dataset

    out_dir = raw_root / "ted2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in cfg["splits"]:
        out_path = out_dir / f"{split}.parquet"
        if out_path.exists():
            logger.info(f"  ted2020/{split}: parquet exists — skipping")
            continue
        logger.info(f"  ted2020/{split}: loading from HF opus-100 en-zh")
        ds = load_dataset(cfg["huggingface"], cfg["config"], split=split)
        ds.to_parquet(out_path)
        logger.info(f"  ted2020/{split}: wrote {len(ds)} rows to {out_path}")


DOWNLOADERS = {
    "news_commentary_v18": download_news_commentary_v18,
    "wikimatrix":          lambda c, r: download_moses_zip("wikimatrix", c, r),
    "open_subtitles":      lambda c, r: download_moses_zip("open_subtitles", c, r),
    "un_pc":               lambda c, r: download_moses_zip("un_pc", c, r),
    "ccmatrix":            lambda c, r: download_moses_zip("ccmatrix", c, r),
    "flores200":           download_flores200,
    "ted2020":             download_ted2020_opus100,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--profile", default="smoke", help="profile defined in data.yaml")
    ap.add_argument("--only", default=None, help="comma-separated source IDs (overrides profile)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.data_config)
    raw_root = Path(cfg.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        wanted = list(cfg.profiles[args.profile].include)

    logger.info(f"profile={args.profile!r}  sources={wanted}")
    failures: list[tuple[str, Exception]] = []
    for src in wanted:
        if src not in cfg.sources:
            logger.error(f"unknown source {src!r} (not in {args.data_config})")
            continue
        if src not in DOWNLOADERS:
            logger.error(f"no downloader registered for {src!r}")
            continue
        logger.info(f"=== {src} ===")
        try:
            DOWNLOADERS[src](cfg.sources[src], raw_root)
        except Exception as e:
            logger.exception(f"  {src}: failed")
            failures.append((src, e))

    if failures:
        logger.error(f"{len(failures)} source(s) failed: {[s for s, _ in failures]}")
        return 1
    logger.info("all sources downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
