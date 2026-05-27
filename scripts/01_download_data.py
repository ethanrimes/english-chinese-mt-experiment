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
    fname = cfg["url"].rsplit("/", 1)[-1]
    dest = raw_root / "news_commentary_v18" / fname
    http_download(cfg["url"], dest)


def download_moses_zip(source_id: str, cfg: dict, raw_root: Path) -> None:
    url = cfg["url"]
    fname = url.rsplit("/", 1)[-1]
    dest = raw_root / source_id / fname
    http_download(url, dest)


def download_flores200(cfg: dict, raw_root: Path) -> None:
    """FLORES-200 via Facebook's public NLLB tarball.

    The tarball flattens to flores200_dataset/{dev,devtest}/{lang_code}.{split}
    with one sentence per line. We extract eng_Latn and zho_Hans, zip them
    line-by-line (FLORES guarantees same row count + alignment across languages),
    and save dev + devtest parquet for fast reload.
    """
    import tarfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = raw_root / "flores200"
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / "flores200_dataset.tar.gz"
    http_download(cfg["url"], tarball)

    # Extract the four files we need. Tar members may be prefixed with './'.
    needed = {f"flores200_dataset/{s}/{L}.{s}" for s in cfg["splits"] for L in ("eng_Latn", "zho_Hans")}
    extracted: dict[str, list[str]] = {}
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            normalized = member.name.lstrip("./")
            if normalized in needed:
                f = tf.extractfile(member)
                if f is None:
                    continue
                extracted[normalized] = f.read().decode("utf-8").splitlines()

    for split in cfg["splits"]:
        out_path = out_dir / f"{split}.parquet"
        if out_path.exists():
            logger.info(f"  flores200/{split}: parquet exists — skipping write")
            continue
        en_lines = extracted.get(f"flores200_dataset/{split}/eng_Latn.{split}", [])
        zh_lines = extracted.get(f"flores200_dataset/{split}/zho_Hans.{split}", [])
        if not en_lines or not zh_lines:
            logger.error(f"  flores200/{split}: missing extracted lang file(s)")
            continue
        if len(en_lines) != len(zh_lines):
            logger.warning(f"  flores200/{split}: en={len(en_lines)} zh={len(zh_lines)} mismatch; truncating")
        n = min(len(en_lines), len(zh_lines))
        rows = [{"eng_Latn": en_lines[i].strip(), "zho_Hans": zh_lines[i].strip(), "id": i} for i in range(n)]
        pq.write_table(pa.Table.from_pylist(rows), out_path)
        logger.info(f"  flores200/{split}: wrote {len(rows)} rows to {out_path}")


def download_ted2020_opus100(cfg: dict, raw_root: Path) -> None:
    """opus-100 zh-en train+validation. We pull the parquet files directly via
    huggingface_hub (current versions of `datasets` no longer support the
    loading-script that ships with this dataset)."""
    from huggingface_hub import hf_hub_download

    out_dir = raw_root / "ted2020"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg["huggingface"]            # Helsinki-NLP/opus-100
    cfg_name = cfg["config"]              # 'en-zh'
    for split in cfg["splits"]:
        out_path = out_dir / f"{split}.parquet"
        if out_path.exists():
            logger.info(f"  ted2020/{split}: parquet exists — skipping")
            continue
        remote = f"{cfg_name}/{split}-00000-of-00001.parquet"
        logger.info(f"  ted2020/{split}: hf_hub_download {repo!r} {remote!r}")
        local = hf_hub_download(repo_id=repo, filename=remote, repo_type="dataset")
        import shutil
        shutil.copyfile(local, out_path)
        # Quick row count
        try:
            import pyarrow.parquet as pq
            n = pq.read_metadata(out_path).num_rows
            logger.info(f"  ted2020/{split}: wrote parquet with {n:,} rows")
        except Exception:
            logger.info(f"  ted2020/{split}: wrote parquet (size unknown)")


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
