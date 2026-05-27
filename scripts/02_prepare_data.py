"""Filter, dedupe, and unify all downloaded sources into a single parquet table.

Pipeline per row:
  raw -> loader -> filter chain (length, script, lang-id) -> deduper -> write

Output:
  data/processed/all_pairs.parquet     # all kept pairs with a 'source' column
  data/processed/source_stats.json     # per-source kept/dropped counters
  data/splits/dev.parquet              # 5K held-out pairs for validation
  data/splits/test_flores_devtest.parquet  # FLORES devtest, for final eval
  data/splits/english_probe.parquet    # English-only probe for forgetting monitor
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.data.dedup import Deduper  # noqa: E402
from ecmt.data.filters import apply_chain, build_filter_chain  # noqa: E402
from ecmt.data.loaders import LOADERS, load_flores200  # noqa: E402
from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402

DEV_PAIRS = 5000
ENG_PROBE_PAIRS = 5000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--only", default=None, help="comma-separated source IDs")
    ap.add_argument("--no-lid", action="store_true", help="skip language-ID filter (faster, lower quality)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.data_config)
    raw_root = Path(cfg.raw_root)
    processed_root = Path(cfg.processed_root)
    splits_root = Path(cfg.splits_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        wanted = list(cfg.profiles[args.profile].include)

    # FLORES first — used both as eval set and as the dedup blocklist.
    flores = {}
    if "flores200" in wanted:
        logger.info("=== flores200: loading for eval + dedup blocklist ===")
        flores = load_flores200(raw_root)
        if "dev" in flores:
            pq.write_table(
                pa.Table.from_pylist(flores["dev"]),
                splits_root / "dev_flores_dev.parquet",
            )
        if "devtest" in flores:
            pq.write_table(
                pa.Table.from_pylist(flores["devtest"]),
                splits_root / "test_flores_devtest.parquet",
            )

    eval_pool = (flores.get("dev", []) + flores.get("devtest", [])) if flores else None
    deduper = Deduper(eval_pool=eval_pool)

    # Build filter chain (LID is the expensive one; can be disabled for smoke tests).
    filter_cfg = dict(cfg.filters)
    chain = build_filter_chain(filter_cfg) if not args.no_lid else build_filter_chain({**filter_cfg, "lid_confidence": 0.0})
    if args.no_lid:
        # drop the LangIdFilter (last in chain)
        from ecmt.data.filters import LangIdFilter
        chain = [f for f in chain if not isinstance(f, LangIdFilter)]
        logger.warning("LID filter disabled (--no-lid)")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "kept": 0, "dropped_filter": 0, "dropped_dedup": 0})
    kept_rows: list[dict] = []

    for src in wanted:
        if src == "flores200":
            continue
        if src not in LOADERS:
            logger.warning(f"no loader for {src!r}; skipping")
            continue
        logger.info(f"=== {src}: load + filter + dedup ===")
        loader = LOADERS[src]
        try:
            it = loader(raw_root, cfg.sources[src])
        except Exception as e:
            logger.exception(f"  {src}: loader failed: {e!r}")
            continue
        for row in tqdm(it, desc=src, unit=" rows"):
            stats[src]["in"] += 1
            if not apply_chain(row, chain):
                stats[src]["dropped_filter"] += 1
                continue
            if not deduper.keep(row):
                stats[src]["dropped_dedup"] += 1
                continue
            kept_rows.append(row)
            stats[src]["kept"] += 1

    if not kept_rows:
        logger.error("no rows kept — aborting before write")
        return 1

    # Shuffle deterministically and carve out dev + english probe before writing main.
    rng = random.Random(args.seed)
    rng.shuffle(kept_rows)

    n_dev = min(DEV_PAIRS, max(0, len(kept_rows) - 100))
    dev_rows = kept_rows[:n_dev]
    rest = kept_rows[n_dev:]

    # English-only probe for catastrophic-forgetting monitor: just the English side of the
    # next slice. Saving as parquet with an 'en' column is enough for PPL eval.
    n_probe = min(ENG_PROBE_PAIRS, max(0, len(rest) - 100))
    probe_rows = [{"en": r["en"]} for r in rest[:n_probe]]
    rest = rest[n_probe:]

    logger.info(f"writing {len(rest)} rows -> processed/all_pairs.parquet")
    pq.write_table(pa.Table.from_pylist(rest), processed_root / "all_pairs.parquet")
    pq.write_table(pa.Table.from_pylist(dev_rows), splits_root / "dev.parquet")
    pq.write_table(pa.Table.from_pylist(probe_rows), splits_root / "english_probe.parquet")

    stats_path = processed_root / "source_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(dict(stats), fh, indent=2, ensure_ascii=False)
    logger.info(f"per-source stats -> {stats_path}")

    grand_kept = sum(s["kept"] for s in stats.values())
    grand_in = sum(s["in"] for s in stats.values())
    logger.info(f"summary: kept {grand_kept} / {grand_in} pairs ({grand_kept / max(grand_in, 1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
