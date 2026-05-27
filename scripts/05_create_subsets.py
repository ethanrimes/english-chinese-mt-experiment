"""Materialize the data-scaling subsets.

Reads `data/processed/all_pairs.parquet` and writes one parquet per scale:

    data/splits/scale_10000.parquet     (strict prefix subset of scale_50000)
    data/splits/scale_50000.parquet     (strict prefix subset of scale_100000)
    data/splits/scale_100000.parquet    (strict prefix subset of scale_500000)
    data/splits/scale_500000.parquet

Stratification:
  Each scale draws from sources in the proportions of `source_mix` in
  configs/data.yaml. Within each source we sort by a stable per-row hash so
  the smaller scales are *exact prefixes* of the larger ones — a controlled
  experiment requires this.

Bidirectional expansion (en2zh + zh2en duplication) happens at training time,
not here — keeping the parquet single-direction makes the file half the size
and lets the trainer apply on-the-fly augmentations later if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecmt.utils.config import load_config  # noqa: E402
from ecmt.utils.logging_setup import setup_logging  # noqa: E402


def _stable_hash(row: dict, seed: int) -> int:
    h = xxhash.xxh64(seed=seed)
    h.update(row["en"].encode("utf-8"))
    h.update(b"\x00")
    h.update(row["zh"].encode("utf-8"))
    return h.intdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--sizes", default="10000,50000,100000,500000,1000000,5000000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.data_config)
    processed = Path(cfg.processed_root) / "all_pairs.parquet"
    splits = Path(cfg.splits_root)
    splits.mkdir(parents=True, exist_ok=True)

    if not processed.exists():
        logger.error(f"{processed} not found; run scripts/02_prepare_data.py first")
        return 1

    sizes = sorted({int(s) for s in args.sizes.split(",")})
    biggest = sizes[-1]

    table = pq.read_table(processed)
    all_rows = table.to_pylist()
    logger.info(f"loaded {len(all_rows)} unified pairs from {processed}")

    by_source: dict[str, list[dict]] = {}
    for r in all_rows:
        by_source.setdefault(r["source"], []).append(r)

    # Pick the mix based on the largest requested scale. The same mix is used
    # for *all* requested scales in a single call (so nesting holds even when
    # scales straddle the threshold — train all-small or all-large together).
    use_large = biggest >= int(cfg.get("large_scale_threshold", 750_000))
    mix_cfg_key = "source_mix_large_scale" if use_large else "source_mix"
    if mix_cfg_key not in cfg:
        logger.warning(f"{mix_cfg_key} not found in data.yaml; falling back to source_mix")
        mix_cfg_key = "source_mix"
    mix = dict(cfg[mix_cfg_key])
    logger.info(f"using mix '{mix_cfg_key}' for biggest scale {biggest:,}")
    # Drop sources that have no mix weight (e.g., flores200 if it sneaks in).
    by_source = {s: rs for s, rs in by_source.items() if s in mix}

    # Sort each source deterministically so the smaller scales are exact prefixes.
    for s in by_source:
        by_source[s].sort(key=lambda r: _stable_hash(r, args.seed))
        logger.info(f"  source {s}: {len(by_source[s]):,} eligible rows")

    # Plan source allocations. First pass: weight × biggest. Second pass: any
    # source that wants more than it has gets capped, and its shortfall is
    # redistributed across sources that still have headroom (proportional to
    # their remaining capacity). This preserves the source diversity intent
    # while actually using available data.
    allocations: dict[str, int] = {}
    headroom: dict[str, int] = {}
    shortfall = 0
    for src, weight in mix.items():
        want = int(round(weight * biggest))
        if src not in by_source:
            # Absent sources still create shortfall — we want their share filled
            # from whoever has headroom (typically UN at large scales).
            logger.warning(f"  source {src} declared in mix but absent from data; allocating 0; shortfall +{want:,}")
            allocations[src] = 0
            shortfall += want
            continue
        avail = len(by_source[src])
        if want > avail:
            shortfall += want - avail
            allocations[src] = avail
            logger.info(f"  {src}: cap at {avail:,} (wanted {want:,}); shortfall +{want - avail:,}")
        else:
            allocations[src] = want
            headroom[src] = avail - want

    # Distribute shortfall to sources with remaining capacity, proportional to that capacity.
    while shortfall > 0 and headroom:
        total_room = sum(headroom.values())
        if total_room == 0:
            break
        distributed_this_round = 0
        for src, room in list(headroom.items()):
            if shortfall <= 0:
                break
            extra = min(room, max(1, (room * shortfall) // total_room))
            allocations[src] += extra
            headroom[src] -= extra
            if headroom[src] == 0:
                del headroom[src]
            shortfall -= extra
            distributed_this_round += extra
        if distributed_this_round == 0:
            break

    big_pool: list[dict] = []
    for src, n in allocations.items():
        if n == 0 or src not in by_source:
            continue
        big_pool.extend(by_source[src][:n])
        weight_pct = (mix.get(src, 0.0)) * 100.0
        actual_pct = (n / biggest) * 100.0 if biggest else 0.0
        logger.info(
            f"  {src}: contributing {n:,} rows  (mix-weight {weight_pct:.1f}%, actual {actual_pct:.1f}% of {biggest:,})"
        )

    # Final shuffle of the biggest pool, deterministic and consistent across calls.
    big_pool.sort(key=lambda r: _stable_hash(r, args.seed + 1))

    if len(big_pool) < biggest:
        logger.warning(
            f"biggest requested scale is {biggest:,} but only {len(big_pool):,} pairs available "
            "across sources — downscaling targets accordingly"
        )

    # Write each scale as a strict prefix.
    for n in sizes:
        actual = min(n, len(big_pool))
        if actual < n:
            logger.warning(f"  scale {n:,}: only {actual:,} pairs available")
        out = splits / f"scale_{n}.parquet"
        pq.write_table(pa.Table.from_pylist(big_pool[:actual]), out)
        logger.info(f"  wrote {out.name}: {actual:,} pairs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
