"""Deduplication.

Two-stage:
  1. Exact dedup on normalized (en, zh) pair using xxhash for speed.
  2. Eval-leakage dedup: any pair whose normalized en or zh matches a FLORES-200
     dev/devtest sentence is dropped, no matter the source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import xxhash

_WS_NORM = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_NORM.sub(" ", s.strip().lower())


def _pair_hash(en: str, zh: str) -> int:
    h = xxhash.xxh64()
    h.update(_norm(en).encode("utf-8"))
    h.update(b"\x00")
    h.update(_norm(zh).encode("utf-8"))
    return h.intdigest()


def _side_hash(s: str) -> int:
    return xxhash.xxh64(_norm(s).encode("utf-8")).intdigest()


class Deduper:
    def __init__(self, eval_pool: Iterable[dict[str, str]] | None = None):
        self._seen_pair: set[int] = set()
        self._eval_en: set[int] = set()
        self._eval_zh: set[int] = set()
        if eval_pool is not None:
            for r in eval_pool:
                self._eval_en.add(_side_hash(r["en"]))
                self._eval_zh.add(_side_hash(r["zh"]))

    def keep(self, row: dict[str, str]) -> bool:
        h = _pair_hash(row["en"], row["zh"])
        if h in self._seen_pair:
            return False
        if _side_hash(row["en"]) in self._eval_en:
            return False
        if _side_hash(row["zh"]) in self._eval_zh:
            return False
        self._seen_pair.add(h)
        return True
