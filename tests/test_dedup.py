"""Tests for src/ecmt/data/dedup.py."""

from __future__ import annotations

from ecmt.data.dedup import Deduper


def test_pair_dedup():
    d = Deduper()
    assert d.keep({"en": "Hello", "zh": "你好"}) is True
    assert d.keep({"en": "Hello", "zh": "你好"}) is False  # exact dup
    # whitespace/case normalization
    assert d.keep({"en": "  hello ", "zh": "你好"}) is False


def test_eval_leakage_block():
    eval_pool = [{"en": "This is a FLORES sentence.", "zh": "这是FLORES句子。"}]
    d = Deduper(eval_pool=eval_pool)
    # Should reject any training pair that matches en or zh of an eval row.
    assert d.keep({"en": "this is a flores sentence.", "zh": "完全不同"}) is False
    assert d.keep({"en": "totally different", "zh": "这是flores句子。"}) is False
    # Unrelated pair passes.
    assert d.keep({"en": "totally different", "zh": "完全不同"}) is True
