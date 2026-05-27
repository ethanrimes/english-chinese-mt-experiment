"""Tests for src/ecmt/data/filters.py — the most likely source of silent data loss."""

from __future__ import annotations

from ecmt.data.filters import length_filter, script_ratio_filter


def test_length_filter_basic():
    f = length_filter(min_tokens=3, max_tokens=200, ratio_low=0.5, ratio_high=2.0)
    assert f({"en": "this is a sentence", "zh": "这是一句话"}) is True
    assert f({"en": "hi", "zh": "你好"}) is False             # too short en
    assert f({"en": "a " * 250, "zh": "x" * 5}) is False       # too long en


def test_length_filter_ratio():
    f = length_filter(min_tokens=3, max_tokens=200, ratio_low=0.5, ratio_high=2.0)
    # 10 en words, only 1 zh char — way too small zh side
    assert f({"en": "one two three four five six seven eight nine ten", "zh": "a"}) is False


def test_script_ratio_chinese():
    f = script_ratio_filter(min_zh_han_ratio=0.3, min_en_ascii_ratio=0.8)
    # Good case: clean en + clean zh.
    assert f({"en": "Hello world how are you", "zh": "你好世界你好吗"}) is True
    # zh side is pure ASCII — should be rejected.
    assert f({"en": "Hello", "zh": "hello world"}) is False
    # en side has too few ASCII letters (mostly punctuation).
    assert f({"en": "!!! @#$ %%% &&&", "zh": "你好世界"}) is False


def test_script_ratio_mixed_zh():
    f = script_ratio_filter(min_zh_han_ratio=0.3, min_en_ascii_ratio=0.8)
    # zh sentence with some English code-switching but still >=30% Han.
    assert f({"en": "He uses an iPhone now", "zh": "他现在用iPhone手机"}) is True
