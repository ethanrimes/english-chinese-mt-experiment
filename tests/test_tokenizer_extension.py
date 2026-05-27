"""Smoke test for tokenizer extension's `_keep_piece` filter — we want to be
sure we don't accidentally inject any Latin-letter pieces into the base
SmolLM2 vocabulary, since that would corrupt the English-side tokenization.
"""

from __future__ import annotations

from ecmt.model.tokenizer_extension import _keep_piece


def test_keep_pure_chinese():
    assert _keep_piece("你好") is True
    assert _keep_piece("▁世界") is True   # SP word-start marker
    assert _keep_piece("汉字") is True


def test_reject_pure_ascii():
    assert _keep_piece("hello") is False
    assert _keep_piece("the") is False
    assert _keep_piece("▁is") is False


def test_reject_code_switched():
    # Mixed pieces are dangerous — could collide with English vocab merges.
    assert _keep_piece("hello你好") is False
    assert _keep_piece("iPhone手机") is False


def test_reject_empty():
    assert _keep_piece("") is False
    assert _keep_piece("▁") is False


def test_reject_punct_or_digit_only():
    assert _keep_piece("123") is False
    assert _keep_piece("...") is False
