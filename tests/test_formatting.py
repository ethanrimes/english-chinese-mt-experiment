"""Tests for src/ecmt/data/formatting.py — prompt rendering and bidir expansion."""

from __future__ import annotations

from ecmt.data.formatting import expand_to_bidirectional, format_example

TEMPLATE = "{direction}\n<|src|> {src} <|tgt|> {tgt}"
TOKENS = {"en2zh": "<|en2zh|>", "zh2en": "<|zh2en|>"}


def test_format_en2zh():
    ex = format_example(
        {"en": "Hello world", "zh": "你好世界", "source": "test"},
        direction="en2zh",
        template=TEMPLATE,
        direction_tokens=TOKENS,
    )
    assert ex["direction"] == "en2zh"
    assert "Hello world" in ex["text"]
    assert "你好世界" in ex["text"]
    assert "<|en2zh|>" in ex["text"]
    # source side comes before target side
    assert ex["text"].index("Hello world") < ex["text"].index("你好世界")


def test_format_zh2en():
    ex = format_example(
        {"en": "Hello world", "zh": "你好世界", "source": "test"},
        direction="zh2en",
        template=TEMPLATE,
        direction_tokens=TOKENS,
    )
    assert ex["direction"] == "zh2en"
    assert ex["text"].index("你好世界") < ex["text"].index("Hello world")


def test_bidirectional_expansion_doubles_rows():
    rows = [
        {"en": "one", "zh": "一", "source": "x"},
        {"en": "two", "zh": "二", "source": "x"},
    ]
    out = list(expand_to_bidirectional(rows, template=TEMPLATE, direction_tokens=TOKENS))
    assert len(out) == 4
    # both directions present
    dirs = sorted(o["direction"] for o in out)
    assert dirs == ["en2zh", "en2zh", "zh2en", "zh2en"]
