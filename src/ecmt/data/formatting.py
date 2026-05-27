"""Build training examples from parallel rows.

Each parallel row produces TWO training examples (one per direction) so the
model learns both en→zh and zh→en from the same data.

The prompt template lives in configs/training.yaml so it can be tweaked without
code changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def format_example(row: dict[str, str], direction: str, *, template: str, direction_tokens: dict[str, str]) -> dict[str, Any]:
    """Render one training example.

    `row` has fields {en, zh, source}. `direction` is 'en2zh' or 'zh2en'.
    Returns a dict with at least a "text" field, ready for SFTTrainer.
    """
    if direction == "en2zh":
        src, tgt = row["en"], row["zh"]
    elif direction == "zh2en":
        src, tgt = row["zh"], row["en"]
    else:
        raise ValueError(f"unknown direction: {direction}")
    text = template.format(direction=direction_tokens[direction], src=src, tgt=tgt)
    return {
        "text": text,
        "direction": direction,
        "source": row.get("source", "unknown"),
    }


def expand_to_bidirectional(
    rows: Iterable[dict[str, str]],
    *,
    template: str,
    direction_tokens: dict[str, str],
) -> Iterable[dict[str, Any]]:
    for r in rows:
        yield format_example(r, "en2zh", template=template, direction_tokens=direction_tokens)
        yield format_example(r, "zh2en", template=template, direction_tokens=direction_tokens)
