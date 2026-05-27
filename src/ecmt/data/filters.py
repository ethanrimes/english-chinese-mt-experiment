"""Row-level filters for parallel data.

Each filter is a pure function `(row) -> bool` returning True to keep.
The pipeline applies them in order; filters are designed to be cheap-first
so we discard junk before paying for language ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

Row = dict[str, str]

# Unicode ranges that count as "Han" for the Chinese-side script ratio check.
# CJK Unified Ideographs + Extension A + Extension B + compatibility.
_HAN_RE = re.compile(
    r"[一-鿿㐀-䶿\U00020000-\U0002A6DF\U0002A700-\U0002B73F\U0002B740-\U0002B81F豈-﫿]"
)
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_WS_RE = re.compile(r"\s+")


def _word_count(s: str) -> int:
    return len(_WS_RE.split(s.strip())) if s.strip() else 0


def length_filter(min_tokens: int, max_tokens: int, ratio_low: float, ratio_high: float) -> Callable[[Row], bool]:
    def _f(row: Row) -> bool:
        en_n = _word_count(row["en"])
        zh_n = len(row["zh"])  # character-count proxy for Chinese
        if en_n < min_tokens or en_n > max_tokens:
            return False
        if zh_n < min_tokens or zh_n > max_tokens:
            return False
        # rough zh-chars-per-en-word ratio gate
        if en_n == 0:
            return False
        ratio = zh_n / en_n
        if ratio < ratio_low or ratio > ratio_high * 4:  # 4× because zh chars ≈ 1.5-2.5× en words
            return False
        return True

    return _f


def script_ratio_filter(min_zh_han_ratio: float, min_en_ascii_ratio: float) -> Callable[[Row], bool]:
    def _f(row: Row) -> bool:
        zh = row["zh"]
        en = row["en"]
        if not zh or not en:
            return False
        zh_han = len(_HAN_RE.findall(zh))
        if zh_han / max(len(zh), 1) < min_zh_han_ratio:
            return False
        en_letters = len(_ASCII_LETTER_RE.findall(en))
        if en_letters / max(len(en), 1) < min_en_ascii_ratio:
            return False
        return True

    return _f


@dataclass
class LangIdFilter:
    """fasttext-based LID. Lazily loads the model on first call."""

    min_confidence: float = 0.7
    _model: object | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from ftlangdetect import detect  # type: ignore[import-not-found]
            self._detect = detect
            self._model = "ftlangdetect"
        except ImportError as e:
            raise RuntimeError(
                "fasttext-langdetect is required for LID filtering. "
                "Install with `pip install fasttext-langdetect`."
            ) from e

    def __call__(self, row: Row) -> bool:
        self._load()
        try:
            en_pred = self._detect(text=row["en"].replace("\n", " "), low_memory=False)
            zh_pred = self._detect(text=row["zh"].replace("\n", " "), low_memory=False)
        except Exception:
            return False
        if en_pred.get("lang") != "en" or en_pred.get("score", 0.0) < self.min_confidence:
            return False
        # fasttext labels Chinese as 'zh' (sometimes 'zh-cn'); accept any zh*
        zlang = zh_pred.get("lang", "")
        if not zlang.startswith("zh") or zh_pred.get("score", 0.0) < self.min_confidence:
            return False
        return True


def build_filter_chain(filters_cfg: dict) -> list[Callable[[Row], bool]]:
    """Compose the default filter chain from a config dict (configs/data.yaml -> filters)."""
    chain: list[Callable[[Row], bool]] = [
        length_filter(
            min_tokens=int(filters_cfg["min_tokens"]),
            max_tokens=int(filters_cfg["max_tokens"]),
            ratio_low=float(filters_cfg["ratio_low"]),
            ratio_high=float(filters_cfg["ratio_high"]),
        ),
        script_ratio_filter(
            min_zh_han_ratio=float(filters_cfg["min_zh_han_ratio"]),
            min_en_ascii_ratio=float(filters_cfg["min_en_ascii_ratio"]),
        ),
        LangIdFilter(min_confidence=float(filters_cfg["lid_confidence"])),
    ]
    return chain


def apply_chain(row: Row, chain: list[Callable[[Row], bool]]) -> bool:
    for f in chain:
        if not f(row):
            return False
    return True
