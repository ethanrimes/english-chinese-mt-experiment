"""Config loading + dotted-key overrides.

The training/sweep configs are YAML; we use OmegaConf so that hierarchical keys
like `trainer.learning_rate` can be overridden from the CLI or from a sweep
file without bespoke parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected mapping at top level of {path}, got {type(cfg).__name__}")
    return cfg


def apply_overrides(cfg: DictConfig, overrides: dict[str, Any] | None) -> DictConfig:
    """Apply a dict of dotted-key overrides onto cfg, returning a new config."""
    if not overrides:
        return cfg
    merged = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    if not isinstance(merged, DictConfig):
        raise TypeError("Override merge produced a non-mapping config")
    return merged


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
