"""Logging setup — one place to configure loguru for scripts and the training run.

Each training run gets a rotating file logger under its run dir, plus stdout.
Optional JSONL metrics sink for downstream analysis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(
    run_dir: str | Path | None = None,
    *,
    level: str = "INFO",
    structured_metrics: bool = True,
) -> None:
    """Initialize loguru handlers. Idempotent — safe to call from any script."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        enqueue=False,
    )

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            run_dir / "train.log",
            level=level,
            rotation="50 MB",
            retention=10,
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )
        if structured_metrics:
            metrics_path = run_dir / "metrics.jsonl"
            _MetricsSink.attach(metrics_path)


class _MetricsSink:
    """Sink that filters records bound with extra={'metrics': True} to JSONL."""

    _attached_path: Path | None = None

    @classmethod
    def attach(cls, path: Path) -> None:
        cls._attached_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            cls._write,
            filter=lambda record: record["extra"].get("metrics") is True,
            level="DEBUG",
            enqueue=True,
        )

    @classmethod
    def _write(cls, msg: Any) -> None:
        if cls._attached_path is None:
            return
        record = msg.record
        payload = {
            "ts": record["time"].isoformat(),
            "msg": record["message"],
            **{k: v for k, v in record["extra"].items() if k != "metrics"},
        }
        with cls._attached_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_metric(step: int, **kwargs: Any) -> None:
    """Emit a structured metric record. Picked up by the JSONL sink."""
    logger.bind(metrics=True, step=step, **kwargs).info(f"step={step} {kwargs}")
