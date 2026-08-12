"""Atomic parquet checkpointing + the one enforced month-key derivation, so
chunked pipelines (Phase 6 restructure) can resume after a kill without
redoing completed months and never chunk by the ambiguous Int8 `month`
column (it collides Dec-2024 with Dec-2025 in panel.parquet)."""
from __future__ import annotations

from pathlib import Path

import polars as pl


def month_key_expr(col: str = "interval_start") -> pl.Expr:
    return pl.col(col).dt.strftime("%Y-%m")


def write_checkpoint(df: pl.DataFrame, path: Path) -> None:
    """Writes to a .tmp sibling then renames into place -- a kill mid-write
    never leaves a checkpoint file that `exists()` but is truncated, which a
    naive skip-if-exists resume check would otherwise trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def checkpoint_path(directory: Path, month_key: str) -> Path:
    return directory / f"{month_key}.parquet"


def is_checkpointed(directory: Path, month_key: str) -> bool:
    return checkpoint_path(directory, month_key).exists()
