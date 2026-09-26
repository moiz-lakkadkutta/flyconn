"""Shared helpers for converters: provenance records and Parquet writing."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import flyconn
from flyconn.data.download import sha256_file


def input_record(path: Path) -> dict[str, Any]:
    """Describe one input file (name, size, SHA-256) for provenance."""
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_provenance(
    out_dir: Path,
    *,
    dataset: str,
    version: str,
    converter: str,
    inputs: list[Path],
    counts: dict[str, int],
    **extra: Any,
) -> dict[str, Any]:
    """Write ``provenance.json`` next to the converted tables and return it."""
    record: dict[str, Any] = {
        "dataset": dataset,
        "version": version,
        "converter": converter,
        "flyconn_version": flyconn.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "inputs": [input_record(p) for p in inputs],
        "counts": counts,
        **extra,
    }
    (out_dir / "provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def write_parquet(table: pa.Table, path: Path) -> None:
    """Write a table with zstd compression and row groups sized for pushdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=1_000_000)
