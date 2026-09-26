"""On-disk cache layout: ``<root>/raw/<name>/<version>/`` and ``<root>/store/<name>/<version>/``."""

from __future__ import annotations

import os
from pathlib import Path

from flyconn.data.registry import DatasetSpec


def cache_root() -> Path:
    """Cache root: ``$FLYCONN_CACHE`` if set, else ``~/.cache/flyconn``."""
    env = os.environ.get("FLYCONN_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "flyconn"


def raw_dir(spec: DatasetSpec) -> Path:
    """Directory holding the downloaded primary files of ``spec``."""
    return cache_root() / "raw" / spec.name / spec.version


def store_dir(spec: DatasetSpec) -> Path:
    """Directory holding the converted harmonized Parquet tables of ``spec``."""
    return cache_root() / "store" / spec.name / spec.version
