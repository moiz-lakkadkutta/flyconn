"""Golden tests use the real cached data under $FLYCONN_CACHE (default repo .cache/flyconn)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "flyconn"


@pytest.fixture(scope="session", autouse=True)
def _cache_env() -> None:
    os.environ.setdefault("FLYCONN_CACHE", str(REPO_CACHE))
