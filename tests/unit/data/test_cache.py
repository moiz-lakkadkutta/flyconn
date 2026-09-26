"""Tests for the on-disk cache layout."""

from pathlib import Path

import pytest
from flyconn.data.cache import cache_root, raw_dir, store_dir

from flyconn.data.registry import get_dataset


def test_cache_root_honours_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FLYCONN_CACHE", str(tmp_path / "c"))
    assert cache_root() == tmp_path / "c"


def test_cache_root_defaults_under_user_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FLYCONN_CACHE", raising=False)
    assert cache_root().name == "flyconn"
    assert ".cache" in cache_root().parts or "Caches" in cache_root().parts


def test_raw_and_store_dirs_are_per_dataset_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("FLYCONN_CACHE", str(tmp_path))
    spec = get_dataset("malecns@1.0")
    assert raw_dir(spec) == tmp_path / "raw" / "malecns" / "1.0"
    assert store_dir(spec) == tmp_path / "store" / "malecns" / "1.0"
