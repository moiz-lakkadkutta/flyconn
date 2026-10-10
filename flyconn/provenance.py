"""Run environment for provenance records: package versions and git SHA."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import flyconn


def git_sha() -> str | None:
    """HEAD of the checkout flyconn is imported from, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(flyconn.__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def run_environment(packages: Sequence[str] = ("numpy", "scipy", "pandas")) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"flyconn_version": flyconn.__version__, "packages": versions, "git_sha": git_sha()}
