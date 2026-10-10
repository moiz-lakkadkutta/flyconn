"""Run environment for provenance records: package versions and git SHA."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import flyconn


def _git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _source_root(package_dir: Path) -> Path | None:
    """Git toplevel only if it is flyconn's own source tree (not a user repo around a .venv)."""
    top = _git(["rev-parse", "--show-toplevel"], package_dir)
    if not top:
        return None
    if (Path(top) / "flyconn" / "__init__.py").resolve() != (package_dir / "__init__.py").resolve():
        return None
    return Path(top)


def git_sha(package_dir: Path | None = None) -> str | None:
    """HEAD of flyconn's own source checkout, or None (installed package, foreign repo)."""
    pkg = Path(package_dir or Path(flyconn.__file__).parent).resolve()
    root = _source_root(pkg)
    return (_git(["rev-parse", "HEAD"], root) or None) if root is not None else None


def git_dirty(package_dir: Path | None = None) -> bool | None:
    """True if flyconn's source checkout has uncommitted changes; None outside a checkout."""
    pkg = Path(package_dir or Path(flyconn.__file__).parent).resolve()
    root = _source_root(pkg)
    if root is None:
        return None
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    return None if status is None else bool(status)


def run_environment(packages: Sequence[str] = ("numpy", "scipy", "pandas")) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "flyconn_version": flyconn.__version__,
        "packages": versions,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
    }
