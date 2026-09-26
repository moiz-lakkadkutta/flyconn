"""flyconn: research-grade toolkit over public Drosophila connectomes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flyconn")
except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
    __version__ = "0.0.0"

__all__ = ["__version__"]
