"""Dataset registry: pinned, checksummed descriptions of public connectome releases.

Each ``flyconn/data/registry/*.yaml`` file describes one ``name@version``: its
licence, citations, and the primary files with URL, size and checksum. The
registry is data, not behaviour; converters in :mod:`flyconn.data.convert`
know how to turn the files into the harmonized store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any, Literal

import yaml

Level = Literal["meta", "weights", "nt-probs", "synapses"]
LEVELS: tuple[Level, ...] = ("meta", "weights", "nt-probs", "synapses")
"""Download levels in increasing size; each level includes all lower ones."""


@dataclass(frozen=True)
class Citation:
    key: str
    text: str
    doi: str = ""
    url: str = ""


@dataclass(frozen=True)
class FileSpec:
    name: str
    url: str
    bytes: int
    level: Level
    role: str
    sha256: str | None = None
    md5_b64: str | None = None
    note: str = ""

    @property
    def has_checksum(self) -> bool:
        return bool(self.sha256 or self.md5_b64)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    version: str
    title: str
    license: str
    citations: tuple[Citation, ...]
    files: tuple[FileSpec, ...]
    homepage: str = ""
    license_url: str = ""
    release_date: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def file(self, name: str) -> FileSpec:
        for f in self.files:
            if f.name == name:
                return f
        msg = f"{self.ref} has no file {name!r}; known: {[f.name for f in self.files]}"
        raise KeyError(msg)

    def files_for_level(self, level: Level) -> tuple[FileSpec, ...]:
        rank = LEVELS.index(level)
        return tuple(f for f in self.files if LEVELS.index(f.level) <= rank)

    def files_with_role(self, role: str) -> tuple[FileSpec, ...]:
        return tuple(f for f in self.files if f.role == role)


def parse_ref(ref: str) -> tuple[str, str]:
    """Split ``"name@version"`` into its parts."""
    name, sep, version = ref.partition("@")
    if not sep or not name or not version:
        msg = f"dataset reference must look like name@version, got {ref!r}"
        raise ValueError(msg)
    return name, version


def _spec_from_dict(d: dict[str, Any]) -> DatasetSpec:
    citations = tuple(Citation(**c) for c in d.get("citations", []))
    files = tuple(FileSpec(**f) for f in d.get("files", []))
    known = {"name", "version", "title", "license", "citations", "files"}
    known |= {"homepage", "license_url", "release_date", "notes"}
    extra = {k: v for k, v in d.items() if k not in known}
    return DatasetSpec(
        name=str(d["name"]),
        version=str(d["version"]),
        title=d.get("title", ""),
        license=d["license"],
        citations=citations,
        files=files,
        homepage=d.get("homepage", ""),
        license_url=d.get("license_url", ""),
        release_date=str(d.get("release_date", "")),
        notes=d.get("notes", ""),
        extra=extra,
    )


@cache
def _load_all() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    pkg = resources.files("flyconn.data") / "registry"
    for entry in sorted(pkg.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".yaml"):
            continue
        with entry.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        spec = _spec_from_dict(data)
        specs[spec.ref] = spec
    return specs


def list_datasets() -> list[str]:
    """Return all registered ``name@version`` references, sorted."""
    return sorted(_load_all())


def get_dataset(ref: str) -> DatasetSpec:
    """Look up a dataset by ``name@version``."""
    parse_ref(ref)
    try:
        return _load_all()[ref]
    except KeyError:
        msg = f"unknown dataset {ref!r}; registered: {', '.join(list_datasets())}"
        raise KeyError(msg) from None
