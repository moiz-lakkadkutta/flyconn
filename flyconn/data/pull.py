"""``pull``: download a dataset's files for a level, verify, and convert into the store."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flyconn.data.cache import raw_dir, store_dir
from flyconn.data.convert.banc import convert_banc
from flyconn.data.convert.flywire import convert_flywire
from flyconn.data.convert.malecns import convert_malecns
from flyconn.data.convert.shiu import convert_shiu
from flyconn.data.download import DownloadResult, download
from flyconn.data.registry import LEVELS, DatasetSpec, Level, get_dataset

Converter = Callable[[Path, Path, DatasetSpec], dict[str, Any]]


def _convert_malecns(raw: Path, out: Path, spec: DatasetSpec) -> dict[str, Any]:
    return convert_malecns(raw, out)


def _convert_flywire(raw: Path, out: Path, spec: DatasetSpec) -> dict[str, Any]:
    return convert_flywire(raw, out, version=spec.version)


def _convert_shiu(raw: Path, out: Path, spec: DatasetSpec) -> dict[str, Any]:
    return convert_shiu(raw, out)


def _convert_banc(raw: Path, out: Path, spec: DatasetSpec) -> dict[str, Any]:
    return convert_banc(raw, out)


def _convert_lines(raw: Path, out: Path, spec: DatasetSpec) -> dict[str, Any]:
    from flyconn.access.lines import XLSX, convert_meissner_lines

    return convert_meissner_lines(raw / XLSX, out)


CONVERTER_VERSIONS: dict[str, str] = {
    "malecns": "2",  # 2: nt-probs aggregation and per-neuron totals
    "flywire": "1",
    "shiu": "1",
    "flylight_lines": "2",  # 2: raw EM-id text column and count
    "banc": "1",
}
"""Bump a dataset's converter version whenever its output changes; stores are reconverted."""


CONVERTERS: dict[str, Converter] = {
    "flylight_lines": _convert_lines,
    "banc": _convert_banc,
    "malecns": _convert_malecns,
    "flywire": _convert_flywire,
    "shiu": _convert_shiu,
}


@dataclass(frozen=True)
class PullResult:
    ref: str
    level: Level
    raw_dir: Path
    store_dir: Path
    downloads: tuple[DownloadResult, ...] = field(default_factory=tuple)
    converted: bool = False


def _named_progress(
    progress: Callable[[str, int, int | None], None], name: str
) -> Callable[[int, int | None], None]:
    def cb(done: int, total: int | None) -> None:
        progress(name, done, total)

    return cb


def _existing(store: Path) -> tuple[Level | None, str | None]:
    """(level, converter_version) recorded in an existing store, if any."""
    prov = store / "provenance.json"
    if not prov.exists():
        return None, None
    data = json.loads(prov.read_text())
    level = data.get("level")
    return (level if level in LEVELS else None), data.get("converter_version")


def pull(
    ref: str | None = None,
    *,
    spec: DatasetSpec | None = None,
    level: Level = "meta",
    force: bool = False,
    progress: Callable[[str, int, int | None], None] | None = None,
) -> PullResult:
    """Download and convert ``ref`` (``name@version``) up to ``level``.

    Files already present and verified are skipped; conversion runs when the
    store is missing, when any file was (re)downloaded, when the stored level is
    lower than requested, or when ``force`` is set.
    """
    if spec is None:
        if ref is None:
            msg = "pull() needs a dataset ref or a spec"
            raise ValueError(msg)
        spec = get_dataset(ref)
    if level not in LEVELS:
        msg = f"level must be one of {LEVELS}, got {level!r}"
        raise ValueError(msg)
    try:
        converter = CONVERTERS[spec.name]
    except KeyError:
        msg = f"no converter for dataset {spec.name!r}; available: {sorted(CONVERTERS)}"
        raise NotImplementedError(msg) from None

    raw = raw_dir(spec)
    store = store_dir(spec)
    downloads: list[DownloadResult] = []
    for f in spec.files_for_level(level):
        cb = _named_progress(progress, f.name) if progress else None
        downloads.append(
            download(
                f.url,
                raw / f.name,
                sha256=f.sha256,
                md5_b64=f.md5_b64,
                expected_bytes=f.bytes,
                progress=cb,
            )
        )

    stored, stored_version = _existing(store)
    wanted_version = CONVERTER_VERSIONS.get(spec.name, "1")
    needs_convert = (
        force
        or stored is None
        or LEVELS.index(stored) < LEVELS.index(level)
        or stored_version != wanted_version
        or any(not d.skipped for d in downloads)
    )
    if needs_convert:
        store.mkdir(parents=True, exist_ok=True)
        converter(raw, store, spec)
        prov_path = store / "provenance.json"
        prov = json.loads(prov_path.read_text())
        prov.update(
            level=level,
            registry_ref=spec.ref,
            license=spec.license,
            converter_version=wanted_version,
        )
        prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True))
    return PullResult(spec.ref, level, raw, store, tuple(downloads), converted=needs_convert)
