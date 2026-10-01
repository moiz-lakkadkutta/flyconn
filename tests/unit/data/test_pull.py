"""Tests for the pull orchestrator: download registry files for a level, then convert."""

import json
import shutil
import threading
from collections.abc import Iterator
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from flyconn.data.download import sha256_file
from flyconn.data.pull import PullResult, pull
from flyconn.data.registry import FileSpec, get_dataset
from flyconn.data.store import Store


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def local_malecns(
    malecns_raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[object]:
    """Serve the fake MaleCNS files over HTTP and return a spec pointing at them."""
    srv_root = tmp_path / "srv"
    shutil.copytree(malecns_raw, srv_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(srv_root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    real = get_dataset("malecns@1.0")
    files: list[FileSpec] = []
    for f in real.files:
        p = srv_root / f.name
        if p.exists():
            files.append(
                replace(
                    f,
                    url=f"{base}/{f.name}",
                    bytes=p.stat().st_size,
                    sha256=sha256_file(p),
                    md5_b64=None,
                )
            )
    spec = replace(real, files=tuple(files))
    monkeypatch.setenv("FLYCONN_CACHE", str(tmp_path / "cache"))
    try:
        yield spec
    finally:
        server.shutdown()


def test_pull_meta_downloads_small_files_and_converts(local_malecns: object):
    result = pull(spec=local_malecns, level="meta")  # type: ignore[arg-type]
    assert isinstance(result, PullResult)
    assert {d.path.name for d in result.downloads} == {
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "body-neurotransmitters-male-cns-v1.0.feather",
    }
    store = Store(result.store_dir)
    assert len(store.neurons()) == 4
    assert "edges" not in store.tables


def test_pull_weights_adds_edges_and_is_idempotent(local_malecns: object):
    first = pull(spec=local_malecns, level="weights")  # type: ignore[arg-type]
    assert Store(first.store_dir).provenance["counts"]["edges"] == 5
    second = pull(spec=local_malecns, level="weights")  # type: ignore[arg-type]
    assert all(d.skipped for d in second.downloads)
    assert second.converted is False


def test_pull_records_level_in_provenance(local_malecns: object):
    result = pull(spec=local_malecns, level="meta")  # type: ignore[arg-type]
    prov = json.loads((result.store_dir / "provenance.json").read_text())
    assert prov["level"] == "meta"
    assert prov["registry_ref"] == "malecns@1.0"


def test_pull_reconverts_when_converter_version_changes(
    local_malecns: object, monkeypatch: pytest.MonkeyPatch
):
    import flyconn.data.pull as pull_mod

    first = pull(spec=local_malecns, level="meta")  # type: ignore[arg-type]
    assert first.converted is True
    prov = json.loads((first.store_dir / "provenance.json").read_text())
    assert prov["converter_version"] == pull_mod.CONVERTER_VERSIONS["malecns"]
    again = pull(spec=local_malecns, level="meta")  # type: ignore[arg-type]
    assert again.converted is False
    monkeypatch.setitem(pull_mod.CONVERTER_VERSIONS, "malecns", "999")
    third = pull(spec=local_malecns, level="meta")  # type: ignore[arg-type]
    assert third.converted is True


@pytest.mark.parametrize(
    ("ref", "meta_files", "weights_only"),
    [
        (
            "hemibrain@1.2.1",
            {
                "exported-traced-adjacencies-v1.2.tar.gz",
                "hemibrain-v1.2-body-mean-neurotransmitters.feather",
                "Supplemental_file5_hemibrain_meta.csv",
                "hemibrain_121_meta.feather",
            },
            set[str](),
        ),
        (
            "manc@1.2.1",
            {"neurons.csv.gz", "manc_121_meta.feather"},
            {"manc_121_simple_edgelist.feather"},
        ),
    ],
)
def test_hemibrain_and_manc_have_converters_and_pinned_files(
    ref: str, meta_files: set[str], weights_only: set[str]
):
    from flyconn.data.pull import CONVERTER_VERSIONS, CONVERTERS

    spec = get_dataset(ref)
    assert spec.name in CONVERTERS and spec.name in CONVERTER_VERSIONS
    meta = {f.name for f in spec.files_for_level("meta")}
    weights = {f.name for f in spec.files_for_level("weights")}
    assert meta == meta_files
    assert weights - meta == weights_only
    assert all(f.sha256 and f.bytes for f in spec.files)
