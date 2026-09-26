"""Tests for resumable, checksummed downloads (local HTTP server; no network)."""

import hashlib
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from flyconn.data.download import ChecksumMismatchError, download, sha256_file


class _RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with minimal HTTP Range support and quiet logs."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def send_head(self):  # type: ignore[override]
        path = Path(self.translate_path(self.path))
        rng = self.headers.get("Range")
        if rng is None or not path.is_file():
            return super().send_head()
        start = int(rng.removeprefix("bytes=").split("-")[0])
        size = path.stat().st_size
        self.send_response(206)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.send_header("Content-Length", str(size - start))
        self.end_headers()
        fh = path.open("rb")
        fh.seek(start)
        return fh


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "srv"
    root.mkdir()
    payload = bytes(range(256)) * 4000  # 1 MB, deterministic
    (root / "blob.bin").write_bytes(payload)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RangeHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", root / "blob.bin"
    finally:
        server.shutdown()


def test_download_writes_file_and_verifies_sha256(served: tuple[str, Path], tmp_path: Path):
    base, src = served
    dest = tmp_path / "out" / "blob.bin"
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    result = download(f"{base}/blob.bin", dest, sha256=digest, expected_bytes=src.stat().st_size)
    assert result.path == dest
    assert result.bytes == src.stat().st_size
    assert result.sha256 == digest
    assert dest.read_bytes() == src.read_bytes()
    assert not dest.with_suffix(".bin.part").exists()


def test_download_resumes_a_partial_file(served: tuple[str, Path], tmp_path: Path):
    base, src = served
    dest = tmp_path / "blob.bin"
    part = dest.with_name(dest.name + ".part")
    payload = src.read_bytes()
    part.write_bytes(payload[:300_000])
    result = download(f"{base}/blob.bin", dest, sha256=hashlib.sha256(payload).hexdigest())
    assert result.resumed_from == 300_000
    assert dest.read_bytes() == payload


def test_download_raises_on_checksum_mismatch_and_keeps_no_final_file(
    served: tuple[str, Path], tmp_path: Path
):
    base, _ = served
    dest = tmp_path / "blob.bin"
    with pytest.raises(ChecksumMismatchError):
        download(f"{base}/blob.bin", dest, sha256="0" * 64)
    assert not dest.exists()


def test_download_skips_when_destination_already_verified(served: tuple[str, Path], tmp_path: Path):
    base, src = served
    dest = tmp_path / "blob.bin"
    dest.write_bytes(src.read_bytes())
    digest = sha256_file(dest)
    result = download(f"{base}/blob.bin", dest, sha256=digest)
    assert result.skipped is True


def test_download_accepts_gcs_style_base64_md5(served: tuple[str, Path], tmp_path: Path):
    import base64

    base, src = served
    dest = tmp_path / "blob.bin"
    md5_b64 = base64.b64encode(hashlib.md5(src.read_bytes()).digest()).decode()
    result = download(f"{base}/blob.bin", dest, md5_b64=md5_b64)
    assert result.bytes == src.stat().st_size
