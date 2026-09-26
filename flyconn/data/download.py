"""Resumable, checksummed HTTP downloads for registry files.

Downloads go to ``<dest>.part`` and are renamed only after the checksum
verifies, so a ``dest`` that exists is always a verified file. Interrupted
downloads resume with an HTTP ``Range`` request.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

CHUNK = 1 << 20


class ChecksumMismatchError(RuntimeError):
    """Raised when a downloaded file does not match its registered checksum."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    bytes: int
    sha256: str
    resumed_from: int = 0
    skipped: bool = False


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed."""
    return _hash_file(path, "sha256")


def md5_b64_file(path: Path) -> str:
    """Base64 MD5 of a file, the form Google Cloud Storage reports as ``md5Hash``."""
    return base64.b64encode(bytes.fromhex(_hash_file(path, "md5"))).decode()


def _verify(path: Path, sha256: str | None, md5_b64: str | None, expected_bytes: int | None) -> str:
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        msg = f"{path.name}: size {size} != expected {expected_bytes}"
        raise ChecksumMismatchError(msg)
    digest = sha256_file(path)
    if sha256 is not None and digest != sha256.lower():
        msg = f"{path.name}: sha256 {digest} != expected {sha256}"
        raise ChecksumMismatchError(msg)
    if md5_b64 is not None and md5_b64_file(path) != md5_b64:
        msg = f"{path.name}: md5 does not match expected {md5_b64}"
        raise ChecksumMismatchError(msg)
    return digest


def download(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    md5_b64: str | None = None,
    expected_bytes: int | None = None,
    timeout: float = 60.0,
    progress: Callable[[int, int | None], None] | None = None,
) -> DownloadResult:
    """Download ``url`` to ``dest`` resumably and verify its checksum.

    Args:
        url: HTTP(S) URL.
        dest: final path; ``dest.part`` is used while downloading.
        sha256: expected hex SHA-256 (preferred).
        md5_b64: expected base64 MD5 (as GCS reports); used when no SHA-256 is pinned.
        expected_bytes: expected size; checked before hashing.
        timeout: per-request timeout in seconds.
        progress: optional callback ``(bytes_done, bytes_total_or_None)``.

    Returns:
        :class:`DownloadResult`; ``skipped`` is true when ``dest`` already verified.

    Raises:
        ChecksumMismatchError: when size or checksum differ (the partial file is removed).
    """
    dest = Path(dest)
    if dest.exists():
        try:
            digest = _verify(dest, sha256, md5_b64, expected_bytes)
        except ChecksumMismatchError:
            dest.unlink()
        else:
            return DownloadResult(dest, dest.stat().st_size, digest, skipped=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    resumed_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resumed_from}-"} if resumed_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        if resumed_from and resp.status_code != 206:
            # Server ignored the range: start over.
            resumed_from = 0
            part.unlink(missing_ok=True)
        resp.raise_for_status()
        total = resp.headers.get("Content-Length")
        total_bytes = (int(total) + resumed_from) if total else None
        done = resumed_from
        with part.open("ab" if resumed_from else "wb") as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total_bytes)

    try:
        digest = _verify(part, sha256, md5_b64, expected_bytes)
    except ChecksumMismatchError:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return DownloadResult(dest, dest.stat().st_size, digest, resumed_from=resumed_from)
