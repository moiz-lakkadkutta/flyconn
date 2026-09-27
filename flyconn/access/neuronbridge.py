"""NeuronBridge precomputed colour-depth matches: aggregation and cached fetch (CC BY 4.0).

Bucket layout (v3_10_0): ``metadata/by_line/<LINE>.json`` lists a line's images, each
with ``files.CDSResults`` pointing at ``metadata/cdsresults/<imageId>.json`` whose
``results`` are ``CDSMatch`` records (``normalizedScore``, ``matchingPixels``,
``mirrored``, ``image.publishedName`` = ``dataset:version:bodyId`` for EM targets).
Matches are ranked similarity candidates, not expression ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import requests

from flyconn.data.cache import cache_root

BUCKET = "https://janelia-neuronbridge-data-prod.s3.amazonaws.com"
DEFAULT_VERSION = "v3_10_0"
Direction = Literal["em_to_lm", "lm_to_em"]


def parse_published_name(name: str) -> tuple[str, str, str]:
    """Split ``dataset:version:id`` names; bare names give ``("line", "", name)``."""
    parts = name.split(":")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return "line", "", name


def aggregate_cds_matches(
    results: Sequence[dict[str, Any]], *, direction: Direction
) -> pd.DataFrame:
    """Best score per matched entity (line for em_to_lm; dataset+body for lm_to_em) over images."""
    rows: list[dict[str, Any]] = []
    for r in results:
        img = r.get("image", {})
        name = str(img.get("publishedName", ""))
        ds, ver, ident = parse_published_name(name)
        rows.append(
            {
                "line": ident if direction == "em_to_lm" else None,
                "dataset": ds if direction == "lm_to_em" else None,
                "version": ver if direction == "lm_to_em" else None,
                "body_id": ident if direction == "lm_to_em" else None,
                "neuron_type": img.get("neuronType"),
                "library": img.get("libraryName"),
                "normalized_score": float(r.get("normalizedScore", 0.0)),
                "matching_pixels": int(r.get("matchingPixels", 0)),
                "mirrored": bool(r.get("mirrored", False)),
            }
        )
    df = pd.DataFrame(rows)
    key = (
        ["line", "library"]
        if direction == "em_to_lm"
        else ["dataset", "version", "body_id", "neuron_type"]
    )
    if df.empty:
        cols = [*key, "best_normalized_score", "best_matching_pixels", "n_images", "mirrored_any"]
        out = pd.DataFrame({c: [] for c in cols})
    else:
        g = df.groupby(key, dropna=False)
        out = pd.DataFrame(
            {
                "best_normalized_score": g["normalized_score"].max(),
                "best_matching_pixels": g["matching_pixels"].max(),
                "n_images": g.size(),
                "mirrored_any": g["mirrored"].any(),
            }
        ).reset_index()
        out = out.sort_values("best_normalized_score", ascending=False, ignore_index=True)
    out.attrs["note"] = (
        "candidate matches from NeuronBridge colour-depth search; scores are not comparable across "
        "libraries and a match is not an expression call"
    )
    return out


def _get_json(path: str, version: str, timeout: float = 60.0) -> dict[str, Any]:
    cache = cache_root() / "neuronbridge" / version / path
    if cache.exists():
        return cast("dict[str, Any]", json.loads(cache.read_text()))
    resp = requests.get(f"{BUCKET}/{version}/{path}", timeout=timeout)
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return cast("dict[str, Any]", resp.json())


def fetch_line_off_targets(
    line: str, *, version: str = DEFAULT_VERSION, max_images: int = 50
) -> pd.DataFrame:
    """EM bodies matched to a driver line's images (LM -> EM), aggregated per dataset and body.

    Network access to the public NeuronBridge bucket; results are cached under the flyconn
    cache directory. Returns the best score per (dataset, version, body, neuron type).
    """
    meta = _get_json(f"metadata/by_line/{line}.json", version)
    results: list[dict[str, Any]] = []
    for img in meta.get("results", [])[:max_images]:
        cds = img.get("files", {}).get("CDSResults")
        if not cds:
            continue
        data = _get_json(f"metadata/cdsresults/{cds}", version)
        results.extend(data.get("results", []))
    df = aggregate_cds_matches(results, direction="lm_to_em")
    df.attrs["line"] = line
    df.attrs["n_images"] = min(len(meta.get("results", [])), max_images)
    df.attrs["source"] = f"{BUCKET}/{version}"
    df.attrs["license"] = "CC BY 4.0 (NeuronBridge, HHMI Janelia)"
    return df


def local_cache_path() -> Path:
    return cache_root() / "neuronbridge"
