"""Driver lines for screen hits (Meissner 2025 table; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

from flyconn.access.handoff import attach_lines
from flyconn.access.lines import LineCatalog
from flyconn.access.neuronbridge import cached_off_target_count
from flyconn.data.convert.common import write_parquet


def _catalog(tmp_path: Path) -> LineCatalog:
    df = pd.DataFrame(
        {
            "line": ["SS1", "SS2", "SS3"],
            "cell_types": [["GNG232"], ["GNG232", "GNG100"], ["LB3c"]],
            "quality": [2, 1, 1],
            "doi": ["d1", "d2", "d3"],
            "first_author_year": ["A 2020", "B 2021", "C 2022"],
            "genotype": ["g", "g", "g"],
            "em_body_ids": [[], [], []],
        }
    )
    write_parquet(pa.Table.from_pandas(df, preserve_index=False), tmp_path / "lines.parquet")
    (tmp_path / "provenance.json").write_text(json.dumps({"counts": {}}))
    return LineCatalog(tmp_path)


def test_exact_glob_none(tmp_path: Path):
    hits = pd.DataFrame({"candidate": ["GNG232", "LB3", "ZZZ"]})
    out = attach_lines(hits, _catalog(tmp_path))
    g = out.iloc[0]
    assert g["line_match"] == "exact" and g["best_line"] == "SS2" and g["line_quality"] == 1
    assert not g["line_specific"]
    assert g["lines"].startswith("SS2 (q1)") and "SS1 (q2, specific)" in g["lines"]
    assert out.iloc[1]["line_match"] == "glob" and out.iloc[1]["best_line"] == "SS3"
    assert out.iloc[2]["line_match"] == "none" and pd.isna(out.iloc[2]["best_line"])


def test_not_looked_up():
    out = attach_lines(pd.DataFrame({"candidate": ["A"]}), None)
    assert out.iloc[0]["line_match"] == "not looked up"


def test_cached_off_targets_never_fetches(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLYCONN_CACHE", str(tmp_path))
    assert cached_off_target_count("SS9") is None  # nothing cached -> None, no network


def test_specific_line_wins_among_equal_quality(tmp_path: Path):
    df = pd.DataFrame(
        {
            "line": ["AAA_broad", "ZZZ_specific"],
            "cell_types": [["GNG232", "GNG100"], ["GNG232"]],
            "quality": [1, 1],
            "doi": ["d1", "d2"],
            "first_author_year": ["A 2020", "B 2021"],
            "genotype": ["g", "g"],
            "em_body_ids": [[], []],
        }
    )
    write_parquet(pa.Table.from_pandas(df, preserve_index=False), tmp_path / "lines.parquet")
    (tmp_path / "provenance.json").write_text(json.dumps({"counts": {}}))
    out = attach_lines(pd.DataFrame({"candidate": ["GNG232"]}), LineCatalog(tmp_path))
    g = out.iloc[0]
    assert g["best_line"] == "ZZZ_specific" and bool(g["line_specific"])
    assert g["lines"].startswith("ZZZ_specific")
