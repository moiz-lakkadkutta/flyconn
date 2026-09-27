"""Tests for NeuronBridge match aggregation (pure functions over the published JSON schema)."""

import json
from pathlib import Path

import pandas as pd

from flyconn.access.neuronbridge import aggregate_cds_matches, parse_published_name

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "neuronbridge"


def test_parse_published_name_splits_dataset_version_and_id():
    assert parse_published_name("hemibrain:v1.2.1:1912789965") == (
        "hemibrain",
        "v1.2.1",
        "1912789965",
    )
    assert parse_published_name("SS02299") == ("line", "", "SS02299")


def test_aggregate_cds_matches_best_score_per_line_and_library():
    data = json.loads((FIX / "cdsresults_hemibrain_1912789965.json").read_text())
    df = aggregate_cds_matches(data["results"], direction="em_to_lm")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "line",
        "library",
        "best_normalized_score",
        "best_matching_pixels",
        "n_images",
        "mirrored_any",
    }
    assert df["line"].is_unique or df.set_index(["line", "library"]).index.is_unique
    top = df.iloc[0]
    assert top["best_normalized_score"] == df["best_normalized_score"].max()
    assert (df["n_images"] >= 1).all()
    assert len(df) < len(data["results"])  # aggregated over images/channels


def test_aggregate_reports_candidate_status_not_ground_truth():
    data = json.loads((FIX / "cdsresults_hemibrain_1912789965.json").read_text())
    df = aggregate_cds_matches(data["results"], direction="em_to_lm")
    assert df.attrs["note"].lower().startswith("candidate")
