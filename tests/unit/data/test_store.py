"""Tests for the DuckDB-backed Store over harmonized Parquet files."""

from pathlib import Path

import pandas as pd
import pytest
from flyconn.data.store import Store

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.testing import synthetic_connectome


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    neurons, edges = synthetic_connectome(n_neurons=30, n_edges=120, seed=5)
    neurons["dataset"] = "malecns"
    neurons["version"] = "1.0"
    edges["dataset"] = "malecns"
    edges["version"] = "1.0"
    d = tmp_path / "store"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(
        d,
        dataset="malecns",
        version="1.0",
        converter="test",
        inputs=[],
        counts={"neurons": 30, "edges": 120},
    )
    return d


def test_store_exposes_tables_and_provenance(store_dir: Path):
    s = Store(store_dir)
    assert s.ref == "malecns@1.0"
    assert len(s.neurons()) == 30
    assert len(s.edges()) == 120
    assert s.provenance["counts"] == {"neurons": 30, "edges": 120}


def test_neurons_can_select_columns_and_filter(store_dir: Path):
    s = Store(store_dir)
    df = s.neurons(columns=["neuron_id", "cell_type"], where="super_class = 'motor'")
    assert list(df.columns) == ["neuron_id", "cell_type"]
    assert set(df["neuron_id"]) <= set(s.neurons()["neuron_id"])


def test_edges_filter_by_min_weight_and_neuron_subset(store_dir: Path):
    s = Store(store_dir)
    all_edges = s.edges()
    strong = s.edges(min_weight=3)
    assert (strong["weight"] >= 3).all()
    assert len(strong) == (all_edges["weight"] >= 3).sum()
    ids = list(all_edges["pre"].unique()[:5])
    sub = s.edges(pre=ids)
    assert set(sub["pre"]) <= set(ids)


def test_sql_query_sees_views(store_dir: Path):
    s = Store(store_dir)
    df = s.query("SELECT count(*) AS n FROM edges JOIN neurons ON edges.pre = neurons.neuron_id")
    assert isinstance(df, pd.DataFrame)
    assert df["n"].iloc[0] == 120


def test_citations_come_from_registry(store_dir: Path):
    s = Store(store_dir)
    keys = {c.key for c in s.citations()}
    assert "berg2026" in keys
    text = s.citation_text()
    assert "10.1016/j.cell.2026.08.015" in text


def test_missing_edges_file_gives_empty_frame_with_schema(store_dir: Path):
    (store_dir / "edges.parquet").unlink()
    s = Store(store_dir)
    e = s.edges()
    assert list(e.columns)[:5] == ["dataset", "version", "pre", "post", "weight"]
    assert len(e) == 0
