"""Tests for version drift: what changed for neurons of interest between two releases."""

from pathlib import Path

import pandas as pd
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store
from flyconn.uncertainty.versions import diff_versions


def _store(tmp_path: Path, version: str, neurons: pd.DataFrame, edges: pd.DataFrame) -> Store:
    d = tmp_path / version
    d.mkdir()
    neurons = neurons.assign(dataset="flywire", version=version)
    edges = edges.assign(dataset="flywire", version=version)
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(d, dataset="flywire", version=version, converter="t", inputs=[], counts={})
    return Store(d)


@pytest.fixture
def two_versions(tmp_path: Path) -> tuple[Store, Store]:
    old = _store(
        tmp_path,
        "630",
        pd.DataFrame(
            {"neuron_id": [1, 2, 3, 4], "cell_type": ["A", "B", "C", "D"], "nt_pred": ["gaba"] * 4}
        ),
        pd.DataFrame({"pre": [1, 1, 2, 3], "post": [2, 3, 3, 4], "weight": [10, 5, 7, 6]}),
    )
    new = _store(
        tmp_path,
        "783",
        pd.DataFrame(
            {
                "neuron_id": [1, 2, 3, 5],
                "cell_type": ["A", "B2", "C", "E"],
                "nt_pred": ["gaba", "gaba", "acetylcholine", "gaba"],
            }
        ),
        pd.DataFrame({"pre": [1, 1, 2, 5], "post": [2, 3, 3, 3], "weight": [10, 9, 7, 4]}),
    )
    return old, new


def test_neuron_diff_reports_added_removed_and_changed_attributes(
    two_versions: tuple[Store, Store],
):
    old, new = two_versions
    d = diff_versions(old, new)
    assert d.neurons_removed == [4]
    assert d.neurons_added == [5]
    changed = d.neuron_changes.set_index("neuron_id")
    assert changed.loc[2, "cell_type_old"] == "B" and changed.loc[2, "cell_type_new"] == "B2"
    assert (
        changed.loc[3, "nt_pred_old"] == "gaba" and changed.loc[3, "nt_pred_new"] == "acetylcholine"
    )
    assert 1 not in changed.index


def test_edge_diff_for_neurons_of_interest(two_versions: tuple[Store, Store]):
    old, new = two_versions
    d = diff_versions(old, new, neuron_ids=[1, 2, 3])
    e = d.edge_changes.set_index(["pre", "post"])
    assert (
        e.loc[(1, 3), "status"] == "changed"
        and e.loc[(1, 3), "weight_old"] == 5
        and e.loc[(1, 3), "weight_new"] == 9
    )
    assert e.loc[(5, 3), "status"] == "added"
    assert e.loc[(3, 4), "status"] == "removed"
    assert (1, 2) not in e.index  # unchanged edges are not listed
    assert d.summary()["edges_changed"] == 1


def test_diff_flags_ids_absent_from_the_new_release_as_needing_lineage(
    two_versions: tuple[Store, Store],
):
    old, new = two_versions
    d = diff_versions(old, new, neuron_ids=[4])
    assert d.unmatched_ids == [4]
    assert "lineage" in d.caveats[0].lower()
