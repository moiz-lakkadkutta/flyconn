"""Tests for aggregating neuron-level connectivity by a label column."""

import numpy as np
import pandas as pd

from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix
from flyconn.graph.aggregate import aggregate, aggregate_edges


def test_aggregate_by_cell_type_sums_member_edges(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    agg = aggregate(m, by="cell_type")
    assert agg.weights.shape == (len(agg.labels), len(agg.labels))
    assert agg.weights.sum() == m.weights.sum()
    assert agg.counts.sum() == m.n
    # pick a populated label pair and check by brute force
    e = small_store.edges()
    meta = m.meta
    e = e.assign(
        pre_t=meta.loc[e["pre"], "cell_type"].to_numpy(),
        post_t=meta.loc[e["post"], "cell_type"].to_numpy(),
    )
    pair = e.groupby(["pre_t", "post_t"])["weight"].sum()
    (a, b), w = next(iter(pair.items()))
    assert agg.weight(a, b) == w


def test_aggregate_edges_long_form_has_counts_and_normalisation(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    df = aggregate_edges(m, by="super_class")
    assert set(df.columns) >= {
        "pre_label",
        "post_label",
        "weight",
        "n_pre",
        "n_post",
        "n_pairs",
        "weight_per_post",
    }
    assert (df["weight"] > 0).all()
    assert np.isclose(df["weight"].sum(), m.weights.sum())
    row = df.iloc[0]
    assert row["weight_per_post"] == row["weight"] / row["n_post"]


def test_aggregate_signed_uses_signed_weights(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    agg = aggregate(m, by="super_class", signed=True)
    assert np.isclose(agg.weights.sum(), m.signed_weights.sum())


def test_aggregate_drops_unlabelled_neurons_and_reports_them(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    m.meta.loc[m.neuron_ids[:3], "cell_type"] = None
    agg = aggregate(m, by="cell_type")
    assert agg.provenance["unlabelled_neurons"] == 3
    assert agg.counts.sum() == m.n - 3
    assert isinstance(agg.to_frame(), pd.DataFrame)
