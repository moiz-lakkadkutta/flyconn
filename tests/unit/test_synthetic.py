"""Tests for the synthetic connectome fixture generator."""

import numpy as np
import pandas as pd

from flyconn.testing import NT_CLASSES, synthetic_connectome

NEURON_COLUMNS = {
    "dataset",
    "version",
    "neuron_id",
    "super_class",
    "cell_type",
    "side",
    "nt_pred",
    "nt_conf",
}
EDGE_COLUMNS = {"dataset", "version", "pre", "post", "weight"}


def test_returns_neurons_and_edges_with_requested_sizes():
    neurons, edges = synthetic_connectome(n_neurons=40, n_edges=150, seed=3)
    assert isinstance(neurons, pd.DataFrame)
    assert isinstance(edges, pd.DataFrame)
    assert len(neurons) == 40
    assert len(edges) == 150


def test_has_harmonized_columns_including_nt_probabilities():
    neurons, edges = synthetic_connectome(n_neurons=10, n_edges=20, seed=0)
    assert set(neurons.columns) >= NEURON_COLUMNS
    assert set(edges.columns) >= EDGE_COLUMNS
    prob_cols = [f"nt_p_{nt}" for nt in NT_CLASSES]
    assert set(prob_cols) <= set(neurons.columns)
    np.testing.assert_allclose(neurons[prob_cols].sum(axis=1), 1.0, atol=1e-9)
    assert (neurons["nt_pred"] == neurons[prob_cols].idxmax(axis=1).str.removeprefix("nt_p_")).all()


def test_edges_reference_existing_neurons_without_self_loops_or_duplicates():
    neurons, edges = synthetic_connectome(n_neurons=30, n_edges=100, seed=7)
    ids = set(neurons["neuron_id"])
    assert set(edges["pre"]) <= ids
    assert set(edges["post"]) <= ids
    assert (edges["pre"] != edges["post"]).all()
    assert not edges.duplicated(["pre", "post"]).any()
    assert (edges["weight"] >= 1).all()
    assert edges["weight"].dtype == np.int32


def test_is_deterministic_for_a_seed_and_differs_across_seeds():
    a = synthetic_connectome(n_neurons=25, n_edges=60, seed=11)
    b = synthetic_connectome(n_neurons=25, n_edges=60, seed=11)
    c = synthetic_connectome(n_neurons=25, n_edges=60, seed=12)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert not a[1].equals(c[1])


def test_dataset_and_version_are_tagged():
    neurons, edges = synthetic_connectome(n_neurons=5, n_edges=4, seed=0)
    assert (neurons["dataset"] == "synthetic").all()
    assert (neurons["version"] == "0").all()
    assert (edges["dataset"] == "synthetic").all()
