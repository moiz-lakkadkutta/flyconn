"""Tests for native k-hop effective connectivity (matrix powers of input-normalised weights)."""

import numpy as np
import pandas as pd
import pytest

from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.graph.effective import effective_by_hops, effective_connectivity


def _toy() -> ConnectivityMatrix:
    neurons = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "neuron_id": [1, 2, 3],
            "nt_pred": ["acetylcholine", "gaba", "acetylcholine"],
            "input_synapses_total": [10.0, 10.0, 10.0],
        }
    )
    edges = pd.DataFrame(
        {"dataset": "t", "version": "0", "pre": [1, 2, 1], "post": [2, 3, 3], "weight": [5, 10, 2]}
    )
    return ConnectivityMatrix.from_frames(neurons, edges, sign_policy=SignPolicy.ARGMAX)


def test_two_hop_effective_equals_signed_matrix_square():
    m = _toy()
    s = m.input_normalized().toarray() * m.signs[:, None]  # A->B 0.5, B->C -1.0, A->C 0.2
    e2 = effective_connectivity(m, hops=2, signed=True).toarray()
    np.testing.assert_allclose(e2, s @ s)
    assert e2[0, 2] == pytest.approx(0.5 * -1.0)


def test_cumulative_sums_powers_one_to_k():
    m = _toy()
    s = m.input_normalized().toarray() * m.signs[:, None]
    e = effective_connectivity(m, hops=2, signed=True, cumulative=True).toarray()
    np.testing.assert_allclose(e, s + s @ s)


def test_unsigned_ignores_signs():
    m = _toy()
    n = m.input_normalized().toarray()
    e = effective_connectivity(m, hops=2, signed=False).toarray()
    np.testing.assert_allclose(e, n @ n)


def test_effective_by_hops_reports_source_to_target_per_hop():
    m = _toy()
    df = effective_by_hops(m, sources=[1], targets=[3], max_hops=3, signed=True)
    assert list(df["hops"]) == [1, 2, 3]
    assert df.set_index("hops").loc[1, "value"] == pytest.approx(0.2)
    assert df.set_index("hops").loc[2, "value"] == pytest.approx(-0.5)
    assert df.set_index("hops").loc[3, "value"] == pytest.approx(0.0)
    assert set(df.columns) >= {"hops", "value", "n_sources", "n_targets"}


def test_threshold_prunes_small_entries_between_hops():
    m = _toy()
    e = effective_connectivity(m, hops=2, signed=False, threshold=0.3).toarray()
    # A->C (0.2) is pruned at hop 1; A->B (0.5) and B->C (1.0) survive, so A->C at 2 hops is 0.5
    assert e[0, 2] == pytest.approx(0.5)
