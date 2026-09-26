"""Tests for statistics against null-model distributions."""

import numpy as np

from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, effective_by_hops
from flyconn.uncertainty.null_stats import null_distribution


def test_null_distribution_gives_observed_nulls_zscore_and_pvalue(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    ids = m.neuron_ids

    def stat(x: ConnectivityMatrix) -> float:
        return float(
            effective_by_hops(x, ids[:5], ids[-5:], max_hops=2, signed=False)["value"].sum()
        )

    res = null_distribution(stat, m, n_nulls=20, seed=0, null="degree_preserving_rewire")
    assert res.observed == stat(m)
    assert len(res.nulls) == 20
    assert np.isfinite(res.z)
    assert 0 < res.p_two_sided <= 1
    assert res.null_model == "degree_preserving_rewire"
    assert res.provenance["seed"] == 0


def test_null_distribution_supports_sign_shuffle_and_is_seeded(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    ids = m.neuron_ids

    def stat(x: ConnectivityMatrix) -> float:
        return float(
            effective_by_hops(x, ids[:5], ids[-5:], max_hops=2, signed=True)["value"].sum()
        )

    a = null_distribution(stat, m, n_nulls=10, seed=1, null="sign_shuffle")
    b = null_distribution(stat, m, n_nulls=10, seed=1, null="sign_shuffle")
    np.testing.assert_array_equal(a.nulls, b.nulls)
    assert a.summary()["n_nulls"] == 10
