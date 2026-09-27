"""Tests for null models: degree-preserving rewiring and sign shuffles."""

import numpy as np

from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix
from flyconn.uncertainty.nulls import degree_preserving_rewire, shuffle_signs


def test_rewire_preserves_out_degree_out_weight_and_in_degree(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    r = degree_preserving_rewire(m, seed=0)
    w0, w1 = m.weights, r.weights

    def vec(x: object) -> np.ndarray:
        return np.asarray(x).ravel()

    np.testing.assert_array_equal(vec((w0 > 0).sum(axis=1)), vec((w1 > 0).sum(axis=1)))
    np.testing.assert_array_equal(vec(w0.sum(axis=1)), vec(w1.sum(axis=1)))  # out-weight
    np.testing.assert_array_equal(vec((w0 > 0).sum(axis=0)), vec((w1 > 0).sum(axis=0)))
    assert w1.nnz == w0.nnz
    assert (w1.diagonal() == 0).all()
    assert r.provenance["null_model"] == "degree_preserving_rewire"
    assert r.provenance["seed"] == 0


def test_rewire_changes_the_wiring_and_is_seeded(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    a = degree_preserving_rewire(m, seed=1)
    b = degree_preserving_rewire(m, seed=1)
    c = degree_preserving_rewire(m, seed=2)
    assert (a.weights != b.weights).nnz == 0
    assert (a.weights != c.weights).nnz > 0
    assert (a.weights != m.weights).nnz > 0


def test_shuffle_signs_permutes_signs_keeping_their_histogram(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    s = shuffle_signs(m, seed=3)
    assert sorted(s.signs.tolist()) == sorted(m.signs.tolist())
    assert (s.signs != m.signs).any()
    assert (s.weights != m.weights).nnz == 0
    assert s.provenance["null_model"] == "sign_shuffle"
