"""Tests for ConnectivityMatrix: sparse weights, index mapping, thresholds, signs."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy


def test_build_from_store_gives_square_csr_pre_in_rows(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    assert m.n == 40
    assert sp.isspmatrix_csr(m.weights)
    assert m.weights.shape == (40, 40)
    e = small_store.edges()
    i = m.index_of(e["pre"].to_numpy())
    j = m.index_of(e["post"].to_numpy())
    np.testing.assert_array_equal(np.asarray(m.weights[i, j]).ravel(), e["weight"].to_numpy())
    assert m.weights.sum() == e["weight"].sum()


def test_index_round_trips_neuron_ids(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    ids = m.neuron_ids
    assert len(ids) == 40
    np.testing.assert_array_equal(ids[m.index_of(ids)], ids)
    with pytest.raises(KeyError, match="not in matrix"):
        m.index_of(np.array([999_999_999]))


def test_min_weight_threshold_drops_weak_edges(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store, min_weight=3)
    assert m.weights.nnz == (small_store.edges()["weight"] >= 3).sum()
    assert (m.weights.data >= 3).all()
    assert m.provenance["min_weight"] == 3


def test_subset_keeps_only_requested_neurons(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    keep = m.neuron_ids[:10]
    s = m.subset(keep)
    assert s.n == 10
    np.testing.assert_array_equal(s.neuron_ids, keep)
    dense_full = m.weights.toarray()[np.ix_(m.index_of(keep), m.index_of(keep))]
    np.testing.assert_array_equal(s.weights.toarray(), dense_full)


def test_argmax_sign_policy_follows_adr_0007(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store, sign_policy=SignPolicy.ARGMAX)
    s = m.signs  # one entry per neuron (presynaptic sign)
    ids = m.neuron_ids
    meta = small_store.neurons().set_index("neuron_id")
    assert s[m.index_of(np.array([ids[0]]))][0] == 1  # acetylcholine
    assert s[m.index_of(np.array([ids[1]]))][0] == -1  # gaba
    assert s[m.index_of(np.array([ids[2]]))][0] == -1  # histamine inhibitory by default
    assert s[m.index_of(np.array([ids[3]]))][0] == 0  # unknown -> excluded
    # every neuron with a prediction is +/-1, consistent with the table
    for nid, row in meta.iterrows():
        val = s[m.index_of(np.array([nid]))][0]
        if pd.isna(row["nt_pred"]):
            assert val == 0
        elif row["nt_pred"] in {"gaba", "glutamate", "histamine"}:
            assert val == -1
        else:
            assert val == 1


def test_signed_weights_multiply_rows_by_presynaptic_sign(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store, sign_policy=SignPolicy.ARGMAX)
    sw = m.signed_weights.toarray()
    w = m.weights.toarray()
    np.testing.assert_array_equal(sw, w * m.signs[:, None])
    unknown_row = m.index_of(np.array([m.neuron_ids[3]]))[0]
    assert not sw[unknown_row].any()
    assert m.provenance["excluded_unknown_sign_neurons"] == int((m.signs == 0).sum())


def test_probabilistic_policy_gives_expected_sign(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store, sign_policy=SignPolicy.PROBABILISTIC)
    row0 = m.index_of(np.array([m.neuron_ids[0]]))[0]
    assert m.signs[row0] == pytest.approx(0.6 - 0.4)


def test_transmitter_override_makes_glutamate_excitatory(small_store: Store):
    m = ConnectivityMatrix.from_store(
        small_store, sign_policy=SignPolicy.ARGMAX, sign_overrides={"glutamate": 1}
    )
    meta = small_store.neurons().set_index("neuron_id")
    glu = meta.index[meta["nt_pred"] == "glutamate"].to_numpy()
    if len(glu):
        assert (m.signs[m.index_of(glu)] == 1).all()
    assert m.provenance["sign_overrides"] == {"glutamate": 1}


def test_input_normalized_uses_total_inputs_including_fragments(small_store: Store):
    m = ConnectivityMatrix.from_store(small_store)
    norm = m.input_normalized().toarray()
    meta = small_store.neurons().set_index("neuron_id")
    total = meta.loc[m.neuron_ids, "input_synapses_total"].to_numpy(dtype=float)
    np.testing.assert_allclose(norm, m.weights.toarray() / total[None, :])
    assert (norm.sum(axis=0) < 1).all()  # +5 fragment synapses per neuron in the fixture
