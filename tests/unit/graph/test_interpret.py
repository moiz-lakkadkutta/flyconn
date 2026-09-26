"""Parity between flyconn's native effective connectivity and connectome_interpreter (optional)."""

import numpy as np
import pandas as pd
import pytest

from flyconn.graph import ConnectivityMatrix, SignPolicy, effective_connectivity

ci = pytest.importorskip("connectome_interpreter")

from flyconn.graph.interpret import compress_paths_signed, to_interpreter_inputs  # noqa: E402


def _toy() -> ConnectivityMatrix:
    rng = np.random.default_rng(0)
    n = 12
    neurons = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "neuron_id": np.arange(1, n + 1),
            "nt_pred": rng.choice(["acetylcholine", "gaba", "glutamate"], size=n),
        }
    )
    pre, post = np.nonzero(rng.random((n, n)) < 0.35)
    keep = pre != post
    edges = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "pre": pre[keep] + 1,
            "post": post[keep] + 1,
            "weight": rng.integers(1, 20, size=keep.sum()),
        }
    )
    return ConnectivityMatrix.from_frames(neurons, edges, sign_policy=SignPolicy.ARGMAX)


def test_inputs_are_column_normalised_with_pre_in_rows():
    m = _toy()
    inprop, idx_to_sign, idx_to_group = to_interpreter_inputs(m, group="nt_pred")
    col_sums = np.asarray(inprop.sum(axis=0)).ravel()
    assert np.allclose(col_sums[col_sums > 0], 1.0)
    assert set(idx_to_sign.values()) <= {-1, 1}
    assert len(idx_to_group) == m.n


def test_signed_two_hop_matches_native_implementation():
    m = _toy()
    theirs = compress_paths_signed(m, hops=2)  # list of csr per step
    ours = effective_connectivity(m, hops=2, signed=True).toarray()
    np.testing.assert_allclose(theirs[1].toarray(), ours, atol=1e-6)
