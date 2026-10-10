"""Partner-count profiles over a fixed vocabulary."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyconn.compare.profiles import l1_rows, l2_rows, partner_counts
from flyconn.graph.matrix import ConnectivityMatrix


def _tiny() -> ConnectivityMatrix:
    # 1(A) -> 2(B) w4, 1 -> 3(unlabelled) w6, 2 -> 1 w2
    w = sp.csr_matrix(np.array([[0, 4, 6], [2, 0, 0], [0, 0, 0]], dtype=float))
    meta = pd.DataFrame({"neuron_id": [1, 2, 3], "cell_type": ["A", "B", None]})
    return ConnectivityMatrix(
        neuron_ids=np.array([1, 2, 3]), weights=w, signs=np.ones(3), meta=meta
    )


def test_counts_and_profiles_both_directions():
    pc = partner_counts(_tiny(), ["A", "B", None])
    assert pc.vocab.tolist() == ["A", "B"]
    assert pc.out.toarray()[0].tolist() == [0, 4]
    assert pc.inn.toarray()[0].tolist() == [0, 2]  # from neuron 2 (B)
    prof = pc.profiles(direction="both").toarray()
    assert prof.shape == (3, 4)
    assert np.allclose(prof[0], [0, 1, 0, 1])
    assert np.allclose(prof[2], [0, 0, 1, 0])  # neuron 3: no outputs, input from 1 (A)
    out_only = pc.profiles(direction="out").toarray()
    assert out_only.shape == (3, 2)


def test_neuron_three_in_profile():
    pc = partner_counts(_tiny(), ["A", "B", None])
    assert np.allclose(pc.profiles(direction="in").toarray()[2], [1, 0])


def test_unlabelled_fraction_counts_unnamed_partners():
    pc = partner_counts(_tiny(), ["A", "B", None])
    frac = pc.unlabelled_fraction(direction="out")
    assert frac[0] == pytest.approx(0.6)  # 6 of 10 output synapses go to neuron 3
    assert frac[2] == pytest.approx(1.0)  # no outputs at all


def test_mask_blanks_a_label_and_renormalises():
    pc = partner_counts(_tiny(), ["A", "B", None])
    prof = pc.profiles(direction="both", mask=("B",)).toarray()
    assert np.allclose(prof[0], 0)
    assert np.allclose(prof[1], [1, 0, 1, 0])  # 2 -> 1 (A) and 1 (A) -> 2
    assert pc.column_keep(("B",), "both").tolist() == [1, 0, 1, 0]
    assert pc.touching(("B",)).tolist() == [0]


def test_fixed_vocab_and_rows():
    pc = partner_counts(_tiny(), ["A", "B", None], vocab=["B", "Z"])
    assert pc.vocab.tolist() == ["B", "Z"]
    sub = pc.rows(np.array([1]))
    assert sub.neuron_ids.tolist() == [2]
    assert sub.out.shape == (1, 2)


def test_bad_direction_and_length():
    pc = partner_counts(_tiny(), ["A", "B", None])
    with pytest.raises(ValueError, match="direction"):
        pc.profiles(direction="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="labels"):
        partner_counts(_tiny(), ["A"])


def test_row_normalisers():
    x = sp.csr_matrix(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(l1_rows(x).toarray(), [[3 / 7, 4 / 7], [0, 0]])
    assert np.allclose(l2_rows(x).toarray(), [[0.6, 0.8], [0, 0]])
