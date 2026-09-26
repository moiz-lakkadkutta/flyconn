"""Tests for neurotransmitter uncertainty models and sign sampling (ADR-0008)."""

import numpy as np
import pandas as pd
import pytest

from flyconn.data.schema import NT_CLASSES
from flyconn.uncertainty.nt import confidence_model, nt_probabilities, sample_signs


def _neurons() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "neuron_id": [1, 2, 3, 4],
            "nt_pred": ["acetylcholine", "gaba", None, "glutamate"],
            "nt_conf": [0.9, 0.6, np.nan, 1.0],
        }
    )
    for nt in NT_CLASSES:
        df[f"nt_p_{nt}"] = np.nan
    # neuron 1 has real probabilities; others do not
    df.loc[0, "nt_p_acetylcholine"] = 0.7
    df.loc[0, "nt_p_gaba"] = 0.3
    return df


def test_confidence_model_puts_confidence_on_argmax_and_spreads_the_rest():
    p = confidence_model(_neurons())
    assert p.shape == (4, len(NT_CLASSES))
    row = p[1]  # gaba with conf 0.6
    assert row[NT_CLASSES.index("gaba")] == pytest.approx(0.6)
    others = np.delete(row, NT_CLASSES.index("gaba"))
    np.testing.assert_allclose(others, 0.4 / (len(NT_CLASSES) - 1))
    assert np.isnan(p[2]).all()  # unknown stays unknown
    assert p[3][NT_CLASSES.index("glutamate")] == pytest.approx(1.0)


def test_nt_probabilities_prefers_real_probabilities_and_reports_model():
    p, model = nt_probabilities(_neurons())
    assert model[0] == "probabilities"
    assert model[1] == "confidence"
    assert model[2] == "unknown"
    np.testing.assert_allclose(p[0][:3], [0.7, 0.0, 0.3])  # ach, glut, gaba order
    assert np.isnan(p[2]).all()


def test_sample_signs_is_seeded_and_respects_probabilities():
    neurons = _neurons()
    a = np.stack(list(sample_signs(neurons, n=200, seed=1)))
    b = np.stack(list(sample_signs(neurons, n=200, seed=1)))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (200, 4)
    # neuron 1: P(+1) = 0.7 -> mean sign about 0.4
    assert abs(a[:, 0].mean() - 0.4) < 0.15
    # neuron 3 unknown -> always 0
    assert (a[:, 2] == 0).all()
    # neuron 4: conf 1.0 glutamate -> always -1
    assert (a[:, 3] == -1).all()


def test_sample_signs_can_force_the_confidence_model():
    neurons = _neurons()
    a = np.stack(list(sample_signs(neurons, n=300, seed=2, model="confidence")))
    # neuron 1 under confidence model: P(ach)=0.9 -> mean sign about 0.9*1 + 0.1*(mean of others)
    assert a[:, 0].mean() > 0.6
