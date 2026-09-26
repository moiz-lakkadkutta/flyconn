"""Tests for the stability wrapper over path analyses."""

import numpy as np
import pandas as pd

from flyconn.graph import ConnectivityMatrix
from flyconn.uncertainty.stability import path_stability


def _toy() -> ConnectivityMatrix:
    neurons = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "neuron_id": [1, 2, 3, 4],
            "nt_pred": ["acetylcholine", "gaba", "acetylcholine", "acetylcholine"],
            "nt_conf": [1.0, 0.5, 1.0, 1.0],
            "input_synapses_total": [10.0, 20.0, 40.0, 10.0],
        }
    )
    edges = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "pre": [1, 2, 1, 2, 4],
            "post": [2, 3, 3, 4, 3],
            "weight": [10, 8, 4, 5, 6],
        }
    )
    return ConnectivityMatrix.from_frames(neurons, edges)


def test_path_stability_scores_sign_and_presence_across_nt_samples():
    m = _toy()
    res = path_stability(m, sources=[1], targets=[3], max_hops=3, n_samples=100, seed=0)
    df = res.paths
    assert set(df["path"]) == {(1, 3), (1, 2, 3), (1, 2, 4, 3)}
    by = df.set_index("path")
    # A->C has no uncertain neuron on the way: always present, sign +1
    assert by.loc[[(1, 3)], "presence"].iloc[0] == 1.0
    assert by.loc[[(1, 3)], "sign_stability"].iloc[0] == 1.0
    # A->B->C: B is gaba with conf 0.5 -> sign flips in roughly half the samples
    s = by.loc[[(1, 2, 3)], "sign_stability"].iloc[0]
    assert 0.3 < s < 0.85
    assert by.loc[[(1, 2, 3)], "modal_sign"].iloc[0] in (-1, 1)
    assert res.n_samples == 100
    assert res.provenance["seed"] == 0


def test_path_stability_threshold_sweep_reports_presence_per_threshold():
    m = _toy()
    res = path_stability(
        m, sources=[1], targets=[3], max_hops=3, n_samples=10, seed=0, thresholds=[1, 5, 9]
    )
    by = res.paths.set_index("path")
    # A->C weight 4 survives only threshold 1 -> threshold presence 1/3
    assert by.loc[[(1, 3)], "threshold_presence"].iloc[0] == 1 / 3
    assert by.loc[[(1, 2, 3)], "threshold_presence"].iloc[0] == 2 / 3  # weights 10, 8
    assert res.thresholds == (1, 5, 9)
    assert isinstance(res.summary(), pd.DataFrame)
    assert np.isfinite(res.paths["strength_mean"]).all()
