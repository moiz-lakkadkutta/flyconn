"""Tests for 1-3 hop path enumeration with weights and signs."""

import numpy as np
import pandas as pd
import pytest

from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.graph.paths import find_paths


def _toy() -> ConnectivityMatrix:
    # A(ACh) -> B(GABA) -> C(ACh);  A -> C;  B -> D(?) -> C;  C -> A (cycle)
    ids = [1, 2, 3, 4]
    neurons = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "neuron_id": ids,
            "cell_type": ["A", "B", "C", "D"],
            "nt_pred": ["acetylcholine", "gaba", "acetylcholine", None],
            "input_synapses_total": [10.0, 20.0, 40.0, 10.0],
        }
    )
    edges = pd.DataFrame(
        {
            "dataset": "t",
            "version": "0",
            "pre": [1, 2, 1, 2, 4, 3],
            "post": [2, 3, 3, 4, 3, 1],
            "weight": [10, 8, 4, 5, 6, 2],
        }
    )
    return ConnectivityMatrix.from_frames(neurons, edges, sign_policy=SignPolicy.ARGMAX)


def test_direct_paths_only_with_one_hop():
    m = _toy()
    p = find_paths(m, sources=[1], targets=[3], max_hops=1)
    assert len(p) == 1
    assert p.iloc[0]["path"] == (1, 3)
    assert p.iloc[0]["hops"] == 1
    assert p.iloc[0]["weights"] == (4,)


def test_all_paths_up_to_three_hops_without_revisiting_nodes():
    m = _toy()
    p = find_paths(m, sources=[1], targets=[3], max_hops=3)
    paths = set(p["path"])
    assert paths == {(1, 3), (1, 2, 3), (1, 2, 4, 3)}
    assert (p["hops"] == p["path"].map(len) - 1).all()


def test_path_sign_is_product_of_presynaptic_signs_and_unknown_gives_zero():
    m = _toy()
    p = find_paths(m, sources=[1], targets=[3], max_hops=3)
    by = {row["path"]: row for row in p.to_dict("records")}
    assert by[(1, 3)]["signs"] == (1,)
    assert by[(1, 3)]["path_sign"] == 1
    assert by[(1, 2, 3)]["signs"] == (1, -1)
    assert by[(1, 2, 3)]["path_sign"] == -1
    assert by[(1, 2, 4, 3)]["signs"] == (1, -1, 0)
    assert by[(1, 2, 4, 3)]["path_sign"] == 0


def test_path_strength_is_product_of_input_fractions():
    m = _toy()
    p = find_paths(m, sources=[1], targets=[3], max_hops=2)
    by = {row["path"]: row for row in p.to_dict("records")}
    # A->B: 10/20, B->C: 8/40
    assert by[(1, 2, 3)]["strength"] == pytest.approx(0.5 * 0.2)
    assert by[(1, 3)]["strength"] == pytest.approx(4 / 40)
    assert by[(1, 2, 3)]["min_weight"] == 8


def test_min_weight_prunes_edges_along_paths():
    m = _toy()
    p = find_paths(m, sources=[1], targets=[3], max_hops=3, min_weight=5)
    assert set(p["path"]) == {(1, 2, 3), (1, 2, 4, 3)}  # A->C (4) pruned


def test_results_sorted_by_strength_and_carry_labels():
    m = _toy()
    p = find_paths(m, sources=[1], targets=[3], max_hops=3, label="cell_type")
    assert list(p["strength"]) == sorted(p["strength"], reverse=True)
    assert p.iloc[0]["labels"][0] == "A"
    assert isinstance(p["path"].iloc[0], tuple)
    assert np.issubdtype(p["hops"].dtype, np.integer)
