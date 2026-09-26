"""W1 stability and W4 version drift on real data (golden)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.uncertainty import path_stability
from flyconn.uncertainty.versions import diff_versions

pytestmark = pytest.mark.golden
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json"
SETS = json.loads(_FIXTURE.read_text())["sets"]


@pytest.fixture(scope="module")
def shiu630() -> ConnectivityMatrix:
    store = Store(pull("shiu@630", level="weights").store_dir)
    return ConnectivityMatrix.from_store(store, min_weight=5, sign_policy=SignPolicy.ARGMAX)


def test_w1_stability_sugar_to_mn9_under_thresholds_and_nt_sampling(shiu630: ConnectivityMatrix):
    t0 = time.time()
    res = path_stability(
        shiu630,
        SETS["sugar_grn"],
        SETS["mn9"],
        max_hops=3,
        n_samples=100,
        seed=0,
        thresholds=[5, 10, 20],
        min_edge_fraction=0.01,
    )
    dt = time.time() - t0
    s = res.summary()
    print(
        f"\nW1 stability (FlyWire/Shiu v630): {len(res.paths)} paths, thresholds 5/10/20, {dt:.1f}s"
    )
    print(s.to_string(index=False))
    print(
        res.paths.head(8)[
            ["path", "hops", "threshold_presence", "modal_sign", "sign_stability", "strength_mean"]
        ].to_string(index=False)
    )
    assert dt < 300
    assert len(res.paths) >= 31  # at least the threshold-5 paths from the single-run W1 test
    assert res.paths["threshold_presence"].between(1 / 3, 1).all()
    # Shiu signs are pre-baked with confidence 1 -> sign stability is exactly 1 for every path
    assert (res.paths["sign_stability"] == 1.0).all()


def test_w4_version_drift_v630_to_v783_for_sugar_circuit_neurons():
    old = Store(pull("flywire@630", level="weights").store_dir)
    new = Store(pull("flywire@783", level="weights").store_dir)
    ids = SETS["sugar_grn"] + SETS["mn9"] + SETS["abn1"]
    t0 = time.time()
    d = diff_versions(old, new, neuron_ids=ids)
    dt = time.time() - t0
    summ = d.summary()
    print(f"\nW4 v630->v783 for {len(ids)} neurons of interest in {dt:.1f}s: {summ}")
    print("unmatched:", d.unmatched_ids)
    assert dt < 300
    assert summ["neurons_added"] == 139_255 - len(
        set(old.neurons()["neuron_id"]) & set(new.neurons()["neuron_id"])
    )
    assert summ["neurons_removed"] == 127_978 - len(
        set(old.neurons()["neuron_id"]) & set(new.neurons()["neuron_id"])
    )
    # MN9 ipsilateral root id (720575940645521262) is absent from v783 (checked in M2)
    assert 720575940645521262 in d.unmatched_ids
    assert 720575940660219265 not in d.unmatched_ids
    assert d.caveats
