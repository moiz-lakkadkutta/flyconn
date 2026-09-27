"""Validation rung 2: reproduce Shiu et al. 2024 golden results on FlyWire v630.

Model predictions from wiring plus predicted transmitters; 30 trials x 1 s as in the paper.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import LIFNetwork, simulate

pytestmark = pytest.mark.golden
SETS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json").read_text()
)["sets"]
MN9_CONTRA, MN9_IPSI = SETS["mn9"]


@pytest.fixture(scope="module")
def net() -> LIFNetwork:
    m = ConnectivityMatrix.from_store(
        Store(pull("shiu@630", level="weights").store_dir), sign_policy=SignPolicy.ARGMAX
    )
    return LIFNetwork.from_matrix(m)


def _run(
    net: LIFNetwork, rate: float, n_trials: int = 30, device: str = "cpu"
) -> tuple[np.ndarray, float]:
    t0 = time.time()
    res = simulate(
        net,
        stimulate=dict.fromkeys(SETS["sugar_grn"], rate),
        n_steps=10_000,
        n_trials=n_trials,
        seed=0,
        dtype="float32",
        device=device,
        record=False,
    )
    return res.counts, time.time() - t0


def test_sugar_100hz_drives_mn9_within_published_band(net: LIFNetwork):
    counts, dt = _run(net, 100.0)
    ids = list(net.neuron_ids)
    contra = counts[:, ids.index(MN9_CONTRA)].astype(float)
    ipsi = counts[:, ids.index(MN9_IPSI)].astype(float)
    active = int((counts.sum(axis=0) > 0).sum())
    print(
        f"\nShiu 100 Hz sugar, 30 trials, {dt:.0f}s: MN9 contra {contra.mean():.1f} +/- "
        f"{contra.std(ddof=1):.1f} Hz, ipsi {ipsi.mean():.1f} Hz, active {active}, "
        f"spikes {int(counts.sum())}"
    )
    assert 58 <= contra.mean() <= 76  # GOLDEN_RESULTS: 65.7 (ST 1A) / 67.0 (repo) Hz
    assert contra.mean() > ipsi.mean()  # Fig. 1c
    assert abs(active - 404) / 404 < 0.10  # repo example: 404 neurons spiked at 100 Hz (allow 10 %)


def test_sugar_200hz_drives_mn9_within_published_band(net: LIFNetwork):
    counts, dt = _run(net, 200.0)
    ids = list(net.neuron_ids)
    contra = counts[:, ids.index(MN9_CONTRA)].astype(float)
    active = int((counts.sum(axis=0) > 0).sum())
    print(f"\nShiu 200 Hz, 30 trials, {dt:.0f}s: MN9 {contra.mean():.1f} Hz, active {active}")
    assert 83 <= contra.mean() <= 103  # GOLDEN_RESULTS: 93.2 Hz
    assert abs(active - 455) / 455 < 0.10  # paper: 455 activated at 200 Hz (incl. GRNs)


def test_sugar_10hz_barely_activates_network(net: LIFNetwork):
    counts, dt = _run(net, 10.0)
    ids = list(net.neuron_ids)
    contra = counts[:, ids.index(MN9_CONTRA)].astype(float)
    active = int((counts.sum(axis=0) > 0).sum())
    print(f"\nShiu 10 Hz, 30 trials, {dt:.0f}s: MN9 {contra.mean():.2f} Hz, active {active}")
    assert contra.mean() < 1.0  # ST 1A: 0 Hz
    assert abs(active - 45) / 45 < 0.25  # paper: 45 activated at 10 Hz (noisy at the margin)
