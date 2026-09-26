"""MaleCNS per-body NT probabilities (opt-in --level nt-probs) and sampled-sign stability."""

from __future__ import annotations

import time

import numpy as np
import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.uncertainty import nt_probabilities, path_stability

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def malecns_probs() -> Store:
    t0 = time.time()
    r = pull("malecns@1.0", level="nt-probs")
    print(f"\nmalecns nt-probs pull: converted={r.converted} in {time.time() - t0:.0f}s")
    return Store(r.store_dir)


def test_probabilities_cover_neurons_with_presynapses_and_agree_with_consensus(
    malecns_probs: Store,
):
    n = malecns_probs.neurons()
    probs, model = nt_probabilities(n)
    counts = dict(zip(*np.unique(model, return_counts=True), strict=True))
    print("NT model per neuron:", counts)
    assert counts["probabilities"] > 160_000  # 164,620 traced bodies have presynapses (Phase 0)
    # argmax of the mean probabilities agrees with the body-level argmax for most neurons
    from flyconn.data.schema import NT_CLASSES

    has = model == "probabilities"
    argmax = np.array(NT_CLASSES)[np.nanargmax(probs[has], axis=1)]
    consensus = n.loc[has, "nt_pred"].to_numpy(dtype=object)
    known = np.array([c is not None for c in consensus])
    agree = (argmax[known] == consensus[known]).mean()
    print(f"argmax(mean probs) == consensus for {agree:.3f} of neurons with a consensus label")
    assert agree > 0.85
    assert malecns_probs.provenance["nt_probs"]["bodies_with_probabilities"] > 160_000


def test_sampled_signs_change_path_stability_for_lb3_to_mn9(malecns_probs: Store):
    m = ConnectivityMatrix.from_store(malecns_probs, min_weight=5, sign_policy=SignPolicy.ARGMAX)
    meta = m.meta
    sources = meta.index[meta["cell_type"].astype(str).str.match(r"^LB3")].to_numpy()
    targets = meta.index[meta["cell_type"] == "MN9"].to_numpy()
    res = path_stability(
        m,
        sources,
        targets,
        max_hops=3,
        n_samples=200,
        seed=0,
        thresholds=[5, 10, 20],
        min_edge_fraction=0.01,
        label="cell_type",
    )
    print(res.summary().to_string(index=False))
    print(
        res.paths[["labels", "modal_sign", "sign_stability", "threshold_presence"]]
        .head(10)
        .to_string(index=False)
    )
    assert (res.paths["sign_stability"] < 1.0).any()  # real uncertainty shows up
    assert res.paths["sign_stability"].min() > 0.3
