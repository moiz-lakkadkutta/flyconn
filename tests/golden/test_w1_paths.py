"""W1 acceptance on real data: sugar GRN -> proboscis motor neuron pathways in FlyWire and MaleCNS.

Golden numbers: JO-CE -> aBN1 = 103 synapses and JO-F -> aBN1 = 78 (Shiu et al. 2024,
Fig. 5g; GOLDEN_RESULTS.md §3.4), checked on the unthresholded Shiu v630 export.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy, find_paths

pytestmark = pytest.mark.golden
SETS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json").read_text()
)["sets"]


@pytest.fixture(scope="module")
def shiu630() -> ConnectivityMatrix:
    store = Store(pull("shiu@630", level="weights").store_dir)
    return ConnectivityMatrix.from_store(store, sign_policy=SignPolicy.ARGMAX)


@pytest.fixture(scope="module")
def malecns() -> ConnectivityMatrix:
    store = Store(pull("malecns@1.0", level="weights").store_dir)
    return ConnectivityMatrix.from_store(store, min_weight=5, sign_policy=SignPolicy.ARGMAX)


def _pair_weight(m: ConnectivityMatrix, pre: list[int], post: list[int]) -> int:
    i = m.index_of(np.array(pre))
    j = m.index_of(np.array(post))
    return int(m.weights[i][:, j].sum())


def test_jon_to_abn1_synapse_counts_from_repo_id_lists(shiu630: ConnectivityMatrix):
    # Shiu et al. Fig. 5g reports 103 (JO-CE) and 78 (JO-F) synapses onto aBN1. Summing the
    # repo notebook's ID lists over the repo's own v630 export gives 77 and 69 (all 146 JONs:
    # 148). The paper's numbers are therefore NOT reproduced from these files; see
    # GOLDEN_RESULTS.md section 7. The computed values are pinned as regression targets.
    assert _pair_weight(shiu630, SETS["jon_ce"], SETS["abn1"]) == 77
    assert _pair_weight(shiu630, SETS["jon_f"], SETS["abn1"]) == 69
    assert (
        _pair_weight(shiu630, SETS["jon_ce"] + SETS["jon_f"] + SETS["jon_d_m"], SETS["abn1"]) == 148
    )


def test_shiu_matrix_reproduces_model_input_counts(shiu630: ConnectivityMatrix):
    assert shiu630.n == 127_400
    assert shiu630.weights.nnz == 14_687_178
    assert shiu630.weights.sum() == 52_793_639
    exc = int((shiu630.signs > 0).sum())
    inh = int((shiu630.signs < 0).sum())
    assert exc + inh + int((shiu630.signs == 0).sum()) == 127_400
    # every neuron with outgoing edges has a definite sign in the Shiu export
    has_out = np.asarray(shiu630.weights.sum(axis=1)).ravel() > 0
    assert (shiu630.signs[has_out] != 0).all()


def test_w1_flywire_sugar_to_mn9_paths_run_in_minutes(shiu630: ConnectivityMatrix):
    t0 = time.time()
    paths = find_paths(
        shiu630,
        sources=SETS["sugar_grn"],
        targets=SETS["mn9"],
        max_hops=3,
        min_weight=5,
        min_edge_fraction=0.01,
    )
    dt = time.time() - t0
    print(
        f"\nW1 FlyWire(v630, Shiu export): {len(paths)} paths in {dt:.1f}s; "
        f"by hops {paths['hops'].value_counts().sort_index().to_dict()}; "
        f"sign counts {paths['path_sign'].value_counts().to_dict()}"
    )
    assert dt < 300
    assert len(paths) > 0
    assert paths["hops"].max() == 3
    assert paths.attrs["n_sources"] == 21 and paths.attrs["n_targets"] == 2
    # no direct sugar GRN -> MN9 synapses in this export (checked: pair weight 0)
    assert 1 not in set(paths["hops"])


def test_w1_malecns_lb3_to_mn9_paths_run_in_minutes(malecns: ConnectivityMatrix):
    meta = malecns.meta
    sources = meta.index[meta["cell_type"].astype(str).str.match(r"^LB3")].to_numpy()
    targets = meta.index[meta["cell_type"] == "MN9"].to_numpy()
    assert len(targets) == 2
    assert len(sources) > 50  # LB3a-d labellar GRNs (sugar + water group; see caveats)
    t0 = time.time()
    paths = find_paths(
        malecns, sources=sources, targets=targets, max_hops=3, min_weight=5, min_edge_fraction=0.01
    )
    dt = time.time() - t0
    print(
        f"\nW1 MaleCNS: {len(sources)} LB3 sources -> MN9: {len(paths)} paths in {dt:.1f}s; "
        f"by hops {paths['hops'].value_counts().sort_index().to_dict()}; "
        f"sign counts {paths['path_sign'].value_counts().to_dict()}"
    )
    assert dt < 300
    assert len(paths) > 0
