"""Hemibrain v1.2.1 and MANC v1.2.1 on real data: counts against the releases, and compare_type.

Expected numbers and their sources:

* hemibrain v1.2 export README / files: 21,739 non-cropped Traced bodies, 3,550,403 edges,
  summed weight 14,329,229; 662,578 edges with weight >= 5 (computed in
  docs/research/data_flywire_and_others.md section 2.3). The body-mean NT feather covers 21,709
  of the traced bodies, 9,577 of them argmax acetylcholine (same section).
* MANC v1.2.1: 23,665 Traced bodies (neuPrint manc:v1.2.1 and Codex); the sjcabs meta has
  23,650 rows (Traced minus 15 glia) and the simple edgelist 5,305,354 rows (research note
  section 6). neuPrint Traced predictedNt acetylcholine 11,518 (research note section 3.3).
"""

from __future__ import annotations

import time

import pytest

from flyconn.compare import compare_type
from flyconn.data.pull import pull
from flyconn.data.store import Store

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def hemibrain() -> Store:
    t0 = time.time()
    r = pull("hemibrain@1.2.1", level="weights")
    print(f"\nhemibrain@1.2.1 pull: converted={r.converted} in {time.time() - t0:.0f}s")
    return Store(r.store_dir)


@pytest.fixture(scope="module")
def manc() -> Store:
    t0 = time.time()
    r = pull("manc@1.2.1", level="weights")
    print(f"\nmanc@1.2.1 pull: converted={r.converted} in {time.time() - t0:.0f}s")
    return Store(r.store_dir)


def test_hemibrain_counts(hemibrain: Store):
    c = hemibrain.provenance["counts"]
    print("hemibrain counts:", c)
    assert c["neurons"] == 21_739
    assert c["raw_edges"] == c["edges"] == 3_550_403
    assert c["raw_weight_sum"] == 14_329_229
    assert c["with_nt"] == 21_709
    e5 = hemibrain.query("SELECT count(*) n FROM edges WHERE weight >= 5")["n"].iloc[0]
    assert e5 == 662_578
    nt = hemibrain.query("SELECT nt_pred, count(*) n FROM neurons GROUP BY 1").set_index("nt_pred")[
        "n"
    ]
    assert nt["acetylcholine"] == 9_577
    assert hemibrain.provenance["edge_threshold"]["postHighAccuracyThreshold"] == 0


def test_manc_counts(manc: Store):
    c = manc.provenance["counts"]
    print("MANC counts:", c)
    assert c["traced_codex"] == 23_665
    assert c["neurons"] == 23_650
    assert c["glia_excluded"] == 15
    assert c["raw_edges"] == 5_305_354
    nt = manc.query("SELECT nt_pred, count(*) n FROM neurons GROUP BY 1").set_index("nt_pred")["n"]
    assert nt["acetylcholine"] == 11_518
    assert set(nt.index.dropna()) == {"acetylcholine", "gaba", "glutamate"}
    same = manc.query(
        "SELECT count(*) n FROM neurons WHERE manc_121_cell_type IS NOT DISTINCT FROM cell_type"
    )["n"].iloc[0]
    assert same == c["neurons"]


@pytest.mark.parametrize(("other", "cell_type"), [("hemibrain", "EPG"), ("manc", "DNa02")])
def test_compare_type_against_malecns(hemibrain: Store, manc: Store, other: str, cell_type: str):
    male = Store(pull("malecns@1.0", level="weights").store_dir)
    b = hemibrain if other == "hemibrain" else manc
    t0 = time.time()
    res = compare_type(
        male, b, cell_type, direction="out", n_permutations=200, seed=0, min_weight=5
    )
    print(f"\nMaleCNS vs {b.ref} {cell_type} ({time.time() - t0:.0f}s): {res.summary()}")
    assert res.n_a >= 2 and res.n_b >= 2
    assert res.unmatched_fraction_b < 0.05
