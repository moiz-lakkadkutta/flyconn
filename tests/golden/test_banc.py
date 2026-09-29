"""BANC v888 on real data: counts against the paper, and W3 male (MaleCNS) vs female (BANC)."""

from __future__ import annotations

import time

import pytest

from flyconn.compare import compare_type
from flyconn.data.pull import pull
from flyconn.data.store import Store

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def banc() -> Store:
    t0 = time.time()
    r = pull("banc@888", level="weights")
    print(f"\nbanc@888 pull: converted={r.converted} in {time.time() - t0:.0f}s")
    return Store(r.store_dir)


def test_banc_counts(banc: Store):
    c = banc.provenance["counts"]
    print("BANC counts:", c)
    assert (
        c["neurons"] == 155_858
    )  # paper: 155,916 proofread + roughly proofread (Bates et al. 2026)
    assert c["raw_edges"] == 11_752_828
    sc = banc.query("SELECT super_class, count(*) n FROM neurons GROUP BY 1").set_index(
        "super_class"
    )["n"]
    print(sc.to_dict())
    assert abs(sc["descending"] - 1_316) <= 20  # paper: 1,316 DNs
    assert abs(sc["ascending"] - 1_849) <= 20  # paper: 1,849 ANs


@pytest.mark.parametrize("cell_type", ["DNp01", "PFL3", "EPG"])
def test_w3_malecns_vs_banc_with_shared_vocabulary(banc: Store, cell_type: str):
    male = Store(pull("malecns@1.0", level="weights").store_dir)
    t0 = time.time()
    res = compare_type(
        male,
        banc,
        cell_type,
        type_b=cell_type,
        direction="out",
        n_permutations=500,
        seed=0,
        min_weight=5,
        partner_vocabulary="fafb_or_manc",
    )
    print(f"\nW3 MaleCNS vs BANC {cell_type} ({time.time() - t0:.0f}s): {res.summary()}")
    print(res.partner_differences.head(5).to_string(index=False))
    assert res.n_a >= 2 and res.n_b >= 2
    if cell_type == "DNp01":
        # With nerve-cord partners on both sides, most output is now comparable
        # (vs 58 % unmatched against brain-only FlyWire).
        assert res.unmatched_fraction_a < 0.3
