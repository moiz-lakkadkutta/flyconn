"""W3 acceptance: male (MaleCNS) vs female (FlyWire v783) connectivity per type, L/R null."""

from __future__ import annotations

import time

import pytest

from flyconn.compare import compare_type, match_types
from flyconn.data.pull import pull
from flyconn.data.store import Store

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def stores() -> tuple[Store, Store]:
    male = Store(pull("malecns@1.0", level="weights").store_dir)
    female = Store(pull("flywire@783", level="weights").store_dir)
    return male, female


def test_type_matching_covers_thousands_of_shared_types(stores: tuple[Store, Store]):
    male, female = stores
    mapping = match_types(male, female)
    n_types = mapping["type_a"].nunique()
    print(f"\nW3 matching: {n_types} MaleCNS types matched to FlyWire v783 via fafb_783_cell_type")
    assert n_types > 6000
    assert (mapping["source"] == "fafb_783_cell_type").all()


@pytest.mark.parametrize("cell_type", ["PFL3", "DNp01", "EPG", "PAM08"])
def test_w3_compare_type_runs_with_verdict_and_caveats(stores: tuple[Store, Store], cell_type: str):
    male, female = stores
    t0 = time.time()
    res = compare_type(
        male, female, cell_type, direction="out", n_permutations=500, seed=0, min_weight=5
    )
    dt = time.time() - t0
    print(f"\nW3 {cell_type}: {res.summary()} ({dt:.0f}s)")
    print(res.partner_differences.head(5).to_string(index=False))
    assert dt < 300
    assert 0 <= res.p_value <= 1
    assert res.verdict
    assert res.caveats
    if cell_type == "DNp01":
        # giant fibre: most MaleCNS output is to VNC neurons, absent from FlyWire's volume
        assert res.unmatched_fraction_a > 0.5
    else:
        assert res.cross_similarity > 0.7  # matched central types share most partners
