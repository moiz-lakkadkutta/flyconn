"""Random-silencing null: size/super_class-matched sets and empirical p (spec 2.2/2.3)."""

from __future__ import annotations

import pandas as pd

from flyconn.experiments.robust_null import draw_null_sets, null_key, null_p, size_bin
from flyconn.experiments.sweep import Candidate


def _neurons() -> pd.DataFrame:
    ids = list(range(1, 41))
    sc = ["central"] * 30 + ["descending"] * 10
    return pd.DataFrame({"neuron_id": ids, "super_class": sc})


def test_size_bins():
    assert [size_bin(n) for n in (1, 2, 3, 5, 6, 12, 100)] == [1, 2, 4, 4, 8, 16, 128]


def test_sets_match_bin_and_super_class_and_exclude():
    n = _neurons()
    cands = [Candidate("A", (1, 2, 3)), Candidate("D", (31,)), Candidate("K", (4,), "control")]
    excluded = {5, 6, 7}
    sets, skipped = draw_null_sets(n, cands, excluded, n_sets=5, seed=0)
    assert skipped == {}
    sc = dict(zip(n["neuron_id"], n["super_class"], strict=True))
    keys = {s.key for s in sets}
    assert keys == {"central|4", "descending|1", "central|1"}  # control K gets a null too
    assert null_key(n, (1, 2, 3)) == "central|4" and null_key(n, (31,)) == "descending|1"
    for s in sets:
        want_sc, want_bin = s.key.split("|")
        assert len(s.neuron_ids) == int(want_bin)
        assert all(sc[i] == want_sc for i in s.neuron_ids)
        assert not set(s.neuron_ids) & ({1, 2, 3, 31, 4} | excluded)
    assert sum(s.key == "central|4" for s in sets) == 5


def test_deterministic():
    n = _neurons()
    c = [Candidate("A", (1, 2))]
    assert draw_null_sets(n, c, set(), n_sets=4, seed=3) == draw_null_sets(
        n, c, set(), n_sets=4, seed=3
    )
    assert draw_null_sets(n, c, set(), n_sets=4, seed=3) != draw_null_sets(
        n, c, set(), n_sets=4, seed=4
    )


def test_pool_too_small_is_skipped_not_crash():
    n = _neurons()
    sets, skipped = draw_null_sets(
        n, [Candidate("D", tuple(range(31, 41)))], set(), n_sets=3, seed=0
    )
    assert sets == [] and "descending|8" in skipped
    assert "pool" in skipped["descending|8"]


def test_null_p():
    assert null_p(-10.0, [-1.0] * 40) == 1 / 41
    assert null_p(-10.0, [-20.0] * 40) == 1.0
    assert null_p(5.0, [6.0, 1.0, 1.0]) == 2 / 4


def test_null_p_ignores_nan_nulls_and_nan_effect():
    import math

    assert null_p(-10.0, [-1.0, float("nan"), -20.0]) == 2 / 3
    assert math.isnan(null_p(float("nan"), [-1.0, -2.0]))
    assert math.isnan(null_p(-10.0, [float("nan")] * 3))


def test_sets_of_a_key_do_not_depend_on_other_candidates():
    n = _neurons()
    # D's key "descending|2" sorts after B's "central|1", and B's neuron is not descending,
    # so adding B changes nothing about D's pool: only a seed tied to key order could differ
    alone, _ = draw_null_sets(n, [Candidate("D", (31, 32))], set(), n_sets=4, seed=0)
    more, _ = draw_null_sets(
        n, [Candidate("D", (31, 32)), Candidate("B", (1,))], set(), n_sets=4, seed=0
    )
    key = null_key(n, (31, 32))
    assert [s for s in alone if s.key == key] == [s for s in more if s.key == key]
