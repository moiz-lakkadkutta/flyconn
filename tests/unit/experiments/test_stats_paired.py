"""Paired comparison across shared Poisson input streams (M14 D3)."""

from __future__ import annotations

import numpy as np
import pytest

from flyconn.experiments.stats import compare_groups, compare_paired


def test_paired_detects_shift_that_unpaired_misses():
    rng = np.random.default_rng(0)
    control = rng.normal(20, 8, size=10)  # large trial-to-trial spread from shared inputs
    test = control - 1.0 + rng.normal(0, 0.1, size=10)  # small, consistent per-trial drop
    paired = compare_paired(test, control, seed=0)
    unpaired = compare_groups(test, control, seed=0)
    assert paired["difference"] == pytest.approx(unpaired["difference"])
    assert paired["p_value"] < 0.01 < unpaired["p_value"]
    assert paired["ci_high"] < 0
    assert paired["ci_high"] - paired["ci_low"] < unpaired["ci_high"] - unpaired["ci_low"]


def test_paired_all_zero_differences():
    x = np.array([3.0, 5.0, 0.0, 7.0])
    r = compare_paired(x, x.copy())
    assert r["difference"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["p_value"] == 1.0


def test_paired_same_keys_as_unpaired():
    a, b = np.arange(6.0), np.arange(6.0)[::-1]
    assert set(compare_paired(a, b)) == set(compare_groups(a, b))


def test_paired_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal trial counts, got 3 and 4"):
        compare_paired(np.ones(3), np.ones(4))
