"""Tests for effect sizes and control comparisons used in experiment reports."""

import numpy as np

from flyconn.experiments.stats import benjamini_hochberg, cohens_d, compare_groups


def test_cohens_d_and_sign():
    a = np.array([10.0, 12.0, 11.0, 13.0])
    b = np.array([5.0, 6.0, 5.5, 6.5])
    d = cohens_d(a, b)
    assert d > 3
    assert cohens_d(b, a) == -d
    assert cohens_d(a, a) == 0.0


def test_compare_groups_reports_difference_ci_and_pvalue():
    rng = np.random.default_rng(0)
    test = rng.normal(20, 2, size=30)
    ctrl = rng.normal(10, 2, size=30)
    r = compare_groups(test, ctrl, seed=0)
    assert r["mean_test"] > r["mean_control"]
    assert r["difference"] > 8
    assert r["ci_low"] < r["difference"] < r["ci_high"]
    assert r["p_value"] < 0.001
    assert r["cohens_d"] > 3
    assert r["n_test"] == 30 and r["n_control"] == 30


def test_compare_groups_handles_all_zero_control():
    r = compare_groups(np.array([1.0, 2.0, 3.0]), np.zeros(5), seed=0)
    assert r["difference"] == 2.0
    assert np.isfinite(r["cohens_d"])


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.3, 0.9])
    q = benjamini_hochberg(p)
    assert (q >= p).all() and (q <= 1).all()
    assert (np.diff(q[np.argsort(p)]) >= 0).all()
    np.testing.assert_allclose(q, [0.005, 0.025, 0.03333333, 0.375, 0.9], rtol=1e-6)
