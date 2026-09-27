"""Effect sizes, control comparisons and multiple-comparison correction for experiment reports."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import stats as sps


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled s.d. (0 when both groups are constant)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va = a.var(ddof=1) if na > 1 else 0.0
    vb = b.var(ddof=1) if nb > 1 else 0.0
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    diff = float(a.mean() - b.mean())
    if pooled == 0:
        return 0.0 if diff == 0 else float(np.sign(diff) * np.inf)
    return diff / float(pooled)


def compare_groups(
    test: np.ndarray, control: np.ndarray, *, seed: int = 0, n_boot: int = 2000
) -> dict[str, float]:
    """Difference of means with a bootstrap 95 % CI, Mann-Whitney p and Cohen's d.

    Infinite ``cohens_d`` (constant groups that differ) is clipped to ±10 so tables stay finite.
    """
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        diffs[k] = rng.choice(test, len(test)).mean() - rng.choice(control, len(control)).mean()
    if np.all(test == test[0]) and np.all(control == control[0]):
        p = 1.0 if test[0] == control[0] else 0.0
    else:
        mw = sps.mannwhitneyu(test, control, alternative="two-sided")
        p = float(cast("Any", mw).pvalue)
    d = cohens_d(test, control)
    d = float(np.clip(d, -10, 10)) if np.isfinite(d) else float(np.sign(d) * 10)
    return {
        "mean_test": float(test.mean()),
        "mean_control": float(control.mean()),
        "difference": float(test.mean() - control.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": p,
        "cohens_d": d,
        "n_test": len(test),
        "n_control": len(control),
    }


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(q_sorted, 1.0)
    return q
