"""Pure scoring: per-variant labels, axis verdicts, tiers (spec section 2.3)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from flyconn.experiments.robust_scoring import (
    axis_outcome,
    null_outcome,
    score_hits,
    tier,
    variant_label,
)


@pytest.mark.parametrize(
    ("ref", "diff", "lo", "hi", "label"),
    [
        (-10, -8, -12, -4, "holds"),
        (-10, -2, -6, 1, "weakens"),
        (-10, 5, 1, 9, "flips"),
        (-10, 1, -3, 4, "weakens"),
        (10, 8, 4, 12, "holds"),
        (10, 0, -1, 1, "weakens"),
    ],
)
def test_label_table(ref: float, diff: float, lo: float, hi: float, label: str):
    assert variant_label(ref, diff, lo, hi) == label


finite = st.floats(min_value=-50, max_value=50, allow_nan=False).filter(lambda x: abs(x) > 1e-6)


@given(ref=finite, diff=finite, half=st.floats(min_value=0, max_value=30))
def test_negating_the_variant_swaps_holds_and_flips(ref: float, diff: float, half: float):
    lo, hi = diff - half, diff + half
    a = variant_label(ref, diff, lo, hi)
    b = variant_label(ref, -diff, -hi, -lo)
    assert {"holds": "flips", "flips": "holds", "weakens": "weakens"}[a] == b


@given(ref=finite, diff=finite, half=st.floats(min_value=0, max_value=30))
def test_negating_everything_changes_nothing(ref: float, diff: float, half: float):
    lo, hi = diff - half, diff + half
    assert variant_label(ref, diff, lo, hi) == variant_label(-ref, -diff, -hi, -lo)


def test_axis_outcomes():
    assert axis_outcome("w_syn", []) == "not_run"
    assert axis_outcome("w_syn", ["holds", "holds"]) == "robust"
    assert axis_outcome("w_syn", ["holds", "weakens"]) == "fails"
    assert axis_outcome("nt", ["holds"] * 6 + ["weakens"] * 2) == "robust"
    assert axis_outcome("nt", ["holds"] * 5 + ["flips"] * 3) == "fails"
    assert null_outcome(None) == "not_run"
    assert null_outcome(1 / 41) == "robust"
    assert null_outcome(0.05) == "fails"


outcome = st.sampled_from(["robust", "fails", "not_run"])


@given(w=outcome, n=outcome, m=outcome, r=outcome)
def test_every_combination_has_one_tier(w: str, n: str, m: str, r: str):
    oc = {"w_syn": w, "nt": n, "min_weight": m, "random_null": r}
    t, thin = tier(oc)  # type: ignore[arg-type]
    fails = sum(v == "fails" for v in oc.values())
    assert t == ("Strong" if fails == 0 else "Conditional" if fails == 1 else "Fragile")
    assert thin == (sum(v != "not_run" for v in oc.values()) < 2)


def _effects() -> pd.DataFrame:
    rows = [
        ("reference", "reference", "A", "candidate", -10.0, -14.0, -6.0, 0.001),
        ("reference", "reference", "B", "candidate", 5.0, 1.0, 9.0, 0.01),
        ("reference", "reference", "C", "candidate", 1.0, -3.0, 5.0, 0.6),
        ("reference", "reference", "K", "control", -9.0, -12.0, -6.0, 0.001),
        ("w_syn", "wsyn_0.185", "A", "candidate", -9.0, -13.0, -5.0, 0.01),
        ("w_syn", "wsyn_0.209", "A", "candidate", -8.0, -12.0, -4.0, 0.01),
        ("w_syn", "wsyn_0.185", "B", "candidate", 2.0, -1.0, 5.0, 0.3),
        ("w_syn", "wsyn_0.209", "B", "candidate", 4.0, 1.0, 7.0, 0.04),
        ("min_weight", "minw_3", "A", "candidate", -9.0, -13.0, -5.0, 0.01),
        ("min_weight", "minw_3", "B", "candidate", -3.0, -6.0, -1.0, 0.04),
    ]
    cols = ["axis", "variant", "candidate", "role", "difference", "ci_low", "ci_high", "q_value"]
    return pd.DataFrame(rows, columns=pd.Index(cols))


def test_score_hits():
    hits = score_hits(
        _effects(),
        alpha=0.05,
        null_p={"A": 1 / 41, "B": 0.3},
        replicate={"A": "holds", "B": "not testable: unmapped types ['B']"},
    )
    assert list(hits["candidate"]) == ["A", "B"]  # C not a hit, K is a control
    a = hits.iloc[0]
    assert a["tier"] == "Strong" and a["direction"] == "down" and a["nt"] == "not_run"
    assert a["replicate"] == "holds" and not a["thinly_tested"]
    b = hits.iloc[1]
    assert b["tier"] == "Fragile"
    assert set(b["failed_axes"].split(", ")) == {"w_syn", "min_weight", "random_null"}
    assert "wsyn_0.185: weakens" in b["w_syn_detail"]
    assert b["untested_axes"] == "nt"
    assert math.isclose(b["random_null_p"], 0.3)


def test_score_hits_empty():
    eff = _effects()
    eff["q_value"] = 0.9
    assert score_hits(eff, alpha=0.05, null_p={}, replicate={}).empty


def test_score_hits_skips_zero_difference_reference():
    eff = _effects()
    eff.loc[(eff["axis"] == "reference") & (eff["candidate"] == "B"), "difference"] = 0.0
    hits = score_hits(eff, alpha=0.05, null_p={}, replicate={})
    assert list(hits["candidate"]) == ["A"]


def test_score_hits_skips_nan_difference_reference():
    eff = _effects()
    eff.loc[(eff["axis"] == "reference") & (eff["candidate"] == "B"), "difference"] = float("nan")
    hits = score_hits(eff, alpha=0.05, null_p={}, replicate={})
    assert list(hits["candidate"]) == ["A"]
