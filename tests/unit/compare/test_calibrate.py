"""Logistic calibration, thresholds, decisions and calibration files."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flyconn.compare.atlas import build_atlas
from flyconn.compare.calibrate import (
    Calibration,
    LogisticModel,
    Thresholds,
    apply_calls,
    brier,
    calibration_path,
    choose_delta,
    choose_s_floor,
    decide,
    ece,
    fit_logistic,
    uncalibrated_calls,
)
from flyconn.compare.scoring import Scores
from flyconn.data.store import Store


def _synthetic(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(0, 1, n)
    margin = rng.uniform(0, 0.5, n)
    p_true = 1 / (1 + np.exp(-(-6 + 6 * s1 + 10 * margin)))
    y = (rng.uniform(size=n) < p_true).astype(float)
    return np.column_stack([s1, margin]), y


def test_fit_recovers_monotone_calibrated_model():
    x, y = _synthetic()
    model = fit_logistic(x, y, features=("s1", "margin"))
    assert model.coef[1] > 0 and model.coef[2] > 0
    xt, yt = _synthetic(seed=1)
    p = model.predict(xt)
    assert ece(p, yt) < 0.03
    assert brier(p, yt) < 0.25


def test_fit_needs_both_outcomes():
    with pytest.raises(ValueError, match="both"):
        fit_logistic(np.ones((5, 2)), np.ones(5), features=("s1", "margin"))


@settings(max_examples=30, deadline=None)
@given(s=st.floats(0, 0.9), ds=st.floats(0.01, 0.1), m=st.floats(0, 0.4))
def test_probability_rises_with_score_and_margin(s: float, ds: float, m: float):
    model = LogisticModel(features=("s1", "margin"), coef=np.array([-6.0, 6.0, 10.0]))
    lo = model.predict(np.array([[s, m]]))[0]
    assert model.predict(np.array([[s + ds, m]]))[0] > lo
    assert model.predict(np.array([[s, m + ds]]))[0] > lo


def test_ece_and_brier_on_known_values():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert ece(p, y) == pytest.approx(0.1)
    assert brier(p, y) == pytest.approx(0.01)


def test_threshold_choosers():
    open_s1 = np.linspace(0, 1, 101)
    assert choose_s_floor(open_s1, max_false_accept=0.1) == pytest.approx(0.9)
    assert choose_s_floor(np.array([])) == 0.0
    margin = np.r_[np.full(20, 0.005), np.full(20, 0.03), np.full(20, 0.3)]
    correct = np.r_[np.zeros(20), np.r_[np.ones(8), np.zeros(12)], np.ones(20)]
    assert choose_delta(margin, correct) == pytest.approx(0.05)
    assert choose_delta(np.full(20, 0.3), np.ones(20)) == pytest.approx(0.01)


def test_apply_calls_order_unknown_first():
    thr = Thresholds(accept=0.8, s_floor=0.5, delta=0.05)
    calls = apply_calls(np.array([0.99, 0.99, 0.5]), np.array([0.4, 0.9, 0.9]), thr)
    assert calls.tolist() == ["unknown", "type", "ambiguous"]


def test_decide_lists_ambiguous_candidates():
    sc = Scores(
        labels=np.array([["A", "B", "C"], ["A", "B", "C"]], dtype=object),
        s=np.array([[0.9, 0.88, 0.5], [0.95, 0.3, 0.2]]),
        a=np.array([[0.9, 0.88, 0.5], [0.95, 0.3, 0.2]]),
    )
    model = LogisticModel(features=("s1", "margin"), coef=np.array([-6.0, 6.0, 10.0]))
    thr = Thresholds(accept=0.8, s_floor=0.2, delta=0.05)
    out = decide(sc, model, thr)
    assert out["call"].tolist() == ["ambiguous", "type"]
    assert out["label"].tolist() == ["A|B", "A"]
    unc = uncalibrated_calls(sc)
    assert unc["call"].tolist() == ["uncalibrated", "uncalibrated"]
    assert unc["label"].tolist() == ["A", "A"]
    assert unc["p"].isna().all()


def _cal(atlas_hash: str, params: dict[str, object]) -> Calibration:
    m = LogisticModel(features=("s1", "margin"), coef=np.array([-1.0, 2.0, 3.0]))
    thr = Thresholds(accept=0.8, s_floor=0.3, delta=0.02)
    return Calibration(
        reference="flywire@783",
        atlas_params_hash=atlas_hash,
        atlas_params=params,
        variant="raw",
        models={"neuron": m, "group": m},
        thresholds={"neuron": thr, "group": thr},
        fitted_on=["malecns@1.0"],
        metrics={"transfer_ece": {"malecns@1.0": {"neuron": 0.04, "group": 0.05}}},
        provenance={},
    )


def test_calibration_roundtrip_and_compatibility(tmp_path: Path, male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    cal = _cal(atlas.params_hash, atlas.params)
    path = tmp_path / "c.json"
    cal.to_json(path)
    back = Calibration.from_json(path)
    assert back.id == cal.id
    assert np.allclose(back.models["neuron"].coef, [-1.0, 2.0, 3.0])
    assert back.thresholds["group"].s_floor == 0.3
    assert back.incompatibility(atlas) is None
    other = build_atlas(male_female[1], min_weight=10)
    reason = back.incompatibility(other)
    assert reason is not None and "min_weight" in reason


def test_calibration_path_naming():
    p = calibration_path("flywire@783")
    assert p.name == "classify_flywire_783.json"
    assert p.parent.name == "calibration"
