"""Tests for the re-calibration protocol (ADR-0004): pick w_syn from a readout target."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyconn.sim import LIFNetwork, ShiuParams
from flyconn.sim.calibrate import CalibrationResult, calibrate_w_syn, select_w_syn


def _chain() -> LIFNetwork:
    # input(1) -> relay(2) -> readout(3); weights in synapse counts (signed), so w_syn scales them
    w = np.zeros((3, 3))
    w[0, 1] = 150.0
    w[1, 2] = 150.0
    return LIFNetwork(np.array([1, 2, 3]), sp.csr_matrix(w), ShiuParams())


def test_calibration_scans_w_syn_and_reports_readout_curve():
    net = _chain()
    res = calibrate_w_syn(
        net,
        stimulate={1: 100.0},
        readout=[3],
        w_syn_grid_mv=[0.005, 0.275, 0.6],
        rates_hz=[100.0, 200.0],
        n_steps=2000,
        n_trials=2,
        seed=0,
        device="cpu",
        dtype="float64",
    )
    assert isinstance(res, CalibrationResult)
    curve = res.curve
    assert isinstance(curve, pd.DataFrame)
    assert set(curve.columns) >= {"w_syn_mv", "rate_hz", "readout_rate_hz"}
    assert len(curve) == 6
    # 0.005 mV per synapse keeps v below threshold even at 200 Hz drive; 0.6 mV fires
    at200 = curve[curve["rate_hz"] == 200.0].set_index("w_syn_mv")["readout_rate_hz"]
    silent = at200.loc[0.005]
    loud = at200.loc[0.6]
    assert silent == 0.0 and loud > 0.0


def test_calibration_picks_w_syn_closest_to_target_fraction_of_max():
    net = _chain()
    res = calibrate_w_syn(
        net,
        stimulate={1: 100.0},
        readout=[3],
        w_syn_grid_mv=[0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
        rates_hz=[100.0, 200.0],
        target_fraction=0.8,
        reference_rate_hz=100.0,
        max_rate_hz=200.0,
        n_steps=2000,
        n_trials=2,
        seed=0,
        device="cpu",
        dtype="float64",
    )
    assert res.w_syn_mv in {0.1, 0.2, 0.3, 0.4, 0.6, 0.8}
    assert 0 <= res.achieved_fraction <= 1.5
    assert res.provenance["protocol"].startswith("Shiu")
    assert res.provenance["status"] in {"calibrated", "unresolved"}
    assert "uncalibrated" not in res.summary().lower() or res.provenance["status"] == "unresolved"


def _trials(spec: dict[float, tuple[list[float], list[float]]]) -> pd.DataFrame:
    """Per-trial readout table from {w: (rates at reference, rates at max)}."""
    rows = []
    for w, (ref, mx) in spec.items():
        for rate_hz, values in ((100.0, ref), (200.0, mx)):
            for t, v in enumerate(values):
                rows.append({"w_syn_mv": w, "rate_hz": rate_hz, "trial": t, "readout_rate_hz": v})
    return pd.DataFrame(rows)


def test_select_interpolates_a_unique_crossing_and_bootstraps_its_interval():
    trials = _trials(
        {
            0.1: ([10.0, 11.0, 9.0, 10.0], [100.0, 101.0, 99.0, 100.0]),  # 0.10
            0.2: ([60.0, 61.0, 59.0, 60.0], [100.0, 101.0, 99.0, 100.0]),  # 0.60
            0.3: ([100.0, 101.0, 99.0, 100.0], [100.0, 101.0, 99.0, 100.0]),  # 1.00
        }
    )
    sel = select_w_syn(trials, target_fraction=0.8, n_boot=200, seed=0)
    assert sel.status == "calibrated"
    assert sel.crossing_mv == pytest.approx(0.25, abs=1e-6)
    assert sel.w_syn_mv == pytest.approx(0.25, abs=1e-6)
    lo, hi = sel.crossing_ci_mv
    assert lo <= 0.25 <= hi and hi - lo < 0.02
    assert sel.crossing_support > 0.95
    assert list(sel.fractions.columns) == ["w_syn_mv", "fraction", "ci_low", "ci_high"]
    row = sel.fractions.set_index("w_syn_mv").loc[0.2]
    assert row["ci_low"] < 0.6 < row["ci_high"]


def test_select_reports_unresolved_when_the_target_is_never_reached():
    trials = _trials({0.1: ([10.0, 12.0], [100.0, 98.0]), 0.2: ([50.0, 52.0], [100.0, 98.0])})
    sel = select_w_syn(trials, target_fraction=0.8, n_boot=100, seed=0)
    assert sel.status == "unresolved"
    assert sel.crossing_mv is None
    assert sel.w_syn_mv == 0.2  # closest grid point
    assert sel.crossing_support < 0.05


def test_select_reports_ambiguous_when_a_noisy_curve_crosses_more_than_once():
    trials = _trials(
        {
            0.1: ([70.0, 70.0], [100.0, 100.0]),
            0.2: ([90.0, 90.0], [100.0, 100.0]),
            0.3: ([70.0, 70.0], [100.0, 100.0]),
            0.4: ([90.0, 90.0], [100.0, 100.0]),
        }
    )
    sel = select_w_syn(trials, target_fraction=0.8, n_boot=50, seed=0)
    assert sel.status == "ambiguous"
    assert sel.crossing_mv is None


def test_calibration_result_carries_per_trial_rates_and_selection():
    res = calibrate_w_syn(
        _chain(),
        stimulate={1: 100.0},
        readout=[3],
        w_syn_grid_mv=[0.1, 0.3],
        rates_hz=[100.0, 200.0],
        n_steps=1000,
        n_trials=3,
        seed=0,
        device="cpu",
        dtype="float64",
        n_boot=50,
    )
    assert len(res.trials) == 2 * 2 * 3
    assert {"readout_sd_hz", "n_trials"} <= set(res.curve.columns)
    assert res.provenance["status"] in {"calibrated", "unresolved", "ambiguous"}
    assert "crossing_support" in res.provenance
    assert res.selection.fractions.shape[0] == 2
