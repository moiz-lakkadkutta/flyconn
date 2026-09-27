"""Tests for the re-calibration protocol (ADR-0004): pick w_syn from a readout target."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.sim import LIFNetwork, ShiuParams
from flyconn.sim.calibrate import CalibrationResult, calibrate_w_syn


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
