"""Dataset-specific w_syn defaults from the re-calibration protocol (ADR-0009)."""

import pytest

from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import CALIBRATIONS, LIFNetwork, ShiuParams, default_params
from flyconn.testing import synthetic_connectome


def _matrix(dataset: str) -> ConnectivityMatrix:
    neurons, edges = synthetic_connectome(n_neurons=20, n_edges=60, seed=3)
    return ConnectivityMatrix.from_frames(
        neurons, edges, sign_policy=SignPolicy.ARGMAX, provenance={"dataset": dataset}
    )


def test_malecns_defaults_to_the_protocol_value():
    cal = CALIBRATIONS["malecns@1.0"]
    assert cal.w_syn_mv == pytest.approx(0.188)
    assert cal.ci_mv[0] < cal.w_syn_mv < cal.ci_mv[1]
    assert cal.sensitivity_mv[0] <= cal.ci_mv[0] and cal.sensitivity_mv[1] >= cal.ci_mv[1]
    assert default_params("malecns@1.0").w_syn_mv == cal.w_syn_mv
    # everything else keeps Shiu's published constant
    assert default_params("flywire@783").w_syn_mv == 0.275
    assert default_params("banc@888").w_syn_mv == 0.275


def test_network_scales_weights_and_labels_protocol_calibration():
    m = _matrix("malecns@1.0")
    net = LIFNetwork.from_matrix(m)
    assert net.params.w_syn_mv == pytest.approx(0.188)
    ref = LIFNetwork.from_matrix(m, ShiuParams())
    assert net.weights_mv.sum() == pytest.approx(ref.weights_mv.sum() * 0.188 / 0.275)
    note = net.provenance["calibration"]
    assert note.startswith("CALIBRATED BY PROTOCOL")
    assert "not validated" in note and "ADR-0009" in note


def test_explicit_published_constant_on_malecns_is_labelled_uncalibrated():
    net = LIFNetwork.from_matrix(_matrix("malecns@1.0"), ShiuParams())
    assert net.provenance["calibration"].startswith("UNCALIBRATED")


def test_datasets_without_calibration_stay_uncalibrated_and_flywire_unchanged():
    assert (
        LIFNetwork.from_matrix(_matrix("banc@888"))
        .provenance["calibration"]
        .startswith("UNCALIBRATED")
    )
    note = LIFNetwork.from_matrix(_matrix("shiu@630")).provenance["calibration"]
    assert "Shiu" in note and "UNCALIBRATED" not in note
