"""Model parameters must match Shiu et al. 2024 exactly (GOLDEN_RESULTS.md section 3.1)."""

import math

import pytest

from flyconn.sim import ShiuParams


def test_default_parameters_match_shiu_2024():
    p = ShiuParams()
    assert p.v_rest_mv == -52.0
    assert p.v_reset_mv == -52.0
    assert p.v_threshold_mv == -45.0
    assert p.tau_membrane_ms == 20.0
    assert p.tau_synapse_ms == 5.0
    assert p.t_refractory_ms == 2.2
    assert p.delay_ms == 1.8
    assert p.w_syn_mv == 0.275
    assert p.poisson_factor == 250.0
    assert p.dt_ms == 0.1


def test_derived_step_counts_and_exact_coefficients():
    p = ShiuParams()
    assert p.delay_steps == 18
    assert p.refractory_steps == 22
    assert p.poisson_kick_mv == pytest.approx(68.75)
    # Brian2 'exact' integration of dv/dt=(v0-v+g)/tm, dg/dt=-g/ts over one step
    assert p.decay_v == pytest.approx(math.exp(-0.1 / 20.0))
    assert p.decay_g == pytest.approx(math.exp(-0.1 / 5.0))
    expected = (5.0 / (5.0 - 20.0)) * (math.exp(-0.1 / 5.0) - math.exp(-0.1 / 20.0))
    assert p.g_to_v == pytest.approx(expected)
    assert p.g_to_v > 0


def test_parameters_are_immutable_and_serialisable():
    p = ShiuParams()
    with pytest.raises((AttributeError, TypeError)):
        p.w_syn_mv = 1.0  # type: ignore[misc]
    d = p.to_dict()
    assert d["w_syn_mv"] == 0.275 and d["source"].startswith("Shiu")
