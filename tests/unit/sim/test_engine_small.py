"""Hand-checkable dynamics of the LIF engine on tiny networks (float64 CPU reference path).

Tick semantics (see flyconn.sim.engine): an external kick at tick t raises v after the
threshold test of tick t, so the spike is registered at tick t+1; synaptic input reaches g
delay_steps (18) ticks after the presynaptic spike and needs roughly 45 mV to cross the
7 mV gap alone (peak of the exact response is g * 0.1575 for tau 20/5 ms).
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyconn.sim import LIFNetwork, ShiuParams, simulate

P = ShiuParams()
STRONG = 60.0  # mV in g -> peak depolarisation about 9.5 mV > 7 mV gap


def _net(weights_mv: np.ndarray) -> LIFNetwork:
    """Network from a dense signed weight matrix in mV (pre rows, post columns); ids 1..n."""
    return LIFNetwork(
        neuron_ids=np.arange(1, weights_mv.shape[0] + 1),
        weights_mv=sp.csr_matrix(weights_mv),
        params=P,
    )


def test_single_kick_spikes_target_once_on_the_next_tick():
    net = _net(np.zeros((1, 1)))
    res = simulate(net, input_events=[(1, 10)], n_steps=200, dtype="float64", device="cpu")
    assert res.spike_steps(neuron=1, trial=0).tolist() == [11]
    assert res.counts.shape == (1, 1) and res.counts[0, 0] == 1


def test_synaptic_delay_is_18_ticks_and_strong_weight_drives_postsynaptic_spike():
    w = np.zeros((2, 2))
    w[0, 1] = STRONG
    net = _net(w)
    res = simulate(net, input_events=[(1, 5)], n_steps=200, dtype="float64", device="cpu")
    assert res.spike_steps(1).tolist() == [6]
    t1 = res.spike_steps(2)
    assert len(t1) == 1
    # g arrives at tick 6 + 18 = 24; v climbs over the following ~3.6 ms
    assert 24 < t1[0] < 70


def test_weak_weight_alone_does_not_spike():
    w = np.zeros((2, 2))
    w[0, 1] = 20.0  # peak about 3.2 mV
    res = simulate(_net(w), input_events=[(1, 5)], n_steps=200, dtype="float64", device="cpu")
    assert res.counts[0].tolist() == [1, 0]


def test_inhibitory_weight_prevents_spike():
    w = np.zeros((3, 3))
    w[0, 2] = STRONG
    w[1, 2] = -STRONG
    net = _net(w)
    alone = simulate(net, input_events=[(1, 5)], n_steps=200, dtype="float64", device="cpu")
    both = simulate(net, input_events=[(1, 5), (2, 5)], n_steps=200, dtype="float64", device="cpu")
    assert alone.counts[0, 2] == 1
    assert both.counts[0, 2] == 0


def test_refractory_period_drops_kicks_for_22_ticks_unless_poisson_driven():
    net = _net(np.zeros((1, 1)))
    # Externally driven neurons default to refractory 0 (reference rule): both kicks spike.
    res = simulate(net, input_events=[(1, 10), (1, 20)], n_steps=100, dtype="float64", device="cpu")
    assert res.spike_steps(1).tolist() == [11, 21]
    # Forcing the normal refractory period: the second kick lands inside 22 ticks and is dropped.
    res2 = simulate(
        net,
        input_events=[(1, 10), (1, 20)],
        n_steps=100,
        dtype="float64",
        device="cpu",
        no_refractory=[],
    )
    assert res2.spike_steps(1).tolist() == [11]
    # A kick after the window (11 + 22 = 33) spikes again.
    res3 = simulate(
        net,
        input_events=[(1, 10), (1, 40)],
        n_steps=100,
        dtype="float64",
        device="cpu",
        no_refractory=[],
    )
    assert res3.spike_steps(1).tolist() == [11, 41]


def test_poisson_drive_is_seeded_and_rate_is_right():
    net = _net(np.zeros((1, 1)))
    a = simulate(net, stimulate={1: 100.0}, n_steps=10_000, seed=3, dtype="float64", device="cpu")
    b = simulate(net, stimulate={1: 100.0}, n_steps=10_000, seed=3, dtype="float64", device="cpu")
    assert a.counts[0, 0] == b.counts[0, 0]
    assert 60 <= a.counts[0, 0] <= 140  # 1 s at 100 Hz, Bernoulli per 0.1 ms tick


def test_batched_trials_are_independent_and_match_single_runs():
    w = np.zeros((2, 2))
    w[0, 1] = STRONG
    net = _net(w)
    batched = simulate(
        net, stimulate={1: 150.0}, n_steps=2000, n_trials=4, seed=7, dtype="float64", device="cpu"
    )
    assert batched.counts.shape == (4, 2)
    singles = [
        simulate(
            net,
            stimulate={1: 150.0},
            n_steps=2000,
            n_trials=1,
            seed=7,
            trial_offset=k,
            dtype="float64",
            device="cpu",
        )
        for k in range(4)
    ]
    np.testing.assert_array_equal(batched.counts, np.vstack([s.counts for s in singles]))
    assert len({tuple(r) for r in batched.counts.tolist()}) > 1


def test_rates_table_and_provenance():
    net = _net(np.zeros((2, 2)))
    res = simulate(
        net, stimulate={1: 100.0}, n_steps=1000, n_trials=3, seed=1, dtype="float64", device="cpu"
    )
    rates = res.rates()
    assert isinstance(rates, pd.DataFrame)
    assert set(rates.columns) >= {"neuron_id", "rate_hz_mean", "rate_hz_std", "n_trials"}
    assert rates.loc[rates["neuron_id"] == 2, "rate_hz_mean"].iloc[0] == 0.0
    prov = res.provenance
    assert prov["params"]["w_syn_mv"] == 0.275
    assert prov["seed"] == 1 and prov["n_trials"] == 3 and prov["dtype"] == "float64"
    assert prov["label"] == "model prediction" and prov["deterministic"] is True


def test_float32_matches_float64_on_tiny_network():
    w = np.zeros((3, 3))
    w[0, 1] = STRONG
    w[1, 2] = STRONG
    net = _net(w)
    a = simulate(net, input_events=[(1, 5)], n_steps=300, dtype="float64", device="cpu")
    b = simulate(net, input_events=[(1, 5)], n_steps=300, dtype="float32", device="cpu")
    np.testing.assert_array_equal(a.counts, b.counts)
    assert a.counts[0].tolist() == [1, 1, 1]


def test_silencing_zeroes_outgoing_weights_only():
    w = np.zeros((3, 3))
    w[0, 1] = STRONG
    w[1, 2] = STRONG
    net = _net(w)
    res = simulate(
        net, input_events=[(1, 5)], n_steps=300, silence=[3], dtype="float64", device="cpu"
    )
    assert res.counts[0].tolist() == [1, 1, 1]  # neuron 3 still receives input and spikes
    res2 = simulate(
        net, input_events=[(1, 5)], n_steps=300, silence=[2], dtype="float64", device="cpu"
    )
    assert res2.counts[0].tolist() == [1, 1, 0]  # neuron 2's output removed -> 3 silent


def test_accelerator_float32_matches_cpu_reference_when_available():
    import torch

    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("no accelerator")
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    w = np.zeros((3, 3))
    w[0, 1] = STRONG
    w[1, 2] = STRONG
    net = _net(w)
    a = simulate(net, input_events=[(1, 5)], n_steps=300, dtype="float64", device="cpu")
    b = simulate(net, input_events=[(1, 5)], n_steps=300, dtype="float32", device=device)
    np.testing.assert_array_equal(a.counts, b.counts)
