"""Validation rung 1: float64 CPU engine reproduces Brian2 spike for spike on small networks.

The Brian2 model below is the Shiu et al. 2024 ``model.py`` (equations, constants,
``method='linear'``, delay, refractory, reset) with the Poisson drive replaced by
a fixed input spike train delivered as ``v += kick`` so that no RNG differences
can enter the comparison (design borrowed from drosophila-brain-mlx, MIT).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from flyconn.sim import LIFNetwork, ShiuParams, simulate

brian2 = pytest.importorskip("brian2")
pytestmark = pytest.mark.brian

P = ShiuParams()


def _random_net(n: int, density: float, seed: int, scale_mv: float) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(rng.random((n, n)) < density)
    keep = pre != post
    w = rng.integers(1, 400, size=keep.sum()) * P.w_syn_mv
    sign = np.where(rng.random(keep.sum()) < 0.3, -1.0, 1.0)
    return sp.csr_matrix((w * sign * scale_mv, (pre[keep], post[keep])), shape=(n, n))


def _brian2_spikes(
    w: sp.csr_matrix, events: list[tuple[int, int]], n_steps: int, driven: list[int]
) -> set[tuple[int, int]]:
    from brian2 import (
        Hz,
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        second,
    )

    del Hz, second
    n = w.shape[0]
    defaultclock.dt = P.dt_ms * ms
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neu = NeuronGroup(
        n,
        eqs,
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace={
            "v_0": P.v_rest_mv * mV,
            "v_rst": P.v_reset_mv * mV,
            "v_th": P.v_threshold_mv * mV,
            "t_mbr": P.tau_membrane_ms * ms,
            "tau": P.tau_synapse_ms * ms,
        },
    )
    neu.v = P.v_rest_mv * mV
    neu.g = 0 * mV
    neu.rfc = P.t_refractory_ms * ms
    for i in driven:
        neu[i].rfc = 0 * ms
    coo = sp.coo_matrix(w)
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=P.delay_ms * ms)
    syn.connect(i=coo.row, j=coo.col)
    syn.w = coo.data * mV
    idx = np.array([i for i, _ in events])
    times = np.array([s for _, s in events]) * P.dt_ms * ms
    gen = SpikeGeneratorGroup(n, idx, times)
    drive = Synapses(gen, neu, on_pre="v += kick", namespace={"kick": P.poisson_kick_mv * mV})
    drive.connect(j="i")
    mon = SpikeMonitor(neu)
    net = Network(neu, syn, gen, drive, mon)
    net.run(n_steps * P.dt_ms * ms)
    steps = np.rint(np.asarray(mon.t / ms) / P.dt_ms).astype(int)
    return set(zip(steps.tolist(), np.asarray(mon.i).tolist(), strict=True))


@pytest.mark.parametrize(
    ("n", "density", "seed", "n_steps"), [(60, 0.15, 0, 800), (200, 0.06, 1, 1500)]
)
def test_float64_engine_matches_brian2_spike_for_spike(
    n: int, density: float, seed: int, n_steps: int
):
    w = _random_net(n, density, seed, scale_mv=1.0)
    rng = np.random.default_rng(seed + 100)
    driven = sorted(rng.choice(n, size=max(3, n // 10), replace=False).tolist())
    events = [
        (int(i), int(t))
        for i in driven
        for t in np.sort(rng.choice(n_steps - 50, size=12, replace=False))
    ]
    net = LIFNetwork(np.arange(n), w, P)
    ours = simulate(
        net,
        input_events=[(int(i), int(t)) for i, t in events],
        n_steps=n_steps,
        dtype="float64",
        device="cpu",
        no_refractory=driven,
    )
    assert ours.events is not None
    ours_set = set(zip(ours.events[:, 1].tolist(), ours.events[:, 2].tolist(), strict=True))
    theirs = _brian2_spikes(w, events, n_steps, driven)
    assert len(theirs) > 20, "Brian2 reference produced too few spikes to be a meaningful test"
    assert ours_set == theirs
