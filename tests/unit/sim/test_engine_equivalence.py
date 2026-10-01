"""The optimised engine reproduces the plain per-tick loop (v0.2.0) spike for spike on CPU.

The oracle in ``_reference_loop.py`` is the pre-optimisation engine verbatim; these tests
pin that chunked synaptic delivery, batched kicks and chunk-wise recording change nothing:
counts, event lists (order included), truncation and provenance flags must be identical
in float64 and float32.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from flyconn.sim import LIFNetwork, ShiuParams, simulate
from flyconn.sim.engine import DType

from ._reference_loop import simulate_reference

P = ShiuParams()


def _recurrent_net(n: int, density: float, seed: int, scale: float) -> LIFNetwork:
    rng = np.random.default_rng(seed)
    pre, post = np.nonzero(rng.random((n, n)) < density)
    keep = pre != post
    counts = rng.integers(1, 40, size=int(keep.sum())).astype(float)
    sign = np.where(rng.random(int(keep.sum())) < 0.25, -1.0, 1.0)
    w = sp.csr_matrix((counts * sign * P.w_syn_mv * scale, (pre[keep], post[keep])), shape=(n, n))
    return LIFNetwork(np.arange(1000, 1000 + n), w, P)


NET = _recurrent_net(400, 0.03, seed=5, scale=3.0)
STIM = {int(i): 180.0 for i in NET.neuron_ids[:25]}
EVENTS = [
    (int(NET.neuron_ids[30]), 3),
    (int(NET.neuron_ids[30]), 3),  # duplicate event: two kicks summed in one tick
    (int(NET.neuron_ids[30]), 3),
    (int(NET.neuron_ids[2]), 40),  # coincides with this neuron's Poisson drive sometimes
    (int(NET.neuron_ids[31]), 17),
    (int(NET.neuron_ids[31]), 18),
    (int(NET.neuron_ids[32]), 699),  # last tick
    (int(NET.neuron_ids[33]), 5000),  # beyond n_steps: ignored
]

CASES: dict[str, dict[str, Any]] = {
    "poisson_and_events": {"stimulate": STIM, "input_events": EVENTS},
    "default_refractory": {"stimulate": STIM, "no_refractory": []},
    "silenced": {"stimulate": STIM, "silence": [int(i) for i in NET.neuron_ids[40:120]]},
    "events_only": {"input_events": EVENTS},
    "no_input": {},
}


def _assert_same(a: Any, b: Any) -> None:
    np.testing.assert_array_equal(a.counts, b.counts)
    assert (a.events is None) == (b.events is None)
    if a.events is not None:
        np.testing.assert_array_equal(a.events, b.events)
    assert a.provenance["events_truncated"] == b.provenance["events_truncated"]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_engine_matches_per_tick_reference(case: str, dtype: DType):
    kw: dict[str, Any] = dict(
        n_steps=700, n_trials=3, seed=11, dtype=dtype, device="cpu", **CASES[case]
    )
    ours = simulate(NET, **kw)
    ref = simulate_reference(NET, **kw)
    _assert_same(ours, ref)
    if case == "poisson_and_events":
        # the comparison is only meaningful if the network is recurrently active
        assert ref.counts.sum() > 2_000
        assert (ref.counts[:, 25:].sum(axis=0) > 0).sum() > 100


@pytest.mark.parametrize("max_events", [1, 500, 3_000])
def test_event_truncation_matches_reference(max_events: int):
    kw: dict[str, Any] = dict(
        stimulate=STIM, n_steps=400, n_trials=2, seed=3, dtype="float64", device="cpu"
    )
    ours = simulate(NET, max_events=max_events, **kw)
    ref = simulate_reference(NET, max_events=max_events, **kw)
    _assert_same(ours, ref)


def test_record_false_has_no_events_and_same_counts():
    kw: dict[str, Any] = dict(stimulate=STIM, n_steps=500, n_trials=2, seed=9, device="cpu")
    ours = simulate(NET, record=False, **kw)
    ref = simulate_reference(NET, record=False, **kw)
    assert ours.events is None
    _assert_same(ours, ref)


@pytest.mark.parametrize("n_steps", [0, 1, 17, 18, 19, 36, 37])
def test_short_runs_and_chunk_boundaries(n_steps: int):
    kw: dict[str, Any] = dict(
        stimulate=STIM, input_events=EVENTS[:3], n_steps=n_steps, n_trials=2, device="cpu"
    )
    _assert_same(simulate(NET, **kw), simulate_reference(NET, **kw))
