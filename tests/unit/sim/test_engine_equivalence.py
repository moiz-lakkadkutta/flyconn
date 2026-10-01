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
import torch

from flyconn.sim import LIFNetwork, ShiuParams, engine, simulate
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


@pytest.mark.parametrize("n_trials", [1, 3])
def test_odd_sizes_and_partial_last_chunk(n_trials: int):
    odd = _recurrent_net(397, 0.03, seed=8, scale=3.0)  # trials x neurons not a multiple of 8
    kw: dict[str, Any] = dict(
        stimulate={
            int(i): (3000.0 if k < 10 else 300.0) for k, i in enumerate(odd.neuron_ids[:30])
        },
        n_steps=18 * 7 + 5,
        n_trials=n_trials,
        seed=2,
        device="cpu",
    )
    ours, ref = simulate(odd, **kw), simulate_reference(odd, **kw)
    _assert_same(ours, ref)
    assert ref.events is not None and (ref.events[:, 1] >= 18 * 7).sum() > 0


@pytest.mark.parametrize("buf_dtype", [torch.bool, torch.uint8])
@pytest.mark.parametrize(("length", "n_trials", "n"), [(18, 3, 397), (5, 1, 397), (1, 2, 3)])
def test_word_wise_spike_search_equals_nonzero_and_ignores_stale_bytes(
    buf_dtype: torch.dtype, length: int, n_trials: int, n: int
):
    chunk = 18
    size = chunk * n_trials * n
    buf = torch.zeros(-(-size // 8) * 8, dtype=buf_dtype)
    gen = torch.Generator().manual_seed(0)
    buf[:size] = (torch.rand(size, generator=gen) < 0.05).to(buf_dtype)
    buf[length * n_trials * n : size] = True  # stale spikes beyond the current chunk length
    got = engine._Spikes.of(buf, length, n_trials, n)
    want = torch.nonzero(buf[:size].view(chunk, n_trials, n)[:length])
    assert torch.equal(got.dev, want)
    np.testing.assert_array_equal(np.column_stack([got.tick_h, got.trial_h, got.neuron_h]), want)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_compiled_tick_kernel_matches_reference(dtype: DType):
    kw: dict[str, Any] = dict(
        stimulate=STIM, input_events=EVENTS, n_steps=700, n_trials=3, seed=11, dtype=dtype
    )
    ours = simulate(NET, device="cpu", compile=True, **kw)
    _assert_same(ours, simulate_reference(NET, device="cpu", **kw))
    assert ours.provenance["tick_kernel"] in {"torch.compile", "eager"}


def test_compile_off_uses_eager_kernel():
    res = simulate(NET, stimulate=STIM, n_steps=50, device="cpu", compile=False)
    assert res.provenance["tick_kernel"] == "eager"


def test_small_runs_default_to_eager_kernel():
    res = simulate(NET, stimulate=STIM, n_steps=50, device="cpu")
    assert res.provenance["tick_kernel"] == "eager"


def test_compile_failure_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch):
    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no compiler here")

    monkeypatch.setattr(torch, "compile", broken)
    monkeypatch.setattr(engine, "_COMPILED", {})  # forget kernels compiled by earlier tests
    kw: dict[str, Any] = dict(stimulate=STIM, n_steps=300, n_trials=2, seed=4, device="cpu")
    with pytest.warns(RuntimeWarning, match=r"torch\.compile"):
        ours = simulate(NET, compile=True, **kw)
    assert ours.provenance["tick_kernel"] == "eager"
    _assert_same(ours, simulate_reference(NET, **kw))
