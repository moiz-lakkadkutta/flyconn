"""Pre-optimisation per-tick LIF loop, kept verbatim as a test oracle for the chunked engine.

This is ``flyconn.sim.engine.simulate`` as of v0.2.0 (one Python iteration per tick,
per-tick host syncs). The optimised engine must reproduce it spike for spike on CPU.
"""

# ruff: noqa
# pyright: basic
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

import flyconn
from flyconn.sim.engine import DType, LIFNetwork, SimResult, _external_drive, _resolve_device


def simulate_reference(
    net: LIFNetwork,
    *,
    stimulate: Mapping[int, float] | None = None,
    input_events: Sequence[tuple[int, int]] | None = None,
    n_steps: int = 10_000,
    n_trials: int = 1,
    seed: int = 0,
    trial_offset: int = 0,
    dtype: DType = "float32",
    device: str = "auto",
    silence: Sequence[int] | None = None,
    no_refractory: Sequence[int] | None = None,
    record: bool = True,
    max_events: int = 50_000_000,
) -> SimResult:
    """Run ``n_trials`` batched trials of ``n_steps`` ticks.

    Args:
        net: network (weights in mV, pre rows).
        stimulate: neuron id -> Poisson rate in Hz (each event adds ``poisson_kick_mv`` to v).
        input_events: explicit ``(neuron id, step)`` kicks, applied in every trial.
        n_steps, n_trials, seed, trial_offset: trial ``k`` uses seed stream ``trial_offset + k``.
        dtype/device: ``float64`` is CPU-only and is the bit-deterministic reference path.
        silence: neuron ids whose *outgoing* weights are zeroed (Shiu's silencing).
        no_refractory: neuron ids with refractory period 0; defaults to the stimulated and
            explicitly kicked neurons (the reference code's rule for Poisson targets).
        record: keep spike events (trial, step, neuron index) up to ``max_events``.
    """
    p = net.params
    dev = _resolve_device(device)
    if dtype == "float64" and dev.type != "cpu":
        msg = "float64 is only supported on cpu (MPS has no float64; CUDA path is float32)"
        raise ValueError(msg)
    tdt = torch.float64 if dtype == "float64" else torch.float32
    n = net.n
    stimulate = dict(stimulate or {})
    input_events = list(input_events or [])
    stim_idx = net.index_of(list(stimulate)) if stimulate else np.array([], dtype=np.int64)
    rates = np.array([stimulate[i] for i in stimulate], dtype=float)
    ev_idx = [(int(net.index_of([nid])[0]), int(step)) for nid, step in input_events]
    driven = set(stim_idx.tolist()) | {i for i, _ in ev_idx}
    norefr = (
        net.index_of(list(no_refractory))
        if no_refractory is not None
        else np.array(sorted(driven), dtype=np.int64)
    )

    # Weights: zero outgoing rows of silenced neurons; keep CSR arrays for event-driven delivery.
    w = sp.csr_matrix(net.weights_mv, copy=True)
    if silence:
        rows = net.index_of(list(silence))
        mask = np.ones(n)
        mask[rows] = 0.0
        w = sp.csr_matrix(sp.diags(mask) @ w)
        w.eliminate_zeros()
    indptr = torch.as_tensor(w.indptr.astype(np.int64), device=dev)
    indices = torch.as_tensor(w.indices.astype(np.int64), device=dev)
    data = torch.as_tensor(w.data, dtype=tdt, device=dev)
    out_deg = indptr[1:] - indptr[:-1]

    drives = _external_drive(
        n, n_steps, n_trials, trial_offset, seed, stim_idx, rates, p.dt_ms, ev_idx
    )

    v0 = torch.full((n_trials, n), p.v_rest_mv, dtype=tdt, device=dev)
    v = v0.clone()
    g = torch.zeros_like(v)
    rfc = torch.zeros((n_trials, n), dtype=torch.int32, device=dev)
    reload = torch.full((n,), p.refractory_steps, dtype=torch.int32, device=dev)
    if len(norefr):
        reload[torch.as_tensor(norefr, device=dev)] = 0
    ring = torch.zeros((p.delay_steps, n_trials, n), dtype=torch.bool, device=dev)
    counts = torch.zeros((n_trials, n), dtype=torch.int64, device=dev)
    a_v, b_g, c_gv = p.decay_v, p.decay_g, p.g_to_v
    v0_term = p.v_rest_mv * (1.0 - a_v)
    kick = p.poisson_kick_mv
    trial_ix = torch.arange(n_trials, device=dev)
    recorded: list[np.ndarray] = []
    n_recorded = 0

    for t in range(n_steps):
        rfc = torch.clamp(rfc - 1, min=0)
        not_ref = rfc == 0
        v = torch.where(not_ref, v0_term + g * c_gv + v * a_v, v)
        g = torch.where(not_ref, g * b_g, g)
        spike = not_ref & (v > p.v_threshold_mv)

        slot = t % p.delay_steps
        delayed = ring[slot]
        if bool(delayed.any()):
            b_idx, pre = torch.nonzero(delayed, as_tuple=True)
            deg = out_deg[pre]
            total = int(deg.sum())
            if total:
                starts = indptr[pre]
                rep_b = torch.repeat_interleave(b_idx, deg)
                rep_start = torch.repeat_interleave(starts, deg)
                cum = torch.cumsum(deg, 0) - deg
                within = torch.arange(total, device=dev) - torch.repeat_interleave(cum, deg)
                e = rep_start + within
                contrib = torch.zeros(n_trials * n, dtype=tdt, device=dev)
                contrib.index_add_(0, rep_b * n + indices[e], data[e])
                g = g + torch.where(not_ref, contrib.view(n_trials, n), torch.zeros_like(g))

        # External kicks (explicit events and Poisson draws) for this step.
        ext_rows: list[torch.Tensor] = []
        for b, (_steps, neurons, offsets) in enumerate(drives):
            lo, hi = int(offsets[t]), int(offsets[t + 1])
            if hi > lo:
                nb = torch.as_tensor(neurons[lo:hi], device=dev)
                ext_rows.append(torch.stack([torch.full_like(nb, b), nb]))
        if ext_rows:
            ext = torch.cat(ext_rows, dim=1)
            add = torch.zeros(n_trials * n, dtype=tdt, device=dev)
            add.index_add_(
                0, ext[0] * n + ext[1], torch.full((ext.shape[1],), kick, dtype=tdt, device=dev)
            )
            v = v + torch.where(not_ref, add.view(n_trials, n), torch.zeros_like(v))

        v = torch.where(spike, torch.full_like(v, p.v_reset_mv), v)
        g = torch.where(spike, torch.zeros_like(g), g)
        rfc = torch.where(spike, reload.expand_as(rfc), rfc)
        ring[slot] = spike
        counts += spike.to(torch.int64)
        if record and n_recorded < max_events:
            b_s, n_s = torch.nonzero(spike, as_tuple=True)
            if len(b_s):
                arr = np.column_stack(
                    [b_s.cpu().numpy(), np.full(len(b_s), t), n_s.cpu().numpy()]
                ).astype(np.int64)
                recorded.append(arr)
                n_recorded += len(arr)
    del trial_ix

    events = (
        np.concatenate(recorded)
        if recorded
        else (np.zeros((0, 3), dtype=np.int64) if record else None)
    )
    prov: dict[str, Any] = {
        "label": "model prediction",
        "engine": "flyconn.sim.engine",
        "flyconn_version": flyconn.__version__,
        "torch_version": torch.__version__,
        "device": dev.type,
        "dtype": dtype,
        "deterministic": dtype == "float64" and dev.type == "cpu",
        "params": p.to_dict(),
        "n_steps": n_steps,
        "n_trials": n_trials,
        "seed": seed,
        "trial_offset": trial_offset,
        "stimulate": {int(k): float(val) for k, val in stimulate.items()},
        "n_input_events": len(input_events),
        "silence": [int(s) for s in (silence or [])],
        "no_refractory_count": len(norefr),
        "network": net.provenance,
        "events_truncated": record and n_recorded >= max_events,
    }
    return SimResult(net.neuron_ids, counts.cpu().numpy(), n_steps, p.dt_ms, events, prov)
