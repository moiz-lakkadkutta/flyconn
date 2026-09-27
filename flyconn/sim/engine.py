"""Batched leaky integrate-and-fire engine in PyTorch (CPU float64 reference, float32 CPU/MPS/CUDA).

Tick semantics follow Brian2's default schedule for the Shiu et al. 2024 model,
as transcribed by the float64 oracle of drosophila-brain-mlx (MIT; see
ATTRIBUTION.md), which reproduces Brian2 spike-for-spike:

1. refractory counters decrement; neurons with counter 0 get the exact update
   ``v <- v0 + (v - v0)*A + g*C``, ``g <- g*B``;
2. spike if not refractory and ``v > v_th`` (strict);
3. spikes emitted ``delay_steps`` ticks ago add their signed weights to ``g`` and
   external Poisson/explicit events add ``poisson_kick_mv`` to ``v``; both are
   dropped for refractory neurons;
4. spiking neurons reset ``v = v_reset, g = 0`` and reload their refractory
   counter (0 for externally driven neurons, as in the reference code).

Every output is a **model prediction** from wiring plus predicted transmitters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import flyconn
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.sim.params import ShiuParams

Device = Literal["cpu", "mps", "cuda"]
DType = Literal["float32", "float64"]


@dataclass
class LIFNetwork:
    """Neuron ids, signed synaptic weights in mV (pre rows, post columns) and parameters."""

    neuron_ids: np.ndarray
    weights_mv: sp.csr_matrix
    params: ShiuParams = field(default_factory=ShiuParams)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.neuron_ids = np.asarray(self.neuron_ids, dtype=np.int64)
        self.weights_mv = sp.csr_matrix(self.weights_mv, dtype=np.float64)
        self._pos = pd.Index(self.neuron_ids)

    @property
    def n(self) -> int:
        return len(self.neuron_ids)

    def index_of(self, ids: Sequence[int]) -> np.ndarray:
        idx = self._pos.get_indexer(np.asarray(list(ids), dtype=np.int64))
        if (idx < 0).any():
            msg = f"neuron ids not in network: {np.asarray(list(ids))[idx < 0][:5].tolist()}"
            raise KeyError(msg)
        return idx

    @classmethod
    def from_matrix(cls, m: ConnectivityMatrix, params: ShiuParams | None = None) -> LIFNetwork:
        """Signed synapse counts times ``w_syn`` (Shiu: ``Excitatory x Connectivity x w_syn``)."""
        params = params or ShiuParams()
        w = sp.csr_matrix(m.signed_weights, dtype=np.float64) * params.w_syn_mv
        prov = {"matrix": m.provenance, "calibration": _calibration_note(m)}
        return cls(m.neuron_ids, w, params, prov)


def _calibration_note(m: ConnectivityMatrix) -> str:
    ds = str(m.provenance.get("dataset", ""))
    if ds.startswith(("shiu", "flywire")):
        return "constants fitted to FlyWire v630 by Shiu et al. 2024"
    return "UNCALIBRATED: constants were fitted to FlyWire; spike counts are not comparable"


@dataclass
class SimResult:
    """Spike counts per trial and neuron, optional spike events, rates and provenance."""

    neuron_ids: np.ndarray
    counts: np.ndarray  # (n_trials, n_neurons)
    n_steps: int
    dt_ms: float
    events: np.ndarray | None  # (k, 3): trial, step, neuron index
    provenance: dict[str, Any]

    def spike_steps(self, neuron: int, trial: int = 0) -> np.ndarray:
        """Spike ticks of ``neuron`` (id) in ``trial``; requires recorded events."""
        if self.events is None:
            msg = "spike times were not recorded; pass record=True to simulate()"
            raise ValueError(msg)
        idx = int(pd.Index(self.neuron_ids).get_indexer([neuron])[0])
        if idx < 0:
            msg = f"neuron id {neuron} not in result"
            raise KeyError(msg)
        ev = self.events
        sel = (ev[:, 0] == trial) & (ev[:, 2] == idx)
        return np.sort(ev[sel, 1])

    def rates(self) -> pd.DataFrame:
        """Mean and s.d. firing rate (Hz) over trials for neurons that spiked in any trial."""
        seconds = self.n_steps * self.dt_ms / 1000.0
        r = self.counts / seconds
        any_spike = self.counts.sum(axis=0) > 0
        df = pd.DataFrame(
            {
                "neuron_id": self.neuron_ids,
                "rate_hz_mean": r.mean(axis=0),
                "rate_hz_std": r.std(axis=0, ddof=1) if self.counts.shape[0] > 1 else 0.0,
                "spikes_total": self.counts.sum(axis=0),
                "n_trials": self.counts.shape[0],
                "spiked_in_any_trial": any_spike,
            }
        )
        return df.sort_values("rate_hz_mean", ascending=False, ignore_index=True)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        msg = "MPS requested but not available"
        raise RuntimeError(msg)
    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA requested but not available"
        raise RuntimeError(msg)
    return torch.device(device)


def _external_drive(
    n: int,
    n_steps: int,
    n_trials: int,
    trial_offset: int,
    seed: int,
    stim_idx: np.ndarray,
    rates_hz: np.ndarray,
    dt_ms: float,
    events: Sequence[tuple[int, int]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-trial (step, neuron) external event lists as sorted arrays plus step offsets.

    Poisson drive is a Bernoulli draw per step with ``p = rate * dt`` (Brian2
    ``PoissonInput`` with N=1), seeded per trial from ``SeedSequence(seed)``
    so batched and single-trial runs agree.
    """
    seqs = np.random.SeedSequence(seed).spawn(trial_offset + n_trials)[trial_offset:]
    p = rates_hz * dt_ms / 1000.0
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    ev_steps = np.array([s for _, s in events], dtype=np.int64)
    ev_neurons = np.array([i for i, _ in events], dtype=np.int64)
    for sq in seqs:
        rng = np.random.default_rng(sq)
        if len(stim_idx):
            draws = rng.random((n_steps, len(stim_idx))) < p[None, :]
            st, k = np.nonzero(draws)
            steps = np.concatenate([st, ev_steps])
            neurons = np.concatenate([stim_idx[k], ev_neurons])
        else:
            steps, neurons = ev_steps, ev_neurons
        order = np.lexsort((neurons, steps))
        steps, neurons = steps[order], neurons[order]
        offsets = np.searchsorted(steps, np.arange(n_steps + 1))
        out.append((steps, neurons, offsets))
    return out


def simulate(
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
