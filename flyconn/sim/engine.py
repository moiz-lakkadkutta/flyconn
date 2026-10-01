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


# Dense per-chunk synaptic input buffer budget (chunk x trials x neurons values).
_CONTRIB_BUDGET_BYTES = 512 * 2**20


def _chunk_length(delay_steps: int, bytes_per_tick: int) -> int:
    """Ticks per chunk: the largest divisor of ``delay_steps`` whose input buffer fits the budget.

    Spikes emitted inside a chunk reach their targets ``delay_steps`` ticks later, i.e. in a
    later chunk, so all synaptic input of a chunk is known when it starts.
    """
    for length in range(delay_steps, 1, -1):
        if delay_steps % length == 0 and length * bytes_per_tick <= _CONTRIB_BUDGET_BYTES:
            return length
    return 1


@dataclass
class _Kicks:
    """All external kicks of a run as sorted flat keys ``(step * trials + trial) * |S| + s``."""

    targets: np.ndarray  # S: sorted neuron indices that receive any kick
    keys: np.ndarray
    table: np.ndarray  # table[c]: c kicks summed one by one in the run dtype


def _kick_schedule(
    drives: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_steps: int,
    n_trials: int,
    kick: float,
    np_dtype: type[np.floating[Any]],
) -> _Kicks:
    steps_l: list[np.ndarray] = []
    trials_l: list[np.ndarray] = []
    neurons_l: list[np.ndarray] = []
    for b, (steps, neurons, offsets) in enumerate(drives):
        lo, hi = int(offsets[0]), int(offsets[n_steps])
        steps_l.append(steps[lo:hi])
        neurons_l.append(neurons[lo:hi])
        trials_l.append(np.full(hi - lo, b, dtype=np.int64))
    steps = np.concatenate(steps_l) if steps_l else np.zeros(0, dtype=np.int64)
    neurons = np.concatenate(neurons_l) if neurons_l else np.zeros(0, dtype=np.int64)
    trials = np.concatenate(trials_l) if trials_l else np.zeros(0, dtype=np.int64)
    targets = np.unique(neurons)
    s = np.searchsorted(targets, neurons)
    keys = np.sort((steps * n_trials + trials) * len(targets) + s)
    max_c = int(np.unique(keys, return_counts=True)[1].max()) if len(keys) else 0
    # Index-add of c equal kicks into a zero buffer: ((0 + k) + k) + ... in the run dtype.
    table = np.zeros(max_c + 1, dtype=np_dtype)
    for c in range(1, max_c + 1):
        table[c] = table[c - 1] + np_dtype(kick)
    return _Kicks(targets.astype(np.int64), keys, table)


def _chunk_kicks(
    kicks: _Kicks, c0: int, length: int, n_trials: int, dev: torch.device, tdt: torch.dtype
) -> tuple[torch.Tensor, np.ndarray]:
    """Summed kick per (tick, trial, target) for ticks ``c0 .. c0+length-1`` and a per-tick flag."""
    per_tick = n_trials * len(kicks.targets)
    lo, hi = np.searchsorted(kicks.keys, [c0 * per_tick, (c0 + length) * per_tick])
    cnt = np.bincount(kicks.keys[lo:hi] - c0 * per_tick, minlength=length * per_tick)
    cnt = cnt.reshape(length, n_trials, len(kicks.targets))
    has = cnt.reshape(length, -1).any(axis=1)
    if not has.any():
        return torch.zeros(0, dtype=tdt), has
    return torch.as_tensor(kicks.table[cnt], device=dev), has


@dataclass
class _Spikes:
    """Spikes of one chunk: (tick-in-chunk, trial, neuron), on device and on host."""

    dev: torch.Tensor  # (k, 3) int64
    tick_h: np.ndarray
    trial_h: np.ndarray
    neuron_h: np.ndarray

    @classmethod
    def of(cls, spk: torch.Tensor) -> _Spikes:
        nz = torch.nonzero(spk)
        h = nz.cpu().numpy()
        return cls(nz, h[:, 0], h[:, 1], h[:, 2])


def _deliver(
    src: _Spikes,
    contrib: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    data: torch.Tensor,
    out_deg: torch.Tensor,
    out_deg_np: np.ndarray,
    n: int,
    n_trials: int,
    chunk: int,
) -> tuple[torch.Tensor | None, np.ndarray]:
    """Add the signed weights of ``src`` spikes to the dense (chunk, trial, neuron) buffer.

    Per target, weights are summed in presynaptic order from zero, exactly as one
    index_add per tick would; returns the touched flat keys and which ticks got input.
    """
    deg_h = out_deg_np[src.neuron_h]
    has = np.bincount(src.tick_h[deg_h > 0], minlength=chunk)[:chunk] > 0
    total = int(deg_h.sum())
    if total == 0:
        return None, has
    tick, trial, pre = src.dev[:, 0], src.dev[:, 1], src.dev[:, 2]
    deg = out_deg[pre]
    cum = torch.cumsum(deg, 0) - deg
    within = torch.arange(total, device=contrib.device) - torch.repeat_interleave(
        cum, deg, output_size=total
    )
    e = torch.repeat_interleave(indptr[pre], deg, output_size=total) + within
    row = torch.repeat_interleave(tick * n_trials + trial, deg, output_size=total)
    keys = row * n + indices[e]
    contrib.index_add_(0, keys, data[e])
    return keys, has


@dataclass
class _State:
    """Per (trial, neuron) dynamics state plus scratch buffers reused every tick."""

    v: torch.Tensor
    g: torch.Tensor
    rfc: torch.Tensor
    reload: torch.Tensor
    not_ref: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    a_v: float
    b_g: float
    c_gv: float
    v0_term: float
    v_th: float
    v_reset: float

    @classmethod
    def initial(
        cls,
        n_trials: int,
        n: int,
        p: ShiuParams,
        tdt: torch.dtype,
        dev: torch.device,
        norefr: np.ndarray,
    ) -> _State:
        v = torch.full((n_trials, n), p.v_rest_mv, dtype=tdt, device=dev)
        reload = torch.full((n,), p.refractory_steps, dtype=torch.int32, device=dev)
        if len(norefr):
            reload[torch.as_tensor(norefr, device=dev)] = 0
        return cls(
            v=v,
            g=torch.zeros_like(v),
            rfc=torch.zeros((n_trials, n), dtype=torch.int32, device=dev),
            reload=reload.expand(n_trials, n),
            not_ref=torch.zeros((n_trials, n), dtype=torch.bool, device=dev),
            t1=torch.empty_like(v),
            t2=torch.empty_like(v),
            a_v=p.decay_v,
            b_g=p.decay_g,
            c_gv=p.g_to_v,
            v0_term=p.v_rest_mv * (1.0 - p.decay_v),
            v_th=p.v_threshold_mv,
            v_reset=p.v_reset_mv,
        )


def _tick(
    s: _State,
    spike: torch.Tensor,
    syn: torch.Tensor | None,
    kick: torch.Tensor | None,
    kick_targets: torch.Tensor,
) -> None:
    """One tick in place; ``spike`` receives the spike mask.

    Same arithmetic, in the same order, as the plain loop: ``v0_term + g*c + v*a`` for
    non-refractory neurons, ``g*b`` plus delayed input, then kicks, then resets.
    """
    v, g, rfc, nr, t1, t2 = s.v, s.g, s.rfc, s.not_ref, s.t1, s.t2
    rfc.sub_(1).clamp_(min=0)
    torch.eq(rfc, 0, out=nr)
    torch.mul(g, s.c_gv, out=t1)
    t1.add_(s.v0_term)
    torch.mul(v, s.a_v, out=t2)
    t1.add_(t2)
    torch.where(nr, t1, v, out=v)
    torch.mul(g, s.b_g, out=t2)
    if syn is not None:
        t2.add_(syn)
    torch.where(nr, t2, g, out=g)
    torch.gt(v, s.v_th, out=spike)
    spike.logical_and_(nr)
    if kick is not None:
        vs = v.index_select(1, kick_targets)
        add = torch.where(nr.index_select(1, kick_targets), kick, 0.0)
        v.index_copy_(1, kick_targets, vs + add)
    v.masked_fill_(spike, s.v_reset)
    g.masked_fill_(spike, 0.0)
    torch.where(spike, s.reload, rfc, out=rfc)


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
    out_deg_np = np.diff(w.indptr).astype(np.int64)
    indptr = torch.as_tensor(w.indptr.astype(np.int64), device=dev)
    indices = torch.as_tensor(w.indices.astype(np.int64), device=dev)
    data = torch.as_tensor(w.data, dtype=tdt, device=dev)
    out_deg = torch.as_tensor(out_deg_np, device=dev)

    drives = _external_drive(
        n, n_steps, n_trials, trial_offset, seed, stim_idx, rates, p.dt_ms, ev_idx
    )
    np_dtype = np.float64 if dtype == "float64" else np.float32
    kicks = _kick_schedule(drives, n_steps, n_trials, p.poisson_kick_mv, np_dtype)
    kick_targets = torch.as_tensor(kicks.targets, device=dev)

    d = p.delay_steps
    if d < 1:
        msg = "delay_steps must be at least 1"
        raise ValueError(msg)
    chunk = _chunk_length(d, n_trials * n * data.element_size())
    st = _State.initial(n_trials, n, p, tdt, dev, norefr)
    spk = torch.zeros((chunk, n_trials, n), dtype=torch.bool, device=dev)
    contrib = torch.zeros(chunk * n_trials * n, dtype=tdt, device=dev)
    contrib_ticks = contrib.view(chunk, n_trials, n)
    counts = np.zeros((n_trials, n), dtype=np.int64)
    pending: dict[int, _Spikes] = {}
    recorded: list[np.ndarray] = []
    n_recorded = 0

    for k, c0 in enumerate(range(0, n_steps, chunk)):
        length = min(chunk, n_steps - c0)
        # Spikes of the chunk delay_steps ago arrive now (chunk divides delay_steps, so no
        # spike emitted inside this chunk can be delivered inside it).
        src = pending.pop(k - d // chunk, None)
        keys = None
        has_syn = np.zeros(chunk, dtype=bool)
        if src is not None:
            keys, has_syn = _deliver(
                src, contrib, indptr, indices, data, out_deg, out_deg_np, n, n_trials, chunk
            )
        kv, has_kick = _chunk_kicks(kicks, c0, length, n_trials, dev, tdt)
        for ell in range(length):
            _tick(
                st,
                spk[ell],
                contrib_ticks[ell] if has_syn[ell] else None,
                kv[ell] if has_kick[ell] else None,
                kick_targets,
            )
        if keys is not None:
            contrib.index_fill_(0, keys, 0.0)
        spikes = _Spikes.of(spk[:length])
        np.add.at(counts, (spikes.trial_h, spikes.neuron_h), 1)
        if record and n_recorded < max_events and len(spikes.tick_h):
            per_tick = np.bincount(spikes.tick_h, minlength=length)
            before = n_recorded + np.cumsum(per_tick) - per_tick
            keep = (before < max_events)[spikes.tick_h]
            recorded.append(
                np.column_stack(
                    [spikes.trial_h[keep], c0 + spikes.tick_h[keep], spikes.neuron_h[keep]]
                ).astype(np.int64)
            )
            n_recorded += int(keep.sum())
        pending[k] = spikes

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
    return SimResult(net.neuron_ids, counts, n_steps, p.dt_ms, events, prov)
