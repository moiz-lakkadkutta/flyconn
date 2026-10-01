"""Run a declarative experiment: conditions, default controls, statistics, results, report.

The building blocks here (:func:`prepare`, :func:`shared_runs`, :func:`readout_stats`, ...)
are shared with :mod:`flyconn.experiments.sweep`, which runs the shared conditions once and
then one simulation per sweep variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

import flyconn
from flyconn.data.convert.common import write_parquet
from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.experiments.spec import ExperimentSpec, Stimulation, resolve_selection
from flyconn.experiments.stats import benjamini_hochberg, compare_groups
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.graph.signs import SignPolicy
from flyconn.sim.engine import DType, LIFNetwork, simulate
from flyconn.uncertainty.nulls import degree_preserving_rewire, shuffle_signs

# One simulated condition: (condition, control, replicate, counts[n_trials, n_neurons]).
Run = tuple[str, str, int, np.ndarray]


@dataclass
class ExperimentResult:
    out_dir: Path
    conditions: pd.DataFrame  # one row per simulated (condition, control, replicate)
    counts: pd.DataFrame  # long: condition, control, replicate, trial, neuron_id, count
    readouts: pd.DataFrame  # per readout group and comparison: effects and statistics
    top_neurons: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Everything an experiment needs once the dataset is loaded and the network built."""

    out: Path
    store: Store
    neurons: pd.DataFrame
    matrix: ConnectivityMatrix
    net: LIFNetwork
    seconds: float
    background: dict[int, float]
    readout_ids: dict[str, np.ndarray]

    def group_rates(self, counts: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Per-trial mean rate (Hz) over the neurons ``ids``."""
        idx = self.net.index_of(ids.tolist())
        return counts[:, idx].mean(axis=1) / self.seconds


def rates_hz(stims: list[Stimulation], neurons: pd.DataFrame) -> dict[int, float]:
    """Map neuron id -> Poisson rate for a list of stimulation entries (later entries win)."""
    out: dict[int, float] = {}
    for st in stims:
        for nid in resolve_selection(st.select, neurons):
            out[int(nid)] = st.rate_hz
    return out


def simulate_condition(
    net: LIFNetwork,
    spec: ExperimentSpec,
    stimulate: dict[int, float],
    silence: list[int],
) -> np.ndarray:
    """Spike counts ``(trials, neurons)`` for one condition with the spec's seeds and settings."""
    res = simulate(
        net,
        stimulate=stimulate,
        n_steps=spec.n_steps,
        n_trials=spec.trials,
        seed=spec.seed,
        dtype=cast("DType", spec.dtype),
        device=spec.device,
        silence=silence or None,
        record=False,
    )
    return res.counts


def prepare(spec: ExperimentSpec, out_dir: Path | str, store: Store | None) -> Context:
    """Create ``out_dir`` (refusing to overwrite), load the dataset and build the network."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        msg = f"{out} already exists and is not empty; choose a new output directory"
        raise FileExistsError(msg)
    out.mkdir(parents=True, exist_ok=True)
    if store is None:
        store = Store(pull(spec.dataset, level="weights").store_dir)
    neurons = store.neurons()
    m = ConnectivityMatrix.from_store(
        store, min_weight=spec.network.min_weight, sign_policy=SignPolicy(spec.network.sign_policy)
    )
    return Context(
        out=out,
        store=store,
        neurons=neurons,
        matrix=m,
        net=LIFNetwork.from_matrix(m),
        seconds=spec.n_steps * spec.dt_ms / 1000.0,
        background=rates_hz(spec.stimulate, neurons),
        readout_ids={r.name: resolve_selection(r.select, neurons) for r in spec.readouts},
    )


def shared_runs(ctx: Context, spec: ExperimentSpec) -> list[Run]:
    """Baseline (if enabled), stimulated, and the null-connectome controls of ``stimulated``."""
    runs: list[Run] = []
    if spec.controls.unstimulated_baseline:
        runs.append(("baseline", "none", 0, simulate_condition(ctx.net, spec, {}, [])))
    runs.append(("stimulated", "none", 0, simulate_condition(ctx.net, spec, ctx.background, [])))
    for k in range(spec.controls.degree_preserving_rewire):
        null = LIFNetwork.from_matrix(
            degree_preserving_rewire(ctx.matrix, seed=spec.seed + 1000 + k)
        )
        runs.append(
            (
                "stimulated",
                "degree_preserving_rewire",
                k,
                simulate_condition(null, spec, ctx.background, []),
            )
        )
    for k in range(spec.controls.sign_shuffle):
        null = LIFNetwork.from_matrix(shuffle_signs(ctx.matrix, seed=spec.seed + 2000 + k))
        runs.append(
            ("stimulated", "sign_shuffle", k, simulate_condition(null, spec, ctx.background, []))
        )
    return runs


def counts_frame(runs: list[Run], net: LIFNetwork) -> pd.DataFrame:
    """Long-form non-zero spike counts of every run."""
    frames: list[pd.DataFrame] = []
    for cond, ctl, rep, counts in runs:
        t_idx, n_idx = np.nonzero(counts)
        frames.append(
            pd.DataFrame(
                {
                    "condition": cond,
                    "control": ctl,
                    "replicate": rep,
                    "trial": t_idx,
                    "neuron_id": net.neuron_ids[n_idx],
                    "count": counts[t_idx, n_idx],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def conditions_frame(runs: list[Run]) -> pd.DataFrame:
    """One row per run with total spikes and the number of neurons that fired."""
    return pd.DataFrame(
        [
            {
                "condition": cond,
                "control": ctl,
                "replicate": rep,
                "total_spikes": int(c.sum()),
                "active_neurons": int((c.sum(axis=0) > 0).sum()),
            }
            for cond, ctl, rep, c in runs
        ]
    )


def readout_stats(ctx: Context, spec: ExperimentSpec, runs: list[Run]) -> pd.DataFrame:
    """Readout effects: perturbed/stimulated vs stimulated/baseline/controls, BH over all rows."""
    by_key: dict[tuple[str, str], list[np.ndarray]] = {}
    for cond, ctl, _rep, counts in runs:
        by_key.setdefault((cond, ctl), []).append(counts)
    rows: list[dict[str, Any]] = []
    for name, ids in ctx.readout_ids.items():
        stim = ctx.group_rates(by_key[("stimulated", "none")][0], ids)
        comparisons: list[tuple[str, np.ndarray, np.ndarray]] = []
        if ("perturbed", "none") in by_key:
            pert = ctx.group_rates(by_key[("perturbed", "none")][0], ids)
            comparisons.append(("perturbed_vs_stimulated", pert, stim))
        if ("baseline", "none") in by_key:
            base = ctx.group_rates(by_key[("baseline", "none")][0], ids)
            comparisons.append(("stimulated_vs_baseline", stim, base))
        for ctl in ("degree_preserving_rewire", "sign_shuffle"):
            if ("stimulated", ctl) in by_key:
                pooled = np.concatenate(
                    [ctx.group_rates(c, ids) for c in by_key[("stimulated", ctl)]]
                )
                comparisons.append((f"stimulated_vs_{ctl}", stim, pooled))
        for label, test, ctrl in comparisons:
            r = compare_groups(test, ctrl, seed=spec.seed)
            rows.append({"readout": name, "n_neurons": len(ids), "comparison": label, **r})
    readouts = pd.DataFrame(rows)
    if not readouts.empty:
        readouts["q_value"] = benjamini_hochberg(readouts["p_value"].to_numpy())
        readouts["significant"] = readouts["q_value"] < spec.report.alpha
    return readouts


def base_provenance(ctx: Context, spec: ExperimentSpec) -> dict[str, Any]:
    """Provenance common to single experiments and sweeps."""
    return {
        "label": "model prediction",
        "spec": spec.raw | {"name": spec.name},
        "spec_sha256": spec.source_sha256,
        "spec_path": spec.source_path,
        "dataset": ctx.store.ref,
        "dataset_provenance": ctx.store.provenance,
        "network": {
            **ctx.matrix.provenance,
            "calibration": ctx.net.provenance.get("calibration"),
        },
        "seed": spec.seed,
        "trials": spec.trials,
        "n_steps": spec.n_steps,
        "device": spec.device,
        "dtype": spec.dtype,
        "stimulated_neurons": len(ctx.background),
        "readouts": {k: len(v) for k, v in ctx.readout_ids.items()},
        "controls": {
            "degree_preserving_rewire": spec.controls.degree_preserving_rewire,
            "sign_shuffle": spec.controls.sign_shuffle,
            "unstimulated_baseline": spec.controls.unstimulated_baseline,
        },
        "flyconn_version": flyconn.__version__,
        "torch_version": torch.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "citations": [c.__dict__ for c in ctx.store.citations()],
    }


def write_table(df: pd.DataFrame, path: Path) -> None:
    write_parquet(pa.Table.from_pandas(df, preserve_index=False), path)


def write_provenance(prov: dict[str, Any], out: Path) -> None:
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))


def run_experiment(
    spec: ExperimentSpec, *, out_dir: Path | str, store: Store | None = None
) -> ExperimentResult:
    """Run ``spec``; write counts/readouts/conditions Parquet, ``provenance.json``, ``report.html``.

    Conditions: ``baseline`` (no stimulation; only if enabled), ``stimulated`` (background
    stimulation) and, when a perturbation is given, ``perturbed`` (background plus silencing
    or extra stimulation). Controls simulate the *stimulated* condition on degree-preserving
    rewired and sign-shuffled connectomes with the same seeds. Refuses to overwrite ``out_dir``.
    Specs with a ``sweep`` section are run by :func:`flyconn.experiments.sweep.run_sweep`.
    """
    from flyconn.report.html import render_report  # local import: matplotlib is slow to load

    if spec.sweep is not None:
        msg = (
            "spec has a 'sweep' section; use flyconn.experiments.sweep.run_sweep (or `flyconn run`)"
        )
        raise ValueError(msg)
    ctx = prepare(spec, out_dir, store)
    silence: list[int] = []
    perturbed_stim = dict(ctx.background)
    if spec.perturb is not None:
        for s in spec.perturb.silence:
            silence.extend(int(i) for i in resolve_selection(s.select, ctx.neurons))
        perturbed_stim.update(rates_hz(spec.perturb.stimulate, ctx.neurons))

    runs = shared_runs(ctx, spec)
    if spec.perturb is not None:
        at = next(i for i, r in enumerate(runs) if r[:2] == ("stimulated", "none")) + 1
        runs.insert(
            at,
            ("perturbed", "none", 0, simulate_condition(ctx.net, spec, perturbed_stim, silence)),
        )

    counts_df = counts_frame(runs, ctx.net)
    conditions = conditions_frame(runs)
    readouts = readout_stats(ctx, spec, runs)

    # Top neurons by rate in the stimulated condition (and perturbed delta when present).
    by_cond = {(c, k): counts for c, k, _r, counts in runs}
    mean_rate = by_cond[("stimulated", "none")].mean(axis=0) / ctx.seconds
    top_idx = np.argsort(-mean_rate)[: spec.report.top_neurons]
    meta = ctx.matrix.meta
    top = pd.DataFrame(
        {
            "neuron_id": ctx.net.neuron_ids[top_idx],
            "cell_type": meta.iloc[top_idx]["cell_type"].to_numpy()
            if "cell_type" in meta
            else None,
            "rate_hz_stimulated": mean_rate[top_idx],
        }
    )
    if ("perturbed", "none") in by_cond:
        pert_rate = by_cond[("perturbed", "none")].mean(axis=0) / ctx.seconds
        top["rate_hz_perturbed"] = pert_rate[top_idx]
        top["delta_hz"] = top["rate_hz_perturbed"] - top["rate_hz_stimulated"]

    prov = base_provenance(ctx, spec) | {"silenced_neurons": len(silence)}
    out = ctx.out
    write_table(counts_df, out / "counts.parquet")
    write_table(readouts, out / "readouts.parquet")
    write_table(conditions, out / "conditions.parquet")
    write_provenance(prov, out)
    result = ExperimentResult(out, conditions, counts_df, readouts, top, prov)
    (out / "report.html").write_text(render_report(result, spec, ctx.store))
    return result
