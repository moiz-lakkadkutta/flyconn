"""Run a declarative experiment: conditions, default controls, statistics, results, report."""

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


@dataclass
class ExperimentResult:
    out_dir: Path
    conditions: pd.DataFrame  # one row per simulated (condition, control, replicate)
    counts: pd.DataFrame  # long: condition, control, replicate, trial, neuron_id, count
    readouts: pd.DataFrame  # per readout group and comparison: effects and statistics
    top_neurons: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)


def _rates_hz(stims: list[Stimulation], neurons: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for st in stims:
        for nid in resolve_selection(st.select, neurons):
            out[int(nid)] = st.rate_hz
    return out


def _simulate_condition(
    net: LIFNetwork,
    spec: ExperimentSpec,
    stimulate: dict[int, float],
    silence: list[int],
) -> np.ndarray:
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


def run_experiment(
    spec: ExperimentSpec, *, out_dir: Path | str, store: Store | None = None
) -> ExperimentResult:
    """Run ``spec``; write counts/readouts/conditions Parquet, ``provenance.json``, ``report.html``.

    Conditions: ``baseline`` (no stimulation; only if enabled), ``stimulated`` (background
    stimulation) and, when a perturbation is given, ``perturbed`` (background plus silencing
    or extra stimulation). Controls simulate the *stimulated* condition on degree-preserving
    rewired and sign-shuffled connectomes with the same seeds. Refuses to overwrite ``out_dir``.
    """
    from flyconn.report.html import render_report  # local import: matplotlib is slow to load

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
    net = LIFNetwork.from_matrix(m)
    seconds = spec.n_steps * spec.dt_ms / 1000.0

    background = _rates_hz(spec.stimulate, neurons)
    readout_ids = {r.name: resolve_selection(r.select, neurons) for r in spec.readouts}
    silence: list[int] = []
    perturbed_stim = dict(background)
    if spec.perturb is not None:
        for s in spec.perturb.silence:
            silence.extend(int(i) for i in resolve_selection(s.select, neurons))
        perturbed_stim.update(_rates_hz(spec.perturb.stimulate, neurons))

    runs: list[tuple[str, str, int, np.ndarray]] = []  # condition, control, replicate, counts
    if spec.controls.unstimulated_baseline:
        runs.append(("baseline", "none", 0, _simulate_condition(net, spec, {}, [])))
    runs.append(("stimulated", "none", 0, _simulate_condition(net, spec, background, [])))
    if spec.perturb is not None:
        runs.append(
            ("perturbed", "none", 0, _simulate_condition(net, spec, perturbed_stim, silence))
        )
    for k in range(spec.controls.degree_preserving_rewire):
        null_net = LIFNetwork.from_matrix(degree_preserving_rewire(m, seed=spec.seed + 1000 + k))
        runs.append(
            (
                "stimulated",
                "degree_preserving_rewire",
                k,
                _simulate_condition(null_net, spec, background, []),
            )
        )
    for k in range(spec.controls.sign_shuffle):
        null_net = LIFNetwork.from_matrix(shuffle_signs(m, seed=spec.seed + 2000 + k))
        runs.append(
            ("stimulated", "sign_shuffle", k, _simulate_condition(null_net, spec, background, []))
        )

    # Long-form counts table.
    frames = []
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
    counts_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    conditions = pd.DataFrame(
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

    # Readout statistics: per-trial mean rate over the readout group.
    def group_rates(counts: np.ndarray, ids: np.ndarray) -> np.ndarray:
        idx = net.index_of(ids.tolist())
        return counts[:, idx].mean(axis=1) / seconds

    by_key = {(c, k): [] for c, k, _, _ in runs}
    for cond, ctl, _rep, counts in runs:
        by_key[(cond, ctl)].append(counts)
    rows: list[dict[str, Any]] = []
    for name, ids in readout_ids.items():
        stim = group_rates(by_key[("stimulated", "none")][0], ids)
        comparisons: list[tuple[str, np.ndarray, np.ndarray]] = []
        if ("perturbed", "none") in by_key:
            comparisons.append(
                (
                    "perturbed_vs_stimulated",
                    group_rates(by_key[("perturbed", "none")][0], ids),
                    stim,
                )
            )
        if ("baseline", "none") in by_key:
            comparisons.append(
                ("stimulated_vs_baseline", stim, group_rates(by_key[("baseline", "none")][0], ids))
            )
        for ctl in ("degree_preserving_rewire", "sign_shuffle"):
            if ("stimulated", ctl) in by_key:
                pooled = np.concatenate([group_rates(c, ids) for c in by_key[("stimulated", ctl)]])
                comparisons.append((f"stimulated_vs_{ctl}", stim, pooled))
        for label, test, ctrl in comparisons:
            r = compare_groups(test, ctrl, seed=spec.seed)
            rows.append({"readout": name, "n_neurons": len(ids), "comparison": label, **r})
    readouts = pd.DataFrame(rows)
    if not readouts.empty:
        readouts["q_value"] = benjamini_hochberg(readouts["p_value"].to_numpy())
        readouts["significant"] = readouts["q_value"] < spec.report.alpha

    # Top neurons by rate in the stimulated condition (and perturbed delta when present).
    stim_counts = by_key[("stimulated", "none")][0]
    mean_rate = stim_counts.mean(axis=0) / seconds
    top_idx = np.argsort(-mean_rate)[: spec.report.top_neurons]
    meta = m.meta
    top = pd.DataFrame(
        {
            "neuron_id": net.neuron_ids[top_idx],
            "cell_type": meta.iloc[top_idx]["cell_type"].to_numpy()
            if "cell_type" in meta
            else None,
            "rate_hz_stimulated": mean_rate[top_idx],
        }
    )
    if ("perturbed", "none") in by_key:
        pert_rate = by_key[("perturbed", "none")][0].mean(axis=0) / seconds
        top["rate_hz_perturbed"] = pert_rate[top_idx]
        top["delta_hz"] = top["rate_hz_perturbed"] - top["rate_hz_stimulated"]

    prov: dict[str, Any] = {
        "label": "model prediction",
        "spec": spec.raw | {"name": spec.name},
        "spec_sha256": spec.source_sha256,
        "spec_path": spec.source_path,
        "dataset": store.ref,
        "dataset_provenance": store.provenance,
        "network": {**m.provenance, "calibration": net.provenance.get("calibration")},
        "seed": spec.seed,
        "trials": spec.trials,
        "n_steps": spec.n_steps,
        "device": spec.device,
        "dtype": spec.dtype,
        "stimulated_neurons": len(background),
        "silenced_neurons": len(silence),
        "readouts": {k: len(v) for k, v in readout_ids.items()},
        "controls": {
            "degree_preserving_rewire": spec.controls.degree_preserving_rewire,
            "sign_shuffle": spec.controls.sign_shuffle,
            "unstimulated_baseline": spec.controls.unstimulated_baseline,
        },
        "flyconn_version": flyconn.__version__,
        "torch_version": torch.__version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "citations": [c.__dict__ for c in store.citations()],
    }
    write_parquet(pa.Table.from_pandas(counts_df, preserve_index=False), out / "counts.parquet")
    write_parquet(pa.Table.from_pandas(readouts, preserve_index=False), out / "readouts.parquet")
    write_parquet(
        pa.Table.from_pandas(conditions, preserve_index=False), out / "conditions.parquet"
    )
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    result = ExperimentResult(out, conditions, counts_df, readouts, top, prov)
    (out / "report.html").write_text(render_report(result, spec, store))
    return result
