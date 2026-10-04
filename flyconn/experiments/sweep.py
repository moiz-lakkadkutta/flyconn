"""Sweep experiments: one YAML, many variants, one combined result table and report.

Two sweep kinds (exactly one per spec):

``silence_each``
    A silencing screen in the spirit of Shiu et al. 2024: candidate neurons are grouped
    (by cell type, any other neuron column, or one neuron per variant), ranked by their
    summed firing rate in the shared *stimulated* condition, and the ``max_items`` most
    active groups are silenced one at a time. Each variant is compared with the stimulated
    condition. Groups that never fire are not simulated: silencing a neuron zeroes its
    outgoing synapses, which cannot change a run in which it emits no spike.

``rate_hz``
    Every ``stimulate`` entry is set to each listed rate in turn. Each variant is compared
    with the unstimulated baseline (or with the stimulated condition if the baseline is
    disabled), giving readout-vs-rate curves.

The shared conditions (baseline, stimulated, null-connectome controls) run once; each
variant is one batched :func:`~flyconn.sim.engine.simulate` call with the spec's trials and
seeds. Benjamini-Hochberg correction runs across *all* variant x readout rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from flyconn.data.store import Store
from flyconn.experiments.runner import (
    Context,
    base_provenance,
    conditions_frame,
    counts_frame,
    prepare,
    readout_stats,
    shared_runs,
    simulate_condition,
    write_provenance,
    write_table,
)
from flyconn.experiments.spec import (
    ExperimentSpec,
    RateSweep,
    SilenceEachSweep,
    resolve_selection,
)
from flyconn.experiments.stats import benjamini_hochberg, bootstrap_mean_ci, compare_groups
from flyconn.graph.paths import find_paths


@dataclass
class SweepResult:
    out_dir: Path
    kind: str
    sweep: pd.DataFrame  # variant x readout: mean, CI, effect vs reference, p, q, rank
    variants: pd.DataFrame  # one row per candidate variant (simulated or skipped)
    conditions: pd.DataFrame  # shared runs and variant runs
    readouts: pd.DataFrame  # shared comparisons (stimulated vs baseline / controls)
    provenance: dict[str, Any] = field(default_factory=dict)


_CONDITION_COLUMNS: list[str] = [
    "condition",
    "control",
    "replicate",
    "total_spikes",
    "active_neurons",
]


@dataclass
class _Variant:
    label: str
    stimulate: dict[int, float]
    silence: list[int]
    info: dict[str, Any]


def _group_labels(ctx: Context, sweep: SilenceEachSweep, ids: np.ndarray) -> pd.Series:
    """Group label per candidate neuron id (index = neuron_id); unlabelled neurons dropped."""
    if sweep.group_by == "neuron":
        return pd.Series(ids.astype(str), index=ids)
    if sweep.group_by not in ctx.neurons.columns:
        msg = f"sweep group_by: dataset has no column {sweep.group_by!r}"
        raise KeyError(msg)
    col = cast("pd.Series", ctx.neurons.set_index("neuron_id")[sweep.group_by])
    labels = col.reindex(ids)
    labels = cast("pd.Series", labels[labels.notna()])
    return labels.astype(str)


def _pathway_scores(ctx: Context, sweep: SilenceEachSweep) -> pd.DataFrame:
    """Per intermediate neuron: summed strength and count of paths stimulated -> readout."""
    pw = sweep.pathway
    if pw is None:  # pragma: no cover - guarded by spec validation
        msg = "rank_by pathway needs a pathway section"
        raise ValueError(msg)
    sources = sorted(ctx.background)
    targets = ctx.readout_ids[pw.readout].tolist()
    paths = find_paths(
        ctx.matrix,
        sources,
        targets,
        max_hops=pw.max_hops,
        min_weight=pw.min_weight,
        min_edge_fraction=pw.min_edge_fraction,
    )
    excluded = set(sources) | set(targets)
    strength: dict[int, float] = {}
    signed: dict[int, float] = {}
    n_paths: dict[int, int] = {}
    for path, st, sign in zip(paths["path"], paths["strength"], paths["path_sign"], strict=True):
        for nid in set(path[1:-1]) - excluded:
            strength[nid] = strength.get(nid, 0.0) + float(st)
            signed[nid] = signed.get(nid, 0.0) + float(st) * float(sign)
            n_paths[nid] = n_paths.get(nid, 0) + 1
    ids = sorted(strength)
    return pd.DataFrame(
        {
            "neuron_id": np.asarray(ids, dtype=np.int64),
            "path_strength": [strength[i] for i in ids],
            "signed_path_strength": [signed[i] for i in ids],
            "n_paths": [n_paths[i] for i in ids],
        }
    )


def _silence_candidates(
    ctx: Context, sweep: SilenceEachSweep, stim_counts: np.ndarray
) -> tuple[pd.DataFrame, list[_Variant]]:
    ids = resolve_selection(sweep.select, ctx.neurons)
    if sweep.rank_by == "pathway":
        excluded = set(ctx.background) | set(ctx.readout_ids[sweep.pathway.readout].tolist())  # type: ignore[union-attr]
        ids = np.asarray([i for i in ids.tolist() if i not in excluded], dtype=np.int64)
    rate = stim_counts.mean(axis=0) / ctx.seconds
    labels = _group_labels(ctx, sweep, ids)
    per_neuron = pd.DataFrame(
        {
            "neuron_id": labels.index.to_numpy(dtype=np.int64),
            "group": labels.to_numpy(),
            "rate": rate[ctx.net.index_of(labels.index.tolist())],
        }
    )
    agg: dict[str, tuple[str, str]] = {
        "n_neurons": ("neuron_id", "size"),
        "stimulated_rate_hz": ("rate", "sum"),
    }
    if sweep.rank_by == "pathway":
        scores = _pathway_scores(ctx, sweep).set_index("neuron_id")
        for col in ("path_strength", "signed_path_strength", "n_paths"):
            per_neuron[col] = scores[col].reindex(per_neuron["neuron_id"]).fillna(0).to_numpy()
            agg[col] = (col, "sum")
        sort_cols, eligible_col, skip_reason = (
            ["path_strength", "group"],
            "path_strength",
            "not on a path to the readout",
        )
    else:
        sort_cols, eligible_col, skip_reason = (
            ["stimulated_rate_hz", "group"],
            "stimulated_rate_hz",
            "inactive when stimulated",
        )
    groups = (
        per_neuron.groupby("group", sort=True)
        .agg(**agg)  # type: ignore[call-overload]
        .reset_index()
        .sort_values(sort_cols, ascending=[False, True], kind="stable")
        .reset_index(drop=True)
        .rename(columns={"group": "variant"})
    )
    if "n_paths" in groups.columns:
        groups["n_paths"] = groups["n_paths"].astype(int)
    eligible = groups[eligible_col] > 0
    within = np.cumsum(eligible.to_numpy()) <= sweep.max_items
    groups["simulated"] = eligible & within
    groups["reason"] = np.where(
        groups["simulated"], "", np.where(eligible, "beyond max_items", skip_reason)
    )
    members = per_neuron.groupby("group")["neuron_id"].apply(list).to_dict()
    variants = [
        _Variant(
            label=str(lbl),
            stimulate=dict(ctx.background),
            silence=[int(i) for i in members[lbl]],
            info={"n_silenced": int(n)},
        )
        for lbl, n, sim in zip(
            groups["variant"], groups["n_neurons"], groups["simulated"], strict=True
        )
        if sim
    ]
    return groups, variants


def _rate_variants(ctx: Context, sweep: RateSweep) -> tuple[pd.DataFrame, list[_Variant]]:
    variants = [
        _Variant(
            label=f"{r:g} Hz",
            stimulate=dict.fromkeys(ctx.background, r),
            silence=[],
            info={"rate_hz": r},
        )
        for r in sweep.rates_hz
    ]
    table = pd.DataFrame(
        {
            "variant": [v.label for v in variants],
            "rate_hz": sweep.rates_hz,
            "n_neurons": len(ctx.background),
            "simulated": True,
            "reason": "",
        }
    )
    return table, variants


def run_sweep(
    spec: ExperimentSpec, *, out_dir: Path | str, store: Store | None = None
) -> SweepResult:
    """Run a sweep spec; write ``sweep.parquet`` and the usual tables, provenance and report.

    Files: ``sweep.parquet`` (variant x readout), ``variants.parquet`` (every candidate
    variant, including skipped ones and why), ``conditions.parquet`` (all runs),
    ``readouts.parquet`` (shared comparisons against baseline and controls, own BH family),
    ``counts.parquet`` (all non-zero counts of the shared runs; for variant runs only the
    readout neurons, to bound size), ``provenance.json`` and ``report.html``.
    """
    from flyconn.report.html import render_sweep_report  # matplotlib is slow to load

    sweep = spec.sweep
    if sweep is None:
        msg = "spec has no 'sweep' section; use run_experiment"
        raise ValueError(msg)
    ctx = prepare(spec, out_dir, store)
    runs = shared_runs(ctx, spec)
    by_key = {(c, k): counts for c, k, rep, counts in runs if rep == 0}
    stim_counts = by_key[("stimulated", "none")]

    if isinstance(sweep, SilenceEachSweep):
        variants_df, variants = _silence_candidates(ctx, sweep, stim_counts)
        reference = "stimulated"
    else:
        variants_df, variants = _rate_variants(ctx, sweep)
        reference = "baseline" if ("baseline", "none") in by_key else "stimulated"
    ref_counts = by_key[(reference, "none")]

    readout_neurons = np.unique(np.concatenate(list(ctx.readout_ids.values())))
    readout_idx = ctx.net.index_of(readout_neurons.tolist())
    rows: list[dict[str, Any]] = []
    variant_conditions: list[dict[str, Any]] = []
    sparse_counts: list[pd.DataFrame] = []
    for v in variants:
        # A rate equal to the spec's own background rate is the shared stimulated run
        # (same network, stimulation and seeds), so it is reused rather than re-simulated.
        same_as_stim = not v.silence and v.stimulate == ctx.background
        counts = (
            stim_counts
            if same_as_stim
            else simulate_condition(ctx.net, spec, v.stimulate, v.silence)
        )
        condition = f"{sweep.kind}:{v.label}"
        variant_conditions.append(
            {
                "condition": condition,
                "control": "none",
                "replicate": 0,
                "total_spikes": int(counts.sum()),
                "active_neurons": int((counts.sum(axis=0) > 0).sum()),
            }
        )
        sub = counts[:, readout_idx]
        t_idx, n_idx = np.nonzero(sub)
        sparse_counts.append(
            pd.DataFrame(
                {
                    "condition": condition,
                    "control": "none",
                    "replicate": 0,
                    "trial": t_idx,
                    "neuron_id": readout_neurons[n_idx],
                    "count": sub[t_idx, n_idx],
                }
            )
        )
        for name, ids in ctx.readout_ids.items():
            test = ctx.group_rates(counts, ids)
            ref = ctx.group_rates(ref_counts, ids)
            lo, hi = bootstrap_mean_ci(test, seed=spec.seed)
            r = compare_groups(test, ref, seed=spec.seed)
            rows.append(
                {
                    "sweep_kind": sweep.kind,
                    "variant": v.label,
                    **v.info,
                    "readout": name,
                    "n_neurons": len(ids),
                    "mean_hz": r["mean_test"],
                    "mean_ci_low": lo,
                    "mean_ci_high": hi,
                    "reference": reference,
                    "mean_reference_hz": r["mean_control"],
                    "difference": r["difference"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "cohens_d": r["cohens_d"],
                    "p_value": r["p_value"],
                    "n_trials": r["n_test"],
                    "n_trials_reference": r["n_control"],
                }
            )
        del counts

    table = pd.DataFrame(rows)
    if not table.empty:
        table["q_value"] = benjamini_hochberg(table["p_value"].to_numpy())
        table["significant"] = table["q_value"] < spec.report.alpha
        table["rank"] = (
            table.assign(_abs=table["difference"].abs())
            .groupby("readout")["_abs"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        table = table.sort_values(["readout", "rank"], kind="stable").reset_index(drop=True)

    vc = pd.DataFrame(variant_conditions, columns=pd.Index(_CONDITION_COLUMNS))
    variants_df = variants_df.merge(
        vc.assign(variant=[v.label for v in variants])[
            ["variant", "total_spikes", "active_neurons"]
        ],
        on="variant",
        how="left",
    )
    conditions = pd.concat([conditions_frame(runs), vc], ignore_index=True)
    readouts = readout_stats(ctx, spec, runs)
    counts_df = pd.concat([counts_frame(runs, ctx.net), *sparse_counts], ignore_index=True)

    sweep_prov: dict[str, Any] = {"kind": sweep.kind, "reference": reference}
    if isinstance(sweep, SilenceEachSweep):
        sweep_prov |= {
            "select": sweep.select.describe(),
            "group_by": sweep.group_by,
            "max_items": sweep.max_items,
            "candidates": len(variants_df),
            "simulated": int(variants_df["simulated"].sum()),
            "ranked_by": (
                "summed strength of connectome paths from the stimulated neurons to readout "
                f"{sweep.pathway.readout!r} through each candidate (pathway, max_hops "
                f"{sweep.pathway.max_hops}, min_weight {sweep.pathway.min_weight})"
                if sweep.rank_by == "pathway" and sweep.pathway is not None
                else "summed firing rate in the stimulated condition"
            ),
        }
    else:
        sweep_prov |= {"rates_hz": sweep.rates_hz}
    prov = base_provenance(ctx, spec) | {
        "sweep": sweep_prov,
        "n_variants_simulated": len(variants),
        "multiple_comparisons": "Benjamini-Hochberg across all variant x readout rows",
    }
    out = ctx.out
    write_table(table, out / "sweep.parquet")
    write_table(variants_df, out / "variants.parquet")
    write_table(conditions, out / "conditions.parquet")
    write_table(readouts, out / "readouts.parquet")
    write_table(counts_df, out / "counts.parquet")
    write_provenance(prov, out)
    result = SweepResult(out, sweep.kind, table, variants_df, conditions, readouts, prov)
    (out / "report.html").write_text(render_sweep_report(result, spec, ctx.store))
    return result
