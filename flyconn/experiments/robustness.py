"""Robustness-audited silencing screens (M13, ADR-0011).

One axis at a time around the calibrated reference: w_syn, sampled NT signs, network
threshold and a cross-dataset replicate. Every variant silences the reference run's
candidates, so effects are comparable across variants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from flyconn.data.cache import store_dir
from flyconn.data.pull import pull
from flyconn.data.registry import get_dataset
from flyconn.data.store import Store
from flyconn.experiments.robust_null import draw_null_sets, null_key, null_p
from flyconn.experiments.robust_replicate import map_candidates, map_selection, replicate_spec
from flyconn.experiments.robust_scoring import score_hits, variant_label
from flyconn.experiments.runner import rates_hz, write_table
from flyconn.experiments.spec import (
    NO_CONTROLS,
    ExperimentSpec,
    Selection,
    SilenceEachSweep,
    SimParams,
    derive,
    resolve_selection,
)
from flyconn.experiments.sweep import Candidate, run_sweep
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.sim.params import CALIBRATIONS, default_params
from flyconn.uncertainty.stability import path_stability


@dataclass(frozen=True)
class Variant:
    axis: str
    label: str
    spec: ExperimentSpec


def _r6(x: float) -> float:
    return round(float(x), 6)


def w_syn_values(spec: ExperimentSpec) -> tuple[float, ...]:
    """w_syn values of the w_syn axis, sorted, without the reference value."""
    r = spec.robustness
    if r is None or r.w_syn == "off":
        return ()
    ref = spec.params.w_syn_mv or default_params(spec.dataset).w_syn_mv
    if r.w_syn == "auto":
        cal = CALIBRATIONS.get(spec.dataset)
        if cal is None:
            msg = (
                f"robustness w_syn: auto needs a calibration entry for {spec.dataset}; "
                "give an explicit list of mV values or 'off'"
            )
            raise ValueError(msg)
        values = {*cal.ci_mv, *cal.sensitivity_mv}
    elif isinstance(r.w_syn, str):
        msg = f"robustness w_syn must be 'auto', 'off' or a list of mV values, got {r.w_syn!r}"
        raise ValueError(msg)
    else:
        values = set(r.w_syn)
    return tuple(sorted({_r6(v) for v in values} - {_r6(ref)}))


def plan_variants(spec: ExperimentSpec) -> list[Variant]:
    """Reference plus one variant per axis value; variants run without shared controls."""
    r = spec.robustness
    if r is None:
        msg = "spec has no robustness block"
        raise ValueError(msg)
    parent = spec.source_sha256

    def variant(axis: str, label: str, **changes: object) -> Variant:
        s = derive(spec, parent_sha256=parent, robustness=None, controls=NO_CONTROLS, **changes)
        return Variant(axis, label, s)

    out = [Variant("reference", "reference", spec)]
    out += [variant("w_syn", f"wsyn_{w:g}", params=SimParams(w)) for w in w_syn_values(spec)]
    for k in range(r.nt_samples):
        net = replace(spec.network, sign_policy="sampled", sign_seed=spec.seed + 2000 + k)
        out.append(variant("nt", f"nt_s{k}", network=net))
    for mw in r.min_weights:
        if mw != spec.network.min_weight:
            out.append(
                variant("min_weight", f"minw_{mw}", network=replace(spec.network, min_weight=mw))
            )
    if r.replicate:
        label = "rep_" + r.replicate.replace("@", "_").replace(".", "_")
        out.append(variant("replicate", label, dataset=r.replicate, params=SimParams()))
    return out


@dataclass
class RobustScreenResult:
    out_dir: Path
    spec: ExperimentSpec
    hits: pd.DataFrame  # score_hits output + path_stability column
    effects: pd.DataFrame  # every candidate x variant on the primary readout, with label
    null: pd.DataFrame  # key, index, seed, n_neurons, difference
    selection_stability: pd.DataFrame  # variant, axis, jaccard
    provenance: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class _Loaded:
    sweep: pd.DataFrame
    candidates: list[Candidate]
    natural: list[str]


def _plan_hash(spec: ExperimentSpec, plan: list[Variant]) -> str:
    payload = {
        "spec": spec.source_sha256,
        "variants": [(v.label, v.spec.source_sha256) for v in plan],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _open_out(out: Path, plan_hash: str) -> None:
    """Create ``out`` with the plan file, or check an existing one holds the same plan."""
    plan_file = out / "robustness_plan.json"
    if out.exists() and any(out.iterdir()):
        if not plan_file.exists() or json.loads(plan_file.read_text())["hash"] != plan_hash:
            msg = (
                f"{out} holds a different robustness plan (or none); choose a new output "
                "directory to run a changed spec"
            )
            raise FileExistsError(msg)
        return
    out.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps({"hash": plan_hash}, indent=1))


def _run_or_load(
    d: Path,
    spec: ExperimentSpec,
    store: Store,
    *,
    fixed: list[Candidate] | None,
    compute_natural: bool,
) -> _Loaded:
    """Load a finished variant (``done.json``) or run it; an unfinished one is moved aside."""
    done = d / "done.json"
    if done.exists():
        meta = json.loads(done.read_text())
        cands = [
            Candidate(str(c[0]), tuple(int(i) for i in c[1]), str(c[2])) for c in meta["candidates"]
        ]
        return _Loaded(
            pd.read_parquet(d / "sweep.parquet"), cands, [str(x) for x in meta["natural"]]
        )
    if d.exists():
        inc = d.parent / ".incomplete"
        inc.mkdir(exist_ok=True)
        n = 1
        while (inc / f"{d.name}-{n}").exists():
            n += 1
        d.rename(inc / f"{d.name}-{n}")
    res = run_sweep(spec, out_dir=d, store=store, fixed=fixed, compute_natural=compute_natural)
    meta = {
        "candidates": [[c.label, list(c.neuron_ids), c.role] for c in res.candidates],
        "natural": res.natural_selection,
    }
    done.write_text(json.dumps(meta))  # written last: marks the variant complete
    return _Loaded(res.sweep, res.candidates, res.natural_selection)


def _primary_rows(axis: str, label: str, loaded: _Loaded, readout: str) -> pd.DataFrame:
    sw = loaded.sweep
    t = sw.loc[sw["readout"] == readout]
    return pd.DataFrame(
        {
            "axis": axis,
            "variant": label,
            "candidate": [str(x) for x in t["variant"]],
            "role": [str(x) for x in t["role"]],
            "difference": np.asarray(t["difference"], dtype=float),
            "ci_low": np.asarray(t["ci_low"], dtype=float),
            "ci_high": np.asarray(t["ci_high"], dtype=float),
            "q_value": np.asarray(t["q_value"], dtype=float),
        }
    )


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return 1.0 if not sa and not sb else len(sa & sb) / len(sa | sb)


def _replicate_store(ref: str) -> Store | None:
    d = store_dir(get_dataset(ref))
    return Store(d) if (d / "provenance.json").exists() else None


def _reference_differences(reference_rows: pd.DataFrame) -> dict[str, float]:
    """candidate -> difference of a primary-readout table."""
    return {
        str(c): float(d)
        for c, d in zip(reference_rows["candidate"], reference_rows["difference"], strict=True)
    }


def _replicate_mapping(
    spec: ExperimentSpec,
    query: pd.DataFrame,
    rep: pd.DataFrame,
    type_map: pd.DataFrame,
    cands: list[Candidate],
    mapped: list[Candidate],
    reasons: dict[str, str],
) -> dict[str, Any]:
    """The replicate mapping table for provenance: stimulus, readouts and candidates."""

    def sel_entry(sel: Selection) -> dict[str, Any]:
        ids, why = map_selection(sel, query, rep, type_map)
        return {
            "selection": sel.describe(),
            "n_replicate_ids": None if ids is None else len(ids),
            "reason": why,
        }

    by_label = {c.label: c for c in mapped}
    candidates: dict[str, dict[str, Any]] = {}
    for c in cands:
        m = by_label.get(c.label)
        if m is None:
            candidates[c.label] = {"reason": reasons.get(c.label, "not testable")}
            continue
        types = sorted(set(type_map.loc[type_map["type_a"] == c.label, "type_b"].astype(str)))
        candidates[c.label] = {"replicate_types": types, "n_replicate_ids": len(m.neuron_ids)}
    sources = sorted(set(type_map["source"].astype(str))) if len(type_map) else []
    return {
        "type_map_rows": len(type_map),
        "type_map_source": sources,
        "stimulus": [sel_entry(st.select) for st in spec.stimulate],
        "readouts": {ro.name: sel_entry(ro.select) for ro in spec.readouts},
        "candidates": candidates,
    }


def _effect_label(axis: str, ref_difference: float, d: float, lo: float, hi: float) -> str:
    if axis == "reference":
        return "reference"
    if ref_difference == 0:  # no reference direction: never a hit, nothing to compare
        return "weakens"
    return variant_label(ref_difference, d, lo, hi)


def run_robust_screen(
    spec: ExperimentSpec,
    *,
    out_dir: Path | str,
    store: Store | None = None,
    replicate_store: Store | None = None,
) -> RobustScreenResult:
    """Run the reference screen and every robustness variant; score hits (ADR-0011).

    Writes ``robustness_plan.json``, ``variants/<label>/`` (each with ``done.json`` written
    last, so a rerun into the same directory resumes), ``variants/null/``, ``hits.parquet``
    (the effects table), ``null.parquet``, ``selection_stability.parquet`` and
    ``provenance.json``. A changed spec refuses to reuse an output directory.
    """
    r = spec.robustness
    sweep = spec.sweep
    if r is None:
        msg = "spec has no robustness block"
        raise ValueError(msg)
    if not isinstance(sweep, SilenceEachSweep):
        msg = "a robustness screen needs a silence_each sweep"
        raise ValueError(msg)
    plan = plan_variants(spec)
    plan_hash = _plan_hash(spec, plan)
    out = Path(out_dir)
    _open_out(out, plan_hash)
    store = store or Store(pull(spec.dataset, level="weights").store_dir)
    vdir = out / "variants"
    vdir.mkdir(exist_ok=True)
    primary = r.primary_readout

    ref = _run_or_load(vdir / "reference", spec, store, fixed=None, compute_natural=True)
    cands = ref.candidates
    effects = [_primary_rows("reference", "reference", ref, primary)]
    stability: list[dict[str, Any]] = []
    replicate_labels: dict[str, str] = {}
    rep_prov: dict[str, Any] = {}

    for v in plan[1:]:
        if v.axis == "replicate":
            continue
        loaded = _run_or_load(vdir / v.label, v.spec, store, fixed=cands, compute_natural=True)
        effects.append(_primary_rows(v.axis, v.label, loaded, primary))
        stability.append(
            {"variant": v.label, "axis": v.axis, "jaccard": _jaccard(ref.natural, loaded.natural)}
        )

    rep_v = next((v for v in plan if v.axis == "replicate"), None)
    if rep_v is not None:
        rep_store = replicate_store or _replicate_store(rep_v.spec.dataset)
        if rep_store is None:
            why = f"not run: `flyconn data pull {rep_v.spec.dataset} --level weights` first"
            replicate_labels = {c.label: why for c in cands}
            rep_prov = {"reason": why}
        else:
            from flyconn.compare.types import match_types  # heavy import (scipy), kept lazy

            type_map = match_types(store, rep_store)
            q_n, rep_n = store.neurons(), rep_store.neurons()
            rspec, why = replicate_spec(rep_v.spec, spec, q_n, rep_n, type_map)
            mapped, replicate_labels = map_candidates(cands, rep_n, type_map, sweep.group_by)
            rep_prov = _replicate_mapping(
                spec, q_n, rep_n, type_map, cands, mapped, replicate_labels
            )
            rep_prov = {"dataset": rep_store.ref, **rep_prov, "reason": why}
            if rspec is None:
                replicate_labels = {c.label: why for c in cands}
            elif mapped:
                loaded = _run_or_load(
                    vdir / rep_v.label, rspec, rep_store, fixed=mapped, compute_natural=False
                )
                rows = _primary_rows("replicate", rep_v.label, loaded, primary)
                effects.append(rows)
                ref_d = _reference_differences(effects[0])
                for cand, d, lo, hi in zip(
                    rows["candidate"],
                    rows["difference"],
                    rows["ci_low"],
                    rows["ci_high"],
                    strict=True,
                ):
                    replicate_labels[str(cand)] = _effect_label(
                        "replicate", ref_d.get(str(cand), 0.0), float(d), float(lo), float(hi)
                    )
                seen = set(rows["candidate"].astype(str))
                for c in mapped:
                    if c.label not in seen:  # mapped but no row: say so, never the default
                        replicate_labels[c.label] = "not testable: no replicate result"

    neurons = store.neurons()
    null_rows: list[dict[str, Any]] = []
    null_by_cand: dict[str, float | None] = {}
    skipped: dict[str, str] = {}
    if r.random_null_sets > 0:
        excluded = set(rates_hz(spec.stimulate, neurons)) | {
            int(i) for ro in spec.readouts for i in resolve_selection(ro.select, neurons)
        }
        sets, skipped = draw_null_sets(
            neurons, cands, excluded, n_sets=r.random_null_sets, seed=spec.seed
        )
        if sets:
            null_spec = derive(
                spec, parent_sha256=spec.source_sha256, robustness=None, controls=NO_CONTROLS
            )
            fixed = [Candidate(f"null:{s.key}:{s.index}", s.neuron_ids, "null") for s in sets]
            loaded = _run_or_load(
                vdir / "null", null_spec, store, fixed=fixed, compute_natural=False
            )
            null_d = _reference_differences(_primary_rows("null", "null", loaded, primary))
            for s in sets:
                null_rows.append(
                    {
                        "key": s.key,
                        "index": s.index,
                        "seed": s.seed,
                        "n_neurons": len(s.neuron_ids),
                        "difference": null_d[f"null:{s.key}:{s.index}"],
                    }
                )
    null_df = pd.DataFrame(
        null_rows, columns=pd.Index(["key", "index", "seed", "n_neurons", "difference"])
    )
    ref_d = _reference_differences(effects[0])
    control_null_p: dict[str, float | None] = {}
    for c in cands:
        key = null_key(neurons, c.neuron_ids)
        nulls = [float(x) for x in null_df.loc[null_df["key"] == key, "difference"]]
        p = null_p(ref_d.get(c.label, 0.0), nulls) if nulls else None
        (null_by_cand if c.role == "candidate" else control_null_p)[c.label] = p

    eff = pd.concat(effects, ignore_index=True)
    eff["label"] = [
        _effect_label(str(a), ref_d.get(str(c), 0.0), float(d), float(lo), float(hi))
        for a, c, d, lo, hi in zip(
            eff["axis"],
            eff["candidate"],
            eff["difference"],
            eff["ci_low"],
            eff["ci_high"],
            strict=True,
        )
    ]
    hits = score_hits(eff, alpha=spec.report.alpha, null_p=null_by_cand, replicate=replicate_labels)
    hits["path_stability"] = _path_stability(spec, store, hits, cands)

    sel = pd.DataFrame(stability, columns=pd.Index(["variant", "axis", "jaccard"]))
    write_table(eff, out / "hits.parquet")
    write_table(null_df, out / "null.parquet")
    write_table(sel, out / "selection_stability.parquet")
    ref_prov: dict[str, Any] = json.loads((vdir / "reference" / "provenance.json").read_text())
    prov = ref_prov | {
        "robustness": {
            "design": "one axis at a time around the calibrated reference (ADR-0011)",
            "plan_hash": plan_hash,
            "variants": [
                {"axis": v.axis, "label": v.label, "spec_sha256": v.spec.source_sha256}
                for v in plan
            ],
            "w_syn_values": list(w_syn_values(spec)),
            "nt_sign_seeds": [spec.seed + 2000 + k for k in range(r.nt_samples)],
            "min_weights": list(r.min_weights),
            "replicate": rep_prov | {"ref": r.replicate},
            "random_null": {
                "n_sets": r.random_null_sets,
                "seeds": [spec.seed + 3000 + i for i in range(r.random_null_sets)],
                "skipped": skipped,
                "p_rule": "(1 + #null at least as extreme) / (n_sets + 1) < 0.05",
            },
            "extra_candidates": [s.describe() for s in r.extra_candidates],
            "control_null_p": control_null_p,
            "statistics": "unpaired bootstrap CI and Mann-Whitney p; common random inputs "
            "(conservative); no correction across variants",
        }
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return RobustScreenResult(out, spec, hits, eff, null_df, sel, prov)


def _path_stability(
    spec: ExperimentSpec, store: Store, hits: pd.DataFrame, cands: list[Candidate]
) -> list[float]:
    """Mean sign_stability x threshold_presence of stimulus->primary paths through each hit."""
    if hits.empty or spec.robustness is None:
        return []
    neurons = store.neurons()
    sources = sorted(rates_hz(spec.stimulate, neurons))
    readout = next(r for r in spec.readouts if r.name == spec.robustness.primary_readout)
    targets = [int(i) for i in resolve_selection(readout.select, neurons)]
    sweep = spec.sweep
    pathway = sweep.pathway if isinstance(sweep, SilenceEachSweep) else None
    max_hops = pathway.max_hops if pathway is not None else 3
    m = ConnectivityMatrix.from_store(store, min_weight=1)
    try:
        res = path_stability(
            m,
            sources,
            targets,
            max_hops=max_hops,
            n_samples=50,
            seed=spec.seed,
            thresholds=(1, 3, 5),
        )
    except ValueError:  # no stimulus->readout paths at all
        return [float("nan")] * len(hits)
    paths = res.paths
    ids_by = {c.label: set(c.neuron_ids) for c in cands}
    out: list[float] = []
    for lbl in hits["candidate"]:
        ids = ids_by[str(lbl)]
        mask = [bool(ids & set(cast("tuple[int, ...]", p))) for p in paths["path"]]
        sub = paths.loc[mask]
        out.append(
            float((sub["sign_stability"] * sub["threshold_presence"]).mean())
            if not sub.empty
            else float("nan")
        )
    return out
