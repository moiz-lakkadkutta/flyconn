"""Map a screen to a replicate dataset by cell type (ADR-0011; reported, never graded)."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from flyconn.experiments.spec import (
    ExperimentSpec,
    Readout,
    Selection,
    SilenceEachSweep,
    Stimulation,
    derive,
    resolve_selection,
)
from flyconn.experiments.sweep import Candidate


def _rep_ids(rep: pd.DataFrame, types: set[str], side: str | None) -> np.ndarray:
    m = rep["cell_type"].astype(object).isin(sorted(types))
    if side is not None:
        m &= rep["side"].astype(object) == side
    return np.sort(rep.loc[m, "neuron_id"].to_numpy(dtype=np.int64))


def map_selection(
    sel: Selection, query: pd.DataFrame, rep: pd.DataFrame, type_map: pd.DataFrame
) -> tuple[np.ndarray | None, str]:
    if sel.ids is None and sel.cell_type is None and sel.cell_class is None:
        try:
            return resolve_selection(sel, rep), ""
        except (ValueError, KeyError) as e:
            return None, f"selection empty in replicate: {e}"
    ids = resolve_selection(sel, query)
    ct = query.set_index("neuron_id")["cell_type"].reindex(ids).dropna().astype(str)
    types = sorted(set(ct))
    if not types:
        return None, f"{sel.describe()} has no cell types to map"
    tm = type_map[type_map["type_a"].isin(sorted(types))]
    missing = sorted(set(types) - set(tm["type_a"].astype(str)))
    if missing:
        return None, f"unmapped types {missing[:5]}"
    out = _rep_ids(rep, set(tm["type_b"].astype(str)), sel.side)
    if len(out) == 0:
        return None, f"mapped types of {sel.describe()} have no neurons in the replicate"
    return out, ""


def map_candidates(
    candidates: Sequence[Candidate],
    rep: pd.DataFrame,
    type_map: pd.DataFrame,
    group_by: str,
    *,
    exclude: Collection[int] = frozenset(),
) -> tuple[list[Candidate], dict[str, str]]:
    """Map candidates by type; ``exclude`` (replicate stimulus/readout ids) is dropped from each."""
    if group_by != "cell_type":
        reason = "not testable: candidates not grouped by cell type"
        return [], {c.label: reason for c in candidates}
    excluded = {int(i) for i in exclude}
    mapped: list[Candidate] = []
    reasons: dict[str, str] = {}
    for c in candidates:
        tb = set(type_map.loc[type_map["type_a"] == c.label, "type_b"].astype(str))
        ids = _rep_ids(rep, tb, None) if tb else np.array([], dtype=np.int64)
        if len(ids) == 0:
            reasons[c.label] = f"not testable: no replicate type for {c.label}"
            continue
        kept = tuple(int(i) for i in ids if int(i) not in excluded)
        if not kept:
            reasons[c.label] = "not testable: maps only to replicate stimulus/readout neurons"
            continue
        mapped.append(Candidate(c.label, kept, c.role))
    return mapped, reasons


def replicate_spec(
    variant_spec: ExperimentSpec,
    original: ExperimentSpec,
    query: pd.DataFrame,
    rep: pd.DataFrame,
    type_map: pd.DataFrame,
) -> tuple[ExperimentSpec | None, str]:
    """``variant_spec`` with stimulus and readouts re-expressed as replicate neuron ids."""
    stims: list[Stimulation] = []
    for st in original.stimulate:
        ids, why = map_selection(st.select, query, rep, type_map)
        if ids is None:
            return None, f"not testable: stimulus {st.select.describe()}: {why}"
        stims.append(Stimulation(Selection(ids=ids.tolist()), st.rate_hz))
    readouts: list[Readout] = []
    for r in original.readouts:
        ids, why = map_selection(r.select, query, rep, type_map)
        if ids is None:
            return None, f"not testable: readout {r.name}: {why}"
        readouts.append(Readout(r.name, Selection(ids=ids.tolist())))
    sweep = variant_spec.sweep
    if isinstance(sweep, SilenceEachSweep):
        sweep = replace(sweep, select=Selection())  # candidates are fixed; natural not computed
    spec = derive(
        variant_spec,
        parent_sha256=variant_spec.parent_sha256,
        stimulate=stims,
        readouts=readouts,
        sweep=sweep,
    )
    return spec, ""


def replicate_exclusions(
    rspec: ExperimentSpec | None, primary_readout: str, rep: pd.DataFrame
) -> set[int]:
    """Replicate stimulus ids plus the primary readout's ids (as the reference screen excludes)."""
    if rspec is None:
        return set()
    out: set[int] = set()
    for st in rspec.stimulate:
        out.update(int(i) for i in resolve_selection(st.select, rep))
    for ro in rspec.readouts:
        if ro.name == primary_readout:
            out.update(int(i) for i in resolve_selection(ro.select, rep))
    return out
