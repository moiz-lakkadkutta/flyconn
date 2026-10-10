"""Zero-shot cell-type classification by partner-type profile (ADR-0010).

Query neurons are unlabelled; their partners carry reference names through the query
dataset's cross-reference columns. Each neuron (and the selection as a group) is scored
against the reference atlas and called ``type``, ``ambiguous`` or ``unknown`` under a
calibration fitted by the hold-out benchmark, or returned ``uncalibrated`` with raw scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas
from flyconn.compare.calibrate import (
    Calibration,
    decide,
    default_calibration,
    uncalibrated_calls,
)
from flyconn.compare.profiles import partner_counts, vocab_partner_names
from flyconn.compare.scoring import Scores, score
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.provenance import run_environment

HETEROGENEOUS_AGREEMENT = 0.7
UNLABELLED_CAVEAT = 0.3
MIN_VOCAB_OVERLAP = 0.01

BASE_CAVEATS = [
    "Assignments come from wiring similarity alone (partner-type profiles), not "
    "morphology, genetics or transmitter identity.",
    "Partner names come from the query dataset's published cross-references; errors in "
    "those propagate into the profiles.",
    "A type call names the nearest reference label; types split or merged between "
    "datasets are not resolved one to one.",
]


@dataclass
class GroupCall:
    call: str
    label: str
    p: float
    s1: float
    margin: float
    agreement: float
    votes: dict[str, int]
    n: int
    top_labels: list[str] = field(default_factory=list)
    top_s: list[float] = field(default_factory=list)
    n_with_partners: int = 0


@dataclass
class ClassifyResult:
    per_neuron: pd.DataFrame
    group: GroupCall | None
    calibrated: bool
    caveats: list[str]
    provenance: dict[str, Any]


def _resolve_calibration(
    calibration: Calibration | Literal["default"] | None, atlas: Atlas, query_ref: str
) -> tuple[Calibration | None, str]:
    if calibration is None:
        return None, (
            "UNCALIBRATED: no calibration requested; s1 is a raw cosine similarity, "
            "not a probability."
        )
    cal = default_calibration(atlas) if calibration == "default" else calibration
    if cal is None:
        return None, (
            f"UNCALIBRATED: no calibration file for reference {atlas.reference} with these "
            "atlas parameters; s1 is a raw cosine similarity, not a probability."
        )
    reason = cal.incompatibility(atlas)
    if reason is not None:
        return None, f"UNCALIBRATED: the calibration does not match this atlas ({reason})."
    transfer = cal.metrics.get("transfer_ece", {})
    nan = float("nan")
    ece_txt = ", ".join(
        f"{q}: neuron {v.get('neuron', nan):.2f} / group {v.get('group', nan):.2f}"
        for q, v in transfer.items()
    )
    msg = (
        f"Calibrated on {', '.join(cal.fitted_on)} against {cal.reference} (calibration "
        f"{cal.id}); leave-one-dataset-out ECE {ece_txt or 'not measured'}."
    )
    if query_ref in cal.fitted_on:
        msg += f" {query_ref} was part of the calibration data."
    return cal, msg


def _calls(scores: Scores, cal: Calibration | None, level: str) -> pd.DataFrame:
    if cal is None or level not in cal.models:
        return uncalibrated_calls(scores)
    return decide(scores, cal.models[level], cal.thresholds[level])


def _has_cross_reference(meta: pd.DataFrame) -> bool:
    return any(
        col in meta.columns and bool(meta[col].notna().any())
        for col in ("fafb_783_cell_type", "manc_121_cell_type")
    )


def classify_matrix(
    m: ConnectivityMatrix,
    atlas: Atlas,
    *,
    query_ref: str,
    cell_type: str | None = None,
    neuron_ids: Sequence[int] | None = None,
    calibration: Calibration | Literal["default"] | None = "default",
    partner_names: pd.Series | None = None,
    mask: Sequence[str] = (),
    mask_self: bool = True,
    top_k: int = 5,
    query_store_provenance: Mapping[str, Any] | None = None,
) -> ClassifyResult:
    """Classify neurons of an already-built query matrix (see :func:`classify`)."""
    if (cell_type is None) == (neuron_ids is None):
        msg = "pass exactly one of cell_type or neuron_ids"
        raise ValueError(msg)
    if cell_type is not None:
        idx = np.flatnonzero((m.meta["cell_type"].astype(object) == cell_type).to_numpy())
        if len(idx) == 0:
            msg = f"no neurons of type {cell_type!r} in {query_ref}"
            raise KeyError(msg)
    else:
        idx = np.unique(m.index_of(np.unique(np.asarray(neuron_ids, dtype=np.int64))))
    vocabulary = str(atlas.params["vocabulary"])
    no_calibration_reason: str | None = None
    if partner_names is None:
        names = vocab_partner_names(m.meta, set(atlas.vocab.tolist()), vocabulary)
        names_source = vocabulary
        if vocabulary == "fafb_or_manc" and not _has_cross_reference(m.meta):
            names_source = "query cell_type (no cross-reference)"
            no_calibration_reason = (
                f"{query_ref} has no cross-reference columns, so partners are named by its "
                "own cell types (no cross-reference); the calibration assumes cross-referenced "
                "partner names"
            )
    else:
        aligned = partner_names.reindex(m.neuron_ids)
        names = [v if isinstance(v, str) else None for v in aligned.tolist()]
        names_source = "override"
        no_calibration_reason = (
            "partner_names were supplied, so partner names differ from the cross-referenced "
            "names the calibration was fitted on"
        )
    counts_all = partner_counts(m, names, vocab=atlas.vocab.tolist())
    lab, tot = counts_all.labelled_mass(direction=atlas.direction)
    if tot > 0 and lab / tot < MIN_VOCAB_OVERLAP:
        msg = (
            f"only {lab / tot:.2%} of {query_ref} synapses go to partners named in the atlas "
            f"vocabulary ({vocabulary}); check the query's cross-reference columns"
        )
        raise ValueError(msg)
    counts = counts_all.rows(idx)
    own = {str(names[i]) for i in idx.tolist() if isinstance(names[i], str)} if mask_self else set()
    mask_t = tuple(sorted({*mask, *own}))
    atl = atlas.masked(mask_t) if mask_t else atlas
    prof = counts.profiles(direction=atlas.direction, mask=mask_t)
    empty = np.asarray(prof.getnnz(axis=1)) == 0
    if no_calibration_reason is not None and calibration is not None:
        cal: Calibration | None = None
        cal_caveat = f"UNCALIBRATED: {no_calibration_reason}."
    else:
        cal, cal_caveat = _resolve_calibration(calibration, atl, query_ref)

    ns = score(atl, prof, level="neuron", top_k=top_k)
    calls = _calls(ns, cal, "neuron")
    per = pd.DataFrame(
        {
            "neuron_id": counts.neuron_ids,
            "call": calls["call"].to_numpy(dtype=object),
            "label": calls["label"].to_numpy(dtype=object),
            "p": calls["p"].to_numpy(dtype=float),
            "s1": ns.s1,
            "a1": ns.a1,
            "margin": ns.margin,
            "top_labels": [[str(x) for x in r] for r in ns.labels.tolist()],
            "top_s": [[float(x) for x in r] for r in ns.s.tolist()],
            "unlabelled_partner_fraction": counts.unlabelled_fraction(direction=atlas.direction),
            "reason": "",
        }
    )
    per.loc[empty, ["call", "label", "reason"]] = ["unknown", "", "no_labelled_partners"]
    per.loc[empty, "p"] = np.nan

    caveats = [*BASE_CAVEATS, cal_caveat]
    if cal is not None and "neuron" not in cal.models:
        caveats.append(
            "Per-neuron calls are uncalibrated (best label and raw cosine only): in the "
            "benchmark, per-neuron confidence did not transfer between datasets. Use the "
            "group call for a calibrated confidence."
        )
    group: GroupCall | None = None
    if cell_type is not None or len(idx) > 1:
        group = _group_call(atl, prof, empty, ns, cal, top_k)
        if group.call != "unknown" and group.agreement < HETEROGENEOUS_AGREEMENT:
            caveats.append(
                f"heterogeneous selection: only {group.agreement:.0%} of neurons share the "
                f"group's best label {group.top_labels[0]!r} (votes {group.votes}); it may "
                "mix types."
            )
    n_unl = int((per["unlabelled_partner_fraction"] > UNLABELLED_CAVEAT).sum())
    if n_unl:
        caveats.append(
            f"{n_unl} neuron(s) have more than {UNLABELLED_CAVEAT:.0%} of synapses with "
            "partners outside the atlas vocabulary; their profiles cover the rest only."
        )
    top = {str(x) for x in ns.top1[~empty].tolist()}
    fb = sorted(str(x) for x in atl.labels[atl.yardstick_fallback] if x in top)
    if fb:
        caveats.append(
            f"yardstick fallback (atlas median) for {len(fb)} best label(s) with one member "
            f"or one side only: {fb[:10]}"
        )
    if mask_t:
        caveats.append(
            f"partner labels masked: {list(mask_t)}"
            + (
                f" (includes the selection's own cross-reference names {sorted(own)}, as in "
                "the benchmark, so a neuron is not identified by partners of its own type)"
                if own
                else ""
            )
        )
    if any("hemibrain" in r for r in (query_ref, atlas.reference)):
        caveats.append("hemibrain is a truncated volume: similarities are lowered at its edges.")

    prov: dict[str, Any] = {
        "query": query_ref,
        "query_store": dict(query_store_provenance or {}),
        "query_matrix": dict(m.provenance),
        "reference": atlas.reference,
        "atlas_params": atlas.params,
        "atlas_params_hash": atlas.params_hash,
        "atlas_provenance": atlas.provenance,
        "selection": {"cell_type": cell_type, "n_neurons": len(idx)},
        "partner_names": names_source,
        "mask": list(mask_t),
        "top_k": top_k,
        "calibration_id": cal.id if cal is not None else None,
        "environment": run_environment(),
    }
    return ClassifyResult(
        per_neuron=per, group=group, calibrated=cal is not None, caveats=caveats, provenance=prov
    )


def _group_call(
    atlas: Atlas,
    prof: sp.csr_matrix,
    empty: np.ndarray,
    ns: Scores,
    cal: Calibration | None,
    top_k: int,
) -> GroupCall:
    n = len(empty)
    if empty.all():
        return GroupCall("unknown", "", float("nan"), 0.0, 0.0, 0.0, {}, n)
    gp = sp.csr_matrix(prof[np.flatnonzero(~empty)].mean(axis=0))
    gs = score(atlas, gp, level="group", top_k=top_k)
    gcall = _calls(gs, cal, "group")
    tops = pd.Series(ns.top1[~empty].astype(str))
    best = str(gs.top1[0])
    return GroupCall(
        call=str(gcall["call"].iloc[0]),
        label=str(gcall["label"].iloc[0]),
        p=float(gcall["p"].iloc[0]),
        s1=float(gs.s1[0]),
        margin=float(gs.margin[0]),
        agreement=float((tops == best).mean()),
        votes={str(k): int(v) for k, v in tops.value_counts().head(10).items()},
        n=n,
        top_labels=[str(x) for x in gs.labels[0].tolist()],
        top_s=[float(x) for x in gs.s[0].tolist()],
        n_with_partners=int((~empty).sum()),
    )


def classify(
    query: Store,
    atlas: Atlas,
    *,
    cell_type: str | None = None,
    neuron_ids: Sequence[int] | None = None,
    calibration: Calibration | Literal["default"] | None = "default",
    partner_names: pd.Series | None = None,
    mask: Sequence[str] = (),
    mask_self: bool = True,
    top_k: int = 5,
) -> ClassifyResult:
    """Assign reference types to query neurons by partner-type profile.

    Select neurons by ``cell_type`` (the query's own label) or ``neuron_ids``. Partners are
    named with the atlas vocabulary through the query's cross-reference columns, or by
    ``partner_names`` (a Series indexed by neuron id). ``mask`` blanks labels as partner
    names; ``mask_self`` (default) also blanks the selection's own cross-reference names, as
    the benchmark does for the held-out type. ``calibration="default"`` loads
    the shipped calibration for the atlas reference; None returns raw scores.
    """
    m = ConnectivityMatrix.from_store(query, min_weight=int(atlas.params["min_weight"]))
    return classify_matrix(
        m,
        atlas,
        query_ref=query.ref,
        cell_type=cell_type,
        neuron_ids=neuron_ids,
        calibration=calibration,
        partner_names=partner_names,
        mask=mask,
        mask_self=mask_self,
        top_k=top_k,
        query_store_provenance=query.provenance,
    )
