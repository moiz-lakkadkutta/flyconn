"""Version drift: neurons, attributes and edges that changed between two releases (W4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from flyconn.data.store import Store

_ATTRS = ("cell_type", "super_class", "nt_pred", "side", "hemilineage")


@dataclass
class VersionDiff:
    old_ref: str
    new_ref: str
    neurons_added: list[int]
    neurons_removed: list[int]
    neuron_changes: pd.DataFrame
    edge_changes: pd.DataFrame
    unmatched_ids: list[int]
    caveats: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, int]:
        status = self.edge_changes["status"] if len(self.edge_changes) else pd.Series(dtype=str)
        return {
            "neurons_added": len(self.neurons_added),
            "neurons_removed": len(self.neurons_removed),
            "neurons_changed": len(self.neuron_changes),
            "edges_added": int((status == "added").sum()),
            "edges_removed": int((status == "removed").sum()),
            "edges_changed": int((status == "changed").sum()),
            "unmatched_ids": len(self.unmatched_ids),
        }


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    return cast("pd.Series", df[col])


def _neuron_changes(old: pd.DataFrame, new: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    attrs = [a for a in _ATTRS if a in old.columns and a in new.columns]
    o = old.set_index("neuron_id").reindex(ids)
    n = new.set_index("neuron_id").reindex(ids)
    changed = np.zeros(len(ids), dtype=bool)
    out: dict[str, np.ndarray] = {"neuron_id": ids}
    for a in attrs:
        oa = _series(o, a).astype(object).to_numpy()
        na = _series(n, a).astype(object).to_numpy()
        o_na = pd.isna(oa)
        n_na = pd.isna(na)
        changed |= ~((oa == na) | (o_na & n_na))
        out[f"{a}_old"] = oa
        out[f"{a}_new"] = na
    frame = pd.DataFrame(out)
    return cast("pd.DataFrame", frame[changed]).reset_index(drop=True)


def _edge_changes(old: pd.DataFrame, new: pd.DataFrame, ids: np.ndarray | None) -> pd.DataFrame:
    def prep(e: pd.DataFrame, name: str) -> pd.Series:
        if ids is not None:
            keep = _series(e, "pre").isin(list(ids)) | _series(e, "post").isin(list(ids))
            e = cast("pd.DataFrame", e[keep])
        idx = pd.MultiIndex.from_arrays([_series(e, "pre"), _series(e, "post")])
        return pd.Series(_series(e, "weight").to_numpy(dtype=float), index=idx, name=name)

    joined = pd.concat([prep(old, "weight_old"), prep(new, "weight_new")], axis=1)
    w_old = _series(joined, "weight_old")
    w_new = _series(joined, "weight_new")
    status = np.where(w_old.isna(), "added", np.where(w_new.isna(), "removed", "changed"))
    joined["status"] = status
    same = (w_old == w_new).to_numpy()
    joined = cast("pd.DataFrame", joined[~same]).reset_index()
    joined.index.name = None
    return joined.sort_values(["status", "pre", "post"], ignore_index=True)


def diff_versions(old: Store, new: Store, neuron_ids: Sequence[int] | None = None) -> VersionDiff:
    """Compare two converted releases of the same dataset by shared neuron id.

    Identity is by ``neuron_id``; ids of interest that are absent from ``new``
    are reported as ``unmatched_ids`` (root ids change with proofreading, so they
    need a lineage lookup, e.g. CAVE, to be followed).
    """
    on = old.neurons()
    nn = new.neurons()
    old_ids = set(cast("pd.Series", on["neuron_id"]).tolist())
    new_ids = set(cast("pd.Series", nn["neuron_id"]).tolist())
    ids = np.array(sorted(old_ids & new_ids), dtype=np.int64)
    focus = np.asarray(list(neuron_ids), dtype=np.int64) if neuron_ids is not None else None
    if focus is not None:
        ids = ids[np.isin(ids, focus)]
    unmatched = sorted(int(i) for i in (set(focus.tolist()) - new_ids)) if focus is not None else []
    caveats = []
    if unmatched:
        caveats.append(
            f"{len(unmatched)} neuron id(s) of interest are absent from {new.ref}; root ids change "
            "with proofreading, so a lineage lookup (e.g. CAVE) is needed to follow them."
        )
    return VersionDiff(
        old_ref=old.ref,
        new_ref=new.ref,
        neurons_added=sorted(int(i) for i in (new_ids - old_ids)),
        neurons_removed=sorted(int(i) for i in (old_ids - new_ids)),
        neuron_changes=_neuron_changes(on, nn, ids),
        edge_changes=_edge_changes(old.edges(), new.edges(), focus),
        unmatched_ids=unmatched,
        caveats=caveats,
        provenance={"old": old.provenance.get("counts"), "new": new.provenance.get("counts")},
    )
