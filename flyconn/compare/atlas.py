"""Reference atlas for zero-shot classification (ADR-0010).

One centroid per reference label (mean of member partner profiles) plus two yardsticks of
how alike the label's own members are: the neuron yardstick (median leave-one-out cosine
of members to their own centroid) and the group yardstick (cosine between the left and
right centroids, the within-brain variability used by ``compare_type``). Labels with too
few members or one side only take the atlas-wide median and are flagged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.compare.profiles import (
    PartnerCounts,
    ProfileDirection,
    check_direction,
    l2_rows,
    partner_counts,
)
from flyconn.compare.types import partner_labels
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix


def _rowdot(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1), dtype=float).ravel()


def _membership(codes: np.ndarray, n_labels: int, rows: np.ndarray | None = None) -> sp.csr_matrix:
    sel = np.arange(len(codes)) if rows is None else np.flatnonzero(rows)
    return sp.csr_matrix((np.ones(len(sel)), (codes[sel], sel)), shape=(n_labels, len(codes)))


def _fill(x: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(x)) if np.isfinite(x).any() else 1.0
    return np.where(np.isnan(x), med, x)


def _yardsticks(
    prof: sp.csr_matrix, codes: np.ndarray, sides: np.ndarray, n_labels: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    sums = sp.csr_matrix(_membership(codes, n_labels) @ prof)
    n = np.bincount(codes, minlength=n_labels)
    dot = _rowdot(prof, sp.csr_matrix(sums[codes]))
    pp = _rowdot(prof, prof)
    ss = _rowdot(sums, sums)[codes]
    rest = ss - 2 * dot + pp  # |sum of the other members|^2
    ok = (n[codes] >= 2) & (pp > 0) & (rest > 1e-12)
    loo = np.full(len(codes), np.nan)
    loo[ok] = (dot[ok] - pp[ok]) / np.sqrt(pp[ok] * rest[ok])
    neuron = pd.Series(loo).groupby(codes).median().reindex(range(n_labels)).to_numpy(dtype=float)
    left = l2_rows(_membership(codes, n_labels, sides == "left") @ prof)
    right = l2_rows(_membership(codes, n_labels, sides == "right") @ prof)
    both = (_rowdot(left, left) > 0) & (_rowdot(right, right) > 0)
    group = np.where(both, _rowdot(left, right), np.nan)
    fallback = np.isnan(neuron) | np.isnan(group)
    return sums, _fill(neuron), _fill(group), fallback


@dataclass
class Atlas:
    reference: str
    labels: np.ndarray
    member_codes: np.ndarray
    member_sides: np.ndarray
    member_ids: np.ndarray
    member_counts: PartnerCounts
    sums: sp.csr_matrix
    neuron_yardstick: np.ndarray
    group_yardstick: np.ndarray
    yardstick_fallback: np.ndarray
    params: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    mask: tuple[str, ...] = ()

    @property
    def vocab(self) -> np.ndarray:
        return self.member_counts.vocab

    @property
    def direction(self) -> ProfileDirection:
        return self.params["direction"]

    @property
    def n_members(self) -> np.ndarray:
        return np.bincount(self.member_codes, minlength=len(self.labels))

    @property
    def params_hash(self) -> str:
        blob = json.dumps(self.params, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @cached_property
    def centroids(self) -> sp.csr_matrix:
        """Unit-length centroid rows (the mean and the sum point the same way)."""
        return l2_rows(self.sums)

    def masked(self, mask: Sequence[str] = (), *, drop: Sequence[str] = ()) -> Atlas:
        """Copy with ``mask`` blanked as partner labels and ``drop`` labels removed.

        Masking is exact: member profiles that touch a masked label are recomputed and
        their centroids updated. Yardsticks are kept from the unmasked atlas.
        """
        new = [x for x in mask if x not in self.mask]
        full = tuple(sorted({*self.mask, *new}))
        sums = self.sums
        if new:
            mc = self.member_counts
            rows = mc.touching(new)
            if len(rows):
                sub = mc.rows(rows)
                delta = sub.profiles(direction=self.direction, mask=full) - sub.profiles(
                    direction=self.direction, mask=self.mask
                )
                agg = sp.csr_matrix(
                    (np.ones(len(rows)), (self.member_codes[rows], np.arange(len(rows)))),
                    shape=(len(self.labels), len(rows)),
                )
                sums = sp.csr_matrix(sums + agg @ delta)
            sums = sp.csr_matrix(sums @ sp.diags(mc.column_keep(full, self.direction)))
            sums.eliminate_zeros()
        out = replace(self, sums=sums, mask=full)
        if drop:
            out = out._without(np.isin(out.labels, np.asarray(list(drop), dtype=object)))
        return out

    def _without(self, drop_rows: np.ndarray) -> Atlas:
        keep = ~drop_rows
        new_code = np.cumsum(keep) - 1
        rows = np.flatnonzero(keep[self.member_codes])
        return replace(
            self,
            labels=self.labels[keep],
            member_codes=new_code[self.member_codes[rows]],
            member_sides=self.member_sides[rows],
            member_ids=self.member_ids[rows],
            member_counts=self.member_counts.rows(rows),
            sums=sp.csr_matrix(self.sums[np.flatnonzero(keep)]),
            neuron_yardstick=self.neuron_yardstick[keep],
            group_yardstick=self.group_yardstick[keep],
            yardstick_fallback=self.yardstick_fallback[keep],
        )


def _labels_from_groups(
    m: ConnectivityMatrix,
    groups: Mapping[str, Sequence[int]],
    reference: str,
    on_missing: Literal["raise", "drop"],
) -> tuple[np.ndarray, int]:
    lab = np.full(m.n, None, dtype=object)
    pos = pd.Index(m.neuron_ids)
    dropped = 0
    for name, ids in groups.items():
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        idx = pos.get_indexer(ids_arr)
        missing = ids_arr[idx < 0]
        if len(missing) and on_missing == "raise":
            msg = f"group {name!r}: neuron ids not in {reference}: {missing[:5].tolist()}"
            raise KeyError(msg)
        dropped += len(missing)
        idx = idx[idx >= 0]
        if any(v is not None for v in lab[idx].tolist()):
            msg = f"group {name!r}: some neurons are in more than one group"
            raise ValueError(msg)
        lab[idx] = name
    return lab, dropped


def atlas_from_matrix(
    m: ConnectivityMatrix,
    *,
    reference: str,
    groups: Mapping[str, Sequence[int]] | None = None,
    min_weight: int = 5,
    vocabulary: str = "fafb_or_manc",
    direction: ProfileDirection = "both",
    on_missing: Literal["raise", "drop"] = "raise",
    store_provenance: Mapping[str, Any] | None = None,
) -> Atlas:
    """Atlas from an already-built matrix (``min_weight`` is recorded, not re-applied)."""
    check_direction(direction)
    counts = partner_counts(m, partner_labels(m.meta, vocabulary).tolist())
    if groups is None:
        lab = m.meta["cell_type"].astype(object).to_numpy()
        dropped, source, groups_hash = 0, "cell_type", None
    else:
        lab, dropped = _labels_from_groups(m, groups, reference, on_missing)
        source = "groups"
        blob = json.dumps({k: sorted(int(i) for i in v) for k, v in groups.items()}, sort_keys=True)
        groups_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]
    rows = np.flatnonzero([isinstance(v, str) and v != "" for v in lab.tolist()])
    if len(rows) == 0:
        msg = f"{reference}: no labelled reference neurons to build an atlas from"
        raise ValueError(msg)
    codes, uniq = pd.factorize(pd.Series(lab[rows], dtype=object), sort=True)
    labels = np.asarray(uniq, dtype=object)
    member_counts = counts.rows(rows)
    sides = m.meta["side"].astype(object).to_numpy()[rows]
    sums, neuron_y, group_y, fallback = _yardsticks(
        member_counts.profiles(direction=direction), codes, sides, len(labels)
    )
    params: dict[str, Any] = {
        "reference": reference,
        "min_weight": min_weight,
        "vocabulary": vocabulary,
        "direction": direction,
        "labels_from": source,
        "groups_hash": groups_hash,
        "store_hash": hashlib.sha256(
            json.dumps(dict(store_provenance or {}), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    prov: dict[str, Any] = {
        "reference": reference,
        "n_labels": len(labels),
        "n_members": len(rows),
        "group_ids_dropped": dropped,
        "store_provenance": dict(store_provenance or {}),
    }
    return Atlas(
        reference=reference,
        labels=labels,
        member_codes=np.asarray(codes, dtype=np.int64),
        member_sides=sides,
        member_ids=np.asarray(m.neuron_ids)[rows],
        member_counts=member_counts,
        sums=sums,
        neuron_yardstick=neuron_y,
        group_yardstick=group_y,
        yardstick_fallback=fallback,
        params=params,
        provenance=prov,
    )


def build_atlas(
    store: Store,
    *,
    groups: Mapping[str, Sequence[int]] | None = None,
    min_weight: int = 5,
    vocabulary: str = "fafb_or_manc",
    direction: ProfileDirection = "both",
    on_missing: Literal["raise", "drop"] = "raise",
) -> Atlas:
    """Build a reference atlas from a converted dataset (labels: ``cell_type`` or ``groups``)."""
    m = ConnectivityMatrix.from_store(store, min_weight=min_weight)
    return atlas_from_matrix(
        m,
        reference=store.ref,
        groups=groups,
        min_weight=min_weight,
        vocabulary=vocabulary,
        direction=direction,
        on_missing=on_missing,
        store_provenance={"counts": store.provenance.get("counts", {})},
    )
