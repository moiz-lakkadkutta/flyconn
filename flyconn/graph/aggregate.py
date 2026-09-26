"""Aggregate neuron-level connectivity by a label column (cell type, super class, ...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix


@dataclass
class AggregatedMatrix:
    """Label-by-label summed weights; ``counts[k]`` is the number of neurons with label k."""

    by: str
    labels: np.ndarray
    weights: sp.csr_matrix
    counts: np.ndarray
    signed: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._pos = pd.Index(self.labels)

    def weight(self, pre_label: str, post_label: str) -> float:
        i, j = self._pos.get_loc(pre_label), self._pos.get_loc(post_label)
        return float(self.weights[i, j])

    def to_frame(self) -> pd.DataFrame:
        """Long form: one row per non-zero label pair."""
        coo = sp.coo_matrix(self.weights)
        return pd.DataFrame(
            {
                "pre_label": self.labels[coo.row],
                "post_label": self.labels[coo.col],
                "weight": coo.data,
                "n_pre": self.counts[coo.row],
                "n_post": self.counts[coo.col],
            }
        ).sort_values(["pre_label", "post_label"], ignore_index=True)


def _membership(
    m: ConnectivityMatrix, by: str
) -> tuple[np.ndarray, sp.csr_matrix, np.ndarray, int]:
    if by not in m.meta.columns:
        msg = f"no column {by!r} in neuron metadata"
        raise KeyError(msg)
    values = m.meta[by].to_numpy(dtype=object)
    labelled = np.array(
        [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in values]
    )
    codes, labels = pd.factorize(pd.Series(values[labelled]).astype(str), sort=True)
    rows = np.flatnonzero(labelled)
    member = sp.csr_matrix(
        (np.ones(len(rows)), (rows, codes)), shape=(m.n, len(labels)), dtype=float
    )
    counts = np.asarray(member.sum(axis=0)).ravel().astype(int)
    return np.asarray(labels, dtype=object), member, counts, int((~labelled).sum())


def aggregate(m: ConnectivityMatrix, by: str, *, signed: bool = False) -> AggregatedMatrix:
    """Sum (signed) weights between all neurons of each label pair."""
    labels, member, counts, unlabelled = _membership(m, by)
    w = m.signed_weights if signed else m.weights
    agg = sp.csr_matrix(member.T @ w @ member)
    agg.eliminate_zeros()
    prov = dict(m.provenance, aggregated_by=by, signed=signed, unlabelled_neurons=unlabelled)
    return AggregatedMatrix(by, labels, agg, counts, signed, prov)


def aggregate_edges(m: ConnectivityMatrix, by: str, *, signed: bool = False) -> pd.DataFrame:
    """Long-form label-pair table with weights, member counts, connected pair counts."""
    agg = aggregate(m, by, signed=signed)
    df = agg.to_frame()
    labels, member, _, _ = _membership(m, by)
    binary = sp.csr_matrix(m.weights, dtype=float, copy=True)
    binary.data = np.ones_like(binary.data)
    pairs = sp.coo_matrix(member.T @ binary @ member)
    pair_counts = pd.Series(
        pairs.data, index=pd.MultiIndex.from_arrays([labels[pairs.row], labels[pairs.col]])
    )
    key = pd.MultiIndex.from_arrays([df["pre_label"], df["post_label"]])
    df["n_pairs"] = pair_counts.reindex(key).to_numpy().astype(int)
    df["weight_per_post"] = df["weight"] / df["n_post"]
    return df
