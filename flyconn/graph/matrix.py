"""Sparse signed connectivity matrices over the harmonized store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.data.store import Store
from flyconn.graph.signs import SignPolicy, neuron_signs


@dataclass
class ConnectivityMatrix:
    """Presynaptic-in-rows CSR weights plus neuron metadata and per-neuron signs.

    ``weights[i, j]`` is the synapse count from ``neuron_ids[i]`` to
    ``neuron_ids[j]``. ``signs[i]`` is the presynaptic sign of neuron ``i``
    under the chosen :class:`SignPolicy` (0 means unknown, edges excluded).
    """

    neuron_ids: np.ndarray
    weights: sp.csr_matrix
    signs: np.ndarray
    meta: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._pos = pd.Index(self.neuron_ids)

    @property
    def n(self) -> int:
        return len(self.neuron_ids)

    def index_of(self, ids: np.ndarray) -> np.ndarray:
        """Row/column positions of ``ids``; raises ``KeyError`` for unknown ids."""
        idx = self._pos.get_indexer(np.asarray(ids))
        if (idx < 0).any():
            missing = np.asarray(ids)[idx < 0][:5].tolist()
            msg = f"neuron ids not in matrix: {missing}"
            raise KeyError(msg)
        return idx

    @cached_property
    def signed_weights(self) -> sp.csr_matrix:
        """Weights with each row multiplied by its presynaptic sign."""
        return sp.csr_matrix(sp.diags(self.signs) @ self.weights)

    def input_normalized(self, total_column: str = "input_synapses_total") -> sp.csr_matrix:
        """Weights divided by each postsynaptic neuron's total input.

        Uses ``meta[total_column]`` when present (MaleCNS totals include synapses
        from unannotated fragments), else the column sums of ``weights``.
        """
        col = (
            cast("pd.Series", self.meta[total_column])
            if total_column in self.meta.columns
            else None
        )
        if col is not None and bool(col.notna().all()):
            total = col.to_numpy(dtype=float)
        else:
            total = np.asarray(self.weights.sum(axis=0)).ravel().astype(float)
        inv = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
        return sp.csr_matrix(self.weights @ sp.diags(inv))

    def subset(self, ids: np.ndarray) -> ConnectivityMatrix:
        """Restrict to ``ids`` (order preserved)."""
        ids = np.asarray(ids)
        idx = self.index_of(ids)
        sub = sp.csr_matrix(self.weights[idx][:, idx])
        prov = dict(self.provenance, subset=len(ids))
        return ConnectivityMatrix(ids, sub, self.signs[idx], self.meta.iloc[idx].copy(), prov)

    @classmethod
    def from_frames(
        cls,
        neurons: pd.DataFrame,
        edges: pd.DataFrame,
        *,
        min_weight: int = 1,
        sign_policy: SignPolicy = SignPolicy.ARGMAX,
        sign_overrides: Mapping[str, int] | None = None,
        seed: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ConnectivityMatrix:
        """Build from harmonized ``neurons`` and ``edges`` tables."""
        neurons = neurons.reset_index(drop=True)
        ids = neurons["neuron_id"].to_numpy(dtype=np.int64)
        pos = pd.Index(ids)
        e = cast("pd.DataFrame", edges[edges["weight"] >= min_weight])
        pre = cast("pd.Series", e["pre"]).to_numpy()
        post = cast("pd.Series", e["post"]).to_numpy()
        wt = cast("pd.Series", e["weight"]).to_numpy(dtype=np.int64)
        i = pos.get_indexer(pre)
        j = pos.get_indexer(post)
        ok = (i >= 0) & (j >= 0)
        w = sp.csr_matrix((wt[ok], (i[ok], j[ok])), shape=(len(ids), len(ids)))
        w.sum_duplicates()
        rng = np.random.default_rng(seed) if seed is not None else None
        signs = neuron_signs(neurons, sign_policy, sign_overrides, rng)
        prov: dict[str, Any] = {
            **(provenance or {}),
            "min_weight": min_weight,
            "sign_policy": sign_policy.value,
            "sign_overrides": dict(sign_overrides or {}),
            "seed": seed,
            "n_neurons": len(ids),
            "n_edges": int(w.nnz),
            "excluded_unknown_sign_neurons": int((signs == 0).sum()),
        }
        meta = neurons.set_index("neuron_id", drop=False)
        meta.index.name = None
        return cls(ids, w, signs, meta, prov)

    @classmethod
    def from_store(
        cls,
        store: Store,
        *,
        min_weight: int = 1,
        sign_policy: SignPolicy = SignPolicy.ARGMAX,
        sign_overrides: Mapping[str, int] | None = None,
        neuron_ids: np.ndarray | None = None,
        seed: int | None = None,
    ) -> ConnectivityMatrix:
        """Build from a converted dataset; optionally restricted to ``neuron_ids``."""
        neurons = store.neurons()
        if neuron_ids is not None:
            keep = cast("pd.Series", neurons["neuron_id"]).isin(list(np.asarray(neuron_ids)))
            neurons = cast("pd.DataFrame", neurons[keep])
        edges = store.edges(min_weight=min_weight)
        prov = {"dataset": store.ref, "store_provenance": store.provenance.get("counts", {})}
        return cls.from_frames(
            neurons,
            edges,
            min_weight=min_weight,
            sign_policy=sign_policy,
            sign_overrides=sign_overrides,
            seed=seed,
            provenance=prov,
        )
