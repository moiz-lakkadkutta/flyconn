"""Null models: degree-preserving rewiring and sign shuffles (run by default in experiments)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix


def degree_preserving_rewire(
    m: ConnectivityMatrix, seed: int, max_rounds: int = 1000
) -> ConnectivityMatrix:
    """Rewire edges keeping each neuron's out-degree, out-weight distribution and in-degree.

    Every edge keeps its source and weight; destinations are permuted across all
    edges, then duplicate (pre, post) pairs and self-loops are resolved by swapping
    destinations with randomly chosen other edges until none remain. This is the
    configuration-model style shuffle used by Shiu et al. 2024 and the
    drosophila-brain-mlx control (design credit in ATTRIBUTION.md).
    """
    rng = np.random.default_rng(seed)
    coo = sp.coo_matrix(m.weights)
    rows, cols, data = coo.row.copy(), coo.col.copy(), coo.data.copy()
    n_edges = len(rows)
    cols = cols[rng.permutation(n_edges)]
    for _ in range(max_rounds):
        key = rows.astype(np.int64) * m.n + cols
        order = np.argsort(key, kind="stable")
        dup = np.zeros(n_edges, dtype=bool)
        dup[order[1:]] = key[order[1:]] == key[order[:-1]]
        bad = np.flatnonzero(dup | (rows == cols))
        if len(bad) == 0:
            break
        # Sequential pairwise swaps keep the destination multiset (hence in-degrees) intact.
        partners = rng.integers(0, n_edges, size=len(bad))
        for i, j in zip(bad.tolist(), partners.tolist(), strict=True):
            cols[i], cols[j] = cols[j], cols[i]
    else:  # pragma: no cover - pathological graphs
        msg = "could not resolve duplicate edges after max_rounds"
        raise RuntimeError(msg)
    w = sp.csr_matrix((data, (rows, cols)), shape=m.weights.shape)
    w.sum_duplicates()
    prov = dict(m.provenance, null_model="degree_preserving_rewire", seed=seed)
    return ConnectivityMatrix(m.neuron_ids, w, m.signs.copy(), m.meta, prov)


def shuffle_signs(m: ConnectivityMatrix, seed: int) -> ConnectivityMatrix:
    """Permute presynaptic signs across neurons, keeping the wiring and the sign histogram."""
    rng = np.random.default_rng(seed)
    signs = m.signs[rng.permutation(m.n)]
    prov = dict(m.provenance, null_model="sign_shuffle", seed=seed)
    return ConnectivityMatrix(m.neuron_ids, m.weights, signs, m.meta, prov)
