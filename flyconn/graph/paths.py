"""Enumerate 1..k-hop paths between neuron sets with per-edge weights and signs.

The search first prunes the graph to nodes that lie on some source→target walk of
length ≤ ``max_hops`` (forward BFS distance from sources plus backward BFS distance
to targets), then runs a depth-first enumeration of simple paths on that subgraph
with the remaining-distance bound. This keeps whole-CNS 3-hop searches tractable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix

_COLUMNS = [
    "path",
    "hops",
    "weights",
    "signs",
    "path_sign",
    "strength",
    "min_weight",
    "source",
    "target",
]


def _totals(m: ConnectivityMatrix) -> np.ndarray:
    if "input_synapses_total" in m.meta.columns:
        col = cast("pd.Series", m.meta["input_synapses_total"])
        if bool(col.notna().all()):
            return col.to_numpy(dtype=float)
    return np.asarray(m.weights.sum(axis=0)).ravel().astype(float)


def _filtered(
    m: ConnectivityMatrix, min_weight: int, min_edge_fraction: float
) -> tuple[sp.csr_matrix, np.ndarray]:
    """CSR of surviving edges (data = weight) and the per-column input totals."""
    total = _totals(m)
    w = sp.csr_matrix(m.weights, copy=True)
    keep = w.data >= min_weight
    if min_edge_fraction > 0:
        cols = w.indices  # column index per stored entry
        frac = w.data / np.where(total[cols] > 0, total[cols], np.inf)
        keep &= frac >= min_edge_fraction
    w.data = np.where(keep, w.data, 0)
    w.eliminate_zeros()
    return w, total


def _bfs_distances(adj: sp.csr_matrix, start: np.ndarray, max_depth: int) -> np.ndarray:
    """Min hop distance from ``start`` to every node (``max_depth + 1`` = unreachable)."""
    n = adj.shape[0]
    dist = np.full(n, max_depth + 1, dtype=np.int32)
    dist[start] = 0
    frontier = start
    for d in range(1, max_depth + 1):
        if len(frontier) == 0:
            break
        rows = cast("sp.csr_matrix", adj[frontier])
        nxt = np.unique(rows.indices)
        nxt = nxt[dist[nxt] > d]
        dist[nxt] = d
        frontier = nxt
    return dist


def _enumerate(
    m: ConnectivityMatrix,
    w: sp.csr_matrix,
    total: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: set[int],
    to_target: np.ndarray | None,
    max_hops: int,
    min_strength: float,
    label: str | None,
    max_paths: int,
) -> list[dict[str, object]]:
    indptr, indices, data = w.indptr, w.indices, w.data
    signs, ids = m.signs, m.neuron_ids
    labels = m.meta[label].to_numpy(dtype=object) if label else None
    rows: list[dict[str, object]] = []

    def emit(path: list[int], wts: list[int], strength: float) -> None:
        edge_signs = [float(signs[p]) for p in path[:-1]]
        sign = 0.0 if any(s == 0 for s in edge_signs) else float(np.prod(edge_signs))
        row: dict[str, object] = {
            "path": tuple(int(ids[p]) for p in path),
            "hops": len(path) - 1,
            "weights": tuple(wts),
            "signs": tuple(int(s) if s.is_integer() else s for s in edge_signs),
            "path_sign": int(sign) if sign.is_integer() else sign,
            "strength": strength,
            "min_weight": min(wts),
            "source": int(ids[path[0]]),
            "target": int(ids[path[-1]]),
        }
        if labels is not None:
            row["labels"] = tuple(labels[p] for p in path)
        rows.append(row)

    def dfs(path: list[int], wts: list[int], strength: float) -> None:
        if len(rows) >= max_paths:
            return
        node = path[-1]
        depth = len(path) - 1
        if depth > 0 and node in tgt_idx:
            emit(path, wts, strength)
            return  # targets are terminal
        if depth == max_hops:
            return
        remaining = max_hops - depth
        for k in range(indptr[node], indptr[node + 1]):
            nxt = int(indices[k])
            if to_target is not None and to_target[nxt] > remaining - 1:
                continue
            if nxt in path:
                continue
            frac = float(data[k]) / total[nxt] if total[nxt] > 0 else 0.0
            nxt_strength = strength * frac
            if nxt_strength < min_strength:
                continue
            dfs([*path, nxt], [*wts, int(data[k])], nxt_strength)

    for s in src_idx:
        dfs([int(s)], [], 1.0)
    return rows


def _frame(rows: list[dict[str, object]], label: str | None) -> pd.DataFrame:
    cols = [*_COLUMNS, "labels"] if label else list(_COLUMNS)
    if not rows:
        empty: dict[str, list[object]] = {c: [] for c in cols}
        return pd.DataFrame(empty).astype({"hops": int, "min_weight": int, "strength": float})
    df = cast("pd.DataFrame", pd.DataFrame(rows)[cols])
    return df.sort_values(["strength", "hops"], ascending=[False, True], ignore_index=True)


def find_paths(
    m: ConnectivityMatrix,
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    max_hops: int = 3,
    min_weight: int = 1,
    min_edge_fraction: float = 0.0,
    min_strength: float = 0.0,
    label: str | None = None,
    max_paths: int = 1_000_000,
) -> pd.DataFrame:
    """Simple (node-repeat-free) paths from ``sources`` to ``targets`` of length ≤ ``max_hops``.

    Targets are terminal: a path ends when it reaches one. Edge filters: ``min_weight``
    (synapses) and ``min_edge_fraction`` (synapses / postsynaptic neuron's total input,
    applied per hop). ``min_strength`` prunes partial paths whose product of input
    fractions falls below it. Columns: ``path`` (tuple of neuron ids), ``hops``,
    ``weights``, ``signs`` (presynaptic sign per edge), ``path_sign`` (product; 0 if any
    sign unknown), ``strength`` (product of input fractions), ``min_weight``, ``source``,
    ``target`` and, with ``label``, ``labels``. Parameters are recorded in ``df.attrs``.
    """
    if max_hops < 1:
        msg = "max_hops must be >= 1"
        raise ValueError(msg)
    src_idx = m.index_of(np.asarray(list(sources)))
    tgt_arr = m.index_of(np.asarray(list(targets)))
    w, total = _filtered(m, min_weight, min_edge_fraction)

    # Reachability pruning: keep nodes with dist_from_source + dist_to_target <= max_hops.
    from_source = _bfs_distances(w, src_idx, max_hops)
    to_target = _bfs_distances(sp.csr_matrix(w.T), tgt_arr, max_hops)
    on_path = (from_source + to_target) <= max_hops
    on_path[src_idx] = True
    on_path[tgt_arr] |= from_source[tgt_arr] <= max_hops
    keep = sp.diags(on_path.astype(float))
    w_sub = sp.csr_matrix(keep @ w @ keep)
    w_sub.eliminate_zeros()

    rows = _enumerate(
        m,
        w_sub,
        total,
        src_idx,
        set(tgt_arr.tolist()),
        to_target,
        max_hops,
        min_strength,
        label,
        max_paths,
    )
    df = _frame(rows, label)
    df.attrs.update(
        {
            "max_hops": max_hops,
            "min_weight": min_weight,
            "min_edge_fraction": min_edge_fraction,
            "min_strength": min_strength,
            "n_sources": len(src_idx),
            "n_targets": len(tgt_arr),
            "nodes_searched": int(on_path.sum()),
            "truncated": len(rows) >= max_paths,
        }
    )
    return df


def brute_force_paths(
    m: ConnectivityMatrix,
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    max_hops: int = 3,
    min_weight: int = 1,
) -> list[tuple[int, ...]]:
    """Reference implementation without reachability pruning (slow; for validation)."""
    src_idx = m.index_of(np.asarray(list(sources)))
    tgt_idx = set(m.index_of(np.asarray(list(targets))).tolist())
    w, total = _filtered(m, min_weight, 0.0)
    rows = _enumerate(m, w, total, src_idx, tgt_idx, None, max_hops, 0.0, None, 10_000_000)
    return [cast("tuple[int, ...]", r["path"]) for r in rows]
