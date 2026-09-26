"""Enumerate 1..k-hop paths between neuron sets with per-edge weights and signs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix


def find_paths(
    m: ConnectivityMatrix,
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    max_hops: int = 3,
    min_weight: int = 1,
    label: str | None = None,
    max_paths: int = 1_000_000,
) -> pd.DataFrame:
    """Simple (node-repeat-free) paths from ``sources`` to ``targets`` of length ≤ ``max_hops``.

    Targets are terminal: a path ends when it reaches one. Columns: ``path`` (tuple of
    neuron ids), ``hops``, ``weights`` (per edge), ``signs`` (presynaptic sign per edge),
    ``path_sign`` (product; 0 if any sign unknown), ``strength`` (product of input
    fractions ``w / total_input(post)``), ``min_weight``, ``source``, ``target`` and,
    when ``label`` is given, ``labels`` (tuple of ``meta[label]``).
    """
    if max_hops < 1:
        msg = "max_hops must be >= 1"
        raise ValueError(msg)
    src_idx = m.index_of(np.asarray(list(sources)))
    tgt_idx = set(m.index_of(np.asarray(list(targets))).tolist())

    w = sp.csr_matrix(m.weights)
    if min_weight > 1:
        w = w.multiply(w >= min_weight).tocsr()
        w.eliminate_zeros()
    norm = m.input_normalized().tocsr()
    indptr, indices, data = w.indptr, w.indices, w.data
    signs = m.signs
    ids = m.neuron_ids
    labels = m.meta[label].to_numpy(dtype=object) if label else None

    rows: list[dict[str, object]] = []

    def dfs(path: list[int], wts: list[int], strength: float) -> None:
        if len(rows) >= max_paths:
            return
        node = path[-1]
        depth = len(path) - 1
        if depth > 0 and node in tgt_idx:
            edge_signs = tuple(float(signs[p]) for p in path[:-1])
            sign = 0.0 if any(s == 0 for s in edge_signs) else float(np.prod(edge_signs))
            row: dict[str, object] = {
                "path": tuple(int(ids[p]) for p in path),
                "hops": depth,
                "weights": tuple(wts),
                "signs": tuple(int(s) if float(s).is_integer() else s for s in edge_signs),
                "path_sign": int(sign) if sign.is_integer() else sign,
                "strength": strength,
                "min_weight": min(wts),
                "source": int(ids[path[0]]),
                "target": int(ids[node]),
            }
            if labels is not None:
                row["labels"] = tuple(labels[p] for p in path)
            rows.append(row)
            return  # targets are terminal
        if depth == max_hops:
            return
        for k in range(indptr[node], indptr[node + 1]):
            nxt = int(indices[k])
            if nxt in path:
                continue
            dfs([*path, nxt], [*wts, int(data[k])], strength * float(norm[node, nxt]))

    for s in src_idx:
        dfs([int(s)], [], 1.0)

    cols = [
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
    if label:
        cols.append("labels")
    if not rows:
        empty: dict[str, list[object]] = {c: [] for c in cols}
        return pd.DataFrame(empty).astype({"hops": int, "min_weight": int, "strength": float})
    df = cast("pd.DataFrame", pd.DataFrame(rows)[cols])
    return df.sort_values(["strength", "hops"], ascending=[False, True], ignore_index=True)
