"""Native k-hop effective connectivity via powers of the input-normalised (signed) matrix.

This is the small, torch-free implementation used for 1-5 hops and as a parity
check for the optional ``connectome_interpreter`` adapter (ADR-0002). Entry
``E_k[i, j]`` is the sum over all k-step walks from i to j of the product of
per-step input fractions (times presynaptic signs when ``signed``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix


def _base(m: ConnectivityMatrix, signed: bool) -> sp.csr_matrix:
    norm = m.input_normalized()
    if signed:
        return sp.csr_matrix(sp.diags(m.signs) @ norm)
    return norm


def _prune(x: sp.csr_matrix, threshold: float) -> sp.csr_matrix:
    if threshold <= 0:
        return x
    x = x.multiply(abs(x) >= threshold).tocsr()
    x.eliminate_zeros()
    return sp.csr_matrix(x)


def effective_connectivity(
    m: ConnectivityMatrix,
    hops: int,
    *,
    signed: bool = True,
    cumulative: bool = False,
    threshold: float = 0.0,
) -> sp.csr_matrix:
    """Return ``S^hops`` (cumulative: ``S + ... + S^hops``), pruning |x| < threshold per step."""
    if hops < 1:
        msg = "hops must be >= 1"
        raise ValueError(msg)
    s = _prune(_base(m, signed), threshold)
    power = s
    total = s.copy() if cumulative else None
    for _ in range(hops - 1):
        power = _prune(sp.csr_matrix(power @ s), threshold)
        if total is not None:
            total = sp.csr_matrix(total + power)
    return total if total is not None else power


def effective_by_hops(
    m: ConnectivityMatrix,
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    max_hops: int = 3,
    signed: bool = True,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Total effective connectivity from ``sources`` to ``targets`` at each hop count 1..max_hops.

    ``value`` sums the effective matrix over all source x target pairs; ``mean_per_target``
    divides by the number of targets.
    """
    si = m.index_of(np.asarray(list(sources)))
    ti = m.index_of(np.asarray(list(targets)))
    s = _prune(_base(m, signed), threshold)
    power = s
    rows: list[dict[str, float | int]] = []
    for k in range(1, max_hops + 1):
        if k > 1:
            power = _prune(sp.csr_matrix(power @ s), threshold)
        block = power[si][:, ti]
        value = float(block.sum())
        rows.append(
            {
                "hops": k,
                "value": value,
                "mean_per_target": value / len(ti),
                "n_sources": len(si),
                "n_targets": len(ti),
            }
        )
    return pd.DataFrame(rows)
