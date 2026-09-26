"""Stability of path analyses under NT-sign sampling and synapse-threshold sweeps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.graph.paths import find_paths
from flyconn.uncertainty.nt import Model, sample_signs


@dataclass
class StabilityResult:
    """Per-path stability scores plus the parameters that produced them."""

    paths: pd.DataFrame
    n_samples: int
    thresholds: tuple[int, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One row per hop count: path counts and mean stability scores."""
        if self.paths.empty:
            cols = ["hops", "n_paths", "sign_stability", "threshold_presence", "modal_sign_pos"]
            return pd.DataFrame({c: [] for c in cols})
        g = self.paths.groupby("hops")
        return pd.DataFrame(
            {
                "n_paths": g.size(),
                "sign_stability": g["sign_stability"].mean(),
                "threshold_presence": g["threshold_presence"].mean(),
                "modal_sign_pos": g["modal_sign"].apply(_frac_positive),
            }
        ).reset_index()


def _frac_positive(s: pd.Series) -> float:
    return float((s > 0).mean())


_EMPTY_COLUMNS = [
    "path",
    "hops",
    "presence",
    "threshold_presence",
    "modal_sign",
    "sign_stability",
    "strength_mean",
    "strength_std",
]


def path_stability(
    m: ConnectivityMatrix,
    sources: Sequence[int],
    targets: Sequence[int],
    *,
    max_hops: int = 3,
    n_samples: int = 100,
    seed: int = 0,
    thresholds: Sequence[int] | None = None,
    min_edge_fraction: float = 0.0,
    nt_model: Model = "auto",
    label: str | None = None,
) -> StabilityResult:
    """Enumerate paths across synapse thresholds and score their sign stability under NT sampling.

    ``threshold_presence``: fraction of thresholds at which the path exists.
    ``presence``: same quantity (topology does not depend on NT draws).
    ``modal_sign`` / ``sign_stability``: most frequent path sign over ``n_samples``
    sign draws and the fraction of draws that produced it.
    """
    base_threshold = int(m.provenance.get("min_weight", 1))
    ths = tuple(int(t) for t in (thresholds or (base_threshold,)))
    frames: list[pd.DataFrame] = []
    for t in ths:
        df = find_paths(
            m,
            sources,
            targets,
            max_hops=max_hops,
            min_weight=t,
            min_edge_fraction=min_edge_fraction,
            label=label,
        )
        df = df.assign(threshold=t)
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if allp.empty:
        empty = pd.DataFrame({c: [] for c in _EMPTY_COLUMNS})
        return StabilityResult(empty, n_samples, ths, {"seed": seed, "nt_model": nt_model})

    grouped = allp.groupby("path", sort=False)
    paths = grouped.agg(
        hops=("hops", "first"),
        weights=("weights", "first"),
        strength_mean=("strength", "mean"),
        strength_std=("strength", "std"),
        n_thresholds=("threshold", "nunique"),
        source=("source", "first"),
        target=("target", "first"),
    ).reset_index()
    if label:
        paths["labels"] = grouped["labels"].first().to_numpy()
    paths["threshold_presence"] = paths["n_thresholds"] / len(ths)
    paths["presence"] = paths["threshold_presence"]
    paths["strength_std"] = paths["strength_std"].fillna(0.0)

    # Sign stability: sample sign vectors once, evaluate every path's sign per sample.
    samples = np.stack(list(sample_signs(m.meta, n_samples, seed, model=nt_model)))  # (S, n)
    modal = np.zeros(len(paths))
    stab = np.zeros(len(paths))
    for i, path in enumerate(paths["path"]):
        pre_idx = m.index_of(np.asarray(path[:-1]))
        s = samples[:, pre_idx]
        path_sign = np.where((s == 0).any(axis=1), 0.0, np.prod(s, axis=1))
        vals, counts = np.unique(path_sign, return_counts=True)
        k = int(np.argmax(counts))
        modal[i] = vals[k]
        stab[i] = counts[k] / n_samples
    paths["modal_sign"] = modal.astype(int)
    paths["sign_stability"] = stab
    paths = paths.sort_values(
        ["threshold_presence", "sign_stability", "strength_mean"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    prov = {
        "seed": seed,
        "nt_model": nt_model,
        "n_samples": n_samples,
        "thresholds": ths,
        "max_hops": max_hops,
        "min_edge_fraction": min_edge_fraction,
        "matrix": m.provenance,
    }
    return StabilityResult(paths, n_samples, ths, prov)
