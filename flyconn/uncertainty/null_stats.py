"""Compare an observed statistic against degree-preserving or sign-shuffled null models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.uncertainty.nulls import degree_preserving_rewire, shuffle_signs

NullModel = Literal["degree_preserving_rewire", "sign_shuffle"]


@dataclass
class NullResult:
    observed: float
    nulls: np.ndarray
    null_model: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def z(self) -> float:
        sd = float(self.nulls.std(ddof=1)) if len(self.nulls) > 1 else 0.0
        return float((self.observed - self.nulls.mean()) / sd) if sd > 0 else float("inf")

    @property
    def p_two_sided(self) -> float:
        """Empirical two-sided p-value with the +1 correction (never exactly 0)."""
        dev = np.abs(self.nulls - self.nulls.mean())
        obs = abs(self.observed - self.nulls.mean())
        return float((1 + (dev >= obs).sum()) / (len(self.nulls) + 1))

    def summary(self) -> dict[str, float | int | str]:
        return {
            "observed": self.observed,
            "null_mean": float(self.nulls.mean()),
            "null_sd": float(self.nulls.std(ddof=1)) if len(self.nulls) > 1 else 0.0,
            "z": self.z,
            "p_two_sided": self.p_two_sided,
            "n_nulls": len(self.nulls),
            "null_model": self.null_model,
        }


def null_distribution(
    statistic: Callable[[ConnectivityMatrix], float],
    m: ConnectivityMatrix,
    *,
    n_nulls: int = 100,
    seed: int = 0,
    null: NullModel = "degree_preserving_rewire",
) -> NullResult:
    """Evaluate ``statistic`` on ``m`` and on ``n_nulls`` seeded null matrices."""
    make = degree_preserving_rewire if null == "degree_preserving_rewire" else shuffle_signs
    observed = float(statistic(m))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_nulls)
    nulls = np.array([float(statistic(make(m, int(s)))) for s in seeds])
    return NullResult(
        observed, nulls, null, {"seed": seed, "n_nulls": n_nulls, "matrix": m.provenance}
    )
