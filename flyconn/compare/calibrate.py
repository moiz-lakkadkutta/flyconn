"""Calibrated decisions for zero-shot classification (ADR-0010).

A logistic model maps the top candidate's scores to P(top-1 correct). It is fitted only by
the hold-out benchmark (``classify_bench``). Calls, checked in order: ``unknown`` if the best
cosine is below ``s_floor`` (set so at most ``max_false_accept`` of open-set queries pass);
``type`` if P >= ``accept``; otherwise ``ambiguous``, listing every candidate within
``delta`` of the best.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from flyconn.compare.scoring import Scores

if TYPE_CHECKING:
    from flyconn.compare.atlas import Atlas

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "adjusted": ("s1", "a1", "margin"),
    "raw": ("s1", "margin"),
}
MARGIN_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)
CALIBRATION_DIR = Path(__file__).parent / "calibration"


def feature_matrix(scores: Scores, names: Sequence[str]) -> np.ndarray:
    cols = {"s1": scores.s1, "a1": scores.a1, "margin": scores.margin}
    return np.column_stack([cols[n] for n in names]) if len(scores.s) else np.zeros((0, len(names)))


@dataclass
class LogisticModel:
    features: tuple[str, ...]
    coef: np.ndarray  # intercept first

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, len(self.features))
        return np.asarray(expit(self.coef[0] + x @ self.coef[1:]), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"features": list(self.features), "coef": [float(c) for c in self.coef]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogisticModel:
        return cls(features=tuple(d["features"]), coef=np.asarray(d["coef"], dtype=float))


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, features: Sequence[str], l2: float = 1e-4
) -> LogisticModel:
    """L2-penalised logistic regression by L-BFGS (intercept not penalised)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        msg = "calibration needs both correct and incorrect examples"
        raise ValueError(msg)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.r_[0.0, np.ones(design.shape[1] - 1)]

    def nll(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ w
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z) + l2 * np.sum(penalty * w**2))
        grad = design.T @ (expit(z) - y) + 2 * l2 * penalty * w
        return loss, grad

    res = minimize(nll, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    return LogisticModel(features=tuple(features), coef=np.asarray(res.x, dtype=float))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += abs(float(y[sel].mean()) - float(p[sel].mean())) * float(sel.sum())
    return total / len(p)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


@dataclass
class Thresholds:
    accept: float
    s_floor: float
    delta: float


def choose_s_floor(open_s1: np.ndarray, max_false_accept: float = 0.10) -> float:
    """Cosine that at most ``max_false_accept`` of open-set queries (true type absent) reach."""
    open_s1 = np.asarray(open_s1, dtype=float)
    return float(np.quantile(open_s1, 1 - max_false_accept)) if len(open_s1) else 0.0


def choose_delta(
    margin: np.ndarray,
    correct: np.ndarray,
    edges: Sequence[float] = MARGIN_EDGES,
    min_count: int = 10,
) -> float:
    """Upper edge of the largest margin bin where top-1 accuracy is below 50 %."""
    margin = np.asarray(margin, dtype=float)
    correct = np.asarray(correct, dtype=float)
    delta = float(edges[1])
    for lo, hi in pairwise(edges):
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() >= min_count and correct[sel].mean() < 0.5:
            delta = float(hi)
    return delta


def apply_calls(p: np.ndarray, s1: np.ndarray, thr: Thresholds) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    return np.where(
        s1 < thr.s_floor, "unknown", np.where(p >= thr.accept, "type", "ambiguous")
    ).astype(object)


def decide(scores: Scores, model: LogisticModel, thr: Thresholds) -> pd.DataFrame:
    p = model.predict(feature_matrix(scores, model.features))
    calls = apply_calls(p, scores.s1, thr)
    labels: list[str] = []
    for i, call in enumerate(calls.tolist()):
        if call == "type":
            labels.append(str(scores.labels[i, 0]))
        elif call == "ambiguous":
            near = scores.s[i] >= scores.s[i, 0] - thr.delta
            labels.append("|".join(str(x) for x in scores.labels[i][near]))
        else:
            labels.append("")
    return pd.DataFrame({"call": calls, "label": labels, "p": p})


def uncalibrated_calls(scores: Scores) -> pd.DataFrame:
    n = len(scores.s)
    return pd.DataFrame(
        {
            "call": np.full(n, "uncalibrated", dtype=object),
            "label": [str(x) for x in scores.top1.tolist()],
            "p": np.full(n, np.nan),
        }
    )


@dataclass
class Calibration:
    reference: str
    atlas_params_hash: str
    atlas_params: dict[str, Any]
    variant: str
    models: dict[str, LogisticModel]
    thresholds: dict[str, Thresholds]
    fitted_on: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "atlas_params_hash": self.atlas_params_hash,
            "atlas_params": self.atlas_params,
            "variant": self.variant,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "thresholds": {k: asdict(v) for k, v in self.thresholds.items()},
            "fitted_on": list(self.fitted_on),
            "metrics": self.metrics,
            "provenance": self.provenance,
        }

    @property
    def id(self) -> str:
        core = {k: v for k, v in self.to_dict().items() if k != "provenance"}
        blob = json.dumps(core, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, default=str) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> Calibration:
        d = json.loads(Path(path).read_text())
        return cls(
            reference=d["reference"],
            atlas_params_hash=d["atlas_params_hash"],
            atlas_params=d["atlas_params"],
            variant=d["variant"],
            models={k: LogisticModel.from_dict(v) for k, v in d["models"].items()},
            thresholds={k: Thresholds(**v) for k, v in d["thresholds"].items()},
            fitted_on=list(d["fitted_on"]),
            metrics=d.get("metrics", {}),
            provenance=d.get("provenance", {}),
        )

    def incompatibility(self, atlas: Atlas) -> str | None:
        """None if this calibration applies to ``atlas``; else which parameters differ."""
        if atlas.params_hash == self.atlas_params_hash:
            return None
        diff = sorted(
            k
            for k in {*atlas.params, *self.atlas_params}
            if atlas.params.get(k) != self.atlas_params.get(k)
        )
        return "atlas parameters differ: " + ", ".join(
            f"{k}={atlas.params.get(k)!r} (calibrated {self.atlas_params.get(k)!r})" for k in diff
        )


def calibration_path(reference: str) -> Path:
    name, version = reference.split("@", 1)
    return CALIBRATION_DIR / f"classify_{name}_{version}.json"


def default_calibration(atlas: Atlas) -> Calibration | None:
    path = calibration_path(atlas.reference)
    return Calibration.from_json(path) if path.exists() else None
