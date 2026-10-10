"""Scoring for robustness-audited screens: labels, axis verdicts, tiers (ADR-0011)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

Label = Literal["holds", "weakens", "flips"]
Outcome = Literal["robust", "fails", "not_run"]
GRADED_AXES: tuple[str, ...] = ("w_syn", "nt", "min_weight", "random_null")
TIER_ORDER: dict[str, int] = {"Strong": 0, "Conditional": 1, "Fragile": 2}
_SWEEP_AXES: tuple[str, ...] = ("w_syn", "nt", "min_weight")
NT_ROBUST_FRACTION = 0.75
NULL_ALPHA = 0.05


def variant_label(
    reference_difference: float, difference: float, ci_low: float, ci_high: float
) -> Label:
    """holds / weakens / flips of a variant effect relative to the reference direction."""
    ref_sign = float(np.sign(reference_difference))
    if ref_sign == 0:
        msg = "reference difference is zero; no direction to compare"
        raise ValueError(msg)
    excludes_zero = ci_low > 0 or ci_high < 0
    sign = float(np.sign(difference))
    if excludes_zero and sign == ref_sign:
        return "holds"
    if excludes_zero and sign == -ref_sign:
        return "flips"
    return "weakens"


def axis_outcome(axis: str, labels: Sequence[Label]) -> Outcome:
    """Axis verdict from its variant labels (NT axis tolerates a quarter of variants)."""
    if not labels:
        return "not_run"
    n_holds = sum(lbl == "holds" for lbl in labels)
    if axis == "nt":
        need = math.ceil(NT_ROBUST_FRACTION * len(labels))
        return "robust" if n_holds >= need else "fails"
    return "robust" if n_holds == len(labels) else "fails"


def null_outcome(p: float | None) -> Outcome:
    """Verdict of the random-null axis from its empirical p-value."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "not_run"
    return "robust" if p < NULL_ALPHA else "fails"


def tier(outcomes: Mapping[str, Outcome]) -> tuple[str, bool]:
    """(tier, thinly_tested) from graded axis outcomes."""
    graded = [outcomes.get(a, "not_run") for a in GRADED_AXES]
    fails = sum(o == "fails" for o in graded)
    ran = sum(o != "not_run" for o in graded)
    name = "Strong" if fails == 0 else "Conditional" if fails == 1 else "Fragile"
    return name, ran < 2


def score_hits(
    effects: pd.DataFrame,
    *,
    alpha: float,
    null_p: Mapping[str, float | None],
    replicate: Mapping[str, str],
) -> pd.DataFrame:
    """One row per reference hit (candidate, q < alpha, nonzero effect) with verdicts and tier."""
    ref = effects[
        (effects["axis"] == "reference")
        & (effects["role"] == "candidate")
        & (effects["q_value"] < alpha)
        & (effects["difference"] != 0)
    ]
    rows: list[dict[str, Any]] = []
    for _, r in ref.iterrows():
        cand = str(r["candidate"])
        ref_diff = float(r["difference"])
        row: dict[str, Any] = {
            "candidate": cand,
            "difference": ref_diff,
            "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"]),
            "q_value": float(r["q_value"]),
            "direction": "down" if ref_diff < 0 else "up",
        }
        outcomes: dict[str, Outcome] = {}
        for axis in _SWEEP_AXES:
            sub = effects[(effects["axis"] == axis) & (effects["candidate"] == cand)]
            labels: list[Label] = [
                variant_label(ref_diff, float(d), float(lo), float(hi))
                for d, lo, hi in zip(sub["difference"], sub["ci_low"], sub["ci_high"], strict=True)
            ]
            outcomes[axis] = axis_outcome(axis, labels)
            row[axis] = outcomes[axis]
            detail = ", ".join(f"{v}: {lbl}" for v, lbl in zip(sub["variant"], labels, strict=True))
            if axis == "nt":
                row["nt_holds"] = f"{sum(lbl == 'holds' for lbl in labels)}/{len(labels)}"
                row["nt_flips"] = ", ".join(
                    str(v) for v, lbl in zip(sub["variant"], labels, strict=True) if lbl == "flips"
                )
            else:
                row[f"{axis}_detail"] = detail
        p = null_p.get(cand)
        outcomes["random_null"] = null_outcome(p)
        row["random_null"] = outcomes["random_null"]
        row["random_null_p"] = float("nan") if p is None else float(p)
        row["replicate"] = replicate.get(cand, "not run")
        row["tier"], row["thinly_tested"] = tier(outcomes)
        row["failed_axes"] = ", ".join(a for a in GRADED_AXES if outcomes[a] == "fails")
        row["untested_axes"] = ", ".join(a for a in GRADED_AXES if outcomes[a] == "not_run")
        rows.append(row)
    cols = [
        "candidate",
        "difference",
        "ci_low",
        "ci_high",
        "q_value",
        "direction",
        "w_syn",
        "w_syn_detail",
        "nt",
        "nt_holds",
        "nt_flips",
        "min_weight",
        "min_weight_detail",
        "random_null",
        "random_null_p",
        "replicate",
        "tier",
        "thinly_tested",
        "failed_axes",
        "untested_axes",
    ]
    out = pd.DataFrame(rows, columns=pd.Index(cols))
    if out.empty:
        return out
    out["_t"] = [TIER_ORDER[str(t)] for t in out["tier"]]
    out["_a"] = out["difference"].abs()
    out = out.sort_values(["_t", "_a"], ascending=[True, False], kind="stable")
    return out.drop(columns=["_t", "_a"]).reset_index(drop=True)
