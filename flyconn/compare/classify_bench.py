"""Hold-out benchmark for zero-shot classification (ADR-0010).

Truth is the query's published cross-reference to the reference (single names only).
Closed set: the evaluated type is blanked as a partner label in query and atlas, its
centroid stays. Open set: its centroid is also dropped, so the right answer is
unknown/ambiguous. Folds and bootstrap resamples are over types, never neurons.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas
from flyconn.compare.calibrate import (
    FEATURE_SETS,
    Calibration,
    LogisticModel,
    Thresholds,
    apply_calls,
    brier,
    choose_delta,
    choose_s_floor,
    ece,
    fit_logistic,
)
from flyconn.compare.profiles import partner_counts
from flyconn.compare.scoring import Scores, score
from flyconn.compare.types import partner_labels
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.provenance import run_environment

LEVELS = ("neuron", "group")
SELECTION_TOLERANCE = 0.005
"""The adjusted calibrator stays the default unless the raw one beats it by more than this."""


@dataclass
class BenchmarkResult:
    query: str
    reference: str
    records: pd.DataFrame
    n_types_available: int
    n_comma_excluded: int
    query_provenance: dict[str, Any] = field(default_factory=dict)


def truth_labels(
    meta: pd.DataFrame, column: str, reference_labels: set[str]
) -> tuple[np.ndarray, int]:
    """Single-name cross-references present in the reference; count of comma lists dropped."""
    vals = meta[column].astype(object).tolist() if column in meta.columns else [None] * len(meta)
    out: list[str | None] = []
    n_comma = 0
    for v in vals:
        if isinstance(v, str) and "," in v:
            n_comma += 1
            out.append(None)
        elif isinstance(v, str) and v in reference_labels:
            out.append(v)
        else:
            out.append(None)
    return np.asarray(out, dtype=object), n_comma


def _rows(
    query: str, protocol: str, level: str, t: str, ids: np.ndarray, sc: Scores, empty: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "query": query,
            "protocol": protocol,
            "level": level,
            "type": t,
            "neuron_id": int(ids[i]),
            "top1": str(sc.labels[i, 0]),
            "top_labels": [str(x) for x in sc.labels[i].tolist()],
            "s1": float(sc.s1[i]),
            "a1": float(sc.a1[i]),
            "margin": float(sc.margin[i]),
            "correct": bool(sc.labels[i, 0] == t),
            "top3": bool(t in sc.labels[i, :3].tolist()),
            "empty": bool(empty[i]),
        }
        for i in range(len(ids))
    ]


def run_benchmark(
    query_matrix: ConnectivityMatrix,
    atlas: Atlas,
    *,
    query_ref: str,
    truth_column: str = "fafb_783_cell_type",
    max_types: int = 2000,
    seed: int = 0,
    protocols: Sequence[str] = ("closed", "open"),
    top_k: int = 5,
) -> BenchmarkResult:
    names = partner_labels(query_matrix.meta, str(atlas.params["vocabulary"])).tolist()
    counts = partner_counts(query_matrix, names, vocab=atlas.vocab.tolist())
    truth, n_comma = truth_labels(
        query_matrix.meta, truth_column, {str(x) for x in atlas.labels.tolist()}
    )
    labelled = np.flatnonzero([t is not None for t in truth.tolist()])
    by_type: dict[str, np.ndarray] = {
        str(t): np.asarray(v, dtype=np.int64)
        for t, v in pd.Series(labelled).groupby(truth[labelled]).groups.items()
    }
    by_type = {t: labelled[v] for t, v in by_type.items()}
    types = sorted(by_type)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(types, size=min(max_types, len(types)), replace=False).tolist())
    rows: list[dict[str, Any]] = []
    for t in sample:
        idx = np.asarray(by_type[t], dtype=np.int64)
        sub = counts.rows(idx)
        prof = sub.profiles(direction=atlas.direction, mask=(t,))
        empty = np.asarray(prof.getnnz(axis=1)) == 0
        closed = atlas.masked((t,))
        gp = (
            sp.csr_matrix(prof[np.flatnonzero(~empty)].mean(axis=0))
            if (~empty).any()
            else sp.csr_matrix((1, cast("tuple[int, int]", prof.shape)[1]))
        )
        for protocol in protocols:
            atl = closed if protocol == "closed" else closed.masked((), drop=(t,))
            ns = score(atl, prof, level="neuron", top_k=top_k)
            rows += _rows(query_ref, protocol, "neuron", t, sub.neuron_ids, ns, empty)
            gs = score(atl, gp, level="group", top_k=top_k)
            rows += _rows(
                query_ref, protocol, "group", t, np.array([-1]), gs, np.array([bool(empty.all())])
            )
    return BenchmarkResult(
        query=query_ref,
        reference=atlas.reference,
        records=pd.DataFrame(rows),
        n_types_available=len(types),
        n_comma_excluded=n_comma,
        query_provenance=dict(query_matrix.provenance),
    )


def _sel(df: pd.DataFrame, mask: object) -> pd.DataFrame:
    return cast("pd.DataFrame", df[np.asarray(mask, dtype=bool)])


def _col(df: pd.DataFrame, name: str, dtype: type = float) -> np.ndarray:
    return cast("pd.Series", df[name]).to_numpy(dtype=dtype)


def _features(df: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    return cast("pd.DataFrame", df[list(names)]).to_numpy(dtype=float)


def _fit(df: pd.DataFrame, names: Sequence[str]) -> tuple[LogisticModel, bool]:
    y = _col(df, "correct")
    if len(np.unique(y)) < 2:
        rate = float(np.clip(y.mean() if len(y) else 0.5, 1e-3, 1 - 1e-3))
        logit = float(np.clip(np.log(rate / (1 - rate)), -6, 6))
        return LogisticModel(tuple(names), np.r_[logit, np.zeros(len(names))]), True
    return fit_logistic(_features(df, names), y, features=names), False


def _type_bootstrap(df: pd.DataFrame, col: str, n_boot: int, seed: int) -> list[float]:
    """[mean, 2.5 %, 97.5 %] of ``col`` with types resampled (records weighted)."""
    g = cast("pd.DataFrame", df.groupby("type")[col].agg(["sum", "count"]))
    sums, cnts = _col(g, "sum"), _col(g, "count")
    if cnts.sum() == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(g), size=(n_boot, len(g)))
    boot = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    return [
        float(sums.sum() / cnts.sum()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]


def _coverage(calls: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    covered = calls == "type"
    cov = float(covered.mean()) if len(calls) else float("nan")
    acc = float(correct[covered].mean()) if covered.any() else float("nan")
    return cov, acc


def fit_calibration(
    results: Sequence[BenchmarkResult],
    atlas: Atlas,
    *,
    n_folds: int = 5,
    seed: int = 0,
    accept: float = 0.8,
    max_false_accept: float = 0.10,
    n_boot: int = 1000,
    calibrate_levels: Sequence[str] = ("group",),
) -> Calibration:
    """Fit the logistic calibration (metrics for both levels; models for calibrate_levels).

    Per-neuron calls are not calibrated by default: on MaleCNS and BANC against FlyWire v783
    their leave-one-dataset-out ECE was 0.23-0.24 (ADR-0010).
    """
    rec = pd.concat([r.records for r in results], ignore_index=True)
    rec = _sel(rec, ~_col(rec, "empty", bool))
    closed = _sel(rec, _col(rec, "protocol", object) == "closed").copy()
    open_ = _sel(rec, _col(rec, "protocol", object) == "open").copy()
    # folds by reference type, so a type held out from MaleCNS is also held out from BANC
    types = sorted(set(_col(closed, "type", object).tolist()))
    perm = np.random.default_rng(seed).permutation(len(types))
    fold_of = {t: int(f) for t, f in zip(types, perm % n_folds, strict=True)}
    closed = closed.reset_index(drop=True)
    open_ = open_.reset_index(drop=True)
    for df in (closed, open_):
        df["fold"] = [fold_of.get(str(t), -1) for t in _col(df, "type", object).tolist()]
        for variant in FEATURE_SETS:
            df[f"p_{variant}"] = np.nan
            df[f"call_{variant}"] = ""

    def level_of(df: pd.DataFrame, level: str, query: str | None = None) -> pd.DataFrame:
        mask = _col(df, "level", object) == level
        if query is not None:
            mask &= _col(df, "query", object) == query
        return _sel(df, mask)

    def thresholds_for(c: pd.DataFrame, o: pd.DataFrame) -> Thresholds:
        return Thresholds(
            accept=accept,
            s_floor=choose_s_floor(_col(o, "s1"), max_false_accept),
            delta=choose_delta(_col(c, "margin"), _col(c, "correct")),
        )

    thresholds = {
        level: thresholds_for(level_of(closed, level), level_of(open_, level)) for level in LEVELS
    }

    def rate(calls: np.ndarray) -> float:
        return float((calls == "type").mean()) if len(calls) else float("nan")

    # Out of fold: model AND thresholds are chosen without the held-out types, for both the
    # closed-set calls (coverage, accuracy) and the open-set calls (false accept).
    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, names in FEATURE_SETS.items():
        variants[variant] = {}
        for level in LEVELS:
            c, o = level_of(closed, level), level_of(open_, level)
            cf, of = _col(c, "fold"), _col(o, "fold")
            p_c, p_o = np.full(len(c), np.nan), np.full(len(o), np.nan)
            call_c = np.full(len(c), "", dtype=object)
            call_o = np.full(len(o), "", dtype=object)
            for f in range(n_folds):
                tc, to = cf == f, of == f
                if not tc.any() or not (~tc).any():
                    continue
                model, _ = _fit(_sel(c, ~tc), names)
                thr = thresholds_for(_sel(c, ~tc), _sel(o, ~to))
                p_c[tc] = model.predict(_features(_sel(c, tc), names))
                call_c[tc] = apply_calls(p_c[tc], _col(c, "s1")[tc], thr)
                if to.any():
                    p_o[to] = model.predict(_features(_sel(o, to), names))
                    call_o[to] = apply_calls(p_o[to], _col(o, "s1")[to], thr)
            closed.loc[c.index, f"p_{variant}"] = p_c
            closed.loc[c.index, f"call_{variant}"] = call_c
            open_.loc[o.index, f"call_{variant}"] = call_o
            y = _col(c, "correct")
            _, degenerate = _fit(c, names)
            done = call_c != ""
            cov, acc = _coverage(call_c[done], _col(c, "correct", bool)[done])
            variants[variant][level] = {
                "ece_oof": ece(p_c, y),
                "brier_oof": brier(p_c[np.isfinite(p_c)], y[np.isfinite(p_c)]),
                "coverage": cov,
                "accuracy_covered": acc,
                "false_accept_open": rate(call_o[call_o != ""]),
                "degenerate": degenerate,
            }

    def no_worse(level: str) -> bool:
        adj, raw = variants["adjusted"][level], variants["raw"][level]
        return bool(
            adj["ece_oof"] <= raw["ece_oof"] + SELECTION_TOLERANCE
            and (
                np.isnan(raw["accuracy_covered"])
                or adj["accuracy_covered"] >= raw["accuracy_covered"] - SELECTION_TOLERANCE
            )
        )

    variant = "adjusted" if all(no_worse(level) for level in calibrate_levels) else "raw"
    names = FEATURE_SETS[variant]
    models = {level: _fit(level_of(closed, level), names)[0] for level in LEVELS}

    queries = list(dict.fromkeys(r.query for r in results))
    transfer: dict[str, dict[str, float]] = {}
    if len(queries) > 1:
        for q in queries:
            transfer[q] = {}
            for level in LEVELS:
                c = level_of(closed, level)
                is_q = _col(c, "query", object) == q
                model, _ = _fit(_sel(c, ~is_q), names)
                test = _sel(c, is_q)
                transfer[q][level] = ece(
                    model.predict(_features(test, names)), _col(test, "correct")
                )

    per_query: dict[str, dict[str, dict[str, Any]]] = {}
    for q in queries:
        per_query[q] = {}
        for level in LEVELS:
            c, o = level_of(closed, level, q), level_of(open_, level, q)
            oof = _col(c, f"p_{variant}")
            call_c = _col(c, f"call_{variant}", object)
            call_o = _col(o, f"call_{variant}", object)
            done = call_c != ""
            cov, acc = _coverage(call_c[done], _col(c, "correct", bool)[done])
            per_query[q][level] = {
                "n_types": len(set(_col(c, "type", object).tolist())),
                "n_records": len(c),
                "top1": _type_bootstrap(c, "correct", n_boot, seed),
                "top3": _type_bootstrap(c, "top3", n_boot, seed),
                "ece_oof": ece(oof, _col(c, "correct")),
                "coverage": cov,
                "accuracy_covered": acc,
                "false_accept_open": rate(call_o[call_o != ""]),
            }

    metrics: dict[str, Any] = {
        "variant_selected": variant,
        "evaluation": "out_of_fold",
        "calibrated_levels": list(calibrate_levels),
        "selection_rule": (
            "adjusted unless raw has lower out-of-fold ECE or higher accuracy-at-coverage by "
            f"more than {SELECTION_TOLERANCE} at either level"
        ),
        "variants": variants,
        "per_query": per_query,
        "transfer_ece": transfer,
        "n_folds": n_folds,
        "seed": seed,
        "max_false_accept": max_false_accept,
        "benchmark": {
            r.query: {
                "n_types_available": r.n_types_available,
                "n_comma_excluded": r.n_comma_excluded,
            }
            for r in results
        },
    }
    return Calibration(
        reference=atlas.reference,
        atlas_params_hash=atlas.params_hash,
        atlas_params=atlas.params,
        variant=variant,
        models={level: models[level] for level in calibrate_levels},
        thresholds={level: thresholds[level] for level in calibrate_levels},
        fitted_on=queries,
        metrics=metrics,
        provenance={
            "atlas": atlas.provenance,
            "queries": {r.query: r.query_provenance for r in results},
            "environment": run_environment(),
        },
    )


def write_benchmark(out_dir: Path, results: Sequence[BenchmarkResult], cal: Calibration) -> Path:
    """Write per-query records (Parquet) and ``metrics.json``; return the metrics path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.records.to_parquet(out_dir / f"records_{r.query}.parquet", index=False)
    path = out_dir / "metrics.json"
    payload = {
        "calibration_id": cal.id,
        "variant": cal.variant,
        **cal.metrics,
        "thresholds": {k: vars(v) for k, v in cal.thresholds.items()},
        "provenance": cal.provenance,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path
