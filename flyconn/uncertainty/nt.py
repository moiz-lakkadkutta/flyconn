"""Neurotransmitter uncertainty: per-neuron probability models and sign sampling (ADR-0008)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Literal, cast

import numpy as np
import pandas as pd

from flyconn.data.schema import NT_CLASSES
from flyconn.graph.signs import sign_table

Model = Literal["auto", "probabilities", "confidence"]
_K = len(NT_CLASSES)


def confidence_model(neurons: pd.DataFrame) -> np.ndarray:
    """Probability matrix from ``nt_pred`` + ``nt_conf`` only.

    ``P(argmax) = nt_conf`` and the remaining mass is spread uniformly over the
    other classes; a label without a confidence counts as certain; rows without
    a label are all-NaN (unknown).
    """
    labels = cast("pd.Series", neurons["nt_pred"]).to_numpy(dtype=object)
    conf = (
        cast("pd.Series", neurons["nt_conf"]).to_numpy(dtype=float)
        if "nt_conf" in neurons.columns
        else np.full(len(neurons), np.nan)
    )
    p = np.full((len(neurons), _K), np.nan)
    for i, lab in enumerate(labels):
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            continue
        k = NT_CLASSES.index(str(lab))
        c = 1.0 if np.isnan(conf[i]) else float(np.clip(conf[i], 0.0, 1.0))
        p[i] = (1.0 - c) / (_K - 1)
        p[i, k] = c
    return p


def nt_probabilities(neurons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Best available per-neuron probabilities and which model produced each row.

    Rows with any ``nt_p_*`` mass use those (renormalised, model ``probabilities``);
    rows with only a label use :func:`confidence_model` (``confidence``); the rest
    are ``unknown`` (all-NaN).
    """
    cols = [f"nt_p_{nt}" for nt in NT_CLASSES]
    n = len(neurons)
    probs = np.full((n, _K), np.nan)
    if all(c in neurons.columns for c in cols):
        raw = np.nan_to_num(neurons[cols].to_numpy(dtype=float), nan=0.0)
        mass = raw.sum(axis=1)
        has = mass > 0
        probs[has] = raw[has] / mass[has, None]
    conf = confidence_model(neurons)
    model = np.full(n, "unknown", dtype=object)
    use_probs = ~np.isnan(probs).all(axis=1)
    use_conf = ~use_probs & ~np.isnan(conf).all(axis=1)
    probs[use_conf] = conf[use_conf]
    model[use_probs] = "probabilities"
    model[use_conf] = "confidence"
    return probs, model


def sample_signs(
    neurons: pd.DataFrame,
    n: int,
    seed: int,
    *,
    model: Model = "auto",
    overrides: Mapping[str, int] | None = None,
) -> Iterator[np.ndarray]:
    """Yield ``n`` seeded per-neuron sign vectors drawn from the NT probability model.

    Each draw picks one transmitter per neuron from its probabilities and maps it
    through :func:`flyconn.graph.signs.sign_table`; unknown neurons get sign 0.
    """
    if model == "confidence":
        probs = confidence_model(neurons)
    else:
        probs, _ = nt_probabilities(neurons)
        if model == "probabilities":
            pass  # fall back to confidence rows where probabilities are absent
    known = ~np.isnan(probs).all(axis=1)
    cum = np.cumsum(np.nan_to_num(probs, nan=0.0), axis=1)
    cum[known] /= cum[known, -1][:, None]
    table = sign_table(overrides)
    sign_vec = np.array([table[nt] for nt in NT_CLASSES], dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        draws = rng.random(len(neurons))
        choice = (draws[:, None] > cum).sum(axis=1).clip(max=_K - 1)
        signs = np.where(known, sign_vec[choice], 0.0)
        yield signs
