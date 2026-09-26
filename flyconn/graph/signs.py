"""Sign policies: how predicted neurotransmitters become edge signs (ADR-0007)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import numpy as np
import pandas as pd

from flyconn.data.schema import NT_CLASSES

INHIBITORY: frozenset[str] = frozenset({"gaba", "glutamate", "histamine"})
"""Transmitters treated as inhibitory by default (Shiu et al. 2024 rule plus histamine)."""


class SignPolicy(StrEnum):
    """How to turn per-neuron NT predictions into a presynaptic sign."""

    ARGMAX = "argmax"
    """Consensus/argmax label -> +1/-1; unknown -> 0 (edges excluded)."""
    PROBABILISTIC = "probabilistic"
    """Expected sign sum_nt p(nt)*sign(nt) in [-1, 1]; argmax when probabilities are absent."""
    SAMPLED = "sampled"
    """One draw of the transmitter per neuron from p(nt); needs a seed (used by uncertainty)."""
    UNSIGNED = "unsigned"
    """All signs +1 (pure synapse counts)."""


def sign_table(overrides: Mapping[str, int] | None = None) -> dict[str, int]:
    """Transmitter -> sign, with optional per-transmitter overrides (e.g. ``{"glutamate": 1}``)."""
    table = {nt: (-1 if nt in INHIBITORY else 1) for nt in NT_CLASSES}
    for nt, sign in (overrides or {}).items():
        if nt not in table:
            msg = f"unknown transmitter {nt!r}; known: {NT_CLASSES}"
            raise KeyError(msg)
        if sign not in (-1, 1):
            msg = f"sign override for {nt!r} must be -1 or 1, got {sign!r}"
            raise ValueError(msg)
        table[nt] = int(sign)
    return table


def _prob_matrix(neurons: pd.DataFrame) -> np.ndarray | None:
    cols = [f"nt_p_{nt}" for nt in NT_CLASSES]
    if not all(c in neurons.columns for c in cols):
        return None
    probs = neurons[cols].to_numpy(dtype=float)
    probs = np.nan_to_num(probs, nan=0.0)
    total = probs.sum(axis=1)
    if not (total > 0).any():
        return None
    return probs


def neuron_signs(
    neurons: pd.DataFrame,
    policy: SignPolicy = SignPolicy.ARGMAX,
    overrides: Mapping[str, int] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-neuron presynaptic sign for a harmonized ``neurons`` table (row order preserved)."""
    table = sign_table(overrides)
    n = len(neurons)
    if policy is SignPolicy.UNSIGNED:
        return np.ones(n, dtype=float)

    def _label_sign(v: object) -> int:
        return 0 if v is None or (isinstance(v, float) and np.isnan(v)) else table[str(v)]

    argmax = neurons["nt_pred"].map(_label_sign).to_numpy(dtype=float)
    if policy is SignPolicy.ARGMAX:
        return argmax

    probs = _prob_matrix(neurons)
    if probs is None:
        return argmax
    sign_vec = np.array([table[nt] for nt in NT_CLASSES], dtype=float)
    has_probs = probs.sum(axis=1) > 0

    if policy is SignPolicy.PROBABILISTIC:
        expected = (probs / np.where(has_probs, probs.sum(axis=1), 1.0)[:, None]) @ sign_vec
        return np.where(has_probs, expected, argmax)

    if rng is None:
        msg = "SignPolicy.SAMPLED needs a numpy Generator (rng=)"
        raise ValueError(msg)
    normalized = probs / np.where(has_probs, probs.sum(axis=1), 1.0)[:, None]
    cum = normalized.cumsum(axis=1)
    draws = rng.random(n)
    choice = (draws[:, None] > cum).sum(axis=1).clip(max=len(NT_CLASSES) - 1)
    sampled = sign_vec[choice]
    return np.where(has_probs, sampled, argmax)
