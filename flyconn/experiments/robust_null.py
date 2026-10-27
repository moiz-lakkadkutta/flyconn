"""Random-silencing null for robustness screens (ADR-0011).

A hit should matter more than silencing the same number of random neurons of the same
super_class. Sets never contain candidate, stimulus or primary-readout neurons. With
``pool_ids`` (the activity-matched pool, M14 D2) sets are drawn only from those ids.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flyconn.experiments.sweep import Candidate


def size_bin(n: int) -> int:
    if n <= 1:
        return 1
    return int(2 ** round(math.log2(n)))


def null_key(neurons: pd.DataFrame, ids: Sequence[int]) -> str:
    sc = neurons.set_index("neuron_id")["super_class"].reindex(list(ids)).dropna().astype(str)
    major = str(sc.mode().iloc[0]) if not sc.empty else "unknown"
    return f"{major}|{size_bin(len(ids))}"


@dataclass(frozen=True)
class NullSet:
    key: str
    index: int
    seed: int
    neuron_ids: tuple[int, ...]


def draw_null_sets(
    neurons: pd.DataFrame,
    candidates: Sequence[Candidate],
    excluded: set[int],
    *,
    n_sets: int,
    seed: int,
    pool_ids: set[int] | None = None,
) -> tuple[list[NullSet], dict[str, str]]:
    blocked = set(excluded) | {i for c in candidates for i in c.neuron_ids}
    # Controls get a null too: the negative control's null p is the honesty check (spec 4).
    keys = sorted(
        {null_key(neurons, c.neuron_ids) for c in candidates if c.role in ("candidate", "control")}
    )
    sets: list[NullSet] = []
    skipped: dict[str, str] = {}
    for key in keys:
        sc, bin_s = key.split("|")
        size = int(bin_s)
        pool_df = neurons[neurons["super_class"].astype(str) == sc]
        ids = set(pool_df["neuron_id"].astype(np.int64).tolist()) - blocked
        if pool_ids is not None:
            ids &= pool_ids
        pool = np.array(sorted(ids), dtype=np.int64)
        if len(pool) < size:
            kind = f"active {sc}" if pool_ids is not None else sc
            skipped[key] = f"pool of {len(pool)} {kind} neurons is smaller than the set size {size}"
            continue
        for i in range(n_sets):
            s = seed + 3000 + i
            # keyed by the key text, so adding a candidate never reshuffles other keys' sets
            rng = np.random.default_rng([s, zlib.crc32(key.encode())])
            ids = tuple(int(x) for x in np.sort(rng.choice(pool, size=size, replace=False)))
            sets.append(NullSet(key, i, s, ids))
    return sets, skipped


def null_p(effect: float, null_effects: Sequence[float]) -> float:
    """(1 + #null at least as extreme in the effect's direction) / (n + 1).

    NaN null effects are dropped (n counts the rest); a NaN effect, or no finite null
    left, gives NaN (the null axis then scores not_run).
    """
    arr = np.asarray(null_effects, dtype=float)
    arr = arr[~np.isnan(arr)]
    if math.isnan(effect) or len(arr) == 0:
        return float("nan")
    s = 1.0 if effect >= 0 else -1.0
    return float((1 + int((arr * s >= effect * s).sum())) / (len(arr) + 1))
