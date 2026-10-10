"""Cosine scores of query profiles against atlas centroids, with yardstick adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas
from flyconn.compare.profiles import l2_rows

Level = Literal["neuron", "group"]
YARDSTICK_FLOOR = 0.05
"""Lower bound on a yardstick in ``a = s / yardstick`` so near-zero yardsticks cannot explode."""


@dataclass
class Scores:
    """Top-k candidates per query row, best first."""

    labels: np.ndarray  # q x k object
    s: np.ndarray  # q x k cosine
    a: np.ndarray  # q x k cosine / label yardstick

    @property
    def top1(self) -> np.ndarray:
        return self.labels[:, 0]

    @property
    def s1(self) -> np.ndarray:
        return self.s[:, 0]

    @property
    def a1(self) -> np.ndarray:
        return self.a[:, 0]

    @property
    def margin(self) -> np.ndarray:
        """Best minus second-best cosine (the best cosine if the atlas has one label)."""
        return self.s[:, 0] - self.s[:, 1] if self.s.shape[1] > 1 else self.s[:, 0].copy()


def score(
    atlas: Atlas,
    profiles: sp.spmatrix,
    *,
    level: Level = "neuron",
    top_k: int = 5,
    chunk: int = 5000,
) -> Scores:
    """Cosine of each profile row to every centroid; keep the ``top_k`` best per row."""
    q = l2_rows(profiles)
    n_rows, width = cast("tuple[int, int]", q.shape)
    atlas_width = cast("tuple[int, int]", atlas.sums.shape)[1]
    if width != atlas_width:
        msg = f"profile width {width} does not match the atlas width {atlas_width}"
        raise ValueError(msg)
    yard = atlas.neuron_yardstick if level == "neuron" else atlas.group_yardstick
    yard = np.maximum(yard, YARDSTICK_FLOOR)
    k = min(top_k, len(atlas.labels))
    ct = sp.csr_matrix(atlas.centroids.T)
    orders: list[np.ndarray] = [np.zeros((0, k), dtype=np.int64)]
    sims: list[np.ndarray] = [np.zeros((0, k))]
    for start in range(0, n_rows, chunk):
        block = np.asarray((q[start : start + chunk] @ ct).toarray(), dtype=float)
        order = np.argsort(-block, axis=1, kind="stable")[:, :k]
        orders.append(order)
        sims.append(np.take_along_axis(block, order, axis=1))
    order = np.vstack(orders)
    s = np.vstack(sims)
    return Scores(labels=atlas.labels[order], s=s, a=s / yard[order])
