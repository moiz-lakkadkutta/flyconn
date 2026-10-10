"""Partner-type count profiles over a fixed label vocabulary (zero-shot classification).

A profile row holds the fraction of a neuron's output synapses going to partners of each
vocabulary label, followed by the fraction of its input synapses coming from them. Each
half is L1-normalised on its own, so a neuron without inputs (sensory) or outputs (motor)
is described by the other half alone. Partners without a vocabulary label are dropped
from the fractions and counted in ``unlabelled_fraction``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.graph.matrix import ConnectivityMatrix

ProfileDirection = Literal["both", "out", "in"]
_DIRECTIONS = ("both", "out", "in")


def check_direction(direction: str) -> None:
    if direction not in _DIRECTIONS:
        msg = f"unknown profile direction {direction!r}; use one of {_DIRECTIONS}"
        raise ValueError(msg)


def _scale_rows(x: sp.csr_matrix, d: np.ndarray) -> sp.csr_matrix:
    inv = np.divide(1.0, d, out=np.zeros(len(d)), where=d > 0)
    return sp.csr_matrix(sp.diags(inv) @ x)


def l1_rows(x: sp.spmatrix) -> sp.csr_matrix:
    x = sp.csr_matrix(x, dtype=np.float64)
    return _scale_rows(x, np.asarray(abs(x).sum(axis=1), dtype=float).ravel())


def l2_rows(x: sp.spmatrix) -> sp.csr_matrix:
    x = sp.csr_matrix(x, dtype=np.float64)
    return _scale_rows(x, np.sqrt(np.asarray(x.multiply(x).sum(axis=1), dtype=float).ravel()))


def _is_label(v: object) -> bool:
    return isinstance(v, str) and v != ""


@dataclass
class PartnerCounts:
    """Synapse counts per neuron to (``out``) and from (``inn``) partners of each label."""

    neuron_ids: np.ndarray
    vocab: np.ndarray
    out: sp.csr_matrix
    inn: sp.csr_matrix
    out_total: np.ndarray
    in_total: np.ndarray

    def rows(self, idx: np.ndarray) -> PartnerCounts:
        idx = np.asarray(idx, dtype=np.int64)
        return PartnerCounts(
            neuron_ids=self.neuron_ids[idx],
            vocab=self.vocab,
            out=sp.csr_matrix(self.out[idx]),
            inn=sp.csr_matrix(self.inn[idx]),
            out_total=self.out_total[idx],
            in_total=self.in_total[idx],
        )

    def _vocab_keep(self, mask: Sequence[str]) -> np.ndarray:
        return (~np.isin(self.vocab, np.asarray(list(mask), dtype=object))).astype(float)

    def column_keep(self, mask: Sequence[str], direction: ProfileDirection) -> np.ndarray:
        """1/0 per profile column: 0 for columns of masked labels."""
        check_direction(direction)
        keep = self._vocab_keep(mask)
        return np.concatenate([keep, keep]) if direction == "both" else keep

    def profiles(
        self, *, direction: ProfileDirection = "both", mask: Sequence[str] = ()
    ) -> sp.csr_matrix:
        check_direction(direction)
        keep = sp.diags(self._vocab_keep(mask))
        halves: list[sp.csr_matrix] = []
        if direction in ("both", "out"):
            halves.append(l1_rows(sp.csr_matrix(self.out @ keep)))
        if direction in ("both", "in"):
            halves.append(l1_rows(sp.csr_matrix(self.inn @ keep)))
        return sp.csr_matrix(sp.hstack(halves)) if len(halves) == 2 else halves[0]

    def labelled_mass(self, *, direction: ProfileDirection = "both") -> tuple[float, float]:
        """(synapses to/from labelled partners, all synapses) summed over the rows."""
        check_direction(direction)
        lab = tot = 0.0
        if direction in ("both", "out"):
            lab += float(self.out.sum())
            tot += float(self.out_total.sum())
        if direction in ("both", "in"):
            lab += float(self.inn.sum())
            tot += float(self.in_total.sum())
        return lab, tot

    def unlabelled_fraction(self, *, direction: ProfileDirection = "both") -> np.ndarray:
        check_direction(direction)
        lab = np.zeros(len(self.neuron_ids))
        tot = np.zeros(len(self.neuron_ids))
        if direction in ("both", "out"):
            lab += np.asarray(self.out.sum(axis=1), dtype=float).ravel()
            tot += self.out_total
        if direction in ("both", "in"):
            lab += np.asarray(self.inn.sum(axis=1), dtype=float).ravel()
            tot += self.in_total
        safe = np.where(tot > 0, tot, 1.0)
        return np.where(tot > 0, 1.0 - lab / safe, 1.0)

    @cached_property
    def _out_csc(self) -> sp.csc_matrix:
        return sp.csc_matrix(self.out)

    @cached_property
    def _inn_csc(self) -> sp.csc_matrix:
        return sp.csc_matrix(self.inn)

    def touching(self, mask: Sequence[str]) -> np.ndarray:
        """Rows with any synapse to or from a partner of a masked label."""
        cols = np.flatnonzero(np.isin(self.vocab, np.asarray(list(mask), dtype=object)))
        if len(cols) == 0:
            return np.zeros(0, dtype=np.int64)
        hit = np.asarray(self._out_csc[:, cols].sum(axis=1), dtype=float).ravel()
        hit += np.asarray(self._inn_csc[:, cols].sum(axis=1), dtype=float).ravel()
        return np.flatnonzero(hit > 0)


def partner_counts(
    m: ConnectivityMatrix, labels: Sequence[object], vocab: Sequence[str] | None = None
) -> PartnerCounts:
    """Count each neuron's synapses to/from partners of each label in ``vocab``.

    ``labels`` gives one partner label per neuron of ``m`` (None for unlabelled). ``vocab``
    defaults to the sorted distinct labels; pass the atlas vocabulary for query datasets so
    columns line up.
    """
    labels_arr = np.asarray(list(labels), dtype=object)
    if len(labels_arr) != m.n:
        msg = f"labels has {len(labels_arr)} entries for {m.n} neurons"
        raise ValueError(msg)
    if vocab is None:
        vocab_arr = np.array(sorted({str(v) for v in labels_arr if _is_label(v)}), dtype=object)
    else:
        vocab_arr = np.asarray(list(vocab), dtype=object)
    col = {v: i for i, v in enumerate(vocab_arr.tolist())}
    codes = np.array([col.get(v, -1) if _is_label(v) else -1 for v in labels_arr.tolist()])
    ok = codes >= 0
    member = sp.csr_matrix(
        (np.ones(int(ok.sum())), (np.flatnonzero(ok), codes[ok])), shape=(m.n, len(vocab_arr))
    )
    w = sp.csr_matrix(m.weights, dtype=np.float64)
    wt = sp.csr_matrix(w.T)
    return PartnerCounts(
        neuron_ids=np.asarray(m.neuron_ids),
        vocab=vocab_arr,
        out=sp.csr_matrix(w @ member),
        inn=sp.csr_matrix(wt @ member),
        out_total=np.asarray(w.sum(axis=1), dtype=float).ravel(),
        in_total=np.asarray(w.sum(axis=0), dtype=float).ravel(),
    )


XREF_COLUMNS = ("fafb_783_cell_type", "manc_121_cell_type")


def vocab_partner_names(meta: pd.DataFrame, vocab: set[str], vocabulary: str) -> list[object]:
    """Per-neuron partner name: the first cross-reference name that is in ``vocab``.

    Under ``fafb_or_manc`` a neuron may carry both a FlyWire and a MANC name; taking the one
    the atlas actually uses keeps e.g. descending neurons labelled against a MANC atlas.
    Without any cross-reference values this falls back to ``partner_labels`` (own types).
    """
    from flyconn.compare.types import partner_labels

    cols = [c for c in XREF_COLUMNS if c in meta.columns and bool(meta[c].notna().any())]
    if vocabulary != "fafb_or_manc" or not cols:
        return [v if _is_label(v) else None for v in partner_labels(meta, vocabulary).tolist()]
    out: list[object] = [None] * len(meta)
    for c in cols:
        vals = meta[c].astype(object).tolist()
        out = [
            o if o is not None else (v if _is_label(v) and v in vocab else None)
            for o, v in zip(out, vals, strict=True)
        ]
    return out
