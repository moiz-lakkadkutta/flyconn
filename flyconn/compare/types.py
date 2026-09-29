"""Compare a cell type's connectivity between two datasets against left/right variability.

Design (ADR-0003): matching uses the cross-dataset type columns the releases publish
(e.g. MaleCNS ``fafb_783_cell_type``), profiles are partner-type fractions, and the
null model is within-brain left/right variability (Schlegel et al. 2024): the
observed cross-dataset dissimilarity is compared with a permutation distribution
obtained by shuffling dataset labels within each side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix

Direction = Literal["in", "out"]
PartnerVocabulary = Literal["auto", "cell_type", "fafb_or_manc"]


def partner_labels(meta: pd.DataFrame, vocabulary: str) -> pd.Series:
    """Partner label per neuron under a vocabulary.

    ``cell_type``: the dataset's own names. ``fafb_or_manc``: the FlyWire (FAFB v783) name
    where a neuron has one, else its MANC v1.2.1 name. MaleCNS and BANC both publish these
    cross-references, so brain partners get FlyWire names and nerve-cord partners MANC names on
    both sides. A cross-reference column that is entirely empty (e.g. FlyWire, whose own names
    are already FAFB names) is replaced by ``cell_type``.
    """
    if vocabulary == "cell_type":
        return cast("pd.Series", meta["cell_type"]).astype(object)
    if vocabulary != "fafb_or_manc":
        msg = f"unknown partner vocabulary {vocabulary!r}"
        raise ValueError(msg)
    out = pd.Series([None] * len(meta), index=meta.index, dtype=object)
    used = False
    for col in ("fafb_783_cell_type", "manc_121_cell_type"):
        if col in meta.columns and bool(meta[col].notna().any()):
            used = True
            vals = cast("pd.Series", meta[col]).astype(object)
            out = out.where(out.notna(), vals)
    if not used:
        return cast("pd.Series", meta["cell_type"]).astype(object)
    return out.where(out.notna(), None)


def _xref_column(a: Store, b: Store) -> str | None:
    """Column of ``a`` that carries ``b``'s type names, if the schema has one."""
    name, version = b.spec.name, b.spec.version.replace(".", "")
    candidates = {
        "flywire": f"fafb_{version}_cell_type",
        "hemibrain": f"hemibrain_{version}_cell_type",
        "manc": f"manc_{version}_cell_type",
        "malecns": f"malecns_{version}_cell_type",
    }
    col = candidates.get(name)
    cols = set(a.neurons(columns=["neuron_id"]).columns) | set(
        a.query("SELECT * FROM neurons LIMIT 0").columns
    )
    return col if col in cols else None


def match_types(a: Store, b: Store, *, source: str | None = None) -> pd.DataFrame:
    """Map ``a``'s cell types to ``b``'s using a cross-reference column (or exact names).

    Comma-separated cross references (e.g. ``"LB1a,LB1d"``) are split; each token
    that exists as a ``cell_type`` in ``b`` becomes one row.
    """
    col = source or _xref_column(a, b) or "cell_type"
    na = a.query(
        f'SELECT cell_type AS type_a, "{col}" AS xref, count(*) AS n_a FROM neurons '
        "WHERE cell_type IS NOT NULL GROUP BY 1, 2"
    )
    nb = b.query(
        "SELECT cell_type AS type_b, count(*) AS n_b FROM neurons "
        "WHERE cell_type IS NOT NULL GROUP BY 1"
    )
    nb_map = dict(zip(nb["type_b"].tolist(), nb["n_b"].tolist(), strict=True))
    rows: list[dict[str, object]] = []
    for type_a, xref, n_a in zip(na["type_a"], na["xref"], na["n_a"], strict=True):
        if xref is None or (isinstance(xref, float) and np.isnan(xref)):
            continue
        for token in str(xref).split(","):
            token = token.strip()
            if token in nb_map:
                rows.append(
                    {
                        "type_a": type_a,
                        "type_b": token,
                        "n_a": int(n_a),
                        "n_b": int(nb_map[token]),
                        "source": col,
                    }
                )
    cols = ["type_a", "type_b", "n_a", "n_b", "source"]
    df = pd.DataFrame({c: [r[c] for r in rows] for c in cols})
    return df.sort_values(["type_a", "type_b"], ignore_index=True)


def _membership(m: ConnectivityMatrix, by: str) -> tuple[np.ndarray, sp.csr_matrix]:
    values = m.meta[by].astype(object).to_numpy()
    labelled = np.array(
        [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in values]
    )
    labels_arr = np.where(labelled, values, "__unmatched__")
    codes, labels = pd.factorize(pd.Series(labels_arr).astype(str), sort=True)
    member = sp.csr_matrix((np.ones(m.n), (np.arange(m.n), codes)), shape=(m.n, len(labels)))
    return np.asarray(labels, dtype=object), member


def _profiles(
    m: ConnectivityMatrix, idx: np.ndarray, *, direction: Direction, by: str
) -> pd.DataFrame:
    """Per-neuron partner-type fraction vectors (rows = neurons in ``idx``)."""
    labels, member = _membership(m, by)
    w = m.weights if direction == "out" else sp.csr_matrix(m.weights.T)
    block = sp.csr_matrix(sp.csr_matrix(w[idx]) @ member)  # k x L
    dense = block.toarray().astype(float)
    totals = dense.sum(axis=1, keepdims=True)
    frac = np.divide(dense, totals, out=np.zeros_like(dense), where=totals > 0)
    return pd.DataFrame(frac, index=m.neuron_ids[idx], columns=labels)


def type_profile(
    m: ConnectivityMatrix,
    type_label: str,
    *,
    direction: Direction = "out",
    by: str = "cell_type",
    side: str | None = None,
    type_column: str = "cell_type",
) -> pd.Series:
    """Mean partner-type fraction vector of all neurons of ``type_label`` (optionally one side)."""
    meta = m.meta
    mask = meta[type_column].astype(object) == type_label
    if side is not None:
        mask &= meta["side"] == side
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        msg = f"no neurons of type {type_label!r}" + (f" on side {side!r}" if side else "")
        raise ValueError(msg)
    prof = cast("pd.Series", _profiles(m, idx, direction=direction, by=by).mean(axis=0))
    prof = cast("pd.Series", prof[prof > 0])
    return prof.sort_values(ascending=False)


SCHLEGEL_BETWEEN_BRAIN_EFFECT = 0.045
"""Schlegel et al. 2024 (Nature 634:139) Fig. 4d: cosine effect size, across vs within brains."""
SCHLEGEL_BETWEEN_BRAIN_SD = 0.096
BETWEEN_BRAIN_RANGE = SCHLEGEL_BETWEEN_BRAIN_EFFECT + SCHLEGEL_BETWEEN_BRAIN_SD


def verdict_for(p_value: float, statistic: float, alpha: float = 0.05) -> str:
    """Grade: no evidence / detectable but within published between-brain range / beyond it."""
    if p_value >= alpha or statistic <= 0:
        return "no evidence of difference beyond left/right variability"
    if statistic <= BETWEEN_BRAIN_RANGE:
        return (
            "detectable beyond left/right variability but within the between-brain range "
            f"reported by Schlegel et al. 2024 ({SCHLEGEL_BETWEEN_BRAIN_EFFECT} +/- "
            f"{SCHLEGEL_BETWEEN_BRAIN_SD}); not evidence of a sex difference on its own"
        )
    return (
        "different beyond left/right variability and beyond the between-brain range of "
        "Schlegel et al. 2024 (model of wiring; see caveats)"
    )


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


@dataclass
class TypeComparison:
    type_a: str
    type_b: str
    direction: Direction
    n_a: int
    n_b: int
    cross_similarity: float
    within_similarity_a: float
    within_similarity_b: float
    statistic: float  # cross dissimilarity minus mean within (L/R) dissimilarity
    null: np.ndarray
    p_value: float
    partner_differences: pd.DataFrame
    unmatched_fraction_a: float = 0.0
    unmatched_fraction_b: float = 0.0
    caveats: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def z(self) -> float:
        sd = float(self.null.std(ddof=1)) if len(self.null) > 1 else 0.0
        return float((self.statistic - self.null.mean()) / sd) if sd > 0 else 0.0

    @property
    def verdict(self) -> str:
        return verdict_for(self.p_value, self.statistic)

    def summary(self) -> dict[str, object]:
        return {
            "type": f"{self.type_a} ~ {self.type_b}",
            "direction": self.direction,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "cross_similarity": round(self.cross_similarity, 4),
            "within_similarity_a": round(self.within_similarity_a, 4),
            "within_similarity_b": round(self.within_similarity_b, 4),
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "z": round(self.z, 2),
            "unmatched_fraction_a": round(self.unmatched_fraction_a, 3),
            "unmatched_fraction_b": round(self.unmatched_fraction_b, 3),
            "verdict": self.verdict,
        }


def _stat(groups: dict[tuple[str, str], np.ndarray]) -> float:
    """Cross dissimilarity minus mean within-dataset left/right dissimilarity."""
    mean = {k: v.mean(axis=0) for k, v in groups.items() if len(v)}
    a = np.vstack([groups[("a", "left")], groups[("a", "right")]]).mean(axis=0)
    b = np.vstack([groups[("b", "left")], groups[("b", "right")]]).mean(axis=0)
    cross = 1 - _cosine(a, b)
    within = []
    for ds in ("a", "b"):
        if (ds, "left") in mean and (ds, "right") in mean:
            within.append(1 - _cosine(mean[(ds, "left")], mean[(ds, "right")]))
    return cross - (float(np.mean(within)) if within else 0.0)


def compare_type(
    a: Store,
    b: Store,
    type_label: str,
    *,
    type_b: str | None = None,
    direction: Direction = "out",
    n_permutations: int = 1000,
    seed: int = 0,
    min_weight: int = 1,
    partner_vocabulary: PartnerVocabulary = "auto",
) -> TypeComparison:
    """Compare ``type_label`` (in ``a``'s vocabulary) between datasets ``a`` and ``b``.

    Profiles are partner-type fractions, with ``a``'s partners named by the
    cross-reference column pointing at ``b`` so both sides share a vocabulary.
    The statistic is cross-dataset cosine dissimilarity minus the mean
    left/right dissimilarity within each dataset; the null permutes dataset
    labels within each side.
    """
    xref = _xref_column(a, b)
    if type_b is None:
        mapping = match_types(a, b)
        hits = mapping[mapping["type_a"] == type_label]
        if hits.empty:
            msg = f"type {type_label!r} has no match in {b.ref}"
            raise KeyError(msg)
        type_b = str(hits.iloc[0]["type_b"])
    ma = ConnectivityMatrix.from_store(a, min_weight=min_weight)
    mb = ConnectivityMatrix.from_store(b, min_weight=min_weight)
    if partner_vocabulary == "auto":
        by_a, by_b = xref or "cell_type", "cell_type"
    else:
        for m in (ma, mb):
            m.meta["_partner_label"] = partner_labels(m.meta, partner_vocabulary).to_numpy()
        by_a = by_b = "_partner_label"
    groups: dict[tuple[str, str], np.ndarray] = {}
    labels_union: list[str] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for key, m, tlabel, by in (("a", ma, type_label, by_a), ("b", mb, type_b, by_b)):
        meta = m.meta
        for side in ("left", "right"):
            mask = (meta["cell_type"].astype(object) == tlabel) & (meta["side"] == side)
            idx = np.flatnonzero(mask.to_numpy())
            frames[(key, side)] = (
                _profiles(m, idx, direction=direction, by=by) if len(idx) else pd.DataFrame()
            )
    for f in frames.values():
        labels_union.extend(c for c in f.columns if c not in labels_union)
    cols = sorted(c for c in labels_union if c != "__unmatched__")
    unmatched: dict[str, list[float]] = {"a": [], "b": []}
    for k, f in frames.items():
        if len(f) and "__unmatched__" in f.columns:
            unmatched[k[0]].extend(f["__unmatched__"].tolist())
        elif len(f):
            unmatched[k[0]].extend([0.0] * len(f))
    for k, f in list(frames.items()):
        if len(f):
            kept = f.reindex(columns=cols, fill_value=0.0)
            totals = kept.sum(axis=1).to_numpy()
            kept = kept.div(np.where(totals > 0, totals, 1.0), axis=0)
            frames[k] = kept
    for k, f in frames.items():
        groups[k] = (
            f.reindex(columns=cols, fill_value=0.0).to_numpy()
            if len(f)
            else np.zeros((0, len(cols)))
        )
    n_a = sum(len(groups[("a", s)]) for s in ("left", "right"))
    n_b = sum(len(groups[("b", s)]) for s in ("left", "right"))
    if n_a == 0 or n_b == 0:
        msg = f"type {type_label!r}/{type_b!r} missing in one dataset"
        raise ValueError(msg)

    observed = _stat(groups)
    mean_a = np.vstack([groups[("a", "left")], groups[("a", "right")]]).mean(axis=0)
    mean_b = np.vstack([groups[("b", "left")], groups[("b", "right")]]).mean(axis=0)

    def within(ds: str) -> float:
        left, right = groups[(ds, "left")], groups[(ds, "right")]
        return (
            _cosine(left.mean(axis=0), right.mean(axis=0))
            if len(left) and len(right)
            else float("nan")
        )

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for k in range(n_permutations):
        perm: dict[tuple[str, str], np.ndarray] = {}
        for side in ("left", "right"):
            pool = np.vstack([groups[("a", side)], groups[("b", side)]])
            na = len(groups[("a", side)])
            order = rng.permutation(len(pool))
            perm[("a", side)] = pool[order[:na]]
            perm[("b", side)] = pool[order[na:]]
        null[k] = _stat(perm)
    p = float((1 + (null >= observed).sum()) / (n_permutations + 1))

    diff = pd.DataFrame({"partner_type": cols, "fraction_a": mean_a, "fraction_b": mean_b})
    diff["difference"] = diff["fraction_a"] - diff["fraction_b"]
    magnitude = cast("pd.Series", diff["difference"]).abs()
    order = np.argsort(-magnitude.to_numpy())
    diff = diff.iloc[order].reset_index(drop=True)

    caveats = [
        "Connectomes are wiring; partner fractions use synapse counts with dataset-specific "
        "detection and thresholds.",
        f"Type matching relies on the published cross-reference column ({by_a}); partner types "
        "without a match are pooled as '__unmatched__'.",
        "Left/right variability within a brain is the null (Schlegel et al. 2024); individual "
        "variability between animals is not separable from sex.",
    ]
    unmatched_a = float(np.mean(unmatched["a"])) if unmatched["a"] else 0.0
    unmatched_b = float(np.mean(unmatched["b"])) if unmatched["b"] else 0.0
    if max(unmatched_a, unmatched_b) > 0.05:
        caveats.append(
            f"Unmatched partners excluded: {unmatched_a:.0%} of {a.ref} and {unmatched_b:.0%} of "
            f"{b.ref} synapses of this type go to neurons without a cross-dataset match "
            "(e.g. outside the other dataset's volume); the comparison covers the rest."
        )
    if min(n_a, n_b) <= 2:
        caveats.append(
            "Very few neurons per type: the permutation test has little resolution; "
            "treat p-values as descriptive."
        )
    prov = {
        "datasets": [a.ref, b.ref],
        "direction": direction,
        "min_weight": min_weight,
        "n_permutations": n_permutations,
        "seed": seed,
        "partner_vocabulary": partner_vocabulary,
        "partner_vocabulary_a": by_a,
        "partner_vocabulary_b": by_b,
        "matrix_a": ma.provenance,
        "matrix_b": mb.provenance,
    }
    return TypeComparison(
        type_a=type_label,
        type_b=type_b,
        direction=direction,
        n_a=n_a,
        n_b=n_b,
        cross_similarity=_cosine(mean_a, mean_b),
        within_similarity_a=within("a"),
        within_similarity_b=within("b"),
        statistic=observed,
        null=null,
        p_value=p,
        partner_differences=diff,
        unmatched_fraction_a=unmatched_a,
        unmatched_fraction_b=unmatched_b,
        caveats=caveats,
        provenance=prov,
    )
