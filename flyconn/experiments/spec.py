"""Declarative experiment specification (YAML) and neuron selection.

A spec names a dataset, a background stimulation, an optional perturbation
(silence or extra stimulation), readout groups, trial settings and the controls
to run. Controls are **on by default**: degree-preserving rewiring and sign
shuffling of the connectome, each simulated with the same seeds as the main
condition.
"""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Selection:
    """Neuron selection by explicit ids or by attribute (glob patterns allowed)."""

    ids: list[int] | None = None
    cell_type: str | None = None
    super_class: str | None = None
    cell_class: str | None = None
    side: str | None = None

    def describe(self) -> str:
        parts = []
        if self.ids is not None:
            parts.append(f"{len(self.ids)} ids")
        for k in ("cell_type", "super_class", "cell_class", "side"):
            v = getattr(self, k)
            if v is not None:
                parts.append(f"{k}={v}")
        return ", ".join(parts) or "nothing"


@dataclass(frozen=True)
class Stimulation:
    select: Selection
    rate_hz: float


@dataclass(frozen=True)
class Silencing:
    select: Selection


@dataclass(frozen=True)
class Perturbation:
    silence: list[Silencing] = field(default_factory=list)
    stimulate: list[Stimulation] = field(default_factory=list)


@dataclass(frozen=True)
class Readout:
    name: str
    select: Selection


@dataclass(frozen=True)
class NetworkOptions:
    min_weight: int = 1
    sign_policy: str = "argmax"


@dataclass(frozen=True)
class Controls:
    degree_preserving_rewire: int = 5
    sign_shuffle: int = 5
    unstimulated_baseline: bool = True


@dataclass(frozen=True)
class ReportOptions:
    top_neurons: int = 30
    alpha: float = 0.05


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset: str
    stimulate: list[Stimulation]
    readouts: list[Readout]
    perturb: Perturbation | None = None
    network: NetworkOptions = field(default_factory=NetworkOptions)
    duration_ms: float = 1000.0
    trials: int = 30
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"
    dt_ms: float = 0.1
    controls: Controls = field(default_factory=Controls)
    report: ReportOptions = field(default_factory=ReportOptions)
    source_path: str | None = None
    source_sha256: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return round(self.duration_ms / self.dt_ms)


def _selection(d: Any) -> Selection:
    if not isinstance(d, dict):
        msg = f"select must be a mapping, got {d!r}"
        raise ValueError(msg)
    dd = cast("dict[str, Any]", d)
    ids = dd.get("ids")
    return Selection(
        ids=[int(i) for i in ids] if ids is not None else None,
        cell_type=dd.get("cell_type"),
        super_class=dd.get("super_class"),
        cell_class=dd.get("cell_class"),
        side=dd.get("side"),
    )


def _stims(items: Any) -> list[Stimulation]:
    out: list[Stimulation] = []
    for it in items or []:
        out.append(Stimulation(_selection(it["select"]), float(it["rate_hz"])))
    return out


def spec_from_dict(d: dict[str, Any]) -> ExperimentSpec:
    """Build and validate an :class:`ExperimentSpec` from a parsed YAML mapping."""
    for key in ("name", "dataset"):
        if key not in d:
            msg = f"experiment spec needs '{key}'"
            raise ValueError(msg)
    readouts = [Readout(str(r["name"]), _selection(r["select"])) for r in d.get("readouts") or []]
    if not readouts:
        msg = "experiment spec needs at least one readout group"
        raise ValueError(msg)
    perturb_d = d.get("perturb")
    perturb = None
    if perturb_d:
        perturb = Perturbation(
            silence=[Silencing(_selection(s["select"])) for s in perturb_d.get("silence") or []],
            stimulate=_stims(perturb_d.get("stimulate")),
        )
    net = d.get("network") or {}
    ctl = d.get("controls") or {}
    rep = d.get("report") or {}
    trials = int(d.get("trials", 30))
    if trials < 1:
        msg = "trials must be >= 1"
        raise ValueError(msg)
    return ExperimentSpec(
        name=str(d["name"]),
        dataset=str(d["dataset"]),
        stimulate=_stims(d.get("stimulate")),
        readouts=readouts,
        perturb=perturb,
        network=NetworkOptions(
            int(net.get("min_weight", 1)), str(net.get("sign_policy", "argmax"))
        ),
        duration_ms=float(d.get("duration_ms", 1000.0)),
        trials=trials,
        seed=int(d.get("seed", 0)),
        device=str(d.get("device", "auto")),
        dtype=str(d.get("dtype", "float32")),
        controls=Controls(
            int(ctl.get("degree_preserving_rewire", 5)),
            int(ctl.get("sign_shuffle", 5)),
            bool(ctl.get("unstimulated_baseline", True)),
        ),
        report=ReportOptions(int(rep.get("top_neurons", 30)), float(rep.get("alpha", 0.05))),
        raw=d,
    )


def load_spec(path: Path | str) -> ExperimentSpec:
    """Load a YAML experiment spec and record its path and SHA-256 for provenance."""
    path = Path(path)
    text = path.read_text()
    d = yaml.safe_load(text) or {}
    spec = spec_from_dict(d)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return replace(spec, source_path=str(path), source_sha256=digest)


def _matches(value: object, pattern: str) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return fnmatch.fnmatchcase(str(value), pattern)


def resolve_selection(sel: Selection, neurons: pd.DataFrame) -> np.ndarray:
    """Neuron ids matching ``sel`` in a harmonized ``neurons`` table (sorted, unique)."""
    if sel.ids is not None:
        ids = np.array(sorted(set(sel.ids)), dtype=np.int64)
        known = set(cast("pd.Series", neurons["neuron_id"]).tolist())
        missing = [int(i) for i in ids if i not in known]
        if missing:
            msg = f"neuron ids not in dataset: {missing[:10]}"
            raise KeyError(msg)
        return ids
    mask = np.ones(len(neurons), dtype=bool)
    for col in ("cell_type", "super_class", "cell_class", "side"):
        pattern = getattr(sel, col)
        if pattern is None:
            continue
        if col not in neurons.columns:
            msg = f"dataset has no column {col!r}"
            raise KeyError(msg)
        values = cast("pd.Series", neurons[col]).astype(object).to_numpy()
        mask &= np.array([_matches(v, pattern) for v in values])
    ids = cast("pd.Series", neurons["neuron_id"]).to_numpy(dtype=np.int64)[mask]
    if len(ids) == 0:
        msg = f"selection is empty: {sel.describe()}"
        raise ValueError(msg)
    return np.sort(ids)
