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
import json
from dataclasses import asdict, dataclass, field, replace
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
class SimParams:
    """Overrides of simulator constants; ``None`` keeps the dataset default (ADR-0009)."""

    w_syn_mv: float | None = None


@dataclass(frozen=True)
class NetworkOptions:
    min_weight: int = 1
    sign_policy: str = "argmax"
    sign_seed: int | None = None


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
class PathwayRanking:
    """Rank silencing candidates by connectome paths from the stimulated neurons to a readout.

    A candidate's score is the summed strength (product of per-hop input fractions) of all
    simple paths of up to ``max_hops`` synapses from any stimulated neuron to any neuron of
    the readout group ``readout`` that pass *through* it. Stimulated neurons and readout
    neurons themselves are never candidates.
    """

    readout: str
    max_hops: int = 3
    min_weight: int = 5
    min_edge_fraction: float = 0.0


@dataclass(frozen=True)
class SilenceEachSweep:
    """Silence each candidate group one at a time (a silencing screen).

    Candidates are the neurons matched by ``select`` (all neurons when omitted), grouped by
    ``group_by`` (a neuron column such as ``cell_type``, or ``"neuron"`` for one neuron per
    variant). ``rank_by``:

    * ``activity`` (default): groups are ranked by their summed firing rate in the shared
      *stimulated* condition; groups that never fire are reported but not simulated.
    * ``pathway``: groups are ranked by the summed strength of connectome paths from the
      stimulated neurons to the ``pathway.readout`` group running through them (see
      :class:`PathwayRanking`); groups on no such path are reported but not simulated.

    The ``max_items`` top-ranked groups are simulated.
    """

    select: Selection = field(default_factory=Selection)
    group_by: str = "cell_type"
    max_items: int = 50
    rank_by: str = "activity"
    pathway: PathwayRanking | None = None

    kind: str = "silence_each"


@dataclass(frozen=True)
class RateSweep:
    """Re-run the stimulated condition with every ``stimulate`` entry set to each rate."""

    rates_hz: list[float]

    kind: str = "rate_hz"


@dataclass(frozen=True)
class RobustnessOptions:
    primary_readout: str
    readout_proxy: str | None = None
    w_syn: str | tuple[float, ...] = "auto"  # "auto" | "off" | explicit mV values
    nt_samples: int = 8
    min_weights: tuple[int, ...] = (3, 5)
    replicate: str | None = "flywire@783"
    random_null_sets: int = 40
    extra_candidates: tuple[Selection, ...] = ()


Sweep = SilenceEachSweep | RateSweep


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
    sweep: Sweep | None = None
    params: SimParams = field(default_factory=SimParams)
    robustness: RobustnessOptions | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    parent_sha256: str | None = None
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


_SWEEP_KINDS = ("silence_each", "rate_hz")
_RANK_BY = ("activity", "pathway")


def _sweep(d: Any) -> Sweep | None:
    if d is None:
        return None
    if not isinstance(d, dict) or not d:
        msg = f"sweep must be a mapping with exactly one of {_SWEEP_KINDS}, got {d!r}"
        raise ValueError(msg)
    dd = cast("dict[str, Any]", d)
    unknown = [k for k in dd if k not in _SWEEP_KINDS]
    if unknown:
        msg = f"unknown sweep kind(s) {unknown}; expected one of {_SWEEP_KINDS}"
        raise ValueError(msg)
    if len(dd) != 1:
        msg = f"sweep needs exactly one of {_SWEEP_KINDS}, got {sorted(dd)}"
        raise ValueError(msg)
    if "rate_hz" in dd:
        rates_raw = dd["rate_hz"]
        if not isinstance(rates_raw, list) or not rates_raw:
            msg = "sweep rate_hz must be a non-empty list of rates"
            raise ValueError(msg)
        rates = [float(r) for r in cast("list[Any]", rates_raw)]
        if any(r <= 0 for r in rates) or len(set(rates)) != len(rates):
            msg = f"sweep rate_hz values must be positive and distinct, got {rates}"
            raise ValueError(msg)
        return RateSweep(rates_hz=rates)
    se = dd["silence_each"]
    if not isinstance(se, dict):
        msg = "sweep silence_each must be a mapping"
        raise ValueError(msg)
    sed = cast("dict[str, Any]", se)
    rank_by = str(sed.get("rank_by", "activity"))
    if rank_by not in _RANK_BY:
        msg = f"sweep silence_each rank_by must be one of {_RANK_BY}, got {rank_by!r}"
        raise ValueError(msg)
    if rank_by == "activity" and "select" not in sed:
        msg = "sweep silence_each needs a 'select' mapping (optional only with rank_by: pathway)"
        raise ValueError(msg)
    max_items = int(sed.get("max_items", 50))
    if max_items < 1:
        msg = "sweep silence_each max_items must be >= 1"
        raise ValueError(msg)
    pathway: PathwayRanking | None = None
    if rank_by == "pathway":
        pd_raw = sed.get("pathway")
        if not isinstance(pd_raw, dict) or "readout" not in pd_raw:
            msg = "rank_by: pathway needs a 'pathway' mapping with a 'readout' group name"
            raise ValueError(msg)
        pw = cast("dict[str, Any]", pd_raw)
        max_hops = int(pw.get("max_hops", 3))
        if not 2 <= max_hops <= 4:
            msg = f"pathway max_hops must be 2..4 (a relay needs at least 2 hops), got {max_hops}"
            raise ValueError(msg)
        pathway = PathwayRanking(
            readout=str(pw["readout"]),
            max_hops=max_hops,
            min_weight=int(pw.get("min_weight", 5)),
            min_edge_fraction=float(pw.get("min_edge_fraction", 0.0)),
        )
    return SilenceEachSweep(
        select=_selection(sed["select"]) if "select" in sed else Selection(),
        group_by=str(sed.get("group_by", "cell_type")),
        max_items=max_items,
        rank_by=rank_by,
        pathway=pathway,
    )


_ROBUSTNESS_KEYS = (
    "primary_readout",
    "readout_proxy",
    "w_syn",
    "nt_samples",
    "min_weights",
    "replicate",
    "random_null",
    "extra_candidates",
)


def _robustness(
    d: Any, *, dataset: str, sweep: Sweep | None, readout_names: list[str]
) -> RobustnessOptions | None:
    if d is None:
        return None
    if not isinstance(d, dict):
        msg = "robustness must be a mapping"
        raise ValueError(msg)
    dd = cast("dict[str, Any]", d)
    unknown = sorted(set(dd) - set(_ROBUSTNESS_KEYS))
    if unknown:
        msg = f"unknown robustness key(s) {unknown}; expected some of {list(_ROBUSTNESS_KEYS)}"
        raise ValueError(msg)
    if not isinstance(sweep, SilenceEachSweep):
        msg = "robustness needs a silence_each sweep"
        raise ValueError(msg)
    primary = dd.get("primary_readout")
    if primary not in readout_names:
        msg = f"robustness primary_readout must name a readout group; have {readout_names}"
        raise ValueError(msg)
    w_raw = dd.get("w_syn", "auto")
    w_syn: str | tuple[float, ...]
    if w_raw in ("auto", "off"):
        w_syn = str(w_raw)
    elif (
        isinstance(w_raw, list)
        and w_raw
        and all(isinstance(x, int | float) and x > 0 for x in cast("list[Any]", w_raw))
    ):
        w_syn = tuple(float(x) for x in cast("list[Any]", w_raw))
    else:
        msg = (
            f"robustness w_syn must be 'auto', 'off' or a list of positive mV values, got {w_raw!r}"
        )
        raise ValueError(msg)
    nt_samples = int(dd.get("nt_samples", 8))
    if nt_samples < 0:
        msg = "robustness nt_samples must be >= 0"
        raise ValueError(msg)
    mw = [int(x) for x in cast("list[Any]", dd.get("min_weights", [3, 5]) or [])]
    if any(x < 1 for x in mw) or len(set(mw)) != len(mw):
        msg = f"robustness min_weights must be distinct ints >= 1, got {mw}"
        raise ValueError(msg)
    replicate = dd.get("replicate", "flywire@783")
    if replicate is not None and str(replicate) == dataset:
        msg = f"robustness replicate must differ from the dataset ({dataset})"
        raise ValueError(msg)
    null_raw = dd.get("random_null")
    if null_raw is None or null_raw is True:
        n_sets = 40
    elif null_raw is False:
        n_sets = 0  # `random_null: false` disables the null
    elif isinstance(null_raw, int):
        n_sets = null_raw  # `random_null: 0` disables it; a count sets n_sets
    elif isinstance(null_raw, dict):
        n_sets = int(cast("dict[str, Any]", null_raw).get("n_sets", 40))
    else:
        msg = (
            "robustness random_null must be true/false, a set count or {n_sets: N}, "
            f"got {null_raw!r}"
        )
        raise ValueError(msg)
    if n_sets < 0:
        msg = "robustness random_null n_sets must be >= 0"
        raise ValueError(msg)
    extras_raw = dd.get("extra_candidates") or []
    if not isinstance(extras_raw, list) or not all(
        isinstance(x, dict) for x in cast("list[Any]", extras_raw)
    ):
        msg = "robustness extra_candidates must be a list of selection mappings"
        raise ValueError(msg)
    proxy = dd.get("readout_proxy")
    return RobustnessOptions(
        primary_readout=str(primary),
        readout_proxy=str(proxy) if proxy is not None else None,
        w_syn=w_syn,
        nt_samples=nt_samples,
        min_weights=tuple(mw),
        replicate=str(replicate) if replicate is not None else None,
        random_null_sets=n_sets,
        extra_candidates=tuple(_selection(x) for x in cast("list[Any]", extras_raw)),
    )


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
    sweep = _sweep(d.get("sweep"))
    if isinstance(sweep, SilenceEachSweep) and sweep.pathway is not None:
        names = [r.name for r in readouts]
        if sweep.pathway.readout not in names:
            msg = f"pathway readout {sweep.pathway.readout!r} is not a readout group; have {names}"
            raise ValueError(msg)
    stimulate = _stims(d.get("stimulate"))
    if sweep is not None and perturb is not None:
        msg = "a sweep spec cannot also have 'perturb'; the sweep defines the perturbations"
        raise ValueError(msg)
    if isinstance(sweep, RateSweep) and not stimulate:
        msg = "a rate_hz sweep needs at least one 'stimulate' entry whose rate it varies"
        raise ValueError(msg)
    net = d.get("network") or {}
    ctl = d.get("controls") or {}
    rep = d.get("report") or {}
    trials = int(d.get("trials", 30))
    if trials < 1:
        msg = "trials must be >= 1"
        raise ValueError(msg)
    sign_policy = str(net.get("sign_policy", "argmax"))
    sign_seed_raw = net.get("sign_seed")
    if sign_policy == "sampled" and sign_seed_raw is None:
        msg = "sign_policy: sampled needs network.sign_seed (an int)"
        raise ValueError(msg)
    if sign_seed_raw is not None and sign_policy != "sampled":
        msg = "network.sign_seed is only valid with sign_policy: sampled"
        raise ValueError(msg)
    params_d = cast("dict[str, Any]", d.get("params") or {})
    unknown_params = sorted(set(params_d) - {"w_syn_mv"})
    if unknown_params:
        msg = f"unknown params {unknown_params}; supported: ['w_syn_mv']"
        raise ValueError(msg)
    w_raw = params_d.get("w_syn_mv")
    if w_raw is not None and float(w_raw) <= 0:
        msg = f"params.w_syn_mv must be > 0, got {w_raw}"
        raise ValueError(msg)
    robustness = _robustness(
        d.get("robustness"),
        dataset=str(d["dataset"]),
        sweep=sweep,
        readout_names=[r.name for r in readouts],
    )
    return ExperimentSpec(
        name=str(d["name"]),
        dataset=str(d["dataset"]),
        stimulate=stimulate,
        readouts=readouts,
        perturb=perturb,
        network=NetworkOptions(
            int(net.get("min_weight", 1)),
            sign_policy,
            int(sign_seed_raw) if sign_seed_raw is not None else None,
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
        sweep=sweep,
        params=SimParams(float(w_raw) if w_raw is not None else None),
        robustness=robustness,
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


NO_CONTROLS = Controls(0, 0, False)
_PROVENANCE_FIELDS = ("raw", "source_path", "source_sha256", "parent_sha256")


def canonical_dict(spec: ExperimentSpec) -> dict[str, Any]:
    """JSON-safe dict of every field that affects a run (provenance fields excluded)."""
    d = asdict(spec)
    for k in _PROVENANCE_FIELDS:
        d.pop(k, None)
    return cast("dict[str, Any]", json.loads(json.dumps(d, default=str)))


def derive(spec: ExperimentSpec, *, parent_sha256: str | None, **changes: Any) -> ExperimentSpec:
    """A changed copy whose ``raw`` and ``source_sha256`` describe the copy, not the parent."""
    s = replace(spec, **changes)
    raw = canonical_dict(s)
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return replace(s, raw=raw, source_path=None, source_sha256=digest, parent_sha256=parent_sha256)


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
