# M13 Robustness-Audited Silencing Screens Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn an M10 `silence_each` screen into a ranked, robustness-graded hit list with
split-GAL4 lines and a suggested experiment per hit (`report.html`, `handoff.csv`).

**Architecture:**
- A `robustness:` block in the YAML spec drives a new orchestrator,
  `flyconn/experiments/robustness.py`.
- The orchestrator runs the existing `run_sweep` once per variant, one axis at a time:
  w_syn, sampled NT signs, network threshold, and a FlyWire replicate. Every variant
  silences the reference run's fixed candidate list.
- It also runs a size- and super_class-matched random-silencing null.
- Pure modules do the scoring: per-variant labels, then axis verdicts, then tiers.
- `flyconn/access/handoff.py` attaches Meissner 2025 driver lines.
- A new report module renders the evidence cards.

**Tech Stack:** Python >= 3.11, numpy, pandas, pyarrow, torch (existing engine), pytest +
hypothesis, ruff, pyright strict, uv.

**Spec:** `docs/superpowers/specs/2026-10-10-robust-screens-design.md` (read it first; this
plan argues from it).

## Global Constraints

- **Location:** work only in the worktree `/Users/moizp/google-fly-m13`, branch
  `m13-robust-screens`.
  - Never `cd` into or switch branches in `/Users/moizp/google-fly`; another session (M12)
    uses that checkout.
  - All commands below run from `/Users/moizp/google-fly-m13`.
- **Numbering:** this milestone is **M13**. ADR is **ADR-0011**. GOLDEN_RESULTS section is
  **6j** (M12 owns ADR-0010 and 6i).
- **Before every commit:** run `scripts/check.sh` (ruff, format, pyright, unit tests).
  - Before committing any new `.md` under `docs/`, run `uv run ruff format docs/`; it
    formats Python blocks inside Markdown.
- **Unit tests never touch the network or real data.** Use the `small_store` fixture
  (`tests/unit/experiments/conftest.py`, dataset name `malecns@1.0`, 40 neurons).
- **Golden tests need the real cache:**
  `FLYCONN_CACHE=/Users/moizp/google-fly/.cache/flyconn uv run pytest -m golden ...`. The
  worktree has no `.cache`.
- **Labelling:** results are model predictions. MaleCNS w_syn is "calibrated by protocol,
  not validated" (ADR-0009). The replicate runs at an uncalibrated w_syn and is never
  graded.
- **Defaults** (verbatim from the spec):
  - `nt_samples: 8`, NT robust threshold `ceil(0.75 x nt_samples)` (6 of 8);
  - `min_weights: [3, 5]` (network.min_weight);
  - `replicate: flywire@783`;
  - `random_null: {n_sets: 40}`;
  - `w_syn: auto` = distinct values of `ci_mv` together with `sensitivity_mv`, excluding
    the reference (MaleCNS: 0.185, 0.192, 0.209).
- **Random null:**
  - sign seeds `spec.seed + 2000 + k` (NT variants); null-set seeds `spec.seed + 3000 + i`;
  - empirical p = `(1 + #null effects at least as extreme in the hit's direction) /
    (n_sets + 1)`, robust when `< 0.05`;
  - excluded from random sets: candidate (and extra-candidate) neurons, stimulus neurons
    and all readout neurons.
- **Variants:** controls are `{degree_preserving_rewire: 0, sign_shuffle: 0,
  unstimulated_baseline: false}`; only the reference keeps the spec's controls.
- **Output directories:** never delete them. Incomplete variant directories move to
  `variants/.incomplete/<label>-<n>/`.
- **Commits:** conventional commits, each ending with the line
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Subagents:** at most 3 in parallel.

## Review Focus

1. **Silent NT fallback.** A store without NT probabilities with `sign_policy: sampled`
   must raise, never silently use argmax signs. Pinned in Task 1.
2. **Interrupted run, then rerun.** It must resume: finished variants are skipped and the
   half-written directory is moved, never deleted, and never refused. Pinned in Task 8.
3. **Extra candidate already selected.** An extra candidate whose label is already a
   ranked candidate must give a clear error, not a duplicated variant. Pinned in Task 3.
4. **Group too large for its pool.** If a candidate group is bigger than the remaining
   pool of its super_class, the null for that key is recorded as skipped with a reason,
   and the random-null axis for that hit is `not_run`, not a crash. Pinned in Task 6.
5. **No reference hits.** The run completes, `handoff.csv` has a header and no rows, and
   the report says "no reference hits". Pinned in Task 10.

---

### Task 1: Simulator parameter override and sampled-sign seed

**Files:**
- Modify: `flyconn/experiments/spec.py` (new `SimParams`; `NetworkOptions.sign_seed`;
  `ExperimentSpec.params`; parsing in `spec_from_dict`)
- Modify: `flyconn/experiments/runner.py` (`sim_params`; `prepare`; `shared_runs` null
  networks)
- Modify: `flyconn/graph/signs.py` (public `has_nt_probabilities`)
- Test: `tests/unit/experiments/test_spec_params.py`

**Interfaces:**
- Produces:
  - `SimParams(w_syn_mv: float | None = None)`
  - `NetworkOptions(min_weight: int = 1, sign_policy: str = "argmax", sign_seed: int | None = None)`
  - `ExperimentSpec.params: SimParams`
  - `runner.sim_params(spec: ExperimentSpec) -> ShiuParams`
  - `signs.has_nt_probabilities(neurons: pd.DataFrame) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
"""Spec-level simulator overrides: w_syn and the sampled-sign seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store
from flyconn.experiments.runner import prepare, sim_params, simulate_condition
from flyconn.experiments.spec import spec_from_dict


def _d(store: Store, **over: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    d: dict[str, object] = {
        "name": "p",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "duration_ms": 60,
        "trials": 3,
        "seed": 1,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
    }
    d.update(over)
    return d


def test_parse_params_and_sign_seed(small_store: Store):
    spec = spec_from_dict(
        _d(
            small_store,
            params={"w_syn_mv": 0.3},
            network={"sign_policy": "sampled", "sign_seed": 7},
        )
    )
    assert spec.params.w_syn_mv == 0.3
    assert spec.network.sign_seed == 7


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"params": {"w_syn_mv": 0}}, "w_syn_mv"),
        ({"params": {"tau": 1}}, "unknown params"),
        ({"network": {"sign_policy": "sampled"}}, "sign_seed"),
        ({"network": {"sign_policy": "argmax", "sign_seed": 3}}, "only valid"),
    ],
)
def test_param_validation(small_store: Store, over: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        spec_from_dict(_d(small_store, **over))


def test_w_syn_override_reaches_the_network(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store, params={"w_syn_mv": 0.6}))
    assert sim_params(spec).w_syn_mv == 0.6
    ctx = prepare(spec, tmp_path / "a", small_store)
    assert ctx.net.params.w_syn_mv == 0.6
    base = spec_from_dict(_d(small_store))
    ctx0 = prepare(base, tmp_path / "b", small_store)
    assert ctx0.net.params.w_syn_mv == pytest.approx(0.188)  # MaleCNS calibration default
    a = simulate_condition(ctx.net, spec, ctx.background, [])
    b = simulate_condition(ctx0.net, base, ctx0.background, [])
    assert not np.array_equal(a, b)


def test_sampled_signs_are_reproducible(small_store: Store, tmp_path: Path):
    def signed(seed: int, out: str) -> np.ndarray:
        spec = spec_from_dict(
            _d(small_store, network={"sign_policy": "sampled", "sign_seed": seed})
        )
        return prepare(spec, tmp_path / out, small_store).matrix.signed_weights.toarray()

    assert np.array_equal(signed(5, "a"), signed(5, "b"))
    assert not np.array_equal(signed(5, "c"), signed(6, "d"))


def test_sampled_without_probabilities_raises(small_store: Store, tmp_path: Path):
    neurons = small_store.neurons()
    for c in [c for c in neurons.columns if c.startswith("nt_p_")]:
        neurons[c] = np.nan
    d = tmp_path / "noprob"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(small_store.edges()), d / "edges.parquet")
    write_provenance(
        d,
        dataset="malecns",
        version="1.0",
        converter="test",
        inputs=[],
        counts={"neurons": len(neurons), "edges": 200},
    )
    store = Store(d)
    spec = spec_from_dict(_d(store, network={"sign_policy": "sampled", "sign_seed": 1}))
    with pytest.raises(ValueError, match="nt-probs"):
        prepare(spec, tmp_path / "x", store)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_spec_params.py -v`
Expected: FAIL (`ImportError: cannot import name 'sim_params'`).

- [ ] **Step 3: Implement**

In `flyconn/graph/signs.py`, add below `_prob_matrix`:

```python
def has_nt_probabilities(neurons: pd.DataFrame) -> bool:
    """True when ``neurons`` carries usable per-neuron NT probabilities (``nt_p_*``)."""
    return _prob_matrix(neurons) is not None
```

In `flyconn/experiments/spec.py`:
- Add after `Readout`:

```python
@dataclass(frozen=True)
class SimParams:
    """Overrides of simulator constants; ``None`` keeps the dataset default (ADR-0009)."""

    w_syn_mv: float | None = None
```

- Change `NetworkOptions` to:

```python
@dataclass(frozen=True)
class NetworkOptions:
    min_weight: int = 1
    sign_policy: str = "argmax"
    sign_seed: int | None = None
```

- Add the field `params: SimParams = field(default_factory=SimParams)` to `ExperimentSpec`,
  directly after `sweep`.
- In `spec_from_dict`, replace the `network=NetworkOptions(...)` argument by building it
  before the `return`:

```python
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
```

  and pass:

```python
network = (
    NetworkOptions(
        int(net.get("min_weight", 1)),
        sign_policy,
        int(sign_seed_raw) if sign_seed_raw is not None else None,
    ),
)
params = (SimParams(float(w_raw) if w_raw is not None else None),)
```

In `flyconn/experiments/runner.py`:
- Imports: add `from dataclasses import dataclass, field, replace`;
  `from flyconn.graph.signs import SignPolicy, has_nt_probabilities`;
  `from flyconn.sim.params import ShiuParams, default_params`.
- Add above `prepare`:

```python
def sim_params(spec: ExperimentSpec) -> ShiuParams:
    """Dataset default parameters (with protocol calibration), plus the spec's overrides."""
    base = default_params(spec.dataset)
    w = spec.params.w_syn_mv
    return base if w is None else replace(base, w_syn_mv=w)
```

- In `prepare`, replace the matrix and net construction:

```python
    neurons = store.neurons()
    policy = SignPolicy(spec.network.sign_policy)
    if policy is SignPolicy.SAMPLED and not has_nt_probabilities(neurons):
        msg = (
            f"sign_policy: sampled needs per-neuron NT probabilities, which {store.ref} lacks "
            "(without them signs would silently fall back to argmax); for MaleCNS run "
            "`flyconn data pull malecns@1.0 --level nt-probs`"
        )
        raise ValueError(msg)
    m = ConnectivityMatrix.from_store(
        store,
        min_weight=spec.network.min_weight,
        sign_policy=policy,
        seed=spec.network.sign_seed,
    )
```

  and `net=LIFNetwork.from_matrix(m, sim_params(spec)),` in the `Context(...)` call.
- In `shared_runs`, pass the same params to both null networks:
  `LIFNetwork.from_matrix(degree_preserving_rewire(ctx.matrix, seed=spec.seed + 1000 + k), ctx.net.params)`
  and
  `LIFNetwork.from_matrix(shuffle_signs(ctx.matrix, seed=spec.seed + 2000 + k), ctx.net.params)`.
- In `base_provenance`, add `"params": {"w_syn_mv": ctx.net.params.w_syn_mv},` and
  `"sign_seed": spec.network.sign_seed,` after `"network"`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments -v`
Expected: PASS (new tests and all existing experiment tests).

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/spec.py flyconn/experiments/runner.py flyconn/graph/signs.py tests/unit/experiments/test_spec_params.py
git commit -m "feat(experiments): w_syn override and sampled-sign seed in specs; refuse sampled signs without NT probabilities

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `robustness:` block parsing

**Files:**
- Modify: `flyconn/experiments/spec.py` (`RobustnessOptions`, `_robustness`, field on
  `ExperimentSpec`)
- Test: `tests/unit/experiments/test_spec_robustness.py`

**Interfaces:**
- Consumes: `Selection`, `_selection` (spec.py).
- Produces:

```python
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
```

`ExperimentSpec.robustness: RobustnessOptions | None = None`

- [ ] **Step 1: Write the failing tests**

```python
"""Parsing and validation of the robustness: block (M13, ADR-0011)."""

from __future__ import annotations

import pytest

from flyconn.data.store import Store
from flyconn.experiments.spec import RobustnessOptions, spec_from_dict


def _d(store: Store, robustness: object, sweep: object | None = None) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "r",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": sweep
        if sweep is not None
        else {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 3}},
        "robustness": robustness,
    }


def test_defaults(small_store: Store):
    spec = spec_from_dict(_d(small_store, {"primary_readout": "targets"}))
    r = spec.robustness
    assert isinstance(r, RobustnessOptions)
    assert r.w_syn == "auto" and r.nt_samples == 8 and r.min_weights == (3, 5)
    assert r.replicate == "flywire@783" and r.random_null_sets == 40
    assert r.extra_candidates == () and r.readout_proxy is None


def test_full_block(small_store: Store):
    spec = spec_from_dict(
        _d(
            small_store,
            {
                "primary_readout": "targets",
                "readout_proxy": "X drives Y (Doe 2020)",
                "w_syn": [0.2, 0.3],
                "nt_samples": 0,
                "min_weights": [],
                "replicate": None,
                "random_null": {"n_sets": 0},
                "extra_candidates": [{"cell_type": "T001"}],
            },
        )
    )
    r = spec.robustness
    assert r is not None
    assert r.w_syn == (0.2, 0.3) and r.nt_samples == 0 and r.min_weights == ()
    assert r.replicate is None and r.random_null_sets == 0
    assert r.extra_candidates[0].cell_type == "T001"


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({}, "primary_readout"),
        ({"primary_readout": "nope"}, "primary_readout"),
        ({"primary_readout": "targets", "bogus": 1}, "unknown robustness"),
        ({"primary_readout": "targets", "w_syn": "maybe"}, "w_syn"),
        ({"primary_readout": "targets", "w_syn": [0]}, "w_syn"),
        ({"primary_readout": "targets", "nt_samples": -1}, "nt_samples"),
        ({"primary_readout": "targets", "min_weights": [0]}, "min_weights"),
        ({"primary_readout": "targets", "min_weights": [3, 3]}, "min_weights"),
        ({"primary_readout": "targets", "random_null": {"n_sets": -1}}, "n_sets"),
        ({"primary_readout": "targets", "replicate": "malecns@1.0"}, "replicate"),
        ({"primary_readout": "targets", "extra_candidates": [3]}, "extra_candidates"),
    ],
)
def test_validation(small_store: Store, block: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        spec_from_dict(_d(small_store, block))


def test_needs_silence_each(small_store: Store):
    with pytest.raises(ValueError, match="silence_each"):
        spec_from_dict(_d(small_store, {"primary_readout": "targets"}, sweep={"rate_hz": [10]}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_spec_robustness.py -v`
Expected: FAIL (`ImportError: cannot import name 'RobustnessOptions'`).

- [ ] **Step 3: Implement**

In `flyconn/experiments/spec.py`, add the `RobustnessOptions` dataclass shown under
Interfaces (after `RateSweep`). Add the field `robustness: RobustnessOptions | None = None`
to `ExperimentSpec` after `params`. Add the parser:

```python
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
    null_d = cast("dict[str, Any]", dd.get("random_null") or {})
    n_sets = int(null_d.get("n_sets", 40))
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
```

In `spec_from_dict`, before the `return`, add:

```python
    robustness = _robustness(
        d.get("robustness"),
        dataset=str(d["dataset"]),
        sweep=sweep,
        readout_names=[r.name for r in readouts],
    )
```

and pass `robustness=robustness,` to `ExperimentSpec(...)`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/spec.py tests/unit/experiments/test_spec_robustness.py
git commit -m "feat(experiments): robustness block in experiment specs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fixed candidates, extra candidates and natural selection in `run_sweep`

**Files:**
- Modify: `flyconn/experiments/sweep.py`
- Test: `tests/unit/experiments/test_sweep_fixed.py`

**Interfaces:**
- Consumes: `RobustnessOptions.extra_candidates` (Task 2).
- Produces:

```python
@dataclass(frozen=True)
class Candidate:
    label: str
    neuron_ids: tuple[int, ...]
    role: str = "candidate"   # "candidate" | "control" | "null"

def run_sweep(
    spec: ExperimentSpec,
    *,
    out_dir: Path | str,
    store: Store | None = None,
    fixed: Sequence[Candidate] | None = None,
    compute_natural: bool = True,
) -> SweepResult
```

- `SweepResult` gains `candidates: list[Candidate]` (what was simulated, in order) and
  `natural_selection: list[str]` (labels the spec's own ranking would select; empty when
  `compute_natural=False` or for rate sweeps).
- Sweep table rows gain a `role` column.
- Extra candidates (`spec.robustness.extra_candidates`) are appended as `role="control"`
  when `fixed is None`.

- [ ] **Step 1: Write the failing tests**

```python
"""run_sweep with a fixed candidate table, extra (control) candidates and natural selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyconn.data.store import Store
from flyconn.experiments.spec import spec_from_dict
from flyconn.experiments.sweep import Candidate, run_sweep


def _d(store: Store, **over: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    d: dict[str, object] = {
        "name": "fx",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 2}},
        "duration_ms": 60,
        "trials": 3,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
    }
    d.update(over)
    return d


def test_natural_selection_recorded(small_store: Store, tmp_path: Path):
    res = run_sweep(spec_from_dict(_d(small_store)), out_dir=tmp_path / "a", store=small_store)
    assert len(res.natural_selection) == 2
    assert [c.label for c in res.candidates] == res.natural_selection
    assert set(res.sweep["role"]) == {"candidate"}


def test_fixed_candidates_are_simulated_as_given(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    natural = run_sweep(spec, out_dir=tmp_path / "a", store=small_store).natural_selection
    ids = small_store.neurons()["neuron_id"].tolist()
    fixed = [Candidate("mine", (ids[20], ids[21]))]
    res = run_sweep(spec, out_dir=tmp_path / "b", store=small_store, fixed=fixed)
    assert set(res.sweep["variant"]) == {"mine"}
    assert res.natural_selection == natural
    res2 = run_sweep(
        spec, out_dir=tmp_path / "c", store=small_store, fixed=fixed, compute_natural=False
    )
    assert res2.natural_selection == []


def test_extra_candidates_are_controls(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    natural = run_sweep(spec, out_dir=tmp_path / "a", store=small_store).natural_selection
    types = sorted(set(small_store.neurons()["cell_type"]) - set(natural))
    spec2 = spec_from_dict(
        _d(
            small_store,
            robustness={
                "primary_readout": "targets",
                "extra_candidates": [{"cell_type": types[0]}],
            },
        )
    )
    res = run_sweep(spec2, out_dir=tmp_path / "b", store=small_store)
    roles = dict(zip(res.sweep["variant"], res.sweep["role"], strict=True))
    assert roles[types[0]] == "control"
    assert all(roles[n] == "candidate" for n in natural)


def test_extra_candidate_colliding_with_ranked_one_raises(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    natural = run_sweep(spec, out_dir=tmp_path / "a", store=small_store).natural_selection
    spec2 = spec_from_dict(
        _d(
            small_store,
            robustness={
                "primary_readout": "targets",
                "extra_candidates": [{"cell_type": natural[0]}],
            },
        )
    )
    with pytest.raises(ValueError, match="already a ranked candidate"):
        run_sweep(spec2, out_dir=tmp_path / "b", store=small_store)


def test_fixed_not_allowed_for_rate_sweeps(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store, sweep={"rate_hz": [10, 50]}))
    with pytest.raises(ValueError, match="silence_each"):
        run_sweep(spec, out_dir=tmp_path / "a", store=small_store, fixed=[Candidate("x", (1,))])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_sweep_fixed.py -v`
Expected: FAIL (`ImportError: cannot import name 'Candidate'`).

- [ ] **Step 3: Implement**

In `flyconn/experiments/sweep.py`:
- Imports: add `from collections.abc import Sequence`; add `Selection` to the spec imports.
- Add after `_Variant`:

```python
@dataclass(frozen=True)
class Candidate:
    """One silencing group: its label, its neuron ids and its role in the screen."""

    label: str
    neuron_ids: tuple[int, ...]
    role: str = "candidate"


def _extra_candidates(
    ctx: Context, sweep: SilenceEachSweep, extra: Sequence[Selection], taken: set[str]
) -> list[Candidate]:
    """Extra selections grouped like the sweep, always simulated, labelled ``control``."""
    out: list[Candidate] = []
    for sel in extra:
        labels = _group_labels(ctx, sweep, resolve_selection(sel, ctx.neurons))
        for lbl, grp in labels.groupby(labels, sort=True):
            name = str(lbl)
            if name in taken:
                msg = f"extra candidate {name!r} is already a ranked candidate"
                raise ValueError(msg)
            taken.add(name)
            out.append(Candidate(name, tuple(int(i) for i in grp.index), "control"))
    return out
```

- Add to `SweepResult` (after `provenance`):
  `candidates: list[Candidate] = field(default_factory=list)` and
  `natural_selection: list[str] = field(default_factory=list)`.
- Change the `run_sweep` signature to the one under Interfaces. Replace the block
  `if isinstance(sweep, SilenceEachSweep): ... else: ...` with:

```python
    natural: list[str] = []
    candidates: list[Candidate] = []
    if isinstance(sweep, SilenceEachSweep):
        if compute_natural:
            natural_df, natural_variants = _silence_candidates(ctx, sweep, stim_counts)
            natural = [v.label for v in natural_variants]
        else:
            natural_df, natural_variants = pd.DataFrame({"variant": pd.Series(dtype=str)}), []
        if fixed is None:
            candidates = [Candidate(v.label, tuple(v.silence)) for v in natural_variants]
            extra = spec.robustness.extra_candidates if spec.robustness is not None else ()
            candidates += _extra_candidates(ctx, sweep, extra, set(natural))
        else:
            candidates = list(fixed)
        variants = [
            _Variant(
                label=c.label,
                stimulate=dict(ctx.background),
                silence=list(c.neuron_ids),
                info={"n_silenced": len(c.neuron_ids), "role": c.role},
            )
            for c in candidates
        ]
        variants_df = _variants_table(natural_df, candidates, fixed=fixed is not None)
        reference = "stimulated"
    else:
        if fixed is not None:
            msg = "fixed candidates are only valid for silence_each sweeps"
            raise ValueError(msg)
        variants_df, variants = _rate_variants(ctx, sweep)
        for v in variants:
            v.info["role"] = "candidate"
        reference = "baseline" if ("baseline", "none") in by_key else "stimulated"
```

  `_silence_candidates` is unchanged.
- Add the helper that builds `variants.parquet`. Without `fixed`, it keeps the natural
  table, including unsimulated rows with their `reason`. Existing tests rely on this
  (`test_sweep.py:161-166`, `:441-449`). Controls are appended to that table. With
  `fixed`, the table lists exactly the fixed candidates, and ranking columns are merged in
  where labels match.

```python
def _variants_table(
    natural_df: pd.DataFrame, candidates: list[Candidate], *, fixed: bool
) -> pd.DataFrame:
    controls = [c for c in candidates if c.role != "candidate"] if not fixed else candidates
    extra = pd.DataFrame(
        {
            "variant": [c.label for c in controls],
            "role": [c.role for c in controls],
            "n_neurons": [len(c.neuron_ids) for c in controls],
            "simulated": True,
            "reason": "" if fixed else "extra candidate (control)",
        }
    )
    if fixed:
        cols = [c for c in natural_df.columns if c not in extra.columns]
        if cols and "variant" in natural_df.columns:
            extra = extra.merge(natural_df[["variant", *cols]], on="variant", how="left")
        return extra
    base = natural_df.assign(role="candidate")
    return pd.concat([base, extra], ignore_index=True) if not extra.empty else base
```
- In `sweep_prov` for silence_each, add `"natural_selection": natural,` and
  `"fixed_candidates": fixed is not None,`.
- At the end, build the result with the new fields:

```python
    result = SweepResult(
        out, sweep.kind, table, variants_df, conditions, readouts, prov, candidates, natural
    )
```

- The `role` key travels into the sweep table through `**v.info`. The existing report
  shows it with no change needed.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments -v`
Expected: PASS. The existing `test_sweep.py` tests (the default path, including
unsimulated rows with their `reason`) must still pass unchanged.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/sweep.py tests/unit/experiments/test_sweep_fixed.py
git commit -m "feat(sweep): fixed candidate tables, extra control candidates and natural selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Derived specs and `plan_variants`

**Files:**
- Modify: `flyconn/experiments/spec.py` (`ExperimentSpec.parent_sha256`, `canonical_dict`,
  `derive`, `NO_CONTROLS`)
- Modify: `flyconn/experiments/runner.py` (`base_provenance` records `parent_spec_sha256`)
- Create: `flyconn/experiments/robustness.py` (`Variant`, `w_syn_values`, `plan_variants`)
- Test: `tests/unit/experiments/test_robust_plan.py`

**Interfaces:**
- Consumes: `SimParams`, `NetworkOptions.sign_seed` (Task 1), `RobustnessOptions` (Task 2).
- Produces:

```python
NO_CONTROLS: Controls  # Controls(0, 0, False)
def canonical_dict(spec: ExperimentSpec) -> dict[str, Any]
def derive(spec: ExperimentSpec, *, parent_sha256: str | None, **changes: Any) -> ExperimentSpec

@dataclass(frozen=True)
class Variant:
    axis: str    # "reference" | "w_syn" | "nt" | "min_weight" | "replicate"
    label: str
    spec: ExperimentSpec

def w_syn_values(spec: ExperimentSpec) -> tuple[float, ...]
def plan_variants(spec: ExperimentSpec) -> list[Variant]
```

- [ ] **Step 1: Write the failing tests**

```python
"""plan_variants: one axis at a time around the calibrated reference (ADR-0011)."""

from __future__ import annotations

import pytest

from flyconn.data.store import Store
from flyconn.experiments.robustness import plan_variants, w_syn_values
from flyconn.experiments.spec import derive, spec_from_dict


def _d(
    store: Store, robustness: dict[str, object], dataset: str | None = None
) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "plan",
        "dataset": dataset or store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 3}},
        "robustness": {"primary_readout": "targets", **robustness},
    }


def test_full_plan_labels(small_store: Store):
    spec = spec_from_dict(_d(small_store, {}))
    labels = [v.label for v in plan_variants(spec)]
    assert labels == [
        "reference",
        "wsyn_0.185",
        "wsyn_0.192",
        "wsyn_0.209",
        *[f"nt_s{k}" for k in range(8)],
        "minw_3",
        "minw_5",
        "rep_flywire_783",
    ]


def test_auto_w_syn_from_calibration(small_store: Store):
    assert w_syn_values(spec_from_dict(_d(small_store, {}))) == (0.185, 0.192, 0.209)


def test_axes_can_be_switched_off(small_store: Store):
    spec = spec_from_dict(
        _d(
            small_store,
            {"w_syn": "off", "nt_samples": 0, "min_weights": [], "replicate": None},
        )
    )
    assert [v.axis for v in plan_variants(spec)] == ["reference"]


def test_auto_without_calibration_fails(small_store: Store):
    spec = spec_from_dict(_d(small_store, {"replicate": None}, dataset="flywire@783"))
    with pytest.raises(ValueError, match="w_syn: auto"):
        plan_variants(spec)


def test_min_weight_equal_to_reference_is_skipped(small_store: Store):
    spec = spec_from_dict(_d(small_store, {"min_weights": [1, 3]}))
    assert [v.label for v in plan_variants(spec) if v.axis == "min_weight"] == ["minw_3"]


def test_variant_specs_are_rehashed_and_stripped(small_store: Store):
    spec = spec_from_dict(_d(small_store, {}))
    by = {v.label: v.spec for v in plan_variants(spec)}
    w = by["wsyn_0.209"]
    assert w.params.w_syn_mv == 0.209
    assert w.raw["params"]["w_syn_mv"] == 0.209
    assert w.robustness is None
    assert w.controls.degree_preserving_rewire == 0 and not w.controls.unstimulated_baseline
    assert w.source_sha256 and w.source_sha256 != spec.source_sha256
    assert w.parent_sha256 == spec.source_sha256
    nt = by["nt_s3"]
    assert nt.network.sign_policy == "sampled" and nt.network.sign_seed == spec.seed + 2003
    assert by["minw_5"].network.min_weight == 5
    rep = by["rep_flywire_783"]
    assert rep.dataset == "flywire@783" and rep.params.w_syn_mv is None


def test_derive_is_deterministic(small_store: Store):
    spec = spec_from_dict(_d(small_store, {}))
    a = derive(spec, parent_sha256="p", name="x")
    b = derive(spec, parent_sha256="p", name="x")
    assert a.source_sha256 == b.source_sha256
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_robust_plan.py -v`
Expected: FAIL (`ModuleNotFoundError: flyconn.experiments.robustness`).

- [ ] **Step 3: Implement**

In `flyconn/experiments/spec.py`:
- Imports: add `import json` and `from dataclasses import asdict`.
- Add the field `parent_sha256: str | None = None` to `ExperimentSpec` after
  `source_sha256`.
- Add after `load_spec`:

```python
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
```

In `flyconn/experiments/runner.py` `base_provenance`, add
`"parent_spec_sha256": spec.parent_sha256,` after `"spec_sha256"`.

Create `flyconn/experiments/robustness.py`:

```python
"""Robustness-audited silencing screens (M13, ADR-0011).

One axis at a time around the calibrated reference: w_syn, sampled NT signs, network
threshold and a cross-dataset replicate. Every variant silences the reference run's
candidates, so effects are comparable across variants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flyconn.experiments.spec import NO_CONTROLS, ExperimentSpec, SimParams, derive
from flyconn.sim.params import CALIBRATIONS, default_params


@dataclass(frozen=True)
class Variant:
    axis: str
    label: str
    spec: ExperimentSpec


def _r6(x: float) -> float:
    return round(float(x), 6)


def w_syn_values(spec: ExperimentSpec) -> tuple[float, ...]:
    """w_syn values of the w_syn axis, sorted, without the reference value."""
    r = spec.robustness
    if r is None or r.w_syn == "off":
        return ()
    ref = spec.params.w_syn_mv or default_params(spec.dataset).w_syn_mv
    if r.w_syn == "auto":
        cal = CALIBRATIONS.get(spec.dataset)
        if cal is None:
            msg = (
                f"robustness w_syn: auto needs a calibration entry for {spec.dataset}; "
                "give an explicit list of mV values or 'off'"
            )
            raise ValueError(msg)
        values = {*cal.ci_mv, *cal.sensitivity_mv}
    else:
        values = set(r.w_syn)
    return tuple(sorted({_r6(v) for v in values} - {_r6(ref)}))


def plan_variants(spec: ExperimentSpec) -> list[Variant]:
    """Reference plus one variant per axis value; variants run without shared controls."""
    r = spec.robustness
    if r is None:
        msg = "spec has no robustness block"
        raise ValueError(msg)
    parent = spec.source_sha256

    def variant(axis: str, label: str, **changes: object) -> Variant:
        s = derive(spec, parent_sha256=parent, robustness=None, controls=NO_CONTROLS, **changes)
        return Variant(axis, label, s)

    out = [Variant("reference", "reference", spec)]
    out += [variant("w_syn", f"wsyn_{w:g}", params=SimParams(w)) for w in w_syn_values(spec)]
    for k in range(r.nt_samples):
        net = replace(spec.network, sign_policy="sampled", sign_seed=spec.seed + 2000 + k)
        out.append(variant("nt", f"nt_s{k}", network=net))
    for mw in r.min_weights:
        if mw != spec.network.min_weight:
            out.append(
                variant("min_weight", f"minw_{mw}", network=replace(spec.network, min_weight=mw))
            )
    if r.replicate:
        label = "rep_" + r.replicate.replace("@", "_").replace(".", "_")
        out.append(variant("replicate", label, dataset=r.replicate, params=SimParams()))
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/spec.py flyconn/experiments/runner.py flyconn/experiments/robustness.py tests/unit/experiments/test_robust_plan.py
git commit -m "feat(robustness): derived variant specs with their own provenance; one-axis-at-a-time plan

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Labels, axis verdicts and tiers (pure scoring)

**Files:**
- Create: `flyconn/experiments/robust_scoring.py`
- Test: `tests/unit/experiments/test_robust_scoring.py`

**Interfaces:**
- Produces:

```python
Label = Literal["holds", "weakens", "flips"]
Outcome = Literal["robust", "fails", "not_run"]
GRADED_AXES: tuple[str, ...] = ("w_syn", "nt", "min_weight", "random_null")
TIER_ORDER: dict[str, int] = {"Strong": 0, "Conditional": 1, "Fragile": 2}

def variant_label(reference_difference: float, difference: float, ci_low: float, ci_high: float) -> Label
def axis_outcome(axis: str, labels: Sequence[Label]) -> Outcome
def null_outcome(p: float | None) -> Outcome
def tier(outcomes: Mapping[str, Outcome]) -> tuple[str, bool]   # (tier, thinly_tested)
def score_hits(
    effects: pd.DataFrame,          # columns: axis, variant, candidate, role, difference, ci_low, ci_high, q_value
    *,
    alpha: float,
    null_p: Mapping[str, float | None],
    replicate: Mapping[str, str],    # candidate -> replicate label or "not testable: ..." / "not run: ..."
) -> pd.DataFrame
```

- `score_hits` returns one row per reference hit, with these columns: `candidate`,
  `difference`, `ci_low`, `ci_high`, `q_value`, `direction` ("down" | "up"), `w_syn`,
  `w_syn_detail`, `nt`, `nt_holds`, `nt_flips`, `min_weight`, `min_weight_detail`,
  `random_null`, `random_null_p`, `replicate`, `tier`, `thinly_tested`, `failed_axes`,
  `untested_axes`.
- Rows are sorted by tier order, then by |difference| descending.

- [ ] **Step 1: Write the failing tests**

```python
"""Pure scoring: per-variant labels, axis verdicts, tiers (spec section 2.3)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from flyconn.experiments.robust_scoring import (
    axis_outcome,
    null_outcome,
    score_hits,
    tier,
    variant_label,
)


@pytest.mark.parametrize(
    ("ref", "diff", "lo", "hi", "label"),
    [
        (-10, -8, -12, -4, "holds"),
        (-10, -2, -6, 1, "weakens"),
        (-10, 5, 1, 9, "flips"),
        (-10, 1, -3, 4, "weakens"),
        (10, 8, 4, 12, "holds"),
        (10, 0, -1, 1, "weakens"),
    ],
)
def test_label_table(ref: float, diff: float, lo: float, hi: float, label: str):
    assert variant_label(ref, diff, lo, hi) == label


finite = st.floats(min_value=-50, max_value=50, allow_nan=False).filter(lambda x: abs(x) > 1e-6)


@given(ref=finite, diff=finite, half=st.floats(min_value=0, max_value=30))
def test_negating_the_variant_swaps_holds_and_flips(ref: float, diff: float, half: float):
    lo, hi = diff - half, diff + half
    a = variant_label(ref, diff, lo, hi)
    b = variant_label(ref, -diff, -hi, -lo)
    assert {"holds": "flips", "flips": "holds", "weakens": "weakens"}[a] == b


@given(ref=finite, diff=finite, half=st.floats(min_value=0, max_value=30))
def test_negating_everything_changes_nothing(ref: float, diff: float, half: float):
    lo, hi = diff - half, diff + half
    assert variant_label(ref, diff, lo, hi) == variant_label(-ref, -diff, -hi, -lo)


def test_axis_outcomes():
    assert axis_outcome("w_syn", []) == "not_run"
    assert axis_outcome("w_syn", ["holds", "holds"]) == "robust"
    assert axis_outcome("w_syn", ["holds", "weakens"]) == "fails"
    assert axis_outcome("nt", ["holds"] * 6 + ["weakens"] * 2) == "robust"
    assert axis_outcome("nt", ["holds"] * 5 + ["flips"] * 3) == "fails"
    assert null_outcome(None) == "not_run"
    assert null_outcome(1 / 41) == "robust"
    assert null_outcome(0.05) == "fails"


outcome = st.sampled_from(["robust", "fails", "not_run"])


@given(w=outcome, n=outcome, m=outcome, r=outcome)
def test_every_combination_has_one_tier(w: str, n: str, m: str, r: str):
    oc = {"w_syn": w, "nt": n, "min_weight": m, "random_null": r}
    t, thin = tier(oc)  # type: ignore[arg-type]
    fails = sum(v == "fails" for v in oc.values())
    assert t == ("Strong" if fails == 0 else "Conditional" if fails == 1 else "Fragile")
    assert thin == (sum(v != "not_run" for v in oc.values()) < 2)


def _effects() -> pd.DataFrame:
    rows = [
        ("reference", "reference", "A", "candidate", -10.0, -14.0, -6.0, 0.001),
        ("reference", "reference", "B", "candidate", 5.0, 1.0, 9.0, 0.01),
        ("reference", "reference", "C", "candidate", 1.0, -3.0, 5.0, 0.6),
        ("reference", "reference", "K", "control", -9.0, -12.0, -6.0, 0.001),
        ("w_syn", "wsyn_0.185", "A", "candidate", -9.0, -13.0, -5.0, 0.01),
        ("w_syn", "wsyn_0.209", "A", "candidate", -8.0, -12.0, -4.0, 0.01),
        ("w_syn", "wsyn_0.185", "B", "candidate", 2.0, -1.0, 5.0, 0.3),
        ("w_syn", "wsyn_0.209", "B", "candidate", 4.0, 1.0, 7.0, 0.04),
        ("min_weight", "minw_3", "A", "candidate", -9.0, -13.0, -5.0, 0.01),
        ("min_weight", "minw_3", "B", "candidate", -3.0, -6.0, -1.0, 0.04),
    ]
    cols = ["axis", "variant", "candidate", "role", "difference", "ci_low", "ci_high", "q_value"]
    return pd.DataFrame(rows, columns=pd.Index(cols))


def test_score_hits():
    hits = score_hits(
        _effects(),
        alpha=0.05,
        null_p={"A": 1 / 41, "B": 0.3},
        replicate={"A": "holds", "B": "not testable: unmapped types ['B']"},
    )
    assert list(hits["candidate"]) == ["A", "B"]  # C not a hit, K is a control
    a = hits.iloc[0]
    assert a["tier"] == "Strong" and a["direction"] == "down" and a["nt"] == "not_run"
    assert a["replicate"] == "holds" and not a["thinly_tested"]
    b = hits.iloc[1]
    assert b["tier"] == "Fragile"
    assert set(b["failed_axes"].split(", ")) == {"w_syn", "min_weight", "random_null"}
    assert "wsyn_0.185: weakens" in b["w_syn_detail"]
    assert b["untested_axes"] == "nt"
    assert math.isclose(b["random_null_p"], 0.3)


def test_score_hits_empty():
    eff = _effects()
    eff["q_value"] = 0.9
    assert score_hits(eff, alpha=0.05, null_p={}, replicate={}).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_robust_scoring.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** `flyconn/experiments/robust_scoring.py`

```python
"""Scoring for robustness-audited screens: labels, axis verdicts, tiers (ADR-0011)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

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
    if not labels:
        return "not_run"
    n_holds = sum(lbl == "holds" for lbl in labels)
    if axis == "nt":
        need = math.ceil(NT_ROBUST_FRACTION * len(labels))
        return "robust" if n_holds >= need else "fails"
    return "robust" if n_holds == len(labels) else "fails"


def null_outcome(p: float | None) -> Outcome:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "not_run"
    return "robust" if p < NULL_ALPHA else "fails"


def tier(outcomes: Mapping[str, Outcome]) -> tuple[str, bool]:
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
    """One row per reference hit (role candidate, q < alpha) with per-axis verdicts and tier."""
    ref = effects[
        (effects["axis"] == "reference")
        & (effects["role"] == "candidate")
        & (effects["q_value"] < alpha)
    ]
    rows: list[dict[str, Any]] = []
    for r in ref.itertuples(index=False):
        cand = str(r.candidate)
        ref_diff = float(cast("float", r.difference))
        row: dict[str, Any] = {
            "candidate": cand,
            "difference": ref_diff,
            "ci_low": float(cast("float", r.ci_low)),
            "ci_high": float(cast("float", r.ci_high)),
            "q_value": float(cast("float", r.q_value)),
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
    out["_t"] = out["tier"].map(TIER_ORDER)
    out["_a"] = out["difference"].abs()
    out = out.sort_values(["_t", "_a"], ascending=[True, False], kind="stable")
    return out.drop(columns=["_t", "_a"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_robust_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/robust_scoring.py tests/unit/experiments/test_robust_scoring.py
git commit -m "feat(robustness): per-variant labels, axis verdicts and tiers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Random-silencing null sets and empirical p

**Files:**
- Create: `flyconn/experiments/robust_null.py`
- Test: `tests/unit/experiments/test_robust_null.py`

**Interfaces:**
- Consumes: `Candidate` (Task 3).
- Produces:

```python
def size_bin(n: int) -> int                      # nearest power of two, >= 1
def null_key(neurons: pd.DataFrame, ids: Sequence[int]) -> str   # "<super_class>|<bin>"
@dataclass(frozen=True)
class NullSet:
    key: str
    index: int
    seed: int
    neuron_ids: tuple[int, ...]
def draw_null_sets(
    neurons: pd.DataFrame, candidates: Sequence[Candidate], excluded: set[int], *, n_sets: int, seed: int
) -> tuple[list[NullSet], dict[str, str]]      # (sets, skipped key -> reason)
def null_p(effect: float, null_effects: Sequence[float]) -> float
```

- [ ] **Step 1: Write the failing tests**

```python
"""Random-silencing null: size/super_class-matched sets and empirical p (spec 2.2/2.3)."""

from __future__ import annotations

import pandas as pd

from flyconn.experiments.robust_null import draw_null_sets, null_key, null_p, size_bin
from flyconn.experiments.sweep import Candidate


def _neurons() -> pd.DataFrame:
    ids = list(range(1, 41))
    sc = ["central"] * 30 + ["descending"] * 10
    return pd.DataFrame({"neuron_id": ids, "super_class": sc})


def test_size_bins():
    assert [size_bin(n) for n in (1, 2, 3, 5, 6, 12, 100)] == [1, 2, 4, 4, 8, 16, 128]


def test_sets_match_bin_and_super_class_and_exclude():
    n = _neurons()
    cands = [Candidate("A", (1, 2, 3)), Candidate("D", (31,)), Candidate("K", (4,), "control")]
    excluded = {5, 6, 7}
    sets, skipped = draw_null_sets(n, cands, excluded, n_sets=5, seed=0)
    assert skipped == {}
    sc = dict(zip(n["neuron_id"], n["super_class"], strict=True))
    keys = {s.key for s in sets}
    assert keys == {"central|4", "descending|1", "central|1"}  # control K gets a null too
    assert null_key(n, (1, 2, 3)) == "central|4" and null_key(n, (31,)) == "descending|1"
    for s in sets:
        want_sc, want_bin = s.key.split("|")
        assert len(s.neuron_ids) == int(want_bin)
        assert all(sc[i] == want_sc for i in s.neuron_ids)
        assert not set(s.neuron_ids) & ({1, 2, 3, 31, 4} | excluded)
    assert sum(s.key == "central|4" for s in sets) == 5


def test_deterministic():
    n = _neurons()
    c = [Candidate("A", (1, 2))]
    assert draw_null_sets(n, c, set(), n_sets=4, seed=3) == draw_null_sets(
        n, c, set(), n_sets=4, seed=3
    )
    assert draw_null_sets(n, c, set(), n_sets=4, seed=3) != draw_null_sets(
        n, c, set(), n_sets=4, seed=4
    )


def test_pool_too_small_is_skipped_not_crash():
    n = _neurons()
    sets, skipped = draw_null_sets(
        n, [Candidate("D", tuple(range(31, 41)))], set(), n_sets=3, seed=0
    )
    assert sets == [] and "descending|8" in skipped
    assert "pool" in skipped["descending|8"]


def test_null_p():
    assert null_p(-10.0, [-1.0] * 40) == 1 / 41
    assert null_p(-10.0, [-20.0] * 40) == 1.0
    assert null_p(5.0, [6.0, 1.0, 1.0]) == 2 / 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_robust_null.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** `flyconn/experiments/robust_null.py`

```python
"""Random-silencing null for robustness screens (ADR-0011).

A hit should matter more than silencing the same number of random neurons of the same
super_class. Sets never contain candidate, stimulus or readout neurons.
"""

from __future__ import annotations

import math
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
) -> tuple[list[NullSet], dict[str, str]]:
    blocked = set(excluded) | {i for c in candidates for i in c.neuron_ids}
    # Controls get a null too: the negative control's null p is the honesty check (spec 4).
    keys = sorted(
        {null_key(neurons, c.neuron_ids) for c in candidates if c.role in ("candidate", "control")}
    )
    sets: list[NullSet] = []
    skipped: dict[str, str] = {}
    for k_idx, key in enumerate(keys):
        sc, bin_s = key.split("|")
        size = int(bin_s)
        pool_df = neurons[neurons["super_class"].astype(str) == sc]
        pool = np.array(
            sorted(set(pool_df["neuron_id"].astype(np.int64).tolist()) - blocked), dtype=np.int64
        )
        if len(pool) < size:
            skipped[key] = f"pool of {len(pool)} {sc} neurons is smaller than the set size {size}"
            continue
        for i in range(n_sets):
            s = seed + 3000 + i
            rng = np.random.default_rng([s, k_idx])
            ids = tuple(int(x) for x in np.sort(rng.choice(pool, size=size, replace=False)))
            sets.append(NullSet(key, i, s, ids))
    return sets, skipped


def null_p(effect: float, null_effects: Sequence[float]) -> float:
    """(1 + #null at least as extreme in the effect's direction) / (n + 1)."""
    s = 1.0 if effect >= 0 else -1.0
    arr = np.asarray(null_effects, dtype=float)
    return float((1 + int((arr * s >= effect * s).sum())) / (len(arr) + 1))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_robust_null.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/robust_null.py tests/unit/experiments/test_robust_null.py
git commit -m "feat(robustness): size- and super_class-matched random-silencing null

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Cross-dataset replicate mapping

**Files:**
- Create: `flyconn/experiments/robust_replicate.py`
- Test: `tests/unit/experiments/test_robust_replicate.py`

**Interfaces:**
- Consumes: `Selection`, `Stimulation`, `Readout`, `derive` (spec.py); `Candidate` (Task 3).
  The `type_map` columns follow `compare.match_types`: `type_a`, `type_b`, `n_a`, `n_b`,
  `source`.
- Produces:

```python
def map_selection(
    sel: Selection, query: pd.DataFrame, rep: pd.DataFrame, type_map: pd.DataFrame
) -> tuple[np.ndarray | None, str]          # (replicate ids or None, reason)
def map_candidates(
    candidates: Sequence[Candidate], rep: pd.DataFrame, type_map: pd.DataFrame, group_by: str
) -> tuple[list[Candidate], dict[str, str]]  # (mapped, label -> "not testable: ..." reason)
def replicate_spec(
    variant_spec: ExperimentSpec, original: ExperimentSpec,
    query: pd.DataFrame, rep: pd.DataFrame, type_map: pd.DataFrame,
) -> tuple[ExperimentSpec | None, str]
```

- [ ] **Step 1: Write the failing tests**

```python
"""Replicate mapping MaleCNS -> FlyWire by cell type (spec 2.2, cross-dataset replicate)."""

from __future__ import annotations

import pandas as pd

from flyconn.experiments.robust_replicate import map_candidates, map_selection
from flyconn.experiments.spec import Selection
from flyconn.experiments.sweep import Candidate

Q = pd.DataFrame(
    {
        "neuron_id": [1, 2, 3, 4, 5, 6],
        "cell_type": ["LB3c", "LB3d", "MN9", "GNG232", "X1", None],
        "super_class": ["sensory", "sensory", "motor", "central", "central", "descending"],
        "side": ["left", "right", "left", "left", "left", "left"],
    }
)
R = pd.DataFrame(
    {
        "neuron_id": [11, 12, 13, 14, 15],
        "cell_type": ["LB3", "LB3", "MN9", "GNG232", "DNx"],
        "super_class": ["sensory", "sensory", "motor", "central", "descending"],
        "side": ["left", "right", "left", "left", "left"],
    }
)
TM = pd.DataFrame(
    {
        "type_a": ["LB3c", "LB3d", "MN9", "GNG232"],
        "type_b": ["LB3", "LB3", "MN9", "GNG232"],
        "n_a": [1, 1, 1, 1],
        "n_b": [2, 2, 1, 1],
        "source": "fafb_783_cell_type",
    }
)


def test_glob_selection_maps_through_types():
    ids, why = map_selection(Selection(cell_type="LB3*"), Q, R, TM)
    assert why == "" and ids is not None and ids.tolist() == [11, 12]


def test_side_filter_is_kept():
    ids, _ = map_selection(Selection(cell_type="LB3*", side="left"), Q, R, TM)
    assert ids is not None and ids.tolist() == [11]


def test_super_class_only_passes_through():
    ids, _ = map_selection(Selection(super_class="descending"), Q, R, TM)
    assert ids is not None and ids.tolist() == [15]


def test_unmapped_type_is_reported():
    ids, why = map_selection(Selection(cell_type="X1"), Q, R, TM)
    assert ids is None and "X1" in why


def test_candidates():
    mapped, reasons = map_candidates(
        [Candidate("GNG232", (4,)), Candidate("X1", (5,)), Candidate("K", (6,), "control")],
        R,
        TM,
        "cell_type",
    )
    assert [(c.label, c.neuron_ids, c.role) for c in mapped] == [("GNG232", (14,), "candidate")]
    assert reasons["X1"].startswith("not testable") and reasons["K"].startswith("not testable")


def test_candidates_need_cell_type_grouping():
    mapped, reasons = map_candidates([Candidate("4", (4,))], R, TM, "neuron")
    assert mapped == [] and "not grouped by cell type" in reasons["4"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_robust_replicate.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** `flyconn/experiments/robust_replicate.py`

```python
"""Map a screen to a replicate dataset by cell type (ADR-0011; reported, never graded)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from flyconn.experiments.spec import (
    ExperimentSpec,
    Readout,
    Selection,
    SilenceEachSweep,
    Stimulation,
    derive,
    resolve_selection,
)
from flyconn.experiments.sweep import Candidate


def _rep_ids(rep: pd.DataFrame, types: set[str], side: str | None) -> np.ndarray:
    m = rep["cell_type"].astype(object).isin(types)
    if side is not None:
        m &= rep["side"].astype(object) == side
    return np.sort(rep.loc[m, "neuron_id"].to_numpy(dtype=np.int64))


def map_selection(
    sel: Selection, query: pd.DataFrame, rep: pd.DataFrame, type_map: pd.DataFrame
) -> tuple[np.ndarray | None, str]:
    if sel.ids is None and sel.cell_type is None and sel.cell_class is None:
        try:
            return resolve_selection(sel, rep), ""
        except (ValueError, KeyError) as e:
            return None, f"selection empty in replicate: {e}"
    ids = resolve_selection(sel, query)
    ct = query.set_index("neuron_id")["cell_type"].reindex(ids).dropna().astype(str)
    types = sorted(set(ct))
    if not types:
        return None, f"{sel.describe()} has no cell types to map"
    tm = type_map[type_map["type_a"].isin(types)]
    missing = sorted(set(types) - set(tm["type_a"].astype(str)))
    if missing:
        return None, f"unmapped types {missing[:5]}"
    out = _rep_ids(rep, set(tm["type_b"].astype(str)), sel.side)
    if len(out) == 0:
        return None, f"mapped types of {sel.describe()} have no neurons in the replicate"
    return out, ""


def map_candidates(
    candidates: Sequence[Candidate], rep: pd.DataFrame, type_map: pd.DataFrame, group_by: str
) -> tuple[list[Candidate], dict[str, str]]:
    if group_by != "cell_type":
        reason = "not testable: candidates not grouped by cell type"
        return [], {c.label: reason for c in candidates}
    mapped: list[Candidate] = []
    reasons: dict[str, str] = {}
    for c in candidates:
        tb = set(type_map.loc[type_map["type_a"] == c.label, "type_b"].astype(str))
        ids = _rep_ids(rep, tb, None) if tb else np.array([], dtype=np.int64)
        if len(ids) == 0:
            reasons[c.label] = f"not testable: no replicate type for {c.label}"
            continue
        mapped.append(Candidate(c.label, tuple(int(i) for i in ids), c.role))
    return mapped, reasons


def replicate_spec(
    variant_spec: ExperimentSpec,
    original: ExperimentSpec,
    query: pd.DataFrame,
    rep: pd.DataFrame,
    type_map: pd.DataFrame,
) -> tuple[ExperimentSpec | None, str]:
    """``variant_spec`` with stimulus and readouts re-expressed as replicate neuron ids."""
    stims: list[Stimulation] = []
    for st in original.stimulate:
        ids, why = map_selection(st.select, query, rep, type_map)
        if ids is None:
            return None, f"not testable: stimulus {st.select.describe()}: {why}"
        stims.append(Stimulation(Selection(ids=ids.tolist()), st.rate_hz))
    readouts: list[Readout] = []
    for r in original.readouts:
        ids, why = map_selection(r.select, query, rep, type_map)
        if ids is None:
            return None, f"not testable: readout {r.name}: {why}"
        readouts.append(Readout(r.name, Selection(ids=ids.tolist())))
    sweep = variant_spec.sweep
    if isinstance(sweep, SilenceEachSweep):
        sweep = replace(sweep, select=Selection())  # candidates are fixed; natural not computed
    spec = derive(
        variant_spec,
        parent_sha256=variant_spec.parent_sha256,
        stimulate=stims,
        readouts=readouts,
        sweep=sweep,
    )
    return spec, ""
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_robust_replicate.py -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/robust_replicate.py tests/unit/experiments/test_robust_replicate.py
git commit -m "feat(robustness): map screens to a replicate dataset by cell type

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `run_robust_screen` orchestration, resume, path and selection stability

**Files:**
- Modify: `flyconn/experiments/robustness.py`
- Test: `tests/unit/experiments/test_robust_screen.py`

**Interfaces:**
- Consumes:
  - `plan_variants`, `Variant` (Task 4);
  - `run_sweep`, `Candidate` (Task 3);
  - `score_hits`, `variant_label` (Task 5);
  - `draw_null_sets`, `null_p`, `null_key` (Task 6);
  - `map_candidates`, `replicate_spec` (Task 7);
  - `compare.match_types`; `uncertainty.path_stability`; `runner.rates_hz`.
- Produces:

```python
@dataclass
class RobustScreenResult:
    out_dir: Path
    spec: ExperimentSpec
    hits: pd.DataFrame                 # score_hits output + path_stability column
    effects: pd.DataFrame              # every candidate x variant on the primary readout, with label
    null: pd.DataFrame                 # key, index, seed, n_neurons, difference
    selection_stability: pd.DataFrame  # variant, axis, jaccard
    provenance: dict[str, Any]

def run_robust_screen(
    spec: ExperimentSpec, *, out_dir: Path | str, store: Store | None = None,
    replicate_store: Store | None = None,
) -> RobustScreenResult
```

- Files written:
  - `robustness_plan.json`;
  - `variants/<label>/` (each with `done.json` written last);
  - `variants/null/`;
  - `hits.parquet` (the effects table);
  - `null.parquet`;
  - `selection_stability.parquet`;
  - `provenance.json`.
- Task 10 adds the report and `handoff.csv`.

- [ ] **Step 1: Write the failing tests**

```python
"""End-to-end robustness screen on the synthetic store (structure, resume, determinism)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store
from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import spec_from_dict
from flyconn.testing import synthetic_connectome


def _d(store: Store, **rob: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "rs",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 2}},
        "duration_ms": 60,
        "trials": 4,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 1, "sign_shuffle": 0},
        "report": {"alpha": 1.0},  # every candidate counts as a hit, to exercise scoring
        "robustness": {
            "primary_readout": "targets",
            "w_syn": [0.3],
            "nt_samples": 2,
            "min_weights": [2],
            "replicate": None,
            "random_null": {"n_sets": 3},
            **rob,
        },
    }


@pytest.fixture
def rep_store(tmp_path: Path) -> Store:
    neurons, edges = synthetic_connectome(n_neurons=40, n_edges=200, seed=22)
    neurons["dataset"] = edges["dataset"] = "flywire"
    neurons["version"] = edges["version"] = "783"
    neurons["input_synapses_total"] = (
        edges.groupby("post")["weight"].sum().reindex(neurons["neuron_id"]).fillna(0).to_numpy() + 5
    )
    d = tmp_path / "rep"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(
        d,
        dataset="flywire",
        version="783",
        converter="test",
        inputs=[],
        counts={"neurons": 40, "edges": 200},
    )
    return Store(d)


def test_outputs_and_axes(small_store: Store, tmp_path: Path):
    res = run_robust_screen(
        spec_from_dict(_d(small_store)), out_dir=tmp_path / "o", store=small_store
    )
    out = tmp_path / "o"
    for f in (
        "robustness_plan.json",
        "hits.parquet",
        "null.parquet",
        "selection_stability.parquet",
        "provenance.json",
    ):
        assert (out / f).exists(), f
    assert set(res.effects["axis"]) == {"reference", "w_syn", "nt", "min_weight"}
    assert {"label"} <= set(res.effects.columns)
    assert len(res.hits) == 2  # alpha 1.0: both natural candidates are hits
    assert {"tier", "path_stability"} <= set(res.hits.columns)
    assert set(res.selection_stability["variant"]) == {"wsyn_0.3", "nt_s0", "nt_s1", "minw_2"}
    assert len(res.null) > 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["robustness"]["control_null_p"] == {}  # no extra candidates in this spec
    assert prov["robustness"]["w_syn_values"] == [0.3]
    for v in ("wsyn_0.3", "nt_s0", "minw_2", "reference", "null"):
        assert (out / "variants" / v / "done.json").exists()


def test_deterministic(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    a = run_robust_screen(spec, out_dir=tmp_path / "a", store=small_store)
    b = run_robust_screen(spec, out_dir=tmp_path / "b", store=small_store)
    pd.testing.assert_frame_equal(a.hits, b.hits)
    pd.testing.assert_frame_equal(a.effects, b.effects)


def test_resume_moves_incomplete_and_skips_done(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    out = tmp_path / "o"
    first = run_robust_screen(spec, out_dir=out, store=small_store)
    (out / "variants" / "nt_s1" / "done.json").unlink()  # simulate a crash inside nt_s1
    shutil.rmtree(out / "variants" / "minw_2")  # and a variant that never started
    done_mtime = (out / "variants" / "wsyn_0.3" / "done.json").stat().st_mtime_ns
    again = run_robust_screen(spec, out_dir=out, store=small_store)
    assert (out / "variants" / ".incomplete" / "nt_s1-1").is_dir()
    assert (out / "variants" / "wsyn_0.3" / "done.json").stat().st_mtime_ns == done_mtime
    pd.testing.assert_frame_equal(first.hits, again.hits)


def test_changed_plan_refuses(small_store: Store, tmp_path: Path):
    out = tmp_path / "o"
    run_robust_screen(spec_from_dict(_d(small_store)), out_dir=out, store=small_store)
    with pytest.raises(FileExistsError, match="plan"):
        run_robust_screen(
            spec_from_dict(_d(small_store, nt_samples=3)), out_dir=out, store=small_store
        )


def test_replicate_axis(small_store: Store, rep_store: Store, tmp_path: Path):
    ids = small_store.neurons()["neuron_id"].tolist()
    d = _d(small_store, replicate="flywire@783")
    # readouts and stimulus by cell type so they can be mapped
    types = small_store.neurons().set_index("neuron_id")["cell_type"]
    d["stimulate"] = [{"select": {"cell_type": str(types[ids[0]])}, "rate_hz": 200}]
    d["readouts"] = [{"name": "targets", "select": {"cell_type": str(types[ids[-1]])}}]
    res = run_robust_screen(
        spec_from_dict(d), out_dir=tmp_path / "o", store=small_store, replicate_store=rep_store
    )
    assert all(isinstance(x, str) and x for x in res.hits["replicate"])
    assert "replicate" in res.provenance["robustness"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_robust_screen.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_robust_screen'`).

- [ ] **Step 3: Implement** (append to `flyconn/experiments/robustness.py`; extend the
  imports)

```python
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flyconn.data.cache import store_dir
from flyconn.data.pull import pull
from flyconn.data.registry import get_dataset
from flyconn.data.store import Store
from flyconn.experiments.robust_null import draw_null_sets, null_key, null_p
from flyconn.experiments.robust_replicate import map_candidates, replicate_spec
from flyconn.experiments.robust_scoring import score_hits, variant_label
from flyconn.experiments.runner import rates_hz
from flyconn.experiments.spec import resolve_selection
from flyconn.experiments.sweep import Candidate, run_sweep
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.uncertainty.stability import path_stability


@dataclass
class RobustScreenResult:
    out_dir: Path
    spec: ExperimentSpec
    hits: pd.DataFrame
    effects: pd.DataFrame
    null: pd.DataFrame
    selection_stability: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Loaded:
    sweep: pd.DataFrame
    candidates: list[Candidate]
    natural: list[str]


def _plan_hash(spec: ExperimentSpec, plan: list[Variant]) -> str:
    payload = {
        "spec": spec.source_sha256,
        "variants": [(v.label, v.spec.source_sha256) for v in plan],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _open_out(out: Path, plan_hash: str) -> None:
    plan_file = out / "robustness_plan.json"
    if out.exists() and any(out.iterdir()):
        if not plan_file.exists() or json.loads(plan_file.read_text())["hash"] != plan_hash:
            msg = (
                f"{out} holds a different robustness plan (or none); choose a new output "
                "directory to run a changed spec"
            )
            raise FileExistsError(msg)
        return
    out.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps({"hash": plan_hash}, indent=1))


def _run_or_load(
    d: Path,
    spec: ExperimentSpec,
    store: Store,
    *,
    fixed: list[Candidate] | None,
    compute_natural: bool,
) -> _Loaded:
    done = d / "done.json"
    if done.exists():
        meta = json.loads(done.read_text())
        cands = [Candidate(c[0], tuple(c[1]), c[2]) for c in meta["candidates"]]
        return _Loaded(pd.read_parquet(d / "sweep.parquet"), cands, list(meta["natural"]))
    if d.exists():
        inc = d.parent / ".incomplete"
        inc.mkdir(exist_ok=True)
        n = 1
        while (inc / f"{d.name}-{n}").exists():
            n += 1
        d.rename(inc / f"{d.name}-{n}")
    res = run_sweep(spec, out_dir=d, store=store, fixed=fixed, compute_natural=compute_natural)
    meta = {
        "candidates": [[c.label, list(c.neuron_ids), c.role] for c in res.candidates],
        "natural": res.natural_selection,
    }
    done.write_text(json.dumps(meta))
    return _Loaded(res.sweep, res.candidates, res.natural_selection)


def _primary_rows(axis: str, label: str, loaded: _Loaded, readout: str) -> pd.DataFrame:
    t = loaded.sweep[loaded.sweep["readout"] == readout]
    return pd.DataFrame(
        {
            "axis": axis,
            "variant": label,
            "candidate": t["variant"].astype(str).to_numpy(),
            "role": t["role"].astype(str).to_numpy(),
            "difference": t["difference"].to_numpy(dtype=float),
            "ci_low": t["ci_low"].to_numpy(dtype=float),
            "ci_high": t["ci_high"].to_numpy(dtype=float),
            "q_value": t["q_value"].to_numpy(dtype=float),
        }
    )


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return 1.0 if not sa and not sb else len(sa & sb) / len(sa | sb)


def _replicate_store(ref: str) -> Store | None:
    d = store_dir(get_dataset(ref))
    return Store(d) if (d / "provenance.json").exists() else None


def run_robust_screen(
    spec: ExperimentSpec,
    *,
    out_dir: Path | str,
    store: Store | None = None,
    replicate_store: Store | None = None,
) -> RobustScreenResult:
    """Run the reference screen and every robustness variant; score hits (ADR-0011)."""
    r = spec.robustness
    if r is None or spec.sweep is None:
        msg = "spec has no robustness block"
        raise ValueError(msg)
    plan = plan_variants(spec)
    plan_hash = _plan_hash(spec, plan)
    out = Path(out_dir)
    _open_out(out, plan_hash)
    store = store or Store(pull(spec.dataset, level="weights").store_dir)
    vdir = out / "variants"
    vdir.mkdir(exist_ok=True)
    primary = r.primary_readout

    ref = _run_or_load(vdir / "reference", spec, store, fixed=None, compute_natural=True)
    cands = ref.candidates
    effects = [_primary_rows("reference", "reference", ref, primary)]
    stability: list[dict[str, Any]] = []
    replicate_labels: dict[str, str] = {}
    rep_prov: dict[str, Any] = {}

    for v in plan[1:]:
        if v.axis == "replicate":
            continue
        loaded = _run_or_load(vdir / v.label, v.spec, store, fixed=cands, compute_natural=True)
        effects.append(_primary_rows(v.axis, v.label, loaded, primary))
        stability.append(
            {"variant": v.label, "axis": v.axis, "jaccard": _jaccard(ref.natural, loaded.natural)}
        )

    rep_v = next((v for v in plan if v.axis == "replicate"), None)
    if rep_v is not None:
        rep_store = replicate_store or _replicate_store(rep_v.spec.dataset)
        if rep_store is None:
            why = f"not run: `flyconn data pull {rep_v.spec.dataset} --level weights` first"
            replicate_labels = {c.label: why for c in cands}
        else:
            from flyconn.compare.types import match_types  # noqa: PLC0415 (heavy import)

            type_map = match_types(store, rep_store)
            q_n, rep_n = store.neurons(), rep_store.neurons()
            rspec, why = replicate_spec(rep_v.spec, spec, q_n, rep_n, type_map)
            mapped, replicate_labels = map_candidates(
                cands,
                rep_n,
                type_map,
                spec.sweep.group_by,  # type: ignore[union-attr]
            )
            rep_prov = {"dataset": rep_store.ref, "type_map_rows": len(type_map)}
            if rspec is None:
                replicate_labels = {c.label: why for c in cands}
            elif mapped:
                loaded = _run_or_load(
                    vdir / rep_v.label, rspec, rep_store, fixed=mapped, compute_natural=False
                )
                rows = _primary_rows("replicate", rep_v.label, loaded, primary)
                effects.append(rows)
                ref_rows = effects[0].set_index("candidate")
                for row in rows.itertuples(index=False):
                    rd = float(ref_rows.loc[str(row.candidate), "difference"])
                    replicate_labels[str(row.candidate)] = (
                        variant_label(
                            rd, float(row.difference), float(row.ci_low), float(row.ci_high)
                        )
                        if rd != 0
                        else "weakens"
                    )

    neurons = store.neurons()
    null_rows: list[dict[str, Any]] = []
    null_by_cand: dict[str, float | None] = {}
    skipped: dict[str, str] = {}
    if r.random_null_sets > 0:
        excluded = set(rates_hz(spec.stimulate, neurons)) | {
            int(i) for ro in spec.readouts for i in resolve_selection(ro.select, neurons)
        }
        sets, skipped = draw_null_sets(
            neurons, cands, excluded, n_sets=r.random_null_sets, seed=spec.seed
        )
        if sets:
            null_spec = derive(
                spec, parent_sha256=spec.source_sha256, robustness=None, controls=NO_CONTROLS
            )
            fixed = [Candidate(f"null:{s.key}:{s.index}", s.neuron_ids, "null") for s in sets]
            loaded = _run_or_load(
                vdir / "null", null_spec, store, fixed=fixed, compute_natural=False
            )
            t = loaded.sweep[loaded.sweep["readout"] == primary].set_index("variant")
            for s in sets:
                null_rows.append(
                    {
                        "key": s.key,
                        "index": s.index,
                        "seed": s.seed,
                        "n_neurons": len(s.neuron_ids),
                        "difference": float(t.loc[f"null:{s.key}:{s.index}", "difference"]),
                    }
                )
    null_df = pd.DataFrame(
        null_rows, columns=pd.Index(["key", "index", "seed", "n_neurons", "difference"])
    )
    ref_eff = effects[0].set_index("candidate")
    control_null_p: dict[str, float | None] = {}
    for c in cands:
        key = null_key(neurons, c.neuron_ids)
        nulls = null_df.loc[null_df["key"] == key, "difference"].to_list()
        p = null_p(float(ref_eff.loc[c.label, "difference"]), nulls) if nulls else None
        (null_by_cand if c.role == "candidate" else control_null_p)[c.label] = p

    eff = pd.concat(effects, ignore_index=True)
    ref_diff = eff[eff["axis"] == "reference"].set_index("candidate")["difference"]
    eff["label"] = [
        "reference"
        if a == "reference"
        else variant_label(float(ref_diff[c]), d, lo, hi)
        if float(ref_diff.get(c, 0.0)) != 0
        else "weakens"
        for a, c, d, lo, hi in zip(
            eff["axis"],
            eff["candidate"],
            eff["difference"],
            eff["ci_low"],
            eff["ci_high"],
            strict=True,
        )
    ]
    hits = score_hits(eff, alpha=spec.report.alpha, null_p=null_by_cand, replicate=replicate_labels)
    hits["path_stability"] = _path_stability(spec, store, hits, cands)

    sel = pd.DataFrame(stability, columns=pd.Index(["variant", "axis", "jaccard"]))
    from flyconn.experiments.runner import write_table  # noqa: PLC0415

    write_table(eff, out / "hits.parquet")
    write_table(null_df, out / "null.parquet")
    write_table(sel, out / "selection_stability.parquet")
    ref_prov = json.loads((vdir / "reference" / "provenance.json").read_text())
    prov = ref_prov | {
        "robustness": {
            "design": "one axis at a time around the calibrated reference (ADR-0011)",
            "plan_hash": plan_hash,
            "variants": [
                {"axis": v.axis, "label": v.label, "spec_sha256": v.spec.source_sha256}
                for v in plan
            ],
            "w_syn_values": list(w_syn_values(spec)),
            "nt_sign_seeds": [spec.seed + 2000 + k for k in range(r.nt_samples)],
            "min_weights": list(r.min_weights),
            "replicate": rep_prov | {"ref": r.replicate},
            "random_null": {
                "n_sets": r.random_null_sets,
                "seeds": [spec.seed + 3000 + i for i in range(r.random_null_sets)],
                "skipped": skipped,
                "p_rule": "(1 + #null at least as extreme) / (n_sets + 1) < 0.05",
            },
            "extra_candidates": [s.describe() for s in r.extra_candidates],
            "control_null_p": control_null_p,
            "statistics": "unpaired bootstrap CI and Mann-Whitney p; common random inputs "
            "(conservative); no correction across variants",
        }
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return RobustScreenResult(out, spec, hits, eff, null_df, sel, prov)


def _path_stability(
    spec: ExperimentSpec, store: Store, hits: pd.DataFrame, cands: list[Candidate]
) -> list[float]:
    """Mean sign_stability x threshold_presence of stimulus->primary paths through each hit."""
    if hits.empty or spec.robustness is None:
        return []
    neurons = store.neurons()
    sources = sorted(rates_hz(spec.stimulate, neurons))
    readout = next(r for r in spec.readouts if r.name == spec.robustness.primary_readout)
    targets = resolve_selection(readout.select, neurons).tolist()
    sweep = spec.sweep
    max_hops = sweep.pathway.max_hops if getattr(sweep, "pathway", None) else 3  # type: ignore[union-attr]
    m = ConnectivityMatrix.from_store(store, min_weight=1)
    res = path_stability(
        m, sources, targets, max_hops=max_hops, n_samples=50, seed=spec.seed, thresholds=(1, 3, 5)
    )
    paths = res.paths
    ids_by = {c.label: set(c.neuron_ids) for c in cands}
    out: list[float] = []
    for lbl in hits["candidate"]:
        ids = ids_by[str(lbl)]
        mask = paths["path"].map(lambda p: bool(ids & set(p)))
        sub = paths[mask]
        out.append(
            float((sub["sign_stability"] * sub["threshold_presence"]).mean())
            if not sub.empty
            else float("nan")
        )
    return out
```

Notes for the implementer:
- `path_stability` may raise when there are no paths at all. If it does, catch
  `ValueError` and return `[float("nan")] * len(hits)`.
- `np` is used by the helper modules; if ruff flags it unused here, remove it.
- Put all imports at module top except the two marked `noqa: PLC0415`; keep the
  `match_types` import lazy because `compare.types` pulls in scipy-heavy code.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/experiments/test_robust_screen.py -v`
Expected: PASS. Takes about 1-2 minutes: each tiny sweep is quick, but there are about 6
variants plus the null run.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/robustness.py tests/unit/experiments/test_robust_screen.py
git commit -m "feat(robustness): run_robust_screen with resume, random null, replicate, path and selection stability

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Driver-line handoff (`attach_lines`) and cached NeuronBridge counts

**Files:**
- Create: `flyconn/access/handoff.py`
- Modify: `flyconn/access/neuronbridge.py` (add `cached_off_target_count`)
- Modify: `flyconn/access/__init__.py` (export `attach_lines`, `load_catalog`)
- Test: `tests/unit/access/test_handoff.py`

**Interfaces:**
- Consumes: `LineCatalog` (existing); `hits` frame with a `candidate` column (Task 5).
- Produces:

```python
def load_catalog() -> LineCatalog | None          # None when flylight_lines is not pulled
def attach_lines(hits: pd.DataFrame, catalog: LineCatalog | None, *, max_lines: int = 5) -> pd.DataFrame
# adds: best_line, line_quality, line_specific, line_match ("exact"|"glob"|"none"|"not looked up"),
#       lines (up to max_lines "LINE (qN[, specific])" joined by "; "), off_target_bodies (Int64, cached only)
def cached_off_target_count(line: str, *, version: str = DEFAULT_VERSION, max_images: int = 50) -> int | None
```

- [ ] **Step 1: Write the failing tests**

```python
"""Driver lines for screen hits (Meissner 2025 table; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

from flyconn.access.handoff import attach_lines
from flyconn.access.lines import LineCatalog
from flyconn.access.neuronbridge import cached_off_target_count
from flyconn.data.convert.common import write_parquet


def _catalog(tmp_path: Path) -> LineCatalog:
    df = pd.DataFrame(
        {
            "line": ["SS1", "SS2", "SS3"],
            "cell_types": [["GNG232"], ["GNG232", "GNG100"], ["LB3c"]],
            "quality": [2, 1, 1],
            "doi": ["d1", "d2", "d3"],
            "first_author_year": ["A 2020", "B 2021", "C 2022"],
            "genotype": ["g", "g", "g"],
            "em_body_ids": [[], [], []],
        }
    )
    write_parquet(pa.Table.from_pandas(df, preserve_index=False), tmp_path / "lines.parquet")
    (tmp_path / "provenance.json").write_text(json.dumps({"counts": {}}))
    return LineCatalog(tmp_path)


def test_exact_glob_none(tmp_path: Path):
    hits = pd.DataFrame({"candidate": ["GNG232", "LB3", "ZZZ"]})
    out = attach_lines(hits, _catalog(tmp_path))
    g = out.iloc[0]
    assert g["line_match"] == "exact" and g["best_line"] == "SS2" and g["line_quality"] == 1
    assert not g["line_specific"]
    assert g["lines"].startswith("SS2 (q1)") and "SS1 (q2, specific)" in g["lines"]
    assert out.iloc[1]["line_match"] == "glob" and out.iloc[1]["best_line"] == "SS3"
    assert out.iloc[2]["line_match"] == "none" and pd.isna(out.iloc[2]["best_line"])


def test_not_looked_up():
    out = attach_lines(pd.DataFrame({"candidate": ["A"]}), None)
    assert out.iloc[0]["line_match"] == "not looked up"


def test_cached_off_targets_never_fetches(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLYCONN_CACHE", str(tmp_path))
    assert cached_off_target_count("SS9") is None  # nothing cached -> None, no network
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/access/test_handoff.py -v`
Expected: FAIL (`ModuleNotFoundError: flyconn.access.handoff`).

- [ ] **Step 3: Implement**

Append to `flyconn/access/neuronbridge.py`:

```python
def cached_off_target_count(
    line: str, *, version: str = DEFAULT_VERSION, max_images: int = 50
) -> int | None:
    """Distinct EM bodies matched to ``line`` from the local cache only; None if not cached."""
    root = cache_root() / "neuronbridge" / version / "metadata"
    meta_path = root / "by_line" / f"{line}.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    results: list[dict[str, Any]] = []
    for img in meta.get("results", [])[:max_images]:
        cds = img.get("files", {}).get("CDSResults")
        if not cds:
            continue
        p = root / "cdsresults" / cds
        if not p.exists():
            return None
        results.extend(json.loads(p.read_text()).get("results", []))
    return int(len(aggregate_cds_matches(results, direction="lm_to_em")))
```

Create `flyconn/access/handoff.py`:

```python
"""Driver lines for robustness-screen hits (Meissner et al. 2025; M13, ADR-0011).

Annotations are curated free text across nomenclatures, so every row records how the
line was matched. Lines are never invented; NeuronBridge counts come from the cache only.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flyconn.access.lines import LineCatalog
from flyconn.access.neuronbridge import cached_off_target_count
from flyconn.data.cache import store_dir
from flyconn.data.registry import get_dataset

CATALOG_REF = "flylight_lines@meissner2025"
PULL_HINT = f"flyconn data pull {CATALOG_REF}"


def load_catalog() -> LineCatalog | None:
    d = store_dir(get_dataset(CATALOG_REF))
    return LineCatalog(d) if (d / "lines.parquet").exists() else None


def _fmt(row: Any) -> str:
    q = "" if pd.isna(row["quality"]) else f"q{int(row['quality'])}"
    return f"{row['line']} ({q}{', specific' if row['specific'] else ''})"


def attach_lines(
    hits: pd.DataFrame, catalog: LineCatalog | None, *, max_lines: int = 5
) -> pd.DataFrame:
    out = hits.copy()
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "best_line",
            "line_quality",
            "line_specific",
            "line_match",
            "lines",
            "off_target_bodies",
        )
    }
    for label in out["candidate"].astype(str):
        match, found = "not looked up", pd.DataFrame()
        if catalog is not None:
            found = catalog.lines_for_type(label)
            match = "exact"
            if found.empty:
                found = catalog.lines_for_type(f"{label}*")
                match = "glob" if not found.empty else "none"
        if found.empty:
            cols["best_line"].append(None)
            cols["line_quality"].append(None)
            cols["line_specific"].append(None)
            cols["lines"].append("")
            cols["off_target_bodies"].append(None)
        else:
            best = found.iloc[0]
            cols["best_line"].append(str(best["line"]))
            cols["line_quality"].append(None if pd.isna(best["quality"]) else int(best["quality"]))
            cols["line_specific"].append(bool(best["specific"]))
            cols["lines"].append("; ".join(_fmt(r) for _, r in found.head(max_lines).iterrows()))
            cols["off_target_bodies"].append(cached_off_target_count(str(best["line"])))
        cols["line_match"].append(match)
    for k, v in cols.items():
        out[k] = v
    out["line_quality"] = out["line_quality"].astype("Int64")
    out["off_target_bodies"] = out["off_target_bodies"].astype("Int64")
    return out
```

Notes for the implementer:
- If `lines_for_type` sorts SS2 (quality 1) before SS1 (quality 2), the test expectation
  holds: it sorts by quality, then line.
- Export `attach_lines` and `load_catalog` from `flyconn/access/__init__.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/access -v`
Expected: PASS.

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/access tests/unit/access/test_handoff.py
git commit -m "feat(access): driver lines for screen hits, with match method and cached NeuronBridge counts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Evidence-card report, `handoff.csv` and CLI

**Files:**
- Create: `flyconn/report/robust_html.py`
- Modify: `flyconn/experiments/robustness.py` (`write_handoff`; call the report at the end
  of `run_robust_screen`)
- Modify: `flyconn/cli/main.py` (`run_cmd` routes specs with `robustness:`)
- Test: `tests/unit/experiments/test_robust_report.py`

**Interfaces:**
- Consumes: `RobustScreenResult` (Task 8); `attach_lines`, `load_catalog`, `PULL_HINT`
  (Task 9).
- Produces:

```python
def suggested_test(row: pd.Series, spec: ExperimentSpec) -> str
def write_handoff(result: RobustScreenResult, catalog: LineCatalog | None) -> pd.DataFrame   # writes handoff.csv
def render_robust_report(result: RobustScreenResult, handoff: pd.DataFrame, *, catalog_loaded: bool) -> str
```

- `handoff.csv` columns, in order: `label`, `tier`, `direction`, `effect_hz`, `ci_low`,
  `ci_high`, `q_value`, `failed_axes`, `untested_axes`, `thinly_tested`, `replicate`,
  `path_stability`, `best_line`, `line_quality`, `line_specific`, `line_match`, `lines`,
  `off_target_bodies`, `suggested_test`.
- `run_robust_screen` gains the keyword `catalog: LineCatalog | None | Literal["auto"] = "auto"`;
  `"auto"` means `load_catalog()`.

- [ ] **Step 1: Write the failing tests**

```python
"""Evidence cards, handoff.csv and the CLI route for robustness specs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from flyconn.cli.main import app
from flyconn.data.store import Store
from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import spec_from_dict

HANDOFF_COLS = [
    "label",
    "tier",
    "direction",
    "effect_hz",
    "ci_low",
    "ci_high",
    "q_value",
    "failed_axes",
    "untested_axes",
    "thinly_tested",
    "replicate",
    "path_stability",
    "best_line",
    "line_quality",
    "line_specific",
    "line_match",
    "lines",
    "off_target_bodies",
    "suggested_test",
]


def _d(store: Store, alpha: float = 1.0) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "card",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 2}},
        "duration_ms": 60,
        "trials": 4,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
        "report": {"alpha": alpha},
        "robustness": {
            "primary_readout": "targets",
            "readout_proxy": "targets drive nothing in particular (test 2026)",
            "w_syn": [0.3],
            "nt_samples": 0,
            "min_weights": [],
            "replicate": None,
            "random_null": {"n_sets": 2},
        },
    }


def test_report_and_handoff(small_store: Store, tmp_path: Path):
    run_robust_screen(
        spec_from_dict(_d(small_store)), out_dir=tmp_path / "o", store=small_store, catalog=None
    )
    h = pd.read_csv(tmp_path / "o" / "handoff.csv")
    assert list(h.columns) == HANDOFF_COLS and len(h) == 2
    assert set(h["line_match"]) == {"not looked up"}
    assert all("model predicts" in s and "targets drive nothing" in s for s in h["suggested_test"])
    html = (tmp_path / "o" / "report.html").read_text()
    assert "Model prediction." in html and "Calibration:" in html
    assert html.count("class='card'") == 2
    assert html.count("Caveats") >= 2
    assert "lines not looked up" in html and "flyconn data pull flylight_lines@meissner2025" in html


def test_no_hits(small_store: Store, tmp_path: Path):
    run_robust_screen(
        spec_from_dict(_d(small_store, alpha=1e-12)),
        out_dir=tmp_path / "o",
        store=small_store,
        catalog=None,
    )
    h = pd.read_csv(tmp_path / "o" / "handoff.csv")
    assert list(h.columns) == HANDOFF_COLS and h.empty
    assert "No reference hits" in (tmp_path / "o" / "report.html").read_text()


def test_cli_routes_robustness_specs(small_store: Store, tmp_path: Path, monkeypatch):
    import flyconn.experiments.robustness as rob

    seen: dict[str, object] = {}

    def fake(spec, *, out_dir, **kw):  # noqa: ANN001, ANN003
        seen["name"] = spec.name
        return rob.RobustScreenResult(
            Path(out_dir),
            spec,
            pd.DataFrame({"tier": []}),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    monkeypatch.setattr(rob, "run_robust_screen", fake)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(_d(small_store)))
    res = CliRunner().invoke(app, ["run", str(p), "--out", str(tmp_path / "o")])
    assert res.exit_code == 0, res.output
    assert seen["name"] == "card" and "robustness screen" in res.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_robust_report.py -v`
Expected: FAIL (`TypeError: unexpected keyword 'catalog'`).

- [ ] **Step 3: Implement**

In `flyconn/experiments/robustness.py`, add:

```python
from typing import Literal

from flyconn.access.handoff import PULL_HINT, attach_lines, load_catalog
from flyconn.access.lines import LineCatalog

HANDOFF_COLUMNS = [
    "label",
    "tier",
    "direction",
    "effect_hz",
    "ci_low",
    "ci_high",
    "q_value",
    "failed_axes",
    "untested_axes",
    "thinly_tested",
    "replicate",
    "path_stability",
    "best_line",
    "line_quality",
    "line_specific",
    "line_match",
    "lines",
    "off_target_bodies",
    "suggested_test",
]


def _stimulus_summary(spec: ExperimentSpec) -> str:
    return "; ".join(f"{s.select.describe()} at {s.rate_hz:g} Hz" for s in spec.stimulate)


def suggested_test(row: pd.Series, spec: ExperimentSpec) -> str:
    r = spec.robustness
    readout = r.primary_readout if r else "readout"
    text = (
        f"Silence {row['candidate']} (e.g. Kir2.1, GtACR1; examples, not recommendations) under "
        f"{_stimulus_summary(spec)}; the model predicts {readout} goes {row['direction']} by "
        f"{abs(float(row['difference'])):.1f} Hz."
    )
    if r and r.readout_proxy:
        text += f" Readout proxy (from the spec): {r.readout_proxy}"
    return text


def write_handoff(result: RobustScreenResult, catalog: LineCatalog | None) -> pd.DataFrame:
    h = attach_lines(result.hits, catalog) if not result.hits.empty else result.hits.copy()
    rows = pd.DataFrame(
        {
            "label": h.get("candidate", pd.Series(dtype=str)),
            "tier": h.get("tier"),
            "direction": h.get("direction"),
            "effect_hz": h.get("difference"),
            "ci_low": h.get("ci_low"),
            "ci_high": h.get("ci_high"),
            "q_value": h.get("q_value"),
            "failed_axes": h.get("failed_axes"),
            "untested_axes": h.get("untested_axes"),
            "thinly_tested": h.get("thinly_tested"),
            "replicate": h.get("replicate"),
            "path_stability": h.get("path_stability"),
            "best_line": h.get("best_line"),
            "line_quality": h.get("line_quality"),
            "line_specific": h.get("line_specific"),
            "line_match": h.get("line_match"),
            "lines": h.get("lines"),
            "off_target_bodies": h.get("off_target_bodies"),
            "suggested_test": [suggested_test(r, result.spec) for _, r in h.iterrows()],
        },
        columns=pd.Index(HANDOFF_COLUMNS),
    )
    rows.to_csv(result.out_dir / "handoff.csv", index=False)
    return rows
```

- Change the `run_robust_screen` signature to add
  `catalog: LineCatalog | None | Literal["auto"] = "auto"`.
- Before its `return`, add:

```python
    result = RobustScreenResult(out, spec, hits, eff, null_df, sel, prov)
    cat = load_catalog() if catalog == "auto" else catalog
    prov["robustness"]["line_catalog"] = (
        cat.provenance if cat is not None else f"not looked up ({PULL_HINT})"
    )
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    handoff = write_handoff(result, cat)
    from flyconn.report.robust_html import render_robust_report  # noqa: PLC0415 (matplotlib)

    (out / "report.html").write_text(
        render_robust_report(result, handoff, catalog_loaded=cat is not None)
    )
    return result
```

Create `flyconn/report/robust_html.py`:

```python
"""Evidence-card report for robustness-audited screens (M13, ADR-0011)."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import pandas as pd

from flyconn.access.handoff import PULL_HINT
from flyconn.report.html import _footer, _header, _table  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from flyconn.experiments.robustness import RobustScreenResult

CAVEATS = (
    "Model predictions from wiring plus Shiu LIF dynamics; not measurements.",
    "MaleCNS w_syn calibrated by protocol, not validated (ADR-0009).",
    "NT signs are predicted; sampled-sign variants assume the published probabilities are calibrated.",
    "The replicate runs at an uncalibrated w_syn on a different sex; disagreement may be biology or calibration.",
    "Line annotations are curated text and may not match EM type names one to one.",
    "'Strong' means robust within the tested axes, not true.",
)


def _e(x: object) -> str:
    return html.escape(str(x))


def _card(row: pd.Series, hit: pd.Series, catalog_loaded: bool) -> str:
    lines = (
        _e(row["lines"]) or "no annotated split-GAL4 in Meissner 2025"
        if catalog_loaded
        else f"lines not looked up (run <code>{_e(PULL_HINT)}</code>)"
    )
    ps = row["path_stability"]
    caveats = "".join(f"<li>{_e(c)}</li>" for c in CAVEATS)
    thin = (
        "<p class='warn'>Thinly tested: fewer than two graded axes ran.</p>"
        if row["thinly_tested"]
        else ""
    )
    return (
        "<div class='card'>"
        f"<h3>{_e(row['label'])} &middot; {_e(row['tier'])}</h3>"
        f"<p>Silencing moves the primary readout <b>{_e(row['direction'])}</b> by "
        f"{float(row['effect_hz']):.2f} Hz (95% CI {float(row['ci_low']):.2f} to "
        f"{float(row['ci_high']):.2f}), q {float(row['q_value']):.2g}.</p>{thin}"
        "<ul>"
        f"<li>w_syn: {_e(hit['w_syn'])} ({_e(hit['w_syn_detail'])})</li>"
        f"<li>NT signs: {_e(hit['nt'])} (holds {_e(hit['nt_holds'])}; flips: {_e(hit['nt_flips']) or 'none'})</li>"
        f"<li>threshold: {_e(hit['min_weight'])} ({_e(hit['min_weight_detail'])})</li>"
        f"<li>random null: {_e(hit['random_null'])} (p {_e(hit['random_null_p'])})</li>"
        f"<li>replicate (supporting, not graded): {_e(row['replicate'])}</li>"
        f"<li>path stability: {'n/a' if pd.isna(ps) else f'{float(ps):.2f}'}</li>"
        f"<li>lines: {lines} [{_e(row['line_match'])}]</li>"
        "</ul>"
        f"<p><b>Suggested test:</b> {_e(row['suggested_test'])}</p>"
        f"<details><summary>Caveats</summary><ul>{caveats}</ul></details>"
        "</div>"
    )


def render_robust_report(
    result: RobustScreenResult, handoff: pd.DataFrame, *, catalog_loaded: bool
) -> str:
    spec, prov = result.spec, result.provenance
    rob = prov.get("robustness", {})
    parts = [
        *_header(f"flyconn robustness screen: {spec.name}", spec.name, prov),
        "<style>.card{border:1px solid #ccc;border-radius:6px;padding:.5rem 1rem;margin:1rem 0}</style>",
        "<h2>Design</h2>",
        f"<p>{_e(rob.get('design'))}. Variants: "
        + ", ".join(_e(v["label"]) for v in rob.get("variants", []))
        + ". Every variant silences the reference run's candidates.</p>",
        f"<p class='note'>{_e(rob.get('statistics'))}.</p>",
        "<h2>Ranked hits</h2>",
    ]
    if handoff.empty:
        parts.append(
            "<p><b>No reference hits</b> at the spec's alpha; variants were still run.</p>"
        )
    else:
        parts.append(
            _table(
                handoff[
                    [
                        "label",
                        "tier",
                        "direction",
                        "effect_hz",
                        "q_value",
                        "failed_axes",
                        "best_line",
                        "line_match",
                    ]
                ]
            )
        )
        hits = result.hits.set_index("candidate")
        for _, row in handoff.iterrows():
            parts.append(_card(row, hits.loc[row["label"]], catalog_loaded))
    parts += [
        "<h2>Selection stability</h2>",
        "<p class='note'>Jaccard overlap between the reference selection and the one each "
        "variant's own ranking would have picked (candidates were fixed regardless).</p>",
        _table(result.selection_stability),
        "<h2>Not hits and controls</h2>",
        _table(
            result.effects[
                (result.effects["axis"] == "reference")
                & ~result.effects["candidate"].isin(handoff["label"])
            ]
        ),
        *_footer(prov),
    ]
    return "\n".join(parts)
```

In `flyconn/cli/main.py` `run_cmd`, before `if spec.sweep is not None:`, add:

```python
    if spec.robustness is not None:
        from flyconn.experiments import robustness as rob

        res = rob.run_robust_screen(spec, out_dir=out)
        tiers = res.hits["tier"].value_counts().to_dict() if not res.hits.empty else {}
        typer.echo(f"robustness screen: {len(res.hits)} reference hits; tiers {tiers}")
        typer.echo(f"model prediction; report: {res.out_dir / 'report.html'}")
        return
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit -v`
Expected: PASS (the whole unit suite).

- [ ] **Step 5: Run the gate and commit**

```bash
scripts/check.sh
git add flyconn/report/robust_html.py flyconn/experiments/robustness.py flyconn/cli/main.py tests/unit/experiments/test_robust_report.py
git commit -m "feat(report): evidence cards and handoff.csv for robustness screens; CLI route

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Golden run on MaleCNS, ADR-0011, docs

**Files:**
- Create: `examples/specs/w2_robust_pathway_mn9.yaml`
- Create: `tests/golden/test_robust_screen.py`
- Create: `benchmarks/robust_screen_malecns.json` (written by the golden test)
- Create: `docs/adr/0011-robustness-audited-screens.md`, `docs/tutorials/robust-screens.md`
- Modify: `docs/GOLDEN_RESULTS.md` (section **6j** after 6h; M12 inserts 6i), `mkdocs.yml`,
  `CHANGELOG.md`, `docs/PROGRESS.md`, `docs/adr/README.md`

**Interfaces:**
- Consumes: everything above via `flyconn run` / `run_robust_screen`.

- [ ] **Step 1: Pick the negative-control type** (read-only query on real data)

```bash
FLYCONN_CACHE=/Users/moizp/google-fly/.cache/flyconn uv run python - <<'EOF'
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.graph.paths import find_paths
s = Store.open("malecns@1.0")
n = s.neurons()
src = n[n.cell_type.fillna("").str.startswith("LB3")].neuron_id.tolist()
dst = n[n.cell_type == "MN9"].neuron_id.tolist()
m = ConnectivityMatrix.from_store(s, min_weight=1)
p = find_paths(m, src, dst, max_hops=3, min_weight=1)
on = {i for path in p["path"] for i in path}
for t in ("LC4", "T4a", "Mi1", "LPLC2"):
    ids = set(n[n.cell_type == t].neuron_id)
    print(t, len(ids), "on path" if ids & on else "no LB3->MN9 route at min_weight 1, <=3 hops")
EOF
```

Use the first type that prints "no LB3->MN9 route". Record the command and its output in
the YAML comment.

- [ ] **Step 2: Write the spec** `examples/specs/w2_robust_pathway_mn9.yaml`

```yaml
# M13 robustness-audited pathway screen (MaleCNS v1.0; w_syn calibrated by protocol, not
# validated; ADR-0009, ADR-0011). Same screen as w2_sweep_pathway_mn9.yaml (GOLDEN_RESULTS
# 6h) plus a robustness block. Negative control: <TYPE FROM STEP 1> has no LB3 -> MN9 route
# within 3 hops at min_weight 1 (checked with find_paths on <date>; see GOLDEN_RESULTS 6j).
name: w2_robust_pathway_mn9
dataset: malecns@1.0
network: {min_weight: 1, sign_policy: argmax}
stimulate:
  - select: {cell_type: "LB3*"}
    rate_hz: 100
sweep:
  silence_each:
    rank_by: pathway
    pathway: {readout: MN9, max_hops: 3, min_weight: 5, min_edge_fraction: 0.01}
    group_by: cell_type
    max_items: 10
readouts:
  - name: MN9
    select: {cell_type: "MN9"}
  - name: descending
    select: {super_class: descending}
duration_ms: 300
trials: 10
seed: 0
device: cpu
dtype: float32
controls: {degree_preserving_rewire: 1, sign_shuffle: 1, unstimulated_baseline: true}
robustness:
  primary_readout: MN9
  readout_proxy: "MN9 motor neurons drive proboscis extension (Shiu et al. 2024, Nature 634:210)"
  w_syn: auto
  nt_samples: 8
  min_weights: [3, 5]
  replicate: flywire@783
  random_null: {n_sets: 40}
  extra_candidates:
    - {cell_type: "<TYPE FROM STEP 1>"}
```

Replace both `<TYPE FROM STEP 1>` markers with the chosen type before committing.

- [ ] **Step 3: Write the golden test** `tests/golden/test_robust_screen.py`

```python
"""M13 golden: robustness-audited pathway screen on MaleCNS (ADR-0011; GOLDEN_RESULTS 6j)."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import pandas as pd
import pytest

from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import load_spec

pytestmark = pytest.mark.golden
ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "examples" / "specs" / "w2_robust_pathway_mn9.yaml"
# GOLDEN_RESULTS 6h pathway screen at 0.188 mV: relay -> (difference, ci_low, ci_high)
SIX_H = {
    "GNG232": (-14.5, -18.8, -9.5),
    "DNge080": (-4.3, -8.7, 0.3),
    "DNge051": (2.5, -2.3, 7.7),
    "GNG215": (-1.7, -6.7, 3.7),
    "GNG132": (1.7, -3.8, 7.3),
    "GNG130": (0.2, -5.5, 6.0),
}


def test_robust_screen_malecns(tmp_path: Path):
    spec = load_spec(SPEC)
    t0 = time.time()
    res = run_robust_screen(spec, out_dir=tmp_path / "robust")
    wall = time.time() - t0
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    out = res.out_dir
    print(f"\nM13 robust screen: {wall:.0f} s, peak RSS {rss_gb:.1f} GB")
    print(res.hits.to_string(index=False))
    print(pd.read_csv(out / "handoff.csv").to_string(index=False))
    print(res.selection_stability.to_string(index=False))

    for f in ("report.html", "handoff.csv", "hits.parquet", "null.parquet", "provenance.json"):
        assert (out / f).exists()
    ref = res.effects[res.effects["axis"] == "reference"].set_index("candidate")
    for relay, (d, lo, hi) in SIX_H.items():
        assert lo <= float(ref.loc[relay, "difference"]) <= hi, relay
    control = ref[ref["role"] == "control"]
    assert len(control) == 1
    assert control.index[0] not in set(res.hits["candidate"])
    control_p = res.provenance["robustness"]["control_null_p"][control.index[0]]
    print(
        f"negative control {control.index[0]}: effect "
        f"{float(control['difference'].iloc[0]):.2f} Hz, random-null p {control_p}"
    )
    assert control_p is not None
    assert rss_gb < 16
    (ROOT / "benchmarks" / "robust_screen_malecns.json").write_text(
        json.dumps(
            {
                "spec": SPEC.name,
                "wall_s": round(wall),
                "peak_rss_gb": round(rss_gb, 2),
                "variants": len(res.provenance["robustness"]["variants"]),
                "null_runs": len(res.null),
                "machine": "M4 Pro, CPU float32",
            },
            indent=1,
        )
    )
```

"Never in the hit list" holds by construction: `score_hits` skips role `control`. The
real honesty check is the control's random-null p, which is printed and must exist.
- After Step 4, if the printed `control_p` is >= 0.05, add `assert control_p >= 0.05`
  as a pinned regression and commit it with the docs.
- If it is < 0.05 (a pathless type beats random silencing), do **not** add the
  assertion and do not tune anything. Record it in 6j as a finding: the random null is
  not specific enough at this size bin. Report it to the owner.

- [ ] **Step 4: Run the golden test**

Run: `FLYCONN_CACHE=/Users/moizp/google-fly/.cache/flyconn uv run pytest tests/golden/test_robust_screen.py -m golden -s -v`
Expected: PASS, in about 30 minutes (extrapolated). Save the printed tables; they are the
6j numbers.

If any of the six 6h relay effects falls outside its 6h CI: stop and investigate. The
reference run must reproduce 6h (same spec, seeds and w_syn; the extra control only
changes q). Do not widen the bounds.

- [ ] **Step 5: Docs**
  - **`docs/GOLDEN_RESULTS.md`:** add `### 6j. Robustness-audited pathway screen (ADR-0011; <run date>)`
    directly after 6h, or after 6i if M12 has merged. Include:
    - the hit table (tier, effect, CI, failed axes, replicate, best line);
    - the selection-stability table;
    - the negative control's effect and null p;
    - wall time and peak RSS;
    - what is robust and what is not, in the style of 6h.
    Every number must come from Step 4's output.
  - **`docs/adr/0011-robustness-audited-screens.md`** (status Accepted). Cover:
    - context: 6h's by-hand fragility;
    - the decision: option A, one axis at a time;
    - alternatives rejected: B, joint Monte Carlo (loses per-axis attribution); C, graph
      prefilter (drops dynamics-driven hits);
    - the label, tier and null-p rules;
    - the replicate being reported, not graded;
    - the absence of a correction across variants;
    - consequences, including the cost.
    Add a row to `docs/adr/README.md`.
  - **`docs/tutorials/robust-screens.md`:**
    - how to write a `robustness:` block;
    - running `flyconn run examples/specs/w2_robust_pathway_mn9.yaml --out DIR`;
    - how to read `handoff.csv`;
    - what the tiers mean;
    - the caveats list;
    - the prerequisite pulls (`malecns@1.0 --level nt-probs`, `flywire@783 --level weights`,
      `flylight_lines@meissner2025`).
  - **`mkdocs.yml`:** add `- Robust screens: tutorials/robust-screens.md` after the
    Tutorials entry.
  - **`CHANGELOG.md`:** add `### M13 robustness-audited screens` under `## [Unreleased]`.
  - **`docs/PROGRESS.md`:** add a new top entry dated with the run date, giving the key
    numbers from 6j and what was not done.
  - Run `uv run ruff format docs/`.

- [ ] **Step 6: Gate and commit**

```bash
uv run ruff format docs/
scripts/check.sh
git add examples/specs/w2_robust_pathway_mn9.yaml tests/golden/test_robust_screen.py benchmarks/robust_screen_malecns.json docs mkdocs.yml CHANGELOG.md
git commit -m "docs(m13): ADR-0011 robustness-audited screens; MaleCNS golden run (GOLDEN_RESULTS 6j); tutorial

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Merge preparation (no merge without the owner's go-ahead)**
  - If `m12-classify` has merged to `main`, run `git merge main` in the worktree.
  - Resolve the append-only conflicts in `CHANGELOG.md`, `docs/PROGRESS.md`,
    `docs/GOLDEN_RESULTS.md` (keep 6i, then 6j) and `mkdocs.yml` (keep both nav lines).
  - Rerun `scripts/check.sh`.
  - Report to the owner. Do not merge into `main` or push without explicit instruction.
