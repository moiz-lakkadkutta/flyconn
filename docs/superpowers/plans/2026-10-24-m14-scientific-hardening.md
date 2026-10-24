# M14 Scientific Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the weak checks that M13 (v0.4.0) left open, so that a robustness tier means what it says. Make the random null informative, add a negative control that can actually fail, use a paired test, make the null seeds and the plan hash stable, and add a sugar-only W2 spec that matches the MaleCNS calibration stimulus.

**Architecture:** Every change lives in the existing M13 modules: `flyconn/experiments/{stats,spec,sweep,robust_null,robust_scoring,robustness}.py` and `flyconn/report/robust_html.py`. No new subsystem. New behaviour comes from new spec keys whose defaults are set for robustness screens only. Plain experiments and plain sweeps produce the same numbers as before.

**Tech Stack:** Python ≥3.11, numpy, pandas, scipy.stats (`wilcoxon`), pytest + hypothesis, `uv`, ruff, pyright.

**Spec:** There is no separate design doc. The requirements come from:
- `docs/PROGRESS.md`, entry 2026-10-14: "Weak checks still open" and "Not done".
- `docs/GOLDEN_RESULTS.md` §6j: the "Weak checks" bullets.
- `docs/PROGRESS.md`, entry 2026-10-06: the W2 specs stimulate all LB3.
- ADR-0011.

Task 7 writes ADR-0012, which records the decisions below.

**Branch:** `m14-hardening` (one branch per milestone).

## Decisions taken by this plan (owner may veto before Task 4 / 5 start)

- **D1. Tier rule for a degenerate null.** At least 50 % of a key's random sets have an effect of exactly 0 (`DEGENERATE_NULL_FRACTION`, unchanged). The random-null axis then scores `uninformative`, which counts as neither a pass nor a fail. It also does not count as "ran" for the thinly-tested flag. Today such a null counts as `robust` (GNG232: p = 1/41, "robust"). This was the pending owner decision from M13.
- **D2. Activity-matched null pool, default on for robustness screens.** Random sets are drawn only from neurons of the same super_class that fire at least one spike in the reference *stimulated* run. A spec can opt out with `random_null: {pool: all}`. Reason: silencing a neuron that never fires cannot change a run (`sweep.py` module docstring). The old pool therefore produced nulls of exactly 0.
- **D3. Paired statistics, default on for robustness screens.** Trial *k* of every condition uses Poisson seed stream *k* (`flyconn/sim/engine.py:164`). So each variant is compared with the stimulated run trial by trial:
  - CI: paired bootstrap of the per-trial differences;
  - p: Wilcoxon signed-rank.

  `cohens_d` keeps the unpaired, pooled-SD definition so it stays comparable across modes. The key is `report.statistics: paired | unpaired`. It is valid only with a `sweep`, and plain sweeps default to `unpaired`.
- **D4. Automatic off-route negative control, default 1 for robustness screens.** The control is the most active group (summed stimulated rate ≥ 1 Hz) that has no neuron on any path from the stimulus to the primary readout within `max_hops` hops at `min_weight` 1. `robustness.negative_controls: 0` turns it off. Any control that is silent under the stimulus is flagged `uninformative: silent under the stimulus`, as LC4 is today.
- **D5. Null seeds keyed by the key string.** The seed changes from `rng([seed + 3000 + i, k_idx])` to `rng([seed + 3000 + i, zlib.crc32(key)])`. Adding a candidate then no longer reshuffles every other key's sets.
- **D6. Plan hash includes store fingerprints.** The fingerprint is the SHA-256 of `{ref, level, sorted input sha256}` from each store's `provenance.json`, for the main store and for the replicate store when one is present.

## Global Constraints

- Python ≥3.11; `uv`; `ruff`; `pyright` strict on public API; `pytest` + `hypothesis`.
- Unit tests never touch the network; use the `small_store` / `xref_store` fixtures (`tests/conftest.py`) built from `flyconn.testing.synthetic_connectome`.
- Every result carries provenance. Every new option and every default it took is written to `provenance.json`.
- Scientific honesty: simulation outputs are model predictions. MaleCNS w_syn is "CALIBRATED BY PROTOCOL, not validated". Controls are on by default.
- Numbers in docs come from code run in this repo, never from memory. GOLDEN_RESULTS §6k numbers are printed by `benchmarks/robust_screen_details.py`.
- Laptop budget: 16 GB RAM. The M13 golden run peaked at 12.6 GB. The auto control adds one `find_paths` call at `min_weight` 1 (204,286 paths on MaleCNS), so record peak RSS again.
- `scripts/check.sh` must pass before every commit. Conventional commits. Each commit leaves tests green.
- Commit trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0146ujhCdFL57yDjC7Thhn4c
  ```
- Behaviour change for robustness specs: new spec fields change `source_sha256`. So an M13 output directory refuses to resume under M14. CHANGELOG states this.

## Review Focus

1. **Paired test with all-zero differences** (a candidate silenced with no effect in any trial). Expected: difference 0, CI (0, 0), p = 1.0, no scipy exception. Pinned in Task 1.
2. **Unequal trial counts in a paired comparison** (future code passes mismatched arrays). Expected: `ValueError` naming both lengths, never silent truncation. Pinned in Task 1.
3. **Activity-matched pool smaller than the set size** (few active neurons of that super_class). Expected: the key is skipped with a reason that names the active pool, and the axis is `not_run`, never a crash and never a silent fall-back to the full pool. Pinned in Task 3.
4. **No off-route active group exists** (small or densely connected networks). Expected: no control is added, and provenance says `"no active off-route group"`. The screen still runs. Pinned in Task 5.
5. **Rerun into an M13-era or other-store output directory.** Expected: `FileExistsError` because the plan hash differs, never a mixed resume. Pinned in Task 2.

---

### Task 1: Paired statistics for sweeps

**Files:**
- Modify: `flyconn/experiments/stats.py` (add `compare_paired`)
- Modify: `flyconn/experiments/spec.py:89-92` (`ReportOptions`), `spec.py:~459` (parsing)
- Modify: `flyconn/experiments/sweep.py` (variant rows use the chosen comparison; provenance)
- Test: `tests/unit/experiments/test_stats_paired.py` (new), `tests/unit/experiments/test_spec_robustness.py`, `tests/unit/experiments/test_sweep_fixed.py`

**Interfaces:**
- Produces: `compare_paired(test: np.ndarray, control: np.ndarray, *, seed: int = 0, n_boot: int = 2000) -> dict[str, float]` with the same keys as `compare_groups`.
- Produces: `ReportOptions.statistics: Literal["paired", "unpaired"]` (default `"unpaired"`; `"paired"` when the spec has a `robustness` block and `report.statistics` is absent).
- Produces: `run_sweep` provenance key `prov["statistics"]`, a string beginning `"paired"` or `"unpaired"`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/experiments/test_stats_paired.py`:

```python
"""Paired comparison across shared Poisson input streams (M14 D3)."""

from __future__ import annotations

import numpy as np
import pytest

from flyconn.experiments.stats import compare_groups, compare_paired


def test_paired_detects_shift_that_unpaired_misses():
    rng = np.random.default_rng(0)
    control = rng.normal(20, 8, size=10)  # large trial-to-trial spread from shared inputs
    test = control - 1.0 + rng.normal(0, 0.1, size=10)  # small, consistent per-trial drop
    paired = compare_paired(test, control, seed=0)
    unpaired = compare_groups(test, control, seed=0)
    assert paired["difference"] == pytest.approx(unpaired["difference"])
    assert paired["p_value"] < 0.01 < unpaired["p_value"]
    assert paired["ci_high"] < 0
    assert paired["ci_high"] - paired["ci_low"] < unpaired["ci_high"] - unpaired["ci_low"]


def test_paired_all_zero_differences():
    x = np.array([3.0, 5.0, 0.0, 7.0])
    r = compare_paired(x, x.copy())
    assert r["difference"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["p_value"] == 1.0


def test_paired_same_keys_as_unpaired():
    a, b = np.arange(6.0), np.arange(6.0)[::-1]
    assert set(compare_paired(a, b)) == set(compare_groups(a, b))


def test_paired_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal trial counts, got 3 and 4"):
        compare_paired(np.ones(3), np.ones(4))
```

Add to `tests/unit/experiments/test_spec_robustness.py` (its builder is `_d(store, robustness, sweep=None)`):

```python
def test_statistics_defaults_and_validation(small_store: Store):
    from flyconn.experiments.spec import spec_from_dict

    rob = _d(small_store, {"primary_readout": "targets"})
    assert spec_from_dict(rob).report.statistics == "paired"
    rob_unpaired = {**rob, "report": {"statistics": "unpaired"}}
    assert spec_from_dict(rob_unpaired).report.statistics == "unpaired"
    plain = _d(small_store, None)  # robustness: None
    assert spec_from_dict(plain).report.statistics == "unpaired"
    with pytest.raises(ValueError, match="report statistics must be 'paired' or 'unpaired'"):
        spec_from_dict({**rob, "report": {"statistics": "welch"}})
    no_sweep = {k: v for k, v in plain.items() if k != "sweep"}
    with pytest.raises(ValueError, match="report statistics: paired needs a sweep"):
        spec_from_dict({**no_sweep, "report": {"statistics": "paired"}})
```

Add to `tests/unit/experiments/test_sweep_fixed.py` (its builder is `_d(store, **over)`):

```python
def test_paired_statistics_recorded_and_used(small_store: Store, tmp_path: Path):
    d = _d(small_store)
    up = run_sweep(spec_from_dict(d), out_dir=tmp_path / "u", store=small_store)
    pd_ = run_sweep(
        spec_from_dict(_d(small_store, report={"statistics": "paired"})),
        out_dir=tmp_path / "p",
        store=small_store,
    )
    assert up.provenance["statistics"].startswith("unpaired")
    assert pd_.provenance["statistics"].startswith("paired")
    assert list(up.sweep["difference"]) == pytest.approx(list(pd_.sweep["difference"]))
    assert list(up.sweep["p_value"]) != list(pd_.sweep["p_value"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/experiments/test_stats_paired.py tests/unit/experiments/test_spec_robustness.py::test_statistics_defaults_and_validation tests/unit/experiments/test_sweep_fixed.py::test_paired_statistics_recorded_and_used -v`
Expected: FAIL with `ImportError: cannot import name 'compare_paired'`.

- [ ] **Step 3: Implement**

In `flyconn/experiments/stats.py`, after `compare_groups`:

```python
def compare_paired(
    test: np.ndarray, control: np.ndarray, *, seed: int = 0, n_boot: int = 2000
) -> dict[str, float]:
    """Per-trial paired difference: trial k of both conditions shares Poisson seed stream k.

    CI is a bootstrap over trials of the mean per-trial difference; p is Wilcoxon
    signed-rank (1.0 when every difference is zero). ``cohens_d`` keeps the unpaired,
    pooled-s.d. definition of :func:`compare_groups` so effect sizes compare across modes.
    """
    test = np.asarray(test, dtype=float)
    control = np.asarray(control, dtype=float)
    if test.shape != control.shape:
        msg = f"paired comparison needs equal trial counts, got {len(test)} and {len(control)}"
        raise ValueError(msg)
    diff = test - control
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    if not np.any(diff):
        p = 1.0
    else:
        p = float(cast("Any", sps.wilcoxon(diff, alternative="two-sided")).pvalue)
    d = cohens_d(test, control)
    d = float(np.clip(d, -10, 10)) if np.isfinite(d) else float(np.sign(d) * 10)
    return {
        "mean_test": float(test.mean()),
        "mean_control": float(control.mean()),
        "difference": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_value": p,
        "cohens_d": d,
        "n_test": len(test),
        "n_control": len(control),
    }
```

In `flyconn/experiments/spec.py`, change `ReportOptions`:

```python
@dataclass(frozen=True)
class ReportOptions:
    top_neurons: int = 30
    alpha: float = 0.05
    statistics: str = "unpaired"  # "paired" | "unpaired"; paired only with a sweep (M14 D3)
```

And replace the `report=ReportOptions(...)` line (~459) with a value computed before the `ExperimentSpec(...)` call:

```python
    stat_default = "paired" if d.get("robustness") is not None else "unpaired"
    statistics = str(rep.get("statistics", stat_default))
    if statistics not in ("paired", "unpaired"):
        msg = f"report statistics must be 'paired' or 'unpaired', got {statistics!r}"
        raise ValueError(msg)
    if statistics == "paired" and sweep is None:
        msg = "report statistics: paired needs a sweep (variants share the stimulated seeds)"
        raise ValueError(msg)
```

```python
report = (
    ReportOptions(int(rep.get("top_neurons", 30)), float(rep.get("alpha", 0.05)), statistics),
)
```

In `flyconn/experiments/sweep.py`, import `compare_paired` alongside `compare_groups`, and in `run_sweep` before the variant loop:

```python
    paired = spec.report.statistics == "paired"
    compare = compare_paired if paired else compare_groups
```

Then replace `r = compare_groups(test, ref, seed=spec.seed)` with `r = compare(test, ref, seed=spec.seed)`. In the `prov = base_provenance(...) | {...}` dict add:

```python
        "statistics": (
            "paired: per-trial differences vs the reference run (shared Poisson seed "
            "streams), trial-bootstrap 95% CI, Wilcoxon signed-rank p"
            if paired
            else "unpaired: bootstrap 95% CI, Mann-Whitney p (conservative given shared "
            "Poisson seed streams)"
        ),
```

In `flyconn/experiments/robustness.py`, change the `"statistics"` entry of the robustness provenance to:

```python
            "statistics": f"{spec.report.statistics} (see sweep provenance); "
            "no correction across variants",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/experiments -q`
Expected: PASS. Existing robust-screen unit tests now run paired. Any assertion that pinned an unpaired p or CI must be checked. If one fails only because the default changed, set `"report": {"alpha": 2.0, "statistics": "unpaired"}` in that test's builder and note it in the commit body. Do not loosen assertions.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/stats.py flyconn/experiments/spec.py flyconn/experiments/sweep.py flyconn/experiments/robustness.py tests/unit/experiments
git commit -m "feat(experiments): paired per-trial statistics for sweeps (M14 D3)"
```

---

### Task 2: Stable null seeds and store-aware plan hash

**Files:**
- Modify: `flyconn/experiments/robust_null.py` (`draw_null_sets` seed)
- Modify: `flyconn/experiments/robustness.py` (`_plan_hash`, `store_fingerprint`, replicate store resolved before the hash)
- Test: `tests/unit/experiments/test_robust_null.py`, `tests/unit/experiments/test_robust_screen.py`

**Interfaces:**
- Produces: `store_fingerprint(store: Store) -> str` (hex SHA-256) in `flyconn.experiments.robustness`.
- Produces: `_plan_hash(spec, plan, fingerprints: dict[str, str]) -> str`; provenance `robustness.store_fingerprints: dict[str, str]` (store ref → fingerprint).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/experiments/test_robust_null.py`:

```python
def test_sets_of_a_key_do_not_depend_on_other_candidates():
    n = _neurons()
    alone, _ = draw_null_sets(n, [Candidate("A", (1, 2))], set(), n_sets=4, seed=0)
    more, _ = draw_null_sets(
        n, [Candidate("A", (1, 2)), Candidate("D", (31,))], set(), n_sets=4, seed=0
    )
    key = null_key(n, (1, 2))
    assert [s for s in alone if s.key == key] == [s for s in more if s.key == key]
```

D's neuron 31 is descending, so adding D leaves the `central|2` pool unchanged. Any difference in A's sets can therefore only come from the seed.

Append to `tests/unit/experiments/test_robust_screen.py`:

```python
def test_changed_store_refuses_resume(small_store: Store, tmp_path: Path):
    from flyconn.experiments.robustness import store_fingerprint

    spec = spec_from_dict(_d(small_store))
    run_robust_screen(spec, out_dir=tmp_path / "o", store=small_store)
    before = store_fingerprint(small_store)
    small_store.provenance["inputs"] = [{"name": "x", "sha256": "0" * 64, "bytes": 1}]
    assert store_fingerprint(small_store) != before
    with pytest.raises(FileExistsError, match="different robustness plan"):
        run_robust_screen(spec, out_dir=tmp_path / "o", store=small_store)


def test_fingerprints_in_provenance(small_store: Store, tmp_path: Path):
    from flyconn.experiments.robustness import store_fingerprint

    res = run_robust_screen(
        spec_from_dict(_d(small_store)), out_dir=tmp_path / "o", store=small_store
    )
    fp = res.provenance["robustness"]["store_fingerprints"]
    assert fp == {small_store.ref: store_fingerprint(small_store)}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/experiments/test_robust_null.py tests/unit/experiments/test_robust_screen.py -k "other_candidates or changed_store or fingerprints" -v`
Expected: the null test FAILS (the sets differ because `k_idx` shifts), and the two screen tests FAIL with `ImportError: store_fingerprint`.

- [ ] **Step 3: Implement**

`robust_null.py`: `import zlib`. In `draw_null_sets`, drop `enumerate` and change the RNG:

```python
    for key in keys:
        ...
        for i in range(n_sets):
            s = seed + 3000 + i
            # keyed by the key text, so adding a candidate never reshuffles other keys' sets
            rng = np.random.default_rng([s, zlib.crc32(key.encode())])
```

`robustness.py`:

```python
def store_fingerprint(store: Store) -> str:
    """SHA-256 of a store's dataset ref, level and input checksums (its provenance.json)."""
    prov = store.provenance
    payload = {
        "ref": store.ref,
        "level": prov.get("level"),
        "inputs": sorted(str(i.get("sha256", "")) for i in prov.get("inputs", [])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _plan_hash(spec: ExperimentSpec, plan: list[Variant], fingerprints: dict[str, str]) -> str:
    payload = {
        "spec": spec.source_sha256,
        "variants": [(v.label, v.spec.source_sha256) for v in plan],
        "stores": fingerprints,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

In `run_robust_screen`, move the store resolution and replicate-store lookup above the hash. Replace the lines from `plan_hash = _plan_hash(spec, plan)` to `store = store or ...` with:

```python
    store = store or Store(pull(spec.dataset, level="weights").store_dir)
    rep_v = next((v for v in plan if v.axis == "replicate"), None)
    rep_store = None
    if rep_v is not None:
        rep_store = replicate_store or _replicate_store(rep_v.spec.dataset)
    fingerprints = {store.ref: store_fingerprint(store)}
    if rep_store is not None:
        fingerprints[rep_store.ref] = store_fingerprint(rep_store)
    plan_hash = _plan_hash(spec, plan, fingerprints)
    out = Path(out_dir)
    _open_out(out, plan_hash)
```

Then, in the replicate section further down, delete the now-duplicate `rep_v = next(...)` and `rep_store = replicate_store or _replicate_store(...)` lines and keep `if rep_v is not None:` / `if rep_store is None:` as they are. Add `"store_fingerprints": fingerprints,` to the `"robustness"` provenance dict.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/experiments -q`
Expected: PASS. If `test_deterministic` in `test_robust_null.py` relied on old seeds by value, it still passes, because it compares two calls.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments/robust_null.py flyconn/experiments/robustness.py tests/unit/experiments
git commit -m "fix(robustness): null seeds keyed by bin; plan hash covers store checksums (M14 D5, D6)"
```

---

### Task 3: Activity-matched random-null pool

**Files:**
- Modify: `flyconn/experiments/spec.py` (`RobustnessOptions.random_null_pool`, parsing of `random_null`)
- Modify: `flyconn/experiments/robust_null.py` (`draw_null_sets(..., pool_ids=...)`)
- Modify: `flyconn/experiments/robustness.py` (`_active_ids`, pass pool, provenance)
- Test: `tests/unit/experiments/test_robust_null.py`, `tests/unit/experiments/test_spec_robustness.py`, `tests/unit/experiments/test_robust_screen.py`

**Interfaces:**
- Consumes: `store_fingerprint` and the moved store resolution (Task 2).
- Produces: `RobustnessOptions.random_null_pool: str = "active"` (`"active" | "all"`).
- Produces: `draw_null_sets(neurons, candidates, excluded, *, n_sets, seed, pool_ids: set[int] | None = None)`. `None` means every neuron of the super_class.
- Produces: provenance `robustness.random_null.pool` (`"active"` or `"all"`) and `robustness.random_null.n_active` (int or None).

- [ ] **Step 1: Write the failing tests**

`test_robust_null.py`:

```python
def test_pool_ids_restricts_and_skip_names_active_pool():
    n = _neurons()
    active = {10, 11, 12, 13, 14}
    sets, skipped = draw_null_sets(
        n, [Candidate("A", (1, 2))], set(), n_sets=6, seed=0, pool_ids=active
    )
    assert sets and all(set(s.neuron_ids) <= active for s in sets)
    sets, skipped = draw_null_sets(
        n, [Candidate("B", (1, 2, 3, 4))], set(), n_sets=3, seed=0, pool_ids={10, 11}
    )
    assert sets == []
    assert skipped["central|4"] == (
        "pool of 2 active central neurons is smaller than the set size 4"
    )
```

`test_spec_robustness.py`:

```python
def test_random_null_pool_option(small_store: Store):
    from flyconn.experiments.spec import spec_from_dict

    base = {"primary_readout": "targets"}
    assert spec_from_dict(_d(small_store, base)).robustness.random_null_pool == "active"  # type: ignore[union-attr]
    d = _d(small_store, {**base, "random_null": {"n_sets": 5, "pool": "all"}})
    assert spec_from_dict(d).robustness.random_null_pool == "all"  # type: ignore[union-attr]
    bad = _d(small_store, {**base, "random_null": {"pool": "silent"}})
    with pytest.raises(ValueError, match="random_null pool must be 'active' or 'all'"):
        spec_from_dict(bad)
```

`test_robust_screen.py`:

```python
def test_active_pool_counts_neurons_that_fired(small_store: Store, tmp_path: Path):
    res = run_robust_screen(
        spec_from_dict(_d(small_store)), out_dir=tmp_path / "o", store=small_store
    )
    rn = res.provenance["robustness"]["random_null"]
    counts = pd.read_parquet(tmp_path / "o" / "variants" / "reference" / "counts.parquet")
    m = (counts["condition"] == "stimulated") & (counts["control"] == "none")
    fired = set(counts.loc[m & (counts["count"] > 0), "neuron_id"].astype(int))
    assert rn["pool"] == "active" and rn["n_active"] == len(fired) > 0
```

`res.null` does not store neuron ids. The subset property ("every set is drawn from the active ids") is therefore pinned by the `draw_null_sets` unit test above. This test pins the wiring.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/experiments -k "pool" -v`
Expected: FAIL (`unexpected keyword argument 'pool_ids'`; `no attribute random_null_pool`).

- [ ] **Step 3: Implement**

`spec.py`: add `random_null_pool: str = "active"` to `RobustnessOptions` (after `random_null_sets`). In `_robustness`, in the `isinstance(null_raw, dict)` branch, read the pool:

```python
    pool = "active"
    ...
    elif isinstance(null_raw, dict):
        nd = cast("dict[str, Any]", null_raw)
        n_sets = int(nd.get("n_sets", 40))
        pool = str(nd.get("pool", "active"))
    ...
    if pool not in ("active", "all"):
        msg = f"robustness random_null pool must be 'active' or 'all', got {pool!r}"
        raise ValueError(msg)
```

Then pass `random_null_pool=pool` to `RobustnessOptions(...)`.

`robust_null.py`, `draw_null_sets` gets the keyword `pool_ids: set[int] | None = None`:

```python
        pool_df = neurons[neurons["super_class"].astype(str) == sc]
        ids = set(pool_df["neuron_id"].astype(np.int64).tolist()) - blocked
        if pool_ids is not None:
            ids &= pool_ids
        pool = np.array(sorted(ids), dtype=np.int64)
        if len(pool) < size:
            kind = f"active {sc}" if pool_ids is not None else sc
            skipped[key] = f"pool of {len(pool)} {kind} neurons is smaller than the set size {size}"
            continue
```

Update the module docstring: "Sets are drawn from neurons of the same super_class; with `pool_ids` (the activity-matched pool, M14 D2) only from those ids."

`robustness.py`:

```python
def _active_ids(ref_dir: Path) -> set[int]:
    """Neurons with at least one spike in the reference stimulated run (counts.parquet)."""
    c = pd.read_parquet(ref_dir / "counts.parquet")
    m = (c["condition"] == "stimulated") & (c["control"] == "none") & (c["count"] > 0)
    return {int(i) for i in c.loc[m, "neuron_id"]}
```

In the null block:

```python
        pool_ids = _active_ids(vdir / "reference") if r.random_null_pool == "active" else None
        sets, skipped = draw_null_sets(
            neurons, cands, excluded, n_sets=r.random_null_sets, seed=spec.seed, pool_ids=pool_ids
        )
```

Initialise `pool_ids: set[int] | None = None` before `if r.random_null_sets > 0:`. In the `"random_null"` provenance dict add `"pool": r.random_null_pool,` and `"n_active": None if pool_ids is None else len(pool_ids),`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/experiments -q`
Expected: PASS. `test_secondary_readout_does_not_empty_null_pool` asserts `len(res.null) > 0`. If the active pool on the synthetic store is too small for that test's bins, set `"random_null": {"n_sets": 3, "pool": "all"}` in that test only, because it is about exclusions, not activity. Say so in the commit body.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments tests/unit/experiments
git commit -m "feat(robustness): activity-matched random-null pool, default for robust screens (M14 D2)"
```

---

### Task 4: `uninformative` null outcome and tier rule

**Files:**
- Modify: `flyconn/experiments/robust_scoring.py` (`Outcome`, `null_outcome`, `tier`, `score_hits`)
- Modify: `flyconn/experiments/robustness.py` (compute degenerate keys before scoring)
- Modify: `flyconn/report/robust_html.py` (`_null_text`)
- Test: `tests/unit/experiments/test_robust_scoring.py`, `tests/unit/experiments/test_robust_report.py`

**Interfaces:**
- Produces: `Outcome = Literal["robust", "fails", "not_run", "uninformative"]`.
- Produces: `null_outcome(p: float | None, *, degenerate: bool = False) -> Outcome`.
- Produces: `score_hits(effects, *, alpha, null_p, replicate, null_degenerate: Mapping[str, bool] | None = None)`.
- `tier()` signature unchanged. `uninformative` counts as neither fails nor ran.

- [ ] **Step 1: Write the failing tests**

In `test_robust_scoring.py`, replace the `outcome` strategy and the combination test:

```python
outcome = st.sampled_from(["robust", "fails", "not_run", "uninformative"])


@given(w=outcome, n=outcome, m=outcome, r=outcome)
def test_every_combination_has_one_tier(w: str, n: str, m: str, r: str):
    oc = {"w_syn": w, "nt": n, "min_weight": m, "random_null": r}
    t, thin = tier(oc)  # type: ignore[arg-type]
    fails = sum(v == "fails" for v in oc.values())
    assert t == ("Strong" if fails == 0 else "Conditional" if fails == 1 else "Fragile")
    assert thin == (sum(v in ("robust", "fails") for v in oc.values()) < 2)


def test_degenerate_null_is_uninformative():
    assert null_outcome(1 / 41, degenerate=True) == "uninformative"
    assert null_outcome(0.5, degenerate=True) == "uninformative"
    assert null_outcome(float("nan"), degenerate=True) == "not_run"
    assert null_outcome(1 / 41) == "robust"


def test_score_hits_degenerate_null_not_counted():
    hits = score_hits(
        _effects(),
        alpha=0.05,
        null_p={"A": 1 / 41, "B": 0.3},
        replicate={},
        null_degenerate={"A": True},
    )
    a = hits.set_index("candidate").loc["A"]
    assert a["random_null"] == "uninformative"
    assert "random_null" not in a["failed_axes"] and "random_null" not in a["untested_axes"]
    assert a["uninformative_axes"] == "random_null"
```

In `test_robust_report.py`:

```python
def test_uninformative_null_renders_reason():
    from flyconn.report.robust_html import _null_text  # pyright: ignore[reportPrivateUsage]

    hit = pd.Series(
        {
            "random_null": "uninformative",
            "random_null_p": 1 / 41,
            "random_null_skipped": "",
            "random_null_note": "degenerate null: 40/40 random sets had no effect; "
            "p cannot go below 1/41 and is weak evidence",
        }
    )
    text = _null_text(hit)
    assert text.startswith("uninformative (p 0.024")
    assert "not counted in the tier" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/experiments/test_robust_scoring.py tests/unit/experiments/test_robust_report.py -v`
Expected: FAIL (`unexpected keyword 'degenerate'` / `'null_degenerate'`; `KeyError: 'uninformative_axes'`).

- [ ] **Step 3: Implement**

`robust_scoring.py`:

```python
Outcome = Literal["robust", "fails", "not_run", "uninformative"]


def null_outcome(p: float | None, *, degenerate: bool = False) -> Outcome:
    """Verdict of the random-null axis from its empirical p-value.

    A degenerate null (most random sets changed nothing) cannot discriminate: it scores
    ``uninformative``, which the tier counts as neither a pass nor a fail (M14 D1).
    """
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "not_run"
    if degenerate:
        return "uninformative"
    return "robust" if p < NULL_ALPHA else "fails"


def tier(outcomes: Mapping[str, Outcome]) -> tuple[str, bool]:
    """(tier, thinly_tested) from graded axis outcomes; uninformative axes neither pass nor fail."""
    graded = [outcomes.get(a, "not_run") for a in GRADED_AXES]
    fails = sum(o == "fails" for o in graded)
    ran = sum(o in ("robust", "fails") for o in graded)
    name = "Strong" if fails == 0 else "Conditional" if fails == 1 else "Fragile"
    return name, ran < 2
```

In `score_hits`, add the parameter `null_degenerate: Mapping[str, bool] | None = None`, and set `degen = null_degenerate or {}` at the top. Replace `outcomes["random_null"] = null_outcome(p)` with `null_outcome(p, degenerate=bool(degen.get(cand, False)))`. After `untested_axes` add:

```python
row["uninformative_axes"] = ", ".join(a for a in GRADED_AXES if outcomes[a] == "uninformative")
```

Insert `"uninformative_axes"` into `cols` after `"untested_axes"`.

`robustness.py`: compute `zero` (the `groupby("key")` dict) **before** `score_hits`, then:

```python
    degenerate = {
        c.label: bool(degenerate_null_note(*zero.get(key_by_cand[c.label], (0, 0))))
        for c in cands
        if c.role == "candidate"
    }
    hits = score_hits(
        eff,
        alpha=spec.report.alpha,
        null_p=null_by_cand,
        replicate=replicate_labels,
        null_degenerate=degenerate,
    )
```

Keep the `random_null_zero_frac` / `random_null_note` columns after it, reusing the same `zero`. Add `"uninformative_axes"` to `HANDOFF_COLUMNS` after `"untested_axes"`, and add `"uninformative_axes": h.get("uninformative_axes"),` in `write_handoff`. Change the provenance `p_rule` to:

```python
                "p_rule": "(1 + #null at least as extreme) / (n_sets + 1) < 0.05; "
                "uninformative (not counted in the tier) when >= 50% of sets have effect 0",
```

`robust_html.py` `_null_text`:

```python
    if hit["random_null"] == "uninformative":
        return (
            f"uninformative (p {_p(hit['random_null_p'])}; most random sets changed "
            "nothing, so the null cannot discriminate; not counted in the tier)"
        )
```

Place this after the `skipped` branch.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/experiments -q`
Expected: PASS.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments flyconn/report/robust_html.py tests/unit/experiments
git commit -m "feat(robustness): degenerate random null scores uninformative, not robust (M14 D1)"
```

---

### Task 5: Off-route active negative control and silent-control flag

**Files:**
- Modify: `flyconn/experiments/spec.py` (`RobustnessOptions.negative_controls`, parsing)
- Modify: `flyconn/experiments/sweep.py` (`_off_route_controls`, `stimulated_rate_hz` in variant info, provenance)
- Modify: `flyconn/experiments/robustness.py` (`control_summary` in provenance and result)
- Modify: `flyconn/report/robust_html.py` (negative-controls table)
- Modify: `examples/specs/w2_robust_pathway_mn9.yaml` (keep LC4 and add `negative_controls: 1`, so the trivial control stays visible next to the real one)
- Test: `tests/unit/experiments/test_sweep_fixed.py`, `tests/unit/experiments/test_spec_robustness.py`, `tests/unit/experiments/test_robust_screen.py`, `tests/unit/experiments/test_robust_report.py`

**Interfaces:**
- Produces: `RobustnessOptions.negative_controls: int = 1`.
- Produces: `MIN_CONTROL_RATE_HZ = 1.0` in `flyconn.experiments.sweep`.
- Produces: `_off_route_controls(ctx: Context, sweep: SilenceEachSweep, primary: str, stim_counts: np.ndarray, n: int, taken: set[str]) -> tuple[list[Candidate], dict[str, Any]]`. Returns candidates with role `"control"` and a provenance dict `{"requested": n, "chosen": [...], "reason": str}`.
- Produces: the sweep row column `stimulated_rate_hz` (summed rate of the silenced group in the stimulated run) for every `silence_each` variant.
- Produces: `RobustScreenResult.controls: pd.DataFrame` with columns `candidate, source, stimulated_rate_hz, difference, ci_low, ci_high, null_p, informative`, also in provenance `robustness.control_summary` (list of dicts).

- [ ] **Step 1: Write the failing tests**

`test_spec_robustness.py`:

```python
def test_negative_controls_option(small_store: Store):
    from flyconn.experiments.spec import spec_from_dict

    base = {"primary_readout": "targets"}
    assert spec_from_dict(_d(small_store, base)).robustness.negative_controls == 1  # type: ignore[union-attr]
    off = _d(small_store, {**base, "negative_controls": 0})
    assert spec_from_dict(off).robustness.negative_controls == 0  # type: ignore[union-attr]
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="negative_controls must be an int >= 0"):
            spec_from_dict(_d(small_store, {**base, "negative_controls": bad}))
```

`test_robust_screen.py`:

```python
def test_auto_negative_control_is_active_and_off_route(small_store: Store, tmp_path: Path):
    from flyconn.graph.matrix import ConnectivityMatrix
    from flyconn.graph.paths import find_paths

    spec = spec_from_dict(_d(small_store, negative_controls=1))
    res = run_robust_screen(spec, out_dir=tmp_path / "o", store=small_store)
    nc = res.provenance["sweep"]["negative_controls"]
    assert nc["requested"] == 1
    if not nc["chosen"]:
        assert nc["reason"] == "no active off-route group"
        return
    label = nc["chosen"][0]
    ctl = res.controls.set_index("candidate").loc[label]
    assert ctl["source"] == "auto" and ctl["stimulated_rate_hz"] >= 1.0 and ctl["informative"]
    n = small_store.neurons()
    ids = set(n.loc[n["cell_type"] == label, "neuron_id"].astype(int))
    stim = [int(i) for i in spec.stimulate[0].select.ids or ()]
    targets = [int(i) for i in spec.readouts[0].select.ids or ()]
    m = ConnectivityMatrix.from_store(small_store, min_weight=1)
    paths = find_paths(m, stim, targets, max_hops=3, min_weight=1)
    on_route = {int(i) for p in paths["path"] for i in p}
    assert not ids & on_route


def test_no_off_route_group_is_recorded_not_crash(small_store: Store, tmp_path: Path):
    d = _d(small_store, negative_controls=1)
    readout_ids = d["readouts"][0]["select"]["ids"]  # type: ignore[index]
    d["sweep"] = {
        "silence_each": {"select": {"ids": readout_ids[:2]}, "group_by": "neuron", "max_items": 2}
    }
    # the only selectable neurons are readout neurons, which are excluded: nothing eligible
    res = run_robust_screen(spec_from_dict(d), out_dir=tmp_path / "o", store=small_store)
    nc = res.provenance["sweep"]["negative_controls"]
    assert nc["chosen"] == [] and nc["reason"] == "no active off-route group"


def test_silent_extra_control_is_flagged(small_store: Store, tmp_path: Path):
    d = _d(small_store, extra_candidates=[{"cell_type": "T007"}], negative_controls=0)
    res = run_robust_screen(spec_from_dict(d), out_dir=tmp_path / "o", store=small_store)
    row = res.controls.set_index("candidate").loc["T007"]
    assert row["source"] == "extra"
    assert bool(row["informative"]) == (row["stimulated_rate_hz"] > 0)
```

(If T007 happens to fire on the synthetic store, the last assertion still pins the rule. Task 7's golden test pins the LC4 = silent case on real data.)

`test_robust_report.py`:

```python
def test_report_lists_controls_with_informative_flag(small_store: Store, tmp_path: Path):
    d = _d(small_store)  # this file's builder: _d(store, alpha=2.0)
    d["robustness"] = {**d["robustness"], "extra_candidates": [{"cell_type": "T007"}]}  # type: ignore[dict-item]
    run_robust_screen(spec_from_dict(d), out_dir=tmp_path / "o", store=small_store)
    html = (tmp_path / "o" / "report.html").read_text()
    assert "<h2>Negative controls</h2>" in html
    assert "T007" in html
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/experiments -k "negative_control or off_route or silent or controls" -v`
Expected: FAIL (`unknown robustness key(s) ['negative_controls']`).

- [ ] **Step 3: Implement**

`spec.py`: add `"negative_controls"` to `_ROBUSTNESS_KEYS`, and add `negative_controls: int = 1` to `RobustnessOptions`. Parse it:

```python
    nc_raw = dd.get("negative_controls", 1)
    if not isinstance(nc_raw, int) or isinstance(nc_raw, bool) or nc_raw < 0:
        msg = f"robustness negative_controls must be an int >= 0, got {nc_raw!r}"
        raise ValueError(msg)
```

Then pass `negative_controls=nc_raw`.

`sweep.py`:

```python
MIN_CONTROL_RATE_HZ = 1.0


def _off_route_controls(
    ctx: Context,
    sweep: SilenceEachSweep,
    primary: str,
    stim_counts: np.ndarray,
    n: int,
    taken: set[str],
) -> tuple[list[Candidate], dict[str, Any]]:
    """Up to ``n`` groups that fire under the stimulus but are on no stimulus -> primary path.

    "Off-route" means no neuron of the group lies on any path of <= max_hops hops at
    min_weight 1 (pathway.max_hops, else 3). Ranked by summed stimulated rate, ties by
    label. Such a control can change the readout only through longer or weaker routes,
    so a large effect is a warning about the model, not a hit (M14 D4).
    """
    info: dict[str, Any] = {"requested": n, "chosen": [], "reason": "off"}
    if n == 0:
        return [], info
    max_hops = sweep.pathway.max_hops if sweep.pathway is not None else 3
    sources = sorted(ctx.background)
    targets = ctx.readout_ids[primary].tolist()
    paths = find_paths(ctx.matrix, sources, targets, max_hops=max_hops, min_weight=1)
    on_route = {int(i) for p in paths["path"] for i in p}
    blocked = set(sources) | {int(i) for i in targets}
    ids = np.asarray(
        [i for i in resolve_selection(sweep.select, ctx.neurons).tolist() if i not in blocked],
        dtype=np.int64,
    )
    labels = _group_labels(ctx, sweep, ids)
    rate = stim_counts.mean(axis=0) / ctx.seconds
    df = pd.DataFrame(
        {
            "neuron_id": labels.index.to_numpy(dtype=np.int64),
            "group": labels.to_numpy(),
            "rate": rate[ctx.net.index_of(labels.index.tolist())],
        }
    )
    df["on_route"] = df["neuron_id"].isin(list(on_route))
    g = (
        df.groupby("group", sort=True)
        .agg(rate=("rate", "sum"), on_route=("on_route", "any"))
        .reset_index()
    )
    g = g[~g["on_route"] & (g["rate"] >= MIN_CONTROL_RATE_HZ) & ~g["group"].isin(list(taken))]
    g = g.sort_values(["rate", "group"], ascending=[False, True], kind="stable").head(n)
    members = df.groupby("group")["neuron_id"].apply(list).to_dict()
    out = [
        Candidate(str(lbl), tuple(int(i) for i in members[lbl]), "control") for lbl in g["group"]
    ]
    info |= {
        "chosen": [c.label for c in out],
        "reason": "" if out else "no active off-route group",
        "max_hops": max_hops,
        "min_rate_hz": MIN_CONTROL_RATE_HZ,
        "n_paths_checked": len(paths),
    }
    return out, info
```

`SilenceEachSweep.select` defaults to `Selection()`, which `resolve_selection` resolves to every neuron (checked 2026-10-24).

In `run_sweep`, inside `if fixed is None:` after the `_extra_candidates` line:

```python
            nc_info: dict[str, Any] = {"requested": 0, "chosen": [], "reason": "off"}
            rob = spec.robustness
            if rob is not None and rob.negative_controls > 0:
                auto, nc_info = _off_route_controls(
                    ctx,
                    sweep,
                    rob.primary_readout,
                    stim_counts,
                    rob.negative_controls,
                    {c.label for c in candidates},
                )
                candidates += auto
```

Initialise `nc_info` before the `if isinstance(sweep, SilenceEachSweep):` block, so that it exists on every path. Add `"negative_controls": nc_info,` to `sweep_prov` in the `SilenceEachSweep` branch.

Add the group's stimulated rate to each silence variant's `info`:

```python
stim_rate = stim_counts.mean(axis=0) / ctx.seconds
variants = [
    _Variant(
        label=c.label,
        stimulate=dict(ctx.background),
        silence=list(c.neuron_ids),
        info={
            "n_silenced": len(c.neuron_ids),
            "role": c.role,
            "stimulated_rate_hz": float(stim_rate[ctx.net.index_of(list(c.neuron_ids))].sum()),
        },
    )
    for c in candidates
]
```

For replicate and null runs (`fixed` from another dataset), `ctx` is that run's own context. The rate is therefore that run's own stimulated rate, which is what the column means.

`robustness.py`: add the field `controls: pd.DataFrame = field(default_factory=pd.DataFrame)` to `RobustScreenResult`, after `selection_stability`, so every positional construction must be updated. Then build it after `ref_d` and `control_null_p`:

```python
auto_labels = set(
    json.loads((vdir / "reference" / "provenance.json").read_text())["sweep"]
    .get("negative_controls", {})
    .get("chosen", [])
)
ref_rows = ref.sweep.loc[ref.sweep["readout"] == primary].set_index("variant")
controls = pd.DataFrame(
    [
        {
            "candidate": c.label,
            "source": "auto" if c.label in auto_labels else "extra",
            "stimulated_rate_hz": float(ref_rows.loc[c.label, "stimulated_rate_hz"]),
            "difference": float(ref_rows.loc[c.label, "difference"]),
            "ci_low": float(ref_rows.loc[c.label, "ci_low"]),
            "ci_high": float(ref_rows.loc[c.label, "ci_high"]),
            "null_p": control_null_p.get(c.label),
            "informative": float(ref_rows.loc[c.label, "stimulated_rate_hz"]) > 0,
        }
        for c in cands
        if c.role == "control"
    ],
    columns=pd.Index(
        [
            "candidate",
            "source",
            "stimulated_rate_hz",
            "difference",
            "ci_low",
            "ci_high",
            "null_p",
            "informative",
        ]
    ),
)
```

Pass `controls` into `RobustScreenResult(out, spec, hits, eff, null_df, sel, controls, prov)`. Add `"control_summary": controls.to_dict(orient="records"),` to the robustness provenance. Write `write_table(controls, out / "controls.parquet")` next to the other tables, and add `controls.parquet` to the docstring's file list.

`robust_html.py`: before `"<h2>Not hits and controls</h2>"`, insert:

```python
("<h2>Negative controls</h2>",)
"<p class='note'>A control is informative only if it fires under the stimulus; "
("a silent control cannot change any run, so passing it says nothing.</p>",)
(_table(result.controls),)
```

`examples/specs/w2_robust_pathway_mn9.yaml`: add `negative_controls: 1` under `robustness:` after `extra_candidates`. Add this comment above it: `# M14: plus one automatic control that fires under LB3 drive but is on no LB3 -> MN9 route (<=3 hops, min_weight 1); LC4 is kept to show a silent control is flagged uninformative.`

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/experiments -q`
Expected: PASS. Check `test_outputs_and_axes`'s file list, and add `controls.parquet` to it if it enumerates outputs exhaustively.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/experiments flyconn/report/robust_html.py examples/specs/w2_robust_pathway_mn9.yaml tests/unit/experiments
git commit -m "feat(robustness): automatic active off-route negative control; flag silent controls (M14 D4)"
```

---

### Task 6: Sugar-only W2 robustness spec

**Files:**
- Create: `examples/specs/w2_robust_pathway_mn9_sugar.yaml`
- Test: `tests/unit/experiments/test_example_specs.py` (create if no example-spec parse test exists. Check with `grep -rl "examples/specs" tests/unit` and extend the existing one if found.)

**Interfaces:**
- Consumes: the spec keys from Tasks 1–5.

- [ ] **Step 1: Write the failing test**

```python
"""Every example spec parses; the sugar-only spec matches the MaleCNS calibration stimulus."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyconn.experiments.spec import load_spec

SPECS = Path(__file__).resolve().parents[3] / "examples" / "specs"


@pytest.mark.parametrize("path", sorted(SPECS.glob("*.yaml")), ids=lambda p: p.name)
def test_example_spec_parses(path: Path):
    load_spec(path)


def test_sugar_spec_stimulates_lb3c_and_lb3d_only():
    s = load_spec(SPECS / "w2_robust_pathway_mn9_sugar.yaml")
    assert [st.select.cell_type for st in s.stimulate] == ["LB3[cd]"]
    assert s.robustness is not None and s.robustness.negative_controls == 1
    assert s.report.statistics == "paired"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/experiments/test_example_specs.py -v`
Expected: FAIL (file not found for the sugar spec).

- [ ] **Step 3: Create the spec**

`examples/specs/w2_robust_pathway_mn9_sugar.yaml`:

```yaml
# M14 sugar-only variant of w2_robust_pathway_mn9.yaml (MaleCNS v1.0; w_syn 0.188 mV,
# calibrated by protocol on LB3c+LB3d, not validated; ADR-0009, ADR-0011, ADR-0012).
# Stimulus: LB3c + LB3d only (sugar-like by output-partner profile, PROGRESS 2026-10-06),
# matching the calibration stimulus; the all-LB3 spec also drives water-like LB3a/LB3b.
name: w2_robust_pathway_mn9_sugar
dataset: malecns@1.0
network: {min_weight: 1, sign_policy: argmax}
stimulate:
  - select: {cell_type: "LB3[cd]"}
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
  negative_controls: 1
```

Note: in FlyWire, sugar GRNs are not typed `LB3c`/`LB3d`. The replicate maps the stimulus with `map_selection` through `match_types`, and its provenance records whatever that mapping yields. If the stimulus cannot be mapped, the replicate is reported "not testable", which is the existing honest path. Do not hand-edit the mapping.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/unit/experiments/test_example_specs.py -v`
Expected: PASS for every example spec.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add examples/specs/w2_robust_pathway_mn9_sugar.yaml tests/unit/experiments/test_example_specs.py
git commit -m "feat(examples): sugar-only (LB3c+d) robustness spec matching the calibration stimulus"
```

---

### Task 7: Golden runs on MaleCNS, ADR-0012 and docs

Real data, run locally: two robust screens of roughly 20 minutes each. Peak RSS must stay under 16 GB. The numbers below are **not known yet**. Write down what the runs print.

**Files:**
- Modify: `tests/golden/test_robust_screen.py` (update pins; add the sugar test)
- Modify: `benchmarks/robust_screen_details.py` (print the controls table and `uninformative_axes`)
- Create: `docs/adr/0012-robustness-hardening.md`
- Modify: `docs/adr/0011-robustness-audited-screens.md` (line ~85: "unpaired" → points to ADR-0012), `docs/adr/README.md` (index)
- Modify: `docs/GOLDEN_RESULTS.md` (new §6k), `docs/tutorials/robust-screens.md`, `CHANGELOG.md` (`[Unreleased]` → M14 section), `docs/PROGRESS.md` (new entry)
- Modify: `benchmarks/robust_screen_malecns.json` (rewritten by the golden test)

- [ ] **Step 1: Extend the details script**

In `benchmarks/robust_screen_details.py`, after the hits table, print `controls.parquet` (all columns), and print `uninformative_axes` for each hit. Run it on a synthetic output to check it works: `uv run python benchmarks/robust_screen_details.py <tmp dir from a unit run>`.

- [ ] **Step 2: Update the golden test with structural pins only**

Change `test_robust_screen_malecns` so that it no longer pins `random_null == "robust"` for GNG232. Pin the properties that M14 guarantees instead:

```python
    controls = pd.read_parquet(out / "controls.parquet").set_index("candidate")
    assert not bool(controls.loc["LC4", "informative"])  # silent under LB3 drive (6j)
    auto = controls[controls["source"] == "auto"]
    assert len(auto) == 1 and bool(auto["informative"].iloc[0])
    rob = res.provenance["robustness"]
    assert rob["random_null"]["pool"] == "active"
    assert set(rob["store_fingerprints"]) == {"malecns@1.0", "flywire@783"}
    assert res.provenance["sweep"]["statistics"].startswith("paired")
```

Keep the 6h pins on the reference **differences**. Paired statistics leave the difference unchanged (it is the mean of per-trial differences, which equals the difference of means), so `SIX_H` differences must still hold to 0.1 Hz. Drop or loosen the CI pins only. Add a near-identical `test_robust_screen_malecns_sugar` on `w2_robust_pathway_mn9_sugar.yaml` that asserts the same structural properties and prints its hits.

- [ ] **Step 3: Run both golden tests and record the output**

```bash
FLYCONN_CACHE=$PWD/.cache/flyconn uv run pytest -m golden tests/golden/test_robust_screen.py -s 2>&1 | tee "$SCRATCH/m14_golden.log"
uv run flyconn run examples/specs/w2_robust_pathway_mn9.yaml --out "$SCRATCH/m14_all"
uv run flyconn run examples/specs/w2_robust_pathway_mn9_sugar.yaml --out "$SCRATCH/m14_sugar"
uv run python benchmarks/robust_screen_details.py "$SCRATCH/m14_all" --paths
uv run python benchmarks/robust_screen_details.py "$SCRATCH/m14_sugar" --paths
```

(`$SCRATCH` is the session scratchpad.) Expected: both tests PASS and peak RSS < 16 GB. If the auto `find_paths` call pushes the peak RSS over 16 GB, stop and report. Do not raise limits.

- [ ] **Step 4: Pin the numbers that were actually printed**

From the log, add pins to the golden tests for:
- GNG232's tier;
- its `random_null` outcome under the activity-matched pool;
- the auto control's label and its effect CI;
- the sugar run's hit list.

Copy the values exactly as printed. Rerun the golden tests until they pass.

- [ ] **Step 5: ADR-0012**

Write `docs/adr/0012-robustness-hardening.md` in the format of ADR-0011 (Context / Decision / Consequences). Decisions D1–D6 as stated at the top of this plan, each with its reason and its cost:
- D1 can make a hit "thinly tested" that was not before;
- D2 can skip bins whose active pool is small;
- D3 changes p-values and CIs of robustness screens relative to 0.4.0;
- D4 adds one `find_paths` call;
- D5 and D6 invalidate M13 output directories for resume.

Add it to `docs/adr/README.md`. In ADR-0011, at the sentence on line ~85 that says the labels use unpaired statistics, add "(superseded for robustness screens by ADR-0012: paired)".

- [ ] **Step 6: GOLDEN_RESULTS §6k, tutorial, CHANGELOG, PROGRESS**

- `docs/GOLDEN_RESULTS.md` §6k, "Hardened robustness screens (ADR-0012; 2026-10-xx)". Cover both specs, with wall time, peak RSS and every number as printed in Step 3. Compare with 6j: what changed for GNG232, LC4 versus the auto control, and the sugar-only hit list. Close with a "Weak checks" paragraph for anything still weak, as 6j does.
- `docs/tutorials/robust-screens.md`: document `report.statistics`, `random_null.pool`, `negative_controls`, the `uninformative` outcome and `controls.parquet`.
- `CHANGELOG.md` under `## [Unreleased]`: a `### M14 scientific hardening` section. Its first line is **Behaviour change** for robustness specs (paired statistics, active null pool, auto negative control, uninformative null, new null seeds; M13 output directories do not resume).
- `docs/PROGRESS.md`: a new top entry "2026-10-xx — M14 scientific hardening (branch m14-hardening)" in the style of the M13 entry: what was built, the measured numbers (pointing to 6k), and what is still open.

- [ ] **Step 7: Gate and commit**

```bash
scripts/check.sh
git add tests/golden/test_robust_screen.py benchmarks docs CHANGELOG.md
git commit -m "docs(m14): golden runs, ADR-0012, GOLDEN_RESULTS 6k, tutorial, changelog, progress"
```

---

## Out of scope (next milestone candidates)

- M12 open items: mode B, the MaleCNS cross-reference leakage check, dataset-difficulty calibration features, kNN voting, the hemibrain benchmark.
- Paired statistics for `run_experiment` perturbations and for the shared-control comparisons in `readout_stats`. Both stay unpaired here.
- CUDA validation of the compiled float32 path.
