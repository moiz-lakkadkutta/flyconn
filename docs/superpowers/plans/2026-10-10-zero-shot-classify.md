# Zero-shot cell-type classification (M12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify unlabelled neurons of one connectome into a reference dataset's types by
partner-type profile, with a calibrated confidence or an "ambiguous"/"unknown" call, and
benchmark it by holding labels out between MaleCNS, BANC and FlyWire.

**Architecture:**
- Each neuron becomes a sparse vector of synapse fractions to and from partner types, named
  in the shared FlyWire-or-MANC vocabulary (`profiles.py`).
- A reference atlas stores one centroid per type and the type's own similarity yardsticks
  (`atlas.py`).
- Queries are scored by cosine to every centroid (`scoring.py`).
- A logistic model fitted by the hold-out benchmark (`classify_bench.py`) turns scores into
  calls (`calibrate.py`).
- `classify.py` is the public entry point; `flyconn classify` is the CLI.

**Tech Stack:** Python ≥3.11, numpy, scipy.sparse, scipy.optimize, pandas, typer, pytest,
hypothesis. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-10-zero-shot-classify-design.md`

## Global Constraints

- Branch `m12-classify`; conventional commits; every commit leaves `scripts/check.sh` green.
- `pyright` strict on the public API; `ruff check` and `ruff format` clean.
- Unit tests never touch the network or the real cache; use the synthetic stores in
  `tests/unit/compare/conftest.py`.
- Golden tests: `pytestmark = pytest.mark.golden`; run with
  `FLYCONN_CACHE=$PWD/.cache/flyconn uv run pytest -m golden <file>`.
- No new dependencies (no scikit-learn); no GPL imports.
- Defaults: `min_weight=5`, `vocabulary="fafb_or_manc"`, `direction="both"`, `top_k=5`,
  `accept=0.8`, `max_false_accept=0.10`, heterogeneity cutoff 0.7, unlabelled-partner
  caveat above 0.3, vocabulary overlap floor 0.01, scoring chunk 5,000 rows.
- Every result carries provenance: dataset refs, atlas params and hash, mask, calibration
  id, seeds, package versions, git SHA.
- Numbers in docs come from code run in this repo (golden-test output), never from memory.
- Peak RSS < 8 GB for any golden test in this milestone; benchmark < 10 min per pair on the
  M4 Pro.

## Review Focus

1. **A query dataset whose cross-reference columns don't match the reference** (e.g.
   hemibrain against FlyWire through a vocabulary it doesn't carry). Expected: a clear
   `ValueError` about vocabulary overlap, not confident garbage. Test in Task 5.
2. **A query neuron or type with no partners named in the vocabulary.** Expected: call
   `unknown`, reason `no_labelled_partners`, no crash, no fake top label. Test in Task 5.
3. **A calibration file fitted with different atlas parameters** (other `min_weight`,
   `direction`, custom groups). Expected: uncalibrated mode with a caveat, never a silent
   mismatch. Test in Task 5.
4. **Reference labels with one neuron or one side only.** Expected: yardstick fallback,
   flagged, no division by zero or NaN in `a1`. Test in Task 2.
5. **Neuron ids passed from a text file in the CLI (whitespace, blank lines, duplicates).**
   Expected: parsed as integers, de-duplicated. Test in Task 7; de-duplication also
   covered in Task 5.

---

## File structure

| File | Responsibility |
|------|----------------|
| `flyconn/provenance.py` (create) | `run_environment()`: package versions, git SHA |
| `flyconn/compare/profiles.py` (create) | `PartnerCounts`, `partner_counts`, row normalisation |
| `flyconn/compare/atlas.py` (create) | `Atlas`, `build_atlas`, `atlas_from_matrix`, masking/dropping |
| `flyconn/compare/scoring.py` (create) | `Scores`, `score` (chunked cosine top-k) |
| `flyconn/compare/calibrate.py` (create) | `LogisticModel`, `fit_logistic`, `ece`, `brier`, `Thresholds`, `Calibration`, `decide`, `uncalibrated_calls`, threshold choosers, default-file loading |
| `flyconn/compare/classify.py` (create) | `classify`, `classify_matrix`, `ClassifyResult`, `GroupCall`, caveats |
| `flyconn/compare/classify_bench.py` (create) | `run_benchmark`, `fit_calibration`, metrics, `write_benchmark` |
| `flyconn/compare/calibration/classify_flywire_783.json` (create, generated) | shipped pooled calibration |
| `flyconn/compare/__init__.py` (modify) | exports |
| `flyconn/cli/classify.py` (create), `flyconn/cli/main.py` (modify) | `flyconn classify run`, `flyconn classify bench` |
| `tests/unit/test_provenance.py`, `tests/unit/compare/test_profiles.py`, `test_atlas.py`, `test_scoring.py`, `test_calibrate.py`, `test_classify.py`, `test_classify_bench.py`, `tests/unit/test_cli_classify.py` (create) | unit tests |
| `tests/golden/test_classify.py` (create) | real-data benchmark + LB3 |
| `benchmarks/classify_flywire_783.json` (generated) | benchmark metrics |
| `docs/adr/0010-zero-shot-classification.md`, `docs/tutorials/classify.md` (create); `CHANGELOG.md`, `docs/PROGRESS.md`, `docs/GOLDEN_RESULTS.md`, `mkdocs.yml`, the spec (modify) | docs |

Unit-test fixture (existing, `tests/unit/compare/conftest.py`, fixture `male_female`):

- Two stores, `malecns@1.0` (male) and `flywire@783` (female).
- Types T0-T7, 2 sides x 3 neurons each, ring wiring T_i -> T_{i+1} (30 synapses) and
  T_i -> T_{i+2} (8).
- In the male store, T3 is perturbed, `fafb_783_cell_type == cell_type`, and two extra
  "T8" neurons (901, 902) have no FlyWire name and receive input from T0.
- The female store has no cross-reference values, so `partner_labels(..., "fafb_or_manc")`
  falls back to `cell_type`.

---

### Task 1: Provenance helper and partner-count profiles

**Files:**
- Create: `flyconn/provenance.py`, `flyconn/compare/profiles.py`
- Test: `tests/unit/test_provenance.py`, `tests/unit/compare/test_profiles.py`

**Interfaces:**
- Consumes: `ConnectivityMatrix` (`flyconn/graph/matrix.py`: fields `neuron_ids`, `weights`
  (CSR, pre in rows), `signs`, `meta`; property `n`).
- Produces:
  - `run_environment() -> dict[str, Any]` with keys `flyconn_version`, `packages`, `git_sha`;
  - `git_sha() -> str | None`;
  - `ProfileDirection = Literal["both", "out", "in"]`;
  - `l1_rows(x) -> csr_matrix`, `l2_rows(x) -> csr_matrix`;
  - `PartnerCounts` (fields `neuron_ids`, `vocab`, `out`, `inn`, `out_total`, `in_total`;
    methods `rows(idx)`, `profiles(*, direction, mask)`, `unlabelled_fraction(*, direction)`,
    `column_keep(mask, direction)`, `touching(mask)`, `labelled_mass(*, direction)`);
  - `partner_counts(m, labels, vocab=None) -> PartnerCounts`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_provenance.py`:

```python
"""Run-environment provenance helper."""

import re

import flyconn
from flyconn.provenance import git_sha, run_environment


def test_run_environment_has_versions_and_sha():
    env = run_environment()
    assert env["flyconn_version"] == flyconn.__version__
    assert set(env["packages"]) >= {"numpy", "scipy", "pandas"}
    sha = git_sha()
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha)
    assert env["git_sha"] == sha
```

`tests/unit/compare/test_profiles.py`:

```python
"""Partner-count profiles over a fixed vocabulary."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyconn.compare.profiles import l1_rows, l2_rows, partner_counts
from flyconn.graph.matrix import ConnectivityMatrix


def _tiny() -> ConnectivityMatrix:
    # 1(A) -> 2(B) w4, 1 -> 3(unlabelled) w6, 2 -> 1 w2
    w = sp.csr_matrix(np.array([[0, 4, 6], [2, 0, 0], [0, 0, 0]], dtype=float))
    meta = pd.DataFrame({"neuron_id": [1, 2, 3], "cell_type": ["A", "B", None]})
    return ConnectivityMatrix(
        neuron_ids=np.array([1, 2, 3]), weights=w, signs=np.ones(3), meta=meta
    )


def test_counts_and_profiles_both_directions():
    pc = partner_counts(_tiny(), ["A", "B", None])
    assert pc.vocab.tolist() == ["A", "B"]
    assert pc.out.toarray()[0].tolist() == [0, 4]
    assert pc.inn.toarray()[0].tolist() == [0, 2]  # from neuron 2 (B)
    prof = pc.profiles(direction="both").toarray()
    assert prof.shape == (3, 4)
    assert np.allclose(prof[0], [0, 1, 0, 1])
    assert np.allclose(prof[2], [0, 0, 1, 0])  # neuron 3: no outputs, input from 1 (A)
    out_only = pc.profiles(direction="out").toarray()
    assert out_only.shape == (3, 2)


def test_neuron_three_in_profile():
    pc = partner_counts(_tiny(), ["A", "B", None])
    assert np.allclose(pc.profiles(direction="in").toarray()[2], [1, 0])


def test_unlabelled_fraction_counts_unnamed_partners():
    pc = partner_counts(_tiny(), ["A", "B", None])
    frac = pc.unlabelled_fraction(direction="out")
    assert frac[0] == pytest.approx(0.6)  # 6 of 10 output synapses go to neuron 3
    assert frac[2] == pytest.approx(1.0)  # no outputs at all


def test_mask_blanks_a_label_and_renormalises():
    pc = partner_counts(_tiny(), ["A", "B", None])
    prof = pc.profiles(direction="both", mask=("B",)).toarray()
    assert np.allclose(prof[0], 0)
    assert np.allclose(prof[1], [1, 0, 1, 0])  # 2 -> 1 (A) and 1 (A) -> 2
    assert pc.column_keep(("B",), "both").tolist() == [1, 0, 1, 0]
    assert pc.touching(("B",)).tolist() == [0]


def test_fixed_vocab_and_rows():
    pc = partner_counts(_tiny(), ["A", "B", None], vocab=["B", "Z"])
    assert pc.vocab.tolist() == ["B", "Z"]
    sub = pc.rows(np.array([1]))
    assert sub.neuron_ids.tolist() == [2]
    assert sub.out.shape == (1, 2)


def test_bad_direction_and_length():
    pc = partner_counts(_tiny(), ["A", "B", None])
    with pytest.raises(ValueError, match="direction"):
        pc.profiles(direction="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="labels"):
        partner_counts(_tiny(), ["A"])


def test_row_normalisers():
    x = sp.csr_matrix(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(l1_rows(x).toarray(), [[3 / 7, 4 / 7], [0, 0]])
    assert np.allclose(l2_rows(x).toarray(), [[0.6, 0.8], [0, 0]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_provenance.py tests/unit/compare/test_profiles.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flyconn.provenance'` / `flyconn.compare.profiles`.

- [ ] **Step 3: Write the implementation**

`flyconn/provenance.py`:

```python
"""Run environment for provenance records: package versions and git SHA."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import flyconn


def git_sha() -> str | None:
    """HEAD of the checkout flyconn is imported from, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(flyconn.__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def run_environment(packages: Sequence[str] = ("numpy", "scipy", "pandas")) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"flyconn_version": flyconn.__version__, "packages": versions, "git_sha": git_sha()}
```

`flyconn/compare/profiles.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_provenance.py tests/unit/compare/test_profiles.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/provenance.py flyconn/compare/profiles.py tests/unit/test_provenance.py tests/unit/compare/test_profiles.py
git commit -m "feat(compare): partner-count profiles over a fixed vocabulary; run provenance helper"
```

If pyright strict complains about sparse return types, wrap the expressions in
`sp.csr_matrix(...)` or `cast(...)` as `flyconn/compare/types.py` does. Don't add
`# type: ignore` to public signatures.

---

### Task 2: Reference atlas

**Files:**
- Create: `flyconn/compare/atlas.py`
- Test: `tests/unit/compare/test_atlas.py`

**Interfaces:**
- Consumes:
  - Task 1: `PartnerCounts`, `partner_counts`, `l2_rows`, `ProfileDirection`;
  - `partner_labels(meta, vocabulary)` from `flyconn/compare/types.py`;
  - `ConnectivityMatrix.from_store(store, min_weight=...)`; `Store.ref`, `Store.provenance`.
- Produces:
  - `Atlas` dataclass. Fields: `reference: str`, `labels: np.ndarray`,
    `member_codes: np.ndarray`, `member_sides: np.ndarray`, `member_ids: np.ndarray`,
    `member_counts: PartnerCounts`, `sums: sp.csr_matrix`, `neuron_yardstick: np.ndarray`,
    `group_yardstick: np.ndarray`, `yardstick_fallback: np.ndarray`,
    `params: dict[str, Any]`, `provenance: dict[str, Any]`, `mask: tuple[str, ...]`.
  - `Atlas` properties: `vocab`, `direction`, `n_members`, `params_hash: str`,
    `centroids: sp.csr_matrix` (L2-normalised, cached).
  - `Atlas.masked(mask=(), *, drop=()) -> Atlas`.
  - `atlas_from_matrix(m, *, reference, groups=None, min_weight=5, vocabulary="fafb_or_manc", direction="both", on_missing="raise", store_provenance=None) -> Atlas`.
  - `build_atlas(store, *, groups=None, min_weight=5, vocabulary="fafb_or_manc", direction="both", on_missing="raise") -> Atlas`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/compare/test_atlas.py`:

```python
"""Reference atlas: centroids, yardsticks, custom groups, masking."""

import numpy as np
import pytest
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas, atlas_from_matrix, build_atlas
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix


def _female_atlas(male_female: tuple[Store, Store], **kw: object) -> Atlas:
    return build_atlas(male_female[1], **kw)  # type: ignore[arg-type]


def test_cell_type_atlas_has_tight_yardsticks(male_female: tuple[Store, Store]):
    atlas = _female_atlas(male_female)
    assert atlas.labels.tolist() == [f"T{i}" for i in range(8)]
    assert atlas.n_members.tolist() == [6] * 8
    assert atlas.centroids.shape == (8, 2 * len(atlas.vocab))
    assert np.all(atlas.neuron_yardstick > 0.9)
    assert np.all(atlas.group_yardstick > 0.9)
    assert not atlas.yardstick_fallback.any()
    assert atlas.params["direction"] == "both"
    assert len(atlas.params_hash) == 16


def test_params_hash_changes_with_parameters(male_female: tuple[Store, Store]):
    a = _female_atlas(male_female)
    b = _female_atlas(male_female, direction="out")
    assert a.params_hash != b.params_hash
    assert b.centroids.shape[1] == len(b.vocab)
    m = ConnectivityMatrix.from_store(male_female[1], min_weight=5)
    other_store = atlas_from_matrix(
        m, reference="flywire@783", store_provenance={"counts": {"neurons": 1}}
    )
    assert other_store.params_hash != a.params_hash  # same name, different store


def test_custom_groups_and_missing_ids(male_female: tuple[Store, Store]):
    female = male_female[1]
    meta = female.neurons(columns=["neuron_id", "cell_type"])
    t0 = meta.loc[meta["cell_type"] == "T0", "neuron_id"].tolist()
    t1 = meta.loc[meta["cell_type"] == "T1", "neuron_id"].tolist()
    atlas = build_atlas(female, groups={"x": t0, "y": t1})
    assert atlas.labels.tolist() == ["x", "y"]
    assert atlas.params["labels_from"] == "groups"
    with pytest.raises(KeyError, match="not in"):
        build_atlas(female, groups={"x": [*t0, 999_999]})
    dropped = build_atlas(female, groups={"x": [*t0, 999_999]}, on_missing="drop")
    assert dropped.provenance["group_ids_dropped"] == 1
    with pytest.raises(ValueError, match="more than one group"):
        build_atlas(female, groups={"x": t0, "y": t0[:1]})


def test_single_member_label_uses_fallback_yardstick(male_female: tuple[Store, Store]):
    female = male_female[1]
    meta = female.neurons(columns=["neuron_id", "cell_type", "side"])
    t0 = meta.loc[meta["cell_type"] == "T0", "neuron_id"].tolist()
    one = meta.loc[(meta["cell_type"] == "T1"), "neuron_id"].tolist()[:1]
    atlas = build_atlas(female, groups={"x": t0, "solo": one})
    solo = atlas.labels.tolist().index("solo")
    assert atlas.yardstick_fallback[solo]
    assert np.isfinite(atlas.neuron_yardstick).all()
    assert np.isfinite(atlas.group_yardstick).all()
    assert atlas.neuron_yardstick[solo] == pytest.approx(atlas.neuron_yardstick[1 - solo])


def test_masked_equals_exact_recompute(male_female: tuple[Store, Store]):
    atlas = _female_atlas(male_female)
    masked = atlas.masked(("T1",))
    assert masked.mask == ("T1",)
    mc = atlas.member_counts
    prof = mc.profiles(direction="both", mask=("T1",))
    agg = sp.csr_matrix(
        (
            np.ones(len(atlas.member_codes)),
            (atlas.member_codes, np.arange(len(atlas.member_codes))),
        ),
        shape=(len(atlas.labels), len(atlas.member_codes)),
    )
    expected = (agg @ prof).toarray()
    assert np.allclose(masked.sums.toarray(), expected, atol=1e-12)
    t1_cols = np.flatnonzero(mc.column_keep(("T1",), "both") == 0)
    assert np.all(masked.centroids.toarray()[:, t1_cols] == 0)
    assert atlas.mask == ()  # original untouched


def test_drop_removes_label_and_its_members(male_female: tuple[Store, Store]):
    atlas = _female_atlas(male_female).masked(("T3",), drop=("T3",))
    assert "T3" not in atlas.labels.tolist()
    assert len(atlas.labels) == 7
    assert len(atlas.member_codes) == 42
    assert atlas.member_codes.max() == 6
    again = atlas.masked(("T2",))  # masking still works after a drop
    assert again.sums.shape[0] == 7


def test_atlas_from_matrix_rejects_empty_labels(male_female: tuple[Store, Store]):
    m = ConnectivityMatrix.from_store(male_female[1], min_weight=5)
    m.meta["cell_type"] = None
    with pytest.raises(ValueError, match="no labelled"):
        atlas_from_matrix(m, reference="flywire@783")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/compare/test_atlas.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flyconn.compare.atlas'`.

- [ ] **Step 3: Write the implementation**

`flyconn/compare/atlas.py`:

```python
"""Reference atlas for zero-shot classification (ADR-0010).

One centroid per reference label (mean of member partner profiles) plus two yardsticks of
how alike the label's own members are: the neuron yardstick (median leave-one-out cosine
of members to their own centroid) and the group yardstick (cosine between the left and
right centroids, the within-brain variability used by ``compare_type``). Labels with too
few members or one side only take the atlas-wide median and are flagged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.compare.profiles import (
    PartnerCounts,
    ProfileDirection,
    check_direction,
    l2_rows,
    partner_counts,
)
from flyconn.compare.types import partner_labels
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix


def _rowdot(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1), dtype=float).ravel()


def _membership(codes: np.ndarray, n_labels: int, rows: np.ndarray | None = None) -> sp.csr_matrix:
    sel = np.arange(len(codes)) if rows is None else np.flatnonzero(rows)
    return sp.csr_matrix((np.ones(len(sel)), (codes[sel], sel)), shape=(n_labels, len(codes)))


def _fill(x: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(x)) if np.isfinite(x).any() else 1.0
    return np.where(np.isnan(x), med, x)


def _yardsticks(
    prof: sp.csr_matrix, codes: np.ndarray, sides: np.ndarray, n_labels: int
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    sums = sp.csr_matrix(_membership(codes, n_labels) @ prof)
    n = np.bincount(codes, minlength=n_labels)
    dot = _rowdot(prof, sp.csr_matrix(sums[codes]))
    pp = _rowdot(prof, prof)
    ss = _rowdot(sums, sums)[codes]
    rest = ss - 2 * dot + pp  # |sum of the other members|^2
    ok = (n[codes] >= 2) & (pp > 0) & (rest > 1e-12)
    loo = np.full(len(codes), np.nan)
    loo[ok] = (dot[ok] - pp[ok]) / np.sqrt(pp[ok] * rest[ok])
    neuron = pd.Series(loo).groupby(codes).median().reindex(range(n_labels)).to_numpy(dtype=float)
    left = l2_rows(_membership(codes, n_labels, sides == "left") @ prof)
    right = l2_rows(_membership(codes, n_labels, sides == "right") @ prof)
    both = (_rowdot(left, left) > 0) & (_rowdot(right, right) > 0)
    group = np.where(both, _rowdot(left, right), np.nan)
    fallback = np.isnan(neuron) | np.isnan(group)
    return sums, _fill(neuron), _fill(group), fallback


@dataclass
class Atlas:
    reference: str
    labels: np.ndarray
    member_codes: np.ndarray
    member_sides: np.ndarray
    member_ids: np.ndarray
    member_counts: PartnerCounts
    sums: sp.csr_matrix
    neuron_yardstick: np.ndarray
    group_yardstick: np.ndarray
    yardstick_fallback: np.ndarray
    params: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    mask: tuple[str, ...] = ()

    @property
    def vocab(self) -> np.ndarray:
        return self.member_counts.vocab

    @property
    def direction(self) -> ProfileDirection:
        return self.params["direction"]

    @property
    def n_members(self) -> np.ndarray:
        return np.bincount(self.member_codes, minlength=len(self.labels))

    @property
    def params_hash(self) -> str:
        blob = json.dumps(self.params, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @cached_property
    def centroids(self) -> sp.csr_matrix:
        """Unit-length centroid rows (the mean and the sum point the same way)."""
        return l2_rows(self.sums)

    def masked(self, mask: Sequence[str] = (), *, drop: Sequence[str] = ()) -> Atlas:
        """Copy with ``mask`` blanked as partner labels and ``drop`` labels removed.

        Masking is exact: member profiles that touch a masked label are recomputed and
        their centroids updated. Yardsticks are kept from the unmasked atlas.
        """
        new = [x for x in mask if x not in self.mask]
        full = tuple(sorted({*self.mask, *new}))
        sums = self.sums
        if new:
            mc = self.member_counts
            rows = mc.touching(new)
            if len(rows):
                sub = mc.rows(rows)
                delta = sub.profiles(direction=self.direction, mask=full) - sub.profiles(
                    direction=self.direction, mask=self.mask
                )
                agg = sp.csr_matrix(
                    (np.ones(len(rows)), (self.member_codes[rows], np.arange(len(rows)))),
                    shape=(len(self.labels), len(rows)),
                )
                sums = sp.csr_matrix(sums + agg @ delta)
            sums = sp.csr_matrix(sums @ sp.diags(mc.column_keep(full, self.direction)))
            sums.eliminate_zeros()
        out = replace(self, sums=sums, mask=full)
        if drop:
            out = out._without(np.isin(out.labels, np.asarray(list(drop), dtype=object)))
        return out

    def _without(self, drop_rows: np.ndarray) -> Atlas:
        keep = ~drop_rows
        new_code = np.cumsum(keep) - 1
        rows = np.flatnonzero(keep[self.member_codes])
        return replace(
            self,
            labels=self.labels[keep],
            member_codes=new_code[self.member_codes[rows]],
            member_sides=self.member_sides[rows],
            member_ids=self.member_ids[rows],
            member_counts=self.member_counts.rows(rows),
            sums=sp.csr_matrix(self.sums[np.flatnonzero(keep)]),
            neuron_yardstick=self.neuron_yardstick[keep],
            group_yardstick=self.group_yardstick[keep],
            yardstick_fallback=self.yardstick_fallback[keep],
        )


def _labels_from_groups(
    m: ConnectivityMatrix,
    groups: Mapping[str, Sequence[int]],
    reference: str,
    on_missing: Literal["raise", "drop"],
) -> tuple[np.ndarray, int]:
    lab = np.full(m.n, None, dtype=object)
    pos = pd.Index(m.neuron_ids)
    dropped = 0
    for name, ids in groups.items():
        ids_arr = np.asarray(list(ids), dtype=np.int64)
        idx = pos.get_indexer(ids_arr)
        missing = ids_arr[idx < 0]
        if len(missing) and on_missing == "raise":
            msg = f"group {name!r}: neuron ids not in {reference}: {missing[:5].tolist()}"
            raise KeyError(msg)
        dropped += len(missing)
        idx = idx[idx >= 0]
        if any(v is not None for v in lab[idx].tolist()):
            msg = f"group {name!r}: some neurons are in more than one group"
            raise ValueError(msg)
        lab[idx] = name
    return lab, dropped


def atlas_from_matrix(
    m: ConnectivityMatrix,
    *,
    reference: str,
    groups: Mapping[str, Sequence[int]] | None = None,
    min_weight: int = 5,
    vocabulary: str = "fafb_or_manc",
    direction: ProfileDirection = "both",
    on_missing: Literal["raise", "drop"] = "raise",
    store_provenance: Mapping[str, Any] | None = None,
) -> Atlas:
    """Atlas from an already-built matrix (``min_weight`` is recorded, not re-applied)."""
    check_direction(direction)
    counts = partner_counts(m, partner_labels(m.meta, vocabulary).tolist())
    if groups is None:
        lab = m.meta["cell_type"].astype(object).to_numpy()
        dropped, source, groups_hash = 0, "cell_type", None
    else:
        lab, dropped = _labels_from_groups(m, groups, reference, on_missing)
        source = "groups"
        blob = json.dumps({k: sorted(int(i) for i in v) for k, v in groups.items()}, sort_keys=True)
        groups_hash = hashlib.sha256(blob.encode()).hexdigest()[:16]
    rows = np.flatnonzero([isinstance(v, str) and v != "" for v in lab.tolist()])
    if len(rows) == 0:
        msg = f"{reference}: no labelled reference neurons to build an atlas from"
        raise ValueError(msg)
    codes, uniq = pd.factorize(pd.Series(lab[rows], dtype=object), sort=True)
    labels = np.asarray(uniq, dtype=object)
    member_counts = counts.rows(rows)
    sides = m.meta["side"].astype(object).to_numpy()[rows]
    sums, neuron_y, group_y, fallback = _yardsticks(
        member_counts.profiles(direction=direction), codes, sides, len(labels)
    )
    params: dict[str, Any] = {
        "reference": reference,
        "min_weight": min_weight,
        "vocabulary": vocabulary,
        "direction": direction,
        "labels_from": source,
        "groups_hash": groups_hash,
        "store_hash": hashlib.sha256(
            json.dumps(dict(store_provenance or {}), sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    }
    prov: dict[str, Any] = {
        "reference": reference,
        "n_labels": len(labels),
        "n_members": len(rows),
        "group_ids_dropped": dropped,
        "store_provenance": dict(store_provenance or {}),
    }
    return Atlas(
        reference=reference,
        labels=labels,
        member_codes=np.asarray(codes, dtype=np.int64),
        member_sides=sides,
        member_ids=np.asarray(m.neuron_ids)[rows],
        member_counts=member_counts,
        sums=sums,
        neuron_yardstick=neuron_y,
        group_yardstick=group_y,
        yardstick_fallback=fallback,
        params=params,
        provenance=prov,
    )


def build_atlas(
    store: Store,
    *,
    groups: Mapping[str, Sequence[int]] | None = None,
    min_weight: int = 5,
    vocabulary: str = "fafb_or_manc",
    direction: ProfileDirection = "both",
    on_missing: Literal["raise", "drop"] = "raise",
) -> Atlas:
    """Build a reference atlas from a converted dataset (labels: ``cell_type`` or ``groups``)."""
    m = ConnectivityMatrix.from_store(store, min_weight=min_weight)
    return atlas_from_matrix(
        m,
        reference=store.ref,
        groups=groups,
        min_weight=min_weight,
        vocabulary=vocabulary,
        direction=direction,
        on_missing=on_missing,
        store_provenance={"counts": store.provenance.get("counts", {})},
    )
```

`atlas_params` include `store_hash`, a hash of the reference store's provenance counts.
Without it, a calibration fitted on the real `flywire@783` would also match any other store
carrying that name (e.g. the synthetic test store), or a re-converted store.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/compare/test_atlas.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/compare/atlas.py tests/unit/compare/test_atlas.py flyconn/compare/profiles.py
git commit -m "feat(compare): reference atlas with centroid profiles, yardsticks and exact masking"
```

---

### Task 3: Scoring

**Files:**
- Create: `flyconn/compare/scoring.py`
- Test: `tests/unit/compare/test_scoring.py`

**Interfaces:**
- Consumes: `Atlas` (Task 2: `centroids`, `labels`, `neuron_yardstick`, `group_yardstick`);
  `l2_rows` (Task 1).
- Produces:
  - `Level = Literal["neuron", "group"]`;
  - `YARDSTICK_FLOOR = 0.05`;
  - `Scores` dataclass: `labels` (q x k object), `s` (q x k), `a` (q x k); properties
    `s1`, `a1`, `margin`, `top1`;
  - `score(atlas, profiles, *, level="neuron", top_k=5, chunk=5000) -> Scores`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/compare/test_scoring.py`:

```python
"""Cosine scoring of query profiles against atlas centroids."""

import numpy as np
import pytest
import scipy.sparse as sp

from flyconn.compare.atlas import build_atlas
from flyconn.compare.scoring import Scores, score
from flyconn.data.store import Store


def test_reference_neurons_score_their_own_label_first(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    prof = atlas.member_counts.profiles(direction="both")
    sc = score(atlas, prof, top_k=3)
    assert isinstance(sc, Scores)
    assert sc.labels.shape == (48, 3)
    truth = atlas.labels[atlas.member_codes]
    assert (sc.top1 == truth).all()
    assert np.all(sc.s[:, 0] >= sc.s[:, 1])
    assert np.allclose(sc.margin, sc.s[:, 0] - sc.s[:, 1])
    assert np.allclose(sc.a1, sc.s1 / np.maximum(atlas.neuron_yardstick[atlas.member_codes], 0.05))


def test_chunking_gives_identical_scores(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    prof = atlas.member_counts.profiles(direction="both")
    a = score(atlas, prof, top_k=4, chunk=5000)
    b = score(atlas, prof, top_k=4, chunk=7)
    assert (a.labels == b.labels).all()
    assert np.array_equal(a.s, b.s)


def test_group_level_uses_group_yardstick(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    prof = sp.csr_matrix(atlas.member_counts.profiles(direction="both")[:6].mean(axis=0))
    sc = score(atlas, prof, level="group", top_k=2)
    j = atlas.labels.tolist().index(sc.top1[0])
    assert sc.a1[0] == pytest.approx(sc.s1[0] / atlas.group_yardstick[j])


def test_empty_profile_and_empty_query(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    width = atlas.sums.shape[1]
    zero = score(atlas, sp.csr_matrix((1, width)))
    assert zero.s1[0] == 0.0
    none = score(atlas, sp.csr_matrix((0, width)))
    assert none.labels.shape == (0, 5)
    with pytest.raises(ValueError, match="width"):
        score(atlas, sp.csr_matrix((1, width + 1)))


def test_top_k_capped_by_label_count(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    sc = score(atlas, atlas.member_counts.profiles(direction="both")[:2], top_k=50)
    assert sc.labels.shape == (2, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/compare/test_scoring.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flyconn.compare.scoring'`.

- [ ] **Step 3: Write the implementation**

`flyconn/compare/scoring.py`:

```python
"""Cosine scores of query profiles against atlas centroids, with yardstick adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    if q.shape[1] != atlas.sums.shape[1]:
        msg = f"profile width {q.shape[1]} does not match the atlas width {atlas.sums.shape[1]}"
        raise ValueError(msg)
    yard = atlas.neuron_yardstick if level == "neuron" else atlas.group_yardstick
    yard = np.maximum(yard, YARDSTICK_FLOOR)
    k = min(top_k, len(atlas.labels))
    ct = sp.csr_matrix(atlas.centroids.T)
    orders: list[np.ndarray] = [np.zeros((0, k), dtype=np.int64)]
    sims: list[np.ndarray] = [np.zeros((0, k))]
    for start in range(0, q.shape[0], chunk):
        block = np.asarray((q[start : start + chunk] @ ct).toarray(), dtype=float)
        order = np.argsort(-block, axis=1, kind="stable")[:, :k]
        orders.append(order)
        sims.append(np.take_along_axis(block, order, axis=1))
    order = np.vstack(orders)
    s = np.vstack(sims)
    return Scores(labels=atlas.labels[order], s=s, a=s / yard[order])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/compare/test_scoring.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/compare/scoring.py tests/unit/compare/test_scoring.py
git commit -m "feat(compare): chunked cosine top-k scoring with yardstick-adjusted scores"
```

---

### Task 4: Calibration and decisions

**Files:**
- Create: `flyconn/compare/calibrate.py`, `flyconn/compare/calibration/.gitkeep`
- Test: `tests/unit/compare/test_calibrate.py`

**Interfaces:**
- Consumes: `Scores` (Task 3); `Atlas.params_hash`, `Atlas.params`, `Atlas.reference`
  (Task 2).
- Produces:
  - `FEATURE_SETS: dict[str, tuple[str, ...]]` = `{"adjusted": ("s1", "a1", "margin"), "raw": ("s1", "margin")}`;
  - `feature_matrix(scores, names) -> np.ndarray`;
  - `LogisticModel(features: tuple[str, ...], coef: np.ndarray)` with `.predict(X)`,
    `.to_dict()`, `.from_dict(d)`;
  - `fit_logistic(X, y, *, features, l2=1e-4) -> LogisticModel`;
  - `ece(p, y, n_bins=10) -> float`, `brier(p, y) -> float`;
  - `Thresholds(accept: float, s_floor: float, delta: float)`;
  - `choose_s_floor(open_s1, max_false_accept=0.10) -> float`;
  - `choose_delta(margin, correct, edges=MARGIN_EDGES, min_count=10) -> float`;
  - `apply_calls(p, s1, thr) -> np.ndarray` of "type" / "ambiguous" / "unknown";
  - `decide(scores, model, thr) -> pd.DataFrame` (columns `call`, `label`, `p`);
  - `uncalibrated_calls(scores) -> pd.DataFrame` (same columns);
  - `Calibration` dataclass with `incompatibility(atlas) -> str | None`, `id`,
    `to_json(path)`, `from_json(path)`;
  - `calibration_path(reference) -> Path`;
  - `default_calibration(atlas) -> Calibration | None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/compare/test_calibrate.py`:

```python
"""Logistic calibration, thresholds, decisions and calibration files."""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flyconn.compare.atlas import build_atlas
from flyconn.compare.calibrate import (
    Calibration,
    LogisticModel,
    Thresholds,
    apply_calls,
    brier,
    calibration_path,
    choose_delta,
    choose_s_floor,
    decide,
    ece,
    fit_logistic,
    uncalibrated_calls,
)
from flyconn.compare.scoring import Scores
from flyconn.data.store import Store


def _synthetic(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(0, 1, n)
    margin = rng.uniform(0, 0.5, n)
    p_true = 1 / (1 + np.exp(-(-6 + 6 * s1 + 10 * margin)))
    y = (rng.uniform(size=n) < p_true).astype(float)
    return np.column_stack([s1, margin]), y


def test_fit_recovers_monotone_calibrated_model():
    x, y = _synthetic()
    model = fit_logistic(x, y, features=("s1", "margin"))
    assert model.coef[1] > 0 and model.coef[2] > 0
    xt, yt = _synthetic(seed=1)
    p = model.predict(xt)
    assert ece(p, yt) < 0.03
    assert brier(p, yt) < 0.25


def test_fit_needs_both_outcomes():
    with pytest.raises(ValueError, match="both"):
        fit_logistic(np.ones((5, 2)), np.ones(5), features=("s1", "margin"))


@settings(max_examples=30, deadline=None)
@given(s=st.floats(0, 0.9), ds=st.floats(0.01, 0.1), m=st.floats(0, 0.4))
def test_probability_rises_with_score_and_margin(s: float, ds: float, m: float):
    model = LogisticModel(features=("s1", "margin"), coef=np.array([-6.0, 6.0, 10.0]))
    lo = model.predict(np.array([[s, m]]))[0]
    assert model.predict(np.array([[s + ds, m]]))[0] > lo
    assert model.predict(np.array([[s, m + ds]]))[0] > lo


def test_ece_and_brier_on_known_values():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert ece(p, y) == pytest.approx(0.1)
    assert brier(p, y) == pytest.approx(0.01)


def test_threshold_choosers():
    open_s1 = np.linspace(0, 1, 101)
    assert choose_s_floor(open_s1, max_false_accept=0.1) == pytest.approx(0.9)
    assert choose_s_floor(np.array([])) == 0.0
    margin = np.r_[np.full(20, 0.005), np.full(20, 0.03), np.full(20, 0.3)]
    correct = np.r_[np.zeros(20), np.r_[np.ones(8), np.zeros(12)], np.ones(20)]
    assert choose_delta(margin, correct) == pytest.approx(0.05)
    assert choose_delta(np.full(20, 0.3), np.ones(20)) == pytest.approx(0.01)


def test_apply_calls_order_unknown_first():
    thr = Thresholds(accept=0.8, s_floor=0.5, delta=0.05)
    calls = apply_calls(np.array([0.99, 0.99, 0.5]), np.array([0.4, 0.9, 0.9]), thr)
    assert calls.tolist() == ["unknown", "type", "ambiguous"]


def test_decide_lists_ambiguous_candidates():
    sc = Scores(
        labels=np.array([["A", "B", "C"], ["A", "B", "C"]], dtype=object),
        s=np.array([[0.9, 0.88, 0.5], [0.95, 0.3, 0.2]]),
        a=np.array([[0.9, 0.88, 0.5], [0.95, 0.3, 0.2]]),
    )
    model = LogisticModel(features=("s1", "margin"), coef=np.array([-6.0, 6.0, 10.0]))
    thr = Thresholds(accept=0.8, s_floor=0.2, delta=0.05)
    out = decide(sc, model, thr)
    assert out["call"].tolist() == ["ambiguous", "type"]
    assert out["label"].tolist() == ["A|B", "A"]
    unc = uncalibrated_calls(sc)
    assert unc["call"].tolist() == ["uncalibrated", "uncalibrated"]
    assert unc["label"].tolist() == ["A", "A"]
    assert unc["p"].isna().all()


def _cal(atlas_hash: str, params: dict[str, object]) -> Calibration:
    m = LogisticModel(features=("s1", "margin"), coef=np.array([-1.0, 2.0, 3.0]))
    thr = Thresholds(accept=0.8, s_floor=0.3, delta=0.02)
    return Calibration(
        reference="flywire@783",
        atlas_params_hash=atlas_hash,
        atlas_params=params,
        variant="raw",
        models={"neuron": m, "group": m},
        thresholds={"neuron": thr, "group": thr},
        fitted_on=["malecns@1.0"],
        metrics={"transfer_ece": {"malecns@1.0": {"neuron": 0.04, "group": 0.05}}},
        provenance={},
    )


def test_calibration_roundtrip_and_compatibility(tmp_path: Path, male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    cal = _cal(atlas.params_hash, atlas.params)
    path = tmp_path / "c.json"
    cal.to_json(path)
    back = Calibration.from_json(path)
    assert back.id == cal.id
    assert np.allclose(back.models["neuron"].coef, [-1.0, 2.0, 3.0])
    assert back.thresholds["group"].s_floor == 0.3
    assert back.incompatibility(atlas) is None
    other = build_atlas(male_female[1], min_weight=10)
    reason = back.incompatibility(other)
    assert reason is not None and "min_weight" in reason


def test_calibration_path_naming():
    p = calibration_path("flywire@783")
    assert p.name == "classify_flywire_783.json"
    assert p.parent.name == "calibration"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/compare/test_calibrate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flyconn.compare.calibrate'`.

- [ ] **Step 3: Write the implementation**

Create the empty directory marker: `touch flyconn/compare/calibration/.gitkeep`.

`flyconn/compare/calibrate.py`:

```python
"""Calibrated decisions for zero-shot classification (ADR-0010).

A logistic model maps the top candidate's scores to P(top-1 correct). It is fitted only by
the hold-out benchmark (``classify_bench``). Calls, checked in order: ``unknown`` if the best
cosine is below ``s_floor`` (set so at most ``max_false_accept`` of open-set queries pass);
``type`` if P >= ``accept``; otherwise ``ambiguous``, listing every candidate within
``delta`` of the best.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from flyconn.compare.scoring import Scores

if TYPE_CHECKING:
    from flyconn.compare.atlas import Atlas

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "adjusted": ("s1", "a1", "margin"),
    "raw": ("s1", "margin"),
}
MARGIN_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)
CALIBRATION_DIR = Path(__file__).parent / "calibration"


def feature_matrix(scores: Scores, names: Sequence[str]) -> np.ndarray:
    cols = {"s1": scores.s1, "a1": scores.a1, "margin": scores.margin}
    return np.column_stack([cols[n] for n in names]) if len(scores.s) else np.zeros((0, len(names)))


@dataclass
class LogisticModel:
    features: tuple[str, ...]
    coef: np.ndarray  # intercept first

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, len(self.features))
        return np.asarray(expit(self.coef[0] + x @ self.coef[1:]), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"features": list(self.features), "coef": [float(c) for c in self.coef]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LogisticModel:
        return cls(features=tuple(d["features"]), coef=np.asarray(d["coef"], dtype=float))


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, features: Sequence[str], l2: float = 1e-4
) -> LogisticModel:
    """L2-penalised logistic regression by L-BFGS (intercept not penalised)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        msg = "calibration needs both correct and incorrect examples"
        raise ValueError(msg)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.r_[0.0, np.ones(design.shape[1] - 1)]

    def nll(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ w
        loss = float(np.sum(np.logaddexp(0.0, z) - y * z) + l2 * np.sum(penalty * w**2))
        grad = design.T @ (expit(z) - y) + 2 * l2 * penalty * w
        return loss, grad

    res = minimize(nll, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    return LogisticModel(features=tuple(features), coef=np.asarray(res.x, dtype=float))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float("nan")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += abs(float(y[sel].mean()) - float(p[sel].mean())) * float(sel.sum())
    return total / len(p)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


@dataclass
class Thresholds:
    accept: float
    s_floor: float
    delta: float


def choose_s_floor(open_s1: np.ndarray, max_false_accept: float = 0.10) -> float:
    """Cosine that at most ``max_false_accept`` of open-set queries (true type absent) reach."""
    open_s1 = np.asarray(open_s1, dtype=float)
    return float(np.quantile(open_s1, 1 - max_false_accept)) if len(open_s1) else 0.0


def choose_delta(
    margin: np.ndarray,
    correct: np.ndarray,
    edges: Sequence[float] = MARGIN_EDGES,
    min_count: int = 10,
) -> float:
    """Upper edge of the largest margin bin where top-1 accuracy is below 50 %."""
    margin = np.asarray(margin, dtype=float)
    correct = np.asarray(correct, dtype=float)
    delta = float(edges[1])
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (margin >= lo) & (margin < hi)
        if sel.sum() >= min_count and correct[sel].mean() < 0.5:
            delta = float(hi)
    return delta


def apply_calls(p: np.ndarray, s1: np.ndarray, thr: Thresholds) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    return np.where(
        s1 < thr.s_floor, "unknown", np.where(p >= thr.accept, "type", "ambiguous")
    ).astype(object)


def decide(scores: Scores, model: LogisticModel, thr: Thresholds) -> pd.DataFrame:
    p = model.predict(feature_matrix(scores, model.features))
    calls = apply_calls(p, scores.s1, thr)
    labels: list[str] = []
    for i, call in enumerate(calls.tolist()):
        if call == "type":
            labels.append(str(scores.labels[i, 0]))
        elif call == "ambiguous":
            near = scores.s[i] >= scores.s[i, 0] - thr.delta
            labels.append("|".join(str(x) for x in scores.labels[i][near]))
        else:
            labels.append("")
    return pd.DataFrame({"call": calls, "label": labels, "p": p})


def uncalibrated_calls(scores: Scores) -> pd.DataFrame:
    n = len(scores.s)
    return pd.DataFrame(
        {
            "call": np.full(n, "uncalibrated", dtype=object),
            "label": [str(x) for x in scores.top1.tolist()],
            "p": np.full(n, np.nan),
        }
    )


@dataclass
class Calibration:
    reference: str
    atlas_params_hash: str
    atlas_params: dict[str, Any]
    variant: str
    models: dict[str, LogisticModel]
    thresholds: dict[str, Thresholds]
    fitted_on: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "atlas_params_hash": self.atlas_params_hash,
            "atlas_params": self.atlas_params,
            "variant": self.variant,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "thresholds": {k: asdict(v) for k, v in self.thresholds.items()},
            "fitted_on": list(self.fitted_on),
            "metrics": self.metrics,
            "provenance": self.provenance,
        }

    @property
    def id(self) -> str:
        core = {k: v for k, v in self.to_dict().items() if k != "provenance"}
        blob = json.dumps(core, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, default=str) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> Calibration:
        d = json.loads(Path(path).read_text())
        return cls(
            reference=d["reference"],
            atlas_params_hash=d["atlas_params_hash"],
            atlas_params=d["atlas_params"],
            variant=d["variant"],
            models={k: LogisticModel.from_dict(v) for k, v in d["models"].items()},
            thresholds={k: Thresholds(**v) for k, v in d["thresholds"].items()},
            fitted_on=list(d["fitted_on"]),
            metrics=d.get("metrics", {}),
            provenance=d.get("provenance", {}),
        )

    def incompatibility(self, atlas: Atlas) -> str | None:
        """None if this calibration applies to ``atlas``; else which parameters differ."""
        if atlas.params_hash == self.atlas_params_hash:
            return None
        diff = sorted(
            k
            for k in {*atlas.params, *self.atlas_params}
            if atlas.params.get(k) != self.atlas_params.get(k)
        )
        return "atlas parameters differ: " + ", ".join(
            f"{k}={atlas.params.get(k)!r} (calibrated {self.atlas_params.get(k)!r})" for k in diff
        )


def calibration_path(reference: str) -> Path:
    name, version = reference.split("@", 1)
    return CALIBRATION_DIR / f"classify_{name}_{version}.json"


def default_calibration(atlas: Atlas) -> Calibration | None:
    path = calibration_path(atlas.reference)
    return Calibration.from_json(path) if path.exists() else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/compare/test_calibrate.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/compare/calibrate.py flyconn/compare/calibration/.gitkeep tests/unit/compare/test_calibrate.py
git commit -m "feat(compare): logistic score calibration, abstention thresholds and calibration files"
```

---

### Task 5: `classify` API

**Files:**
- Create: `flyconn/compare/classify.py`
- Modify: `flyconn/compare/__init__.py`
- Test: `tests/unit/compare/test_classify.py`

**Interfaces:**
- Consumes:
  - Tasks 1-4: `partner_counts`, `Atlas`, `score`, `Scores`, `decide`,
    `uncalibrated_calls`, `Calibration`, `default_calibration`;
  - `partner_labels`; `run_environment`.
- Produces:
  - `GroupCall` dataclass: `call`, `label`, `p`, `s1`, `margin`, `agreement`,
    `votes: dict[str, int]`, `n`, `top_labels: list[str]`, `top_s: list[float]`;
  - `ClassifyResult` dataclass: `per_neuron: pd.DataFrame`, `group: GroupCall | None`,
    `calibrated: bool`, `caveats: list[str]`, `provenance: dict[str, Any]`;
  - `classify(query, atlas, *, cell_type=None, neuron_ids=None, calibration="default", partner_names=None, mask=(), top_k=5) -> ClassifyResult`;
  - `classify_matrix(m, atlas, *, query_ref, ...)` with the same keyword arguments.
- `per_neuron` columns: `neuron_id`, `call`, `label`, `p`, `s1`, `a1`, `margin`,
  `top_labels`, `top_s`, `unlabelled_partner_fraction`, `reason`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/compare/test_classify.py`:

```python
"""classify(): per-neuron and group calls, calibration modes, caveats, errors."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconn.compare import ClassifyResult, build_atlas, classify
from flyconn.compare.calibrate import Calibration, LogisticModel, Thresholds
from flyconn.data.store import Store


def _cal_for(atlas, accept: float = 0.5) -> Calibration:  # type: ignore[no-untyped-def]
    m = LogisticModel(features=("s1", "margin"), coef=np.array([-4.0, 4.0, 20.0]))
    thr = Thresholds(accept=accept, s_floor=0.2, delta=0.02)
    return Calibration(
        reference=atlas.reference,
        atlas_params_hash=atlas.params_hash,
        atlas_params=atlas.params,
        variant="raw",
        models={"neuron": m, "group": m},
        thresholds={"neuron": thr, "group": thr},
        fitted_on=["banc@888"],
        metrics={"transfer_ece": {"banc@888": {"neuron": 0.04, "group": 0.06}}},
    )


def test_unperturbed_type_is_classified_correctly(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=_cal_for(atlas))
    assert isinstance(res, ClassifyResult)
    assert res.calibrated
    assert len(res.per_neuron) == 6
    assert (res.per_neuron["label"] == "T1").all()
    assert (res.per_neuron["call"] == "type").all()
    assert res.group is not None
    assert res.group.label == "T1" and res.group.call == "type"
    assert res.group.agreement == 1.0
    assert res.group.top_labels[0] == "T1"
    assert set(res.per_neuron.columns) >= {
        "neuron_id",
        "call",
        "label",
        "p",
        "s1",
        "a1",
        "margin",
        "top_labels",
        "top_s",
        "unlabelled_partner_fraction",
        "reason",
    }
    assert res.provenance["atlas_params_hash"] == atlas.params_hash
    assert res.provenance["calibration_id"] is not None
    assert "git_sha" in res.provenance["environment"]


def test_uncalibrated_mode_without_calibration(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=None)
    assert not res.calibrated
    assert (res.per_neuron["call"] == "uncalibrated").all()
    assert res.per_neuron["p"].isna().all()
    assert any(c.startswith("UNCALIBRATED") for c in res.caveats)


def test_default_calibration_missing_falls_back(
    male_female: tuple[Store, Store], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import flyconn.compare.calibrate as cal_mod

    monkeypatch.setattr(cal_mod, "CALIBRATION_DIR", tmp_path)
    male, female = male_female
    res = classify(male, build_atlas(female), cell_type="T1")
    assert not res.calibrated
    assert any("no calibration file" in c for c in res.caveats)


def test_mismatched_calibration_falls_back_with_reason(male_female: tuple[Store, Store]):
    male, female = male_female
    other = build_atlas(female, direction="out")
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=_cal_for(other))
    assert not res.calibrated
    assert any("direction" in c for c in res.caveats)


def test_calibration_caveat_quotes_transfer_ece(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=_cal_for(atlas))
    assert any("leave-one-dataset-out" in c and "0.04" in c for c in res.caveats)


def test_neuron_ids_mode_deduplicates_and_single_neuron_has_no_group(
    male_female: tuple[Store, Store],
):
    male, female = male_female
    ids = male.neurons(columns=["neuron_id", "cell_type"])
    one = int(ids.loc[ids["cell_type"] == "T2", "neuron_id"].iloc[0])
    res = classify(male, build_atlas(female), neuron_ids=[one, one], calibration=None)
    assert len(res.per_neuron) == 1
    assert res.group is None


def test_neuron_without_labelled_partners_is_unknown(male_female: tuple[Store, Store]):
    male, female = male_female
    # 901/902 (T8) have no FlyWire name and only receive input from T0; with T0 masked
    # their profiles are empty
    res = classify(male, build_atlas(female), neuron_ids=[901, 902], mask=("T0",), calibration=None)
    assert (res.per_neuron["call"] == "unknown").all()
    assert (res.per_neuron["reason"] == "no_labelled_partners").all()
    assert (res.per_neuron["label"] == "").all()
    assert res.group is not None and res.group.call == "unknown"


def test_errors(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    with pytest.raises(ValueError, match="exactly one"):
        classify(male, atlas)
    with pytest.raises(KeyError, match="no neurons of type"):
        classify(male, atlas, cell_type="NOPE")
    with pytest.raises(KeyError):
        classify(male, atlas, neuron_ids=[123456789])


def test_vocabulary_mismatch_raises(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    names = pd.Series(["zz"] * 50, index=male.neurons(columns=["neuron_id"])["neuron_id"])
    with pytest.raises(ValueError, match="vocabulary"):
        classify(male, atlas, cell_type="T1", partner_names=names)


def test_partner_names_override_is_used(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    names = pd.Series(meta["cell_type"].to_numpy(), index=meta["neuron_id"].to_numpy())
    res = classify(male, atlas, cell_type="T1", partner_names=names, calibration=None)
    assert res.provenance["partner_names"] == "override"
    assert (res.per_neuron["label"] == "T1").all()


def test_heterogeneous_group_is_flagged(male_female: tuple[Store, Store]):
    male, female = male_female
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    t1 = meta.loc[meta["cell_type"] == "T1", "neuron_id"].tolist()[:3]
    t5 = meta.loc[meta["cell_type"] == "T5", "neuron_id"].tolist()[:3]
    res = classify(male, build_atlas(female), neuron_ids=t1 + t5, calibration=None)
    assert res.group is not None
    assert res.group.agreement == pytest.approx(0.5)
    assert any("heterogeneous" in c for c in res.caveats)


def test_result_independent_of_neuron_order(male_female: tuple[Store, Store]):
    male, female = male_female
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    ids = meta.loc[meta["cell_type"] == "T4", "neuron_id"].tolist()
    atlas = build_atlas(female)
    a = classify(male, atlas, neuron_ids=ids, calibration=None)
    b = classify(male, atlas, neuron_ids=ids[::-1], calibration=None)
    pd.testing.assert_frame_equal(
        a.per_neuron.sort_values("neuron_id").reset_index(drop=True),
        b.per_neuron.sort_values("neuron_id").reset_index(drop=True),
    )
    assert a.group is not None and b.group is not None
    assert a.group.s1 == pytest.approx(b.group.s1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/compare/test_classify.py -q`
Expected: FAIL with `ImportError: cannot import name 'ClassifyResult' from 'flyconn.compare'`.

- [ ] **Step 3: Write the implementation**

`flyconn/compare/classify.py`:

```python
"""Zero-shot cell-type classification by partner-type profile (ADR-0010).

Query neurons are unlabelled; their partners carry reference names through the query
dataset's cross-reference columns. Each neuron (and the selection as a group) is scored
against the reference atlas and called ``type``, ``ambiguous`` or ``unknown`` under a
calibration fitted by the hold-out benchmark, or returned ``uncalibrated`` with raw scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas
from flyconn.compare.calibrate import (
    Calibration,
    decide,
    default_calibration,
    uncalibrated_calls,
)
from flyconn.compare.profiles import partner_counts
from flyconn.compare.scoring import Scores, score
from flyconn.compare.types import partner_labels
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.provenance import run_environment

HETEROGENEOUS_AGREEMENT = 0.7
UNLABELLED_CAVEAT = 0.3
MIN_VOCAB_OVERLAP = 0.01

BASE_CAVEATS = [
    "Assignments come from wiring similarity alone (partner-type profiles), not "
    "morphology, genetics or transmitter identity.",
    "Partner names come from the query dataset's published cross-references; errors in "
    "those propagate into the profiles.",
    "A type call names the nearest reference label; types split or merged between "
    "datasets are not resolved one to one.",
]


@dataclass
class GroupCall:
    call: str
    label: str
    p: float
    s1: float
    margin: float
    agreement: float
    votes: dict[str, int]
    n: int
    top_labels: list[str] = field(default_factory=list)
    top_s: list[float] = field(default_factory=list)


@dataclass
class ClassifyResult:
    per_neuron: pd.DataFrame
    group: GroupCall | None
    calibrated: bool
    caveats: list[str]
    provenance: dict[str, Any]


def _resolve_calibration(
    calibration: Calibration | Literal["default"] | None, atlas: Atlas, query_ref: str
) -> tuple[Calibration | None, str]:
    if calibration is None:
        return None, (
            "UNCALIBRATED: no calibration requested; s1 is a raw cosine similarity, "
            "not a probability."
        )
    cal = default_calibration(atlas) if calibration == "default" else calibration
    if cal is None:
        return None, (
            f"UNCALIBRATED: no calibration file for reference {atlas.reference} with these "
            "atlas parameters; s1 is a raw cosine similarity, not a probability."
        )
    reason = cal.incompatibility(atlas)
    if reason is not None:
        return None, f"UNCALIBRATED: the calibration does not match this atlas ({reason})."
    transfer = cal.metrics.get("transfer_ece", {})
    ece_txt = ", ".join(
        f"{q}: neuron {v.get('neuron', float('nan')):.2f} / group {v.get('group', float('nan')):.2f}"
        for q, v in transfer.items()
    )
    msg = (
        f"Calibrated on {', '.join(cal.fitted_on)} against {cal.reference} (calibration "
        f"{cal.id}); leave-one-dataset-out ECE {ece_txt or 'not measured'}."
    )
    if query_ref in cal.fitted_on:
        msg += f" {query_ref} was part of the calibration data."
    return cal, msg


def _calls(scores: Scores, cal: Calibration | None, level: str) -> pd.DataFrame:
    if cal is None:
        return uncalibrated_calls(scores)
    return decide(scores, cal.models[level], cal.thresholds[level])


def classify_matrix(
    m: ConnectivityMatrix,
    atlas: Atlas,
    *,
    query_ref: str,
    cell_type: str | None = None,
    neuron_ids: Sequence[int] | None = None,
    calibration: Calibration | Literal["default"] | None = "default",
    partner_names: pd.Series | None = None,
    mask: Sequence[str] = (),
    top_k: int = 5,
) -> ClassifyResult:
    """Classify neurons of an already-built query matrix (see :func:`classify`)."""
    if (cell_type is None) == (neuron_ids is None):
        msg = "pass exactly one of cell_type or neuron_ids"
        raise ValueError(msg)
    if cell_type is not None:
        idx = np.flatnonzero((m.meta["cell_type"].astype(object) == cell_type).to_numpy())
        if len(idx) == 0:
            msg = f"no neurons of type {cell_type!r} in {query_ref}"
            raise KeyError(msg)
    else:
        idx = np.unique(m.index_of(np.unique(np.asarray(list(neuron_ids or []), dtype=np.int64))))
    vocabulary = str(atlas.params["vocabulary"])
    if partner_names is None:
        names = partner_labels(m.meta, vocabulary).tolist()
        names_source = vocabulary
    else:
        aligned = partner_names.reindex(m.neuron_ids)
        names = [v if isinstance(v, str) else None for v in aligned.tolist()]
        names_source = "override"
    counts_all = partner_counts(m, names, vocab=atlas.vocab.tolist())
    lab, tot = counts_all.labelled_mass(direction=atlas.direction)
    if tot > 0 and lab / tot < MIN_VOCAB_OVERLAP:
        msg = (
            f"only {lab / tot:.2%} of {query_ref} synapses go to partners named in the atlas "
            f"vocabulary ({vocabulary}); check the query's cross-reference columns"
        )
        raise ValueError(msg)
    counts = counts_all.rows(idx)
    mask_t = tuple(mask)
    atl = atlas.masked(mask_t) if mask_t else atlas
    prof = counts.profiles(direction=atlas.direction, mask=mask_t)
    empty = np.asarray(prof.getnnz(axis=1)) == 0
    cal, cal_caveat = _resolve_calibration(calibration, atl, query_ref)

    ns = score(atl, prof, level="neuron", top_k=top_k)
    calls = _calls(ns, cal, "neuron")
    per = pd.DataFrame(
        {
            "neuron_id": counts.neuron_ids,
            "call": calls["call"].to_numpy(dtype=object),
            "label": calls["label"].to_numpy(dtype=object),
            "p": calls["p"].to_numpy(dtype=float),
            "s1": ns.s1,
            "a1": ns.a1,
            "margin": ns.margin,
            "top_labels": [[str(x) for x in r] for r in ns.labels.tolist()],
            "top_s": [[float(x) for x in r] for r in ns.s.tolist()],
            "unlabelled_partner_fraction": counts.unlabelled_fraction(direction=atlas.direction),
            "reason": "",
        }
    )
    per.loc[empty, ["call", "label", "reason"]] = ["unknown", "", "no_labelled_partners"]
    per.loc[empty, "p"] = np.nan

    caveats = [*BASE_CAVEATS, cal_caveat]
    group: GroupCall | None = None
    if cell_type is not None or len(idx) > 1:
        group = _group_call(atl, prof, empty, ns, cal, top_k)
        if group.call != "unknown" and group.agreement < HETEROGENEOUS_AGREEMENT:
            caveats.append(
                f"heterogeneous selection: only {group.agreement:.0%} of neurons share the "
                f"group's best label {group.top_labels[0]!r} (votes {group.votes}); it may "
                "mix types."
            )
    n_unl = int((per["unlabelled_partner_fraction"] > UNLABELLED_CAVEAT).sum())
    if n_unl:
        caveats.append(
            f"{n_unl} neuron(s) have more than {UNLABELLED_CAVEAT:.0%} of synapses with "
            "partners outside the atlas vocabulary; their profiles cover the rest only."
        )
    top = set(per.loc[~empty, "top_labels"].map(lambda r: r[0]).tolist())
    fb = sorted(str(x) for x in atl.labels[atl.yardstick_fallback] if x in top)
    if fb:
        caveats.append(
            f"yardstick fallback (atlas median) for {len(fb)} best label(s) with one member "
            f"or one side only: {fb[:10]}"
        )
    if mask_t:
        caveats.append(f"partner labels masked: {list(mask_t)}")
    if any("hemibrain" in r for r in (query_ref, atlas.reference)):
        caveats.append("hemibrain is a truncated volume: similarities are lowered at its edges.")

    prov: dict[str, Any] = {
        "query": query_ref,
        "reference": atlas.reference,
        "atlas_params": atlas.params,
        "atlas_params_hash": atlas.params_hash,
        "atlas_provenance": atlas.provenance,
        "selection": {"cell_type": cell_type, "n_neurons": len(idx)},
        "partner_names": names_source,
        "mask": list(mask_t),
        "top_k": top_k,
        "calibration_id": cal.id if cal is not None else None,
        "environment": run_environment(),
    }
    return ClassifyResult(
        per_neuron=per, group=group, calibrated=cal is not None, caveats=caveats, provenance=prov
    )


def _group_call(
    atlas: Atlas,
    prof: sp.csr_matrix,
    empty: np.ndarray,
    ns: Scores,
    cal: Calibration | None,
    top_k: int,
) -> GroupCall:
    n = int(prof.shape[0])
    if empty.all():
        return GroupCall("unknown", "", float("nan"), 0.0, 0.0, 0.0, {}, n)
    gp = sp.csr_matrix(prof[np.flatnonzero(~empty)].mean(axis=0))
    gs = score(atlas, gp, level="group", top_k=top_k)
    gcall = _calls(gs, cal, "group")
    tops = pd.Series(ns.top1[~empty].astype(str))
    best = str(gs.top1[0])
    return GroupCall(
        call=str(gcall["call"].iloc[0]),
        label=str(gcall["label"].iloc[0]),
        p=float(gcall["p"].iloc[0]),
        s1=float(gs.s1[0]),
        margin=float(gs.margin[0]),
        agreement=float((tops == best).mean()),
        votes={str(k): int(v) for k, v in tops.value_counts().head(10).items()},
        n=n,
        top_labels=[str(x) for x in gs.labels[0].tolist()],
        top_s=[float(x) for x in gs.s[0].tolist()],
    )


def classify(
    query: Store,
    atlas: Atlas,
    *,
    cell_type: str | None = None,
    neuron_ids: Sequence[int] | None = None,
    calibration: Calibration | Literal["default"] | None = "default",
    partner_names: pd.Series | None = None,
    mask: Sequence[str] = (),
    top_k: int = 5,
) -> ClassifyResult:
    """Assign reference types to query neurons by partner-type profile.

    Select neurons by ``cell_type`` (the query's own label) or ``neuron_ids``. Partners are
    named with the atlas vocabulary through the query's cross-reference columns, or by
    ``partner_names`` (a Series indexed by neuron id). ``mask`` blanks labels as partner
    names (as the benchmark does for the held-out type). ``calibration="default"`` loads
    the shipped calibration for the atlas reference; None returns raw scores.
    """
    m = ConnectivityMatrix.from_store(query, min_weight=int(atlas.params["min_weight"]))
    return classify_matrix(
        m,
        atlas,
        query_ref=query.ref,
        cell_type=cell_type,
        neuron_ids=neuron_ids,
        calibration=calibration,
        partner_names=partner_names,
        mask=mask,
        top_k=top_k,
    )
```

Modify `flyconn/compare/__init__.py` to export the new API:

```python
"""flyconn.compare: cross-dataset type matching, comparison and zero-shot classification."""

from flyconn.compare.atlas import Atlas, build_atlas
from flyconn.compare.calibrate import Calibration
from flyconn.compare.classify import ClassifyResult, GroupCall, classify
from flyconn.compare.types import (
    TypeComparison,
    compare_type,
    match_types,
    partner_labels,
    type_profile,
    verdict_for,
)

__all__ = [
    "Atlas",
    "Calibration",
    "ClassifyResult",
    "GroupCall",
    "TypeComparison",
    "build_atlas",
    "classify",
    "compare_type",
    "match_types",
    "partner_labels",
    "type_profile",
    "verdict_for",
]
```

Check `test_vocabulary_mismatch_raises`: its override Series names every neuron "zz", which
is outside the atlas vocabulary, so the labelled mass is 0 and the `ValueError` fires.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/compare -q`
Expected: PASS (all compare tests, old and new).

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/compare/classify.py flyconn/compare/__init__.py tests/unit/compare/test_classify.py
git commit -m "feat(compare): classify() with per-neuron and group calls, calibration modes and caveats"
```

---

### Task 6: Hold-out benchmark and calibration fitting

**Files:**
- Create: `flyconn/compare/classify_bench.py`
- Test: `tests/unit/compare/test_classify_bench.py`

**Interfaces:**
- Consumes:
  - `Atlas`, `atlas.masked`, `partner_counts`, `score`;
  - `FEATURE_SETS`, `fit_logistic`, `ece`, `brier`, `Thresholds`, `choose_s_floor`,
    `choose_delta`, `apply_calls`, `Calibration`, `LogisticModel`, `run_environment`.
- Produces:
  - `BenchmarkResult(query: str, reference: str, records: pd.DataFrame, n_types_available: int, n_comma_excluded: int)`;
  - `truth_labels(meta, column, reference_labels) -> tuple[np.ndarray, int]`;
  - `run_benchmark(query_matrix, atlas, *, query_ref, truth_column="fafb_783_cell_type", max_types=2000, seed=0, protocols=("closed", "open"), top_k=5) -> BenchmarkResult`;
  - `fit_calibration(results, atlas, *, n_folds=5, seed=0, accept=0.8, max_false_accept=0.10, n_boot=1000) -> Calibration`
    (metrics live in `Calibration.metrics`);
  - `write_benchmark(out_dir, results, cal) -> Path`.
- Records columns: `query`, `protocol`, `level`, `type`, `neuron_id` (-1 for group rows),
  `top1`, `top_labels`, `s1`, `a1`, `margin`, `correct`, `top3`, `empty`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/compare/test_classify_bench.py`:

```python
"""Hold-out benchmark on the synthetic pair."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flyconn.compare.atlas import build_atlas
from flyconn.compare.classify_bench import (
    fit_calibration,
    run_benchmark,
    truth_labels,
    write_benchmark,
)
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix


def _bench(male_female: tuple[Store, Store]):  # type: ignore[no-untyped-def]
    male, female = male_female
    atlas = build_atlas(female)
    mq = ConnectivityMatrix.from_store(male, min_weight=5)
    return atlas, run_benchmark(mq, atlas, query_ref=male.ref, max_types=50, seed=0)


def test_truth_labels_excludes_comma_lists_and_unknown_types():
    meta = pd.DataFrame({"x": ["A", "A,B", None, "Z", "B"]})
    truth, n_comma = truth_labels(meta, "x", {"A", "B"})
    assert truth.tolist() == ["A", None, None, None, "B"]
    assert n_comma == 1


def test_closed_and_open_records(male_female: tuple[Store, Store]):
    _, res = _bench(male_female)
    rec = res.records
    assert set(rec["protocol"]) == {"closed", "open"}
    assert set(rec["level"]) == {"neuron", "group"}
    assert set(rec["type"]) == {f"T{i}" for i in range(8)}
    closed_g = rec[(rec["protocol"] == "closed") & (rec["level"] == "group")]
    assert len(closed_g) == 8
    assert closed_g["correct"].mean() >= 7 / 8  # T3 is perturbed, the rest identical
    open_ = rec[rec["protocol"] == "open"]
    assert not open_["correct"].any()  # the true label was dropped
    assert (open_["top1"] != open_["type"]).all()
    assert (rec.loc[rec["level"] == "group", "neuron_id"] == -1).all()


def test_masking_blanks_the_held_out_type(male_female: tuple[Store, Store]):
    _, res = _bench(male_female)
    rec = res.records
    # every T_i neuron still has partners after masking T_i (ring wiring), so none is empty
    assert not rec.loc[rec["protocol"] == "closed", "empty"].any()


def test_fit_calibration_metrics_and_thresholds(male_female: tuple[Store, Store], tmp_path: Path):
    atlas, res = _bench(male_female)
    # add a second "pair" so transfer ECE is exercised
    res2 = type(res)(
        query="banc@888",
        reference=res.reference,
        records=res.records.assign(query="banc@888"),
        n_types_available=res.n_types_available,
        n_comma_excluded=0,
    )
    cal = fit_calibration([res, res2], atlas, n_folds=2, seed=0, n_boot=50)
    assert cal.fitted_on == ["malecns@1.0", "banc@888"]
    assert cal.variant in {"adjusted", "raw"}
    assert set(cal.models) == {"neuron", "group"}
    assert cal.thresholds["neuron"].s_floor >= 0.0
    m = cal.metrics
    assert set(m["per_query"]) == {"malecns@1.0", "banc@888"}
    top1 = m["per_query"]["malecns@1.0"]["group"]["top1"]
    assert len(top1) == 3 and top1[1] <= top1[0] <= top1[2]
    assert set(m["transfer_ece"]) == {"malecns@1.0", "banc@888"}
    assert set(m["variants"]) == {"adjusted", "raw"}
    assert cal.atlas_params_hash == atlas.params_hash
    out = write_benchmark(tmp_path, [res, res2], cal)
    data = json.loads(out.read_text())
    assert data["calibration_id"] == cal.id
    assert (tmp_path / "records_malecns@1.0.parquet").exists()


def test_benchmark_is_deterministic(male_female: tuple[Store, Store]):
    _, a = _bench(male_female)
    _, b = _bench(male_female)
    pd.testing.assert_frame_equal(a.records, b.records)
    assert np.isfinite(a.records["s1"]).all()
```

The synthetic pair is nearly perfect, so `fit_logistic` may see only "correct" examples
at one level. `fit_calibration` must handle that: when a level's closed records are all
correct or all wrong, use a constant model whose intercept is the empirical log-odds
clipped to [-6, 6], with zero slopes, and record `"degenerate": true` for that level in the
metrics. The synthetic test exercises this path through T3, the open-set records and the
2-fold split.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/compare/test_classify_bench.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flyconn.compare.classify_bench'`.

- [ ] **Step 3: Write the implementation**

`flyconn/compare/classify_bench.py`:

```python
"""Hold-out benchmark for zero-shot classification (ADR-0010).

Truth is the query's published cross-reference to the reference (single names only).
Closed set: the evaluated type is blanked as a partner label in query and atlas, its
centroid stays. Open set: its centroid is also dropped, so the right answer is
unknown/ambiguous. Folds and bootstrap resamples are over types, never neurons.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas
from flyconn.compare.calibrate import (
    FEATURE_SETS,
    Calibration,
    LogisticModel,
    Thresholds,
    apply_calls,
    brier,
    choose_delta,
    choose_s_floor,
    ece,
    fit_logistic,
)
from flyconn.compare.profiles import partner_counts
from flyconn.compare.scoring import Scores, score
from flyconn.compare.types import partner_labels
from flyconn.graph.matrix import ConnectivityMatrix
from flyconn.provenance import run_environment

LEVELS = ("neuron", "group")
SELECTION_TOLERANCE = 0.005
"""The adjusted calibrator stays the default unless the raw one beats it by more than this."""


@dataclass
class BenchmarkResult:
    query: str
    reference: str
    records: pd.DataFrame
    n_types_available: int
    n_comma_excluded: int


def truth_labels(
    meta: pd.DataFrame, column: str, reference_labels: set[str]
) -> tuple[np.ndarray, int]:
    """Single-name cross-references present in the reference; count of comma lists dropped."""
    vals = meta[column].astype(object).tolist() if column in meta.columns else [None] * len(meta)
    out: list[str | None] = []
    n_comma = 0
    for v in vals:
        if isinstance(v, str) and "," in v:
            n_comma += 1
            out.append(None)
        elif isinstance(v, str) and v in reference_labels:
            out.append(v)
        else:
            out.append(None)
    return np.asarray(out, dtype=object), n_comma


def _rows(
    query: str, protocol: str, level: str, t: str, ids: np.ndarray, sc: Scores, empty: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "query": query,
            "protocol": protocol,
            "level": level,
            "type": t,
            "neuron_id": int(ids[i]),
            "top1": str(sc.labels[i, 0]),
            "top_labels": [str(x) for x in sc.labels[i].tolist()],
            "s1": float(sc.s1[i]),
            "a1": float(sc.a1[i]),
            "margin": float(sc.margin[i]),
            "correct": bool(sc.labels[i, 0] == t),
            "top3": bool(t in sc.labels[i, :3].tolist()),
            "empty": bool(empty[i]),
        }
        for i in range(len(ids))
    ]


def run_benchmark(
    query_matrix: ConnectivityMatrix,
    atlas: Atlas,
    *,
    query_ref: str,
    truth_column: str = "fafb_783_cell_type",
    max_types: int = 2000,
    seed: int = 0,
    protocols: Sequence[str] = ("closed", "open"),
    top_k: int = 5,
) -> BenchmarkResult:
    names = partner_labels(query_matrix.meta, str(atlas.params["vocabulary"])).tolist()
    counts = partner_counts(query_matrix, names, vocab=atlas.vocab.tolist())
    truth, n_comma = truth_labels(
        query_matrix.meta, truth_column, {str(x) for x in atlas.labels.tolist()}
    )
    by_type = (
        pd.Series(np.arange(len(truth)))[pd.notna(truth)]
        .groupby(truth[pd.notna(truth)])
        .apply(lambda s: s.to_numpy())
    )
    types = sorted(str(t) for t in by_type.index)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(types, size=min(max_types, len(types)), replace=False).tolist())
    rows: list[dict[str, Any]] = []
    for t in sample:
        idx = np.asarray(by_type[t], dtype=np.int64)
        sub = counts.rows(idx)
        prof = sub.profiles(direction=atlas.direction, mask=(t,))
        empty = np.asarray(prof.getnnz(axis=1)) == 0
        closed = atlas.masked((t,))
        gp = (
            sp.csr_matrix(prof[np.flatnonzero(~empty)].mean(axis=0))
            if (~empty).any()
            else sp.csr_matrix((1, prof.shape[1]))
        )
        for protocol in protocols:
            atl = closed if protocol == "closed" else closed.masked((), drop=(t,))
            ns = score(atl, prof, level="neuron", top_k=top_k)
            rows += _rows(query_ref, protocol, "neuron", t, sub.neuron_ids, ns, empty)
            gs = score(atl, gp, level="group", top_k=top_k)
            rows += _rows(
                query_ref, protocol, "group", t, np.array([-1]), gs, np.array([bool(empty.all())])
            )
    return BenchmarkResult(
        query=query_ref,
        reference=atlas.reference,
        records=pd.DataFrame(rows),
        n_types_available=len(types),
        n_comma_excluded=n_comma,
    )


def _features(df: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    return df[list(names)].to_numpy(dtype=float)


def _fit(df: pd.DataFrame, names: Sequence[str]) -> tuple[LogisticModel, bool]:
    y = df["correct"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        rate = float(np.clip(y.mean() if len(y) else 0.5, 1e-3, 1 - 1e-3))
        logit = float(np.clip(np.log(rate / (1 - rate)), -6, 6))
        return LogisticModel(tuple(names), np.r_[logit, np.zeros(len(names))]), True
    return fit_logistic(_features(df, names), y, features=names), False


def _type_bootstrap(df: pd.DataFrame, col: str, n_boot: int, seed: int) -> list[float]:
    """[mean, 2.5 %, 97.5 %] of ``col`` with types resampled (records weighted)."""
    g = df.groupby("type")[col].agg(["sum", "count"])
    sums, cnts = g["sum"].to_numpy(dtype=float), g["count"].to_numpy(dtype=float)
    if cnts.sum() == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(g), size=(n_boot, len(g)))
    boot = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    return [
        float(sums.sum() / cnts.sum()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    ]


def _coverage(calls: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    covered = calls == "type"
    cov = float(covered.mean()) if len(calls) else float("nan")
    acc = float(correct[covered].mean()) if covered.any() else float("nan")
    return cov, acc


def fit_calibration(
    results: Sequence[BenchmarkResult],
    atlas: Atlas,
    *,
    n_folds: int = 5,
    seed: int = 0,
    accept: float = 0.8,
    max_false_accept: float = 0.10,
    n_boot: int = 1000,
) -> Calibration:
    rec = pd.concat([r.records for r in results], ignore_index=True)
    rec = rec[~rec["empty"]]
    closed = rec[rec["protocol"] == "closed"].copy()
    open_ = rec[rec["protocol"] == "open"].copy()
    keys = closed[["query", "type"]].drop_duplicates().reset_index(drop=True)
    perm = np.random.default_rng(seed).permutation(len(keys))
    keys["fold"] = perm % n_folds
    closed = closed.merge(keys, on=["query", "type"], how="left").reset_index(drop=True)
    for variant in FEATURE_SETS:
        closed[f"p_{variant}"] = np.nan

    thresholds: dict[str, Thresholds] = {}
    for level in LEVELS:
        c = closed[closed["level"] == level]
        o = open_[open_["level"] == level]
        thresholds[level] = Thresholds(
            accept=accept,
            s_floor=choose_s_floor(o["s1"].to_numpy(), max_false_accept),
            delta=choose_delta(c["margin"].to_numpy(), c["correct"].to_numpy(dtype=float)),
        )

    variants: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, names in FEATURE_SETS.items():
        variants[variant] = {}
        for level in LEVELS:
            c = closed[closed["level"] == level]
            o = open_[open_["level"] == level]
            oof = np.full(len(c), np.nan)
            for f in range(n_folds):
                test = (c["fold"] == f).to_numpy()
                if test.any() and (~test).any():
                    model, _ = _fit(c[~test], names)
                    oof[test] = model.predict(_features(c[test], names))
            closed.loc[c.index, f"p_{variant}"] = oof
            y = c["correct"].to_numpy(dtype=float)
            full, degenerate = _fit(c, names)
            calls = apply_calls(oof, c["s1"].to_numpy(), thresholds[level])
            cov, acc = _coverage(calls, c["correct"].to_numpy(dtype=bool))
            open_calls = apply_calls(
                full.predict(_features(o, names)), o["s1"].to_numpy(), thresholds[level]
            )
            variants[variant][level] = {
                "ece_oof": ece(oof, y),
                "brier_oof": brier(oof, y),
                "coverage": cov,
                "accuracy_covered": acc,
                "false_accept_open": float((open_calls == "type").mean())
                if len(o)
                else float("nan"),
                "degenerate": degenerate,
            }

    def no_worse(level: str) -> bool:
        adj, raw = variants["adjusted"][level], variants["raw"][level]
        return bool(
            adj["ece_oof"] <= raw["ece_oof"] + SELECTION_TOLERANCE
            and (
                np.isnan(raw["accuracy_covered"])
                or adj["accuracy_covered"] >= raw["accuracy_covered"] - SELECTION_TOLERANCE
            )
        )

    variant = "adjusted" if all(no_worse(level) for level in LEVELS) else "raw"
    names = FEATURE_SETS[variant]
    models = {level: _fit(closed[closed["level"] == level], names)[0] for level in LEVELS}

    queries = list(dict.fromkeys(r.query for r in results))
    transfer: dict[str, dict[str, float]] = {}
    if len(queries) > 1:
        for q in queries:
            transfer[q] = {}
            for level in LEVELS:
                c = closed[closed["level"] == level]
                train, test = c[c["query"] != q], c[c["query"] == q]
                model, _ = _fit(train, names)
                transfer[q][level] = ece(
                    model.predict(_features(test, names)), test["correct"].to_numpy(dtype=float)
                )

    per_query: dict[str, dict[str, dict[str, Any]]] = {}
    for q in queries:
        per_query[q] = {}
        for level in LEVELS:
            c = closed[(closed["level"] == level) & (closed["query"] == q)]
            oof = c[f"p_{variant}"].to_numpy(dtype=float)
            o = open_[(open_["level"] == level) & (open_["query"] == q)]
            calls = apply_calls(oof, c["s1"].to_numpy(), thresholds[level])
            cov, acc = _coverage(calls, c["correct"].to_numpy(dtype=bool))
            open_calls = apply_calls(
                models[level].predict(_features(o, names)), o["s1"].to_numpy(), thresholds[level]
            )
            per_query[q][level] = {
                "n_types": int(c["type"].nunique()),
                "n_records": len(c),
                "top1": _type_bootstrap(c, "correct", n_boot, seed),
                "top3": _type_bootstrap(c, "top3", n_boot, seed),
                "ece_oof": ece(oof, c["correct"].to_numpy(dtype=float)),
                "coverage": cov,
                "accuracy_covered": acc,
                "false_accept_open": float((open_calls == "type").mean())
                if len(o)
                else float("nan"),
            }

    metrics: dict[str, Any] = {
        "variant_selected": variant,
        "selection_rule": (
            "adjusted unless raw has lower out-of-fold ECE or higher accuracy-at-coverage by "
            f"more than {SELECTION_TOLERANCE} at either level"
        ),
        "variants": variants,
        "per_query": per_query,
        "transfer_ece": transfer,
        "n_folds": n_folds,
        "seed": seed,
        "max_false_accept": max_false_accept,
        "benchmark": {
            r.query: {
                "n_types_available": r.n_types_available,
                "n_comma_excluded": r.n_comma_excluded,
            }
            for r in results
        },
    }
    return Calibration(
        reference=atlas.reference,
        atlas_params_hash=atlas.params_hash,
        atlas_params=atlas.params,
        variant=variant,
        models=models,
        thresholds=thresholds,
        fitted_on=queries,
        metrics=metrics,
        provenance={"atlas": atlas.provenance, "environment": run_environment()},
    )


def write_benchmark(out_dir: Path, results: Sequence[BenchmarkResult], cal: Calibration) -> Path:
    """Write per-query records (Parquet) and ``metrics.json``; return the metrics path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        r.records.to_parquet(out_dir / f"records_{r.query}.parquet", index=False)
    path = out_dir / "metrics.json"
    payload = {
        "calibration_id": cal.id,
        "variant": cal.variant,
        **cal.metrics,
        "thresholds": {k: vars(v) for k, v in cal.thresholds.items()},
        "provenance": cal.provenance,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    return path
```

Out-of-fold probabilities are stored on `closed` as `p_adjusted` / `p_raw` columns, so the
per-query metrics slice them along with the records.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/compare/test_classify_bench.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/compare/classify_bench.py tests/unit/compare/test_classify_bench.py
git commit -m "feat(compare): hold-out classification benchmark with type-fold calibration and transfer ECE"
```

---

### Task 7: CLI

**Files:**
- Create: `flyconn/cli/classify.py`
- Modify: `flyconn/cli/main.py` (register the sub-app next to `data`)
- Test: `tests/unit/test_cli_classify.py`

**Interfaces:**
- Consumes: `build_atlas`, `classify`, `run_benchmark`, `fit_calibration`, `write_benchmark`,
  `calibration_path`, `Store.open`, `ConnectivityMatrix.from_store`.
- Produces: `flyconn classify run` and `flyconn classify bench`; `read_ids(path) -> list[int]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_cli_classify.py`:

```python
"""flyconn classify run / bench on the synthetic stores."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from flyconn.cli import classify as cli_classify
from flyconn.cli.main import app
from flyconn.data.store import Store
from tests.unit.compare.conftest import _make


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Store]:
    s = {
        "malecns@1.0": _make(tmp_path, "malecns", "1.0", seed=1, perturb_type="T3"),
        "flywire@783": _make(tmp_path, "flywire", "783", seed=2, perturb_type=None),
    }
    monkeypatch.setattr(cli_classify.Store, "open", classmethod(lambda cls, ref: s[ref]))
    return s


def test_read_ids_parses_whitespace_blanks_and_duplicates(tmp_path: Path):
    p = tmp_path / "ids.txt"
    p.write_text(" 12\n\n7\n12\n  3 \n")
    assert cli_classify.read_ids(p) == [3, 7, 12]


def test_run_by_type_writes_parquet_and_json(stores: dict[str, Store], tmp_path: Path):
    out = tmp_path / "t1.parquet"
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "run",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--type",
            "T1",
            "--out",
            str(out),
            "--uncalibrated",
        ],
    )
    assert res.exit_code == 0, res.stdout
    df = pd.read_parquet(out)
    assert (df["label"] == "T1").all()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["group"]["top_labels"][0] == "T1"
    assert meta["caveats"]
    assert "T1" in res.stdout


def test_run_with_groups_file(stores: dict[str, Store], tmp_path: Path):
    female = stores["flywire@783"].neurons(columns=["neuron_id", "cell_type"])
    groups = {
        "first": female.loc[female["cell_type"] == "T1", "neuron_id"].tolist(),
        "second": female.loc[female["cell_type"] == "T5", "neuron_id"].tolist(),
    }
    gpath = tmp_path / "groups.json"
    gpath.write_text(json.dumps(groups))
    out = tmp_path / "g.parquet"
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "run",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--type",
            "T1",
            "--groups",
            str(gpath),
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (pd.read_parquet(out)["label"] == "first").all()


def test_bench_writes_metrics(stores: dict[str, Store], tmp_path: Path):
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "bench",
            "--query",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--out",
            str(tmp_path / "b"),
            "--max-types",
            "8",
            "--folds",
            "2",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (tmp_path / "b" / "metrics.json").exists()
```

The `--groups` test uses no `--uncalibrated` flag: custom groups have no calibration file,
so the run must fall back to uncalibrated mode by itself.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_classify.py -q`
Expected: FAIL with `ImportError: cannot import name 'classify' from 'flyconn.cli'`.

- [ ] **Step 3: Write the implementation**

`flyconn/cli/classify.py`:

```python
"""``flyconn classify``: zero-shot cell-type classification by partner profile."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from flyconn.data.store import Store

app = typer.Typer(
    help="Zero-shot cell-type classification by partner profile.", no_args_is_help=True
)


def read_ids(path: Path) -> list[int]:
    """Neuron ids from a text file, one per line; blanks skipped, duplicates removed."""
    tokens = [t.strip() for t in Path(path).read_text().splitlines()]
    return sorted({int(t) for t in tokens if t})


@app.command("run")
def run_cmd(
    query: str = typer.Argument(..., help="query dataset, e.g. malecns@1.0"),
    reference: str = typer.Option(..., "--reference", help="reference dataset, e.g. flywire@783"),
    cell_type: str | None = typer.Option(None, "--type", help="query cell type to classify"),
    ids: Path | None = typer.Option(None, "--ids", help="text file, one neuron id per line"),
    groups: Path | None = typer.Option(None, "--groups", help="JSON {label: [reference ids]}"),
    direction: str = typer.Option("both", "--direction", help="both | out | in"),
    min_weight: int = typer.Option(5, "--min-weight"),
    uncalibrated: bool = typer.Option(False, "--uncalibrated", help="skip calibration"),
    out: Path = typer.Option(..., "--out", help="output .parquet (a .json sidecar is written too)"),
) -> None:
    """Classify query neurons against a reference atlas."""
    from flyconn.compare import build_atlas, classify

    group_map = json.loads(groups.read_text()) if groups is not None else None
    atlas = build_atlas(
        Store.open(reference),
        groups=group_map,
        min_weight=min_weight,
        direction=direction,  # type: ignore[arg-type]
        on_missing="drop" if group_map is not None else "raise",
    )
    res = classify(
        Store.open(query),
        atlas,
        cell_type=cell_type,
        neuron_ids=read_ids(ids) if ids is not None else None,
        calibration=None if uncalibrated else "default",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    res.per_neuron.to_parquet(out, index=False)
    side = {
        "group": vars(res.group) if res.group is not None else None,
        "calibrated": res.calibrated,
        "caveats": res.caveats,
        "provenance": res.provenance,
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str) + "\n")
    if res.group is not None:
        g = res.group
        typer.echo(
            f"group: {g.call} {g.label or g.top_labels[0]} (s1 {g.s1:.3f}, margin {g.margin:.3f}, "
            f"agreement {g.agreement:.0%} of {g.n})"
        )
    typer.echo(res.per_neuron["call"].value_counts().to_string())
    for c in res.caveats:
        typer.echo(f"caveat: {c}")
    typer.echo(f"wrote {out}")


@app.command("bench")
def bench_cmd(
    query: list[str] = typer.Option(..., "--query", help="query dataset(s) with a cross-reference"),
    reference: str = typer.Option(..., "--reference"),
    out: Path = typer.Option(..., "--out", help="output directory"),
    truth_column: str = typer.Option("fafb_783_cell_type", "--truth-column"),
    max_types: int = typer.Option(2000, "--max-types"),
    folds: int = typer.Option(5, "--folds"),
    seed: int = typer.Option(0, "--seed"),
    write_calibration: bool = typer.Option(
        False, "--write-calibration", help="write the shipped calibration file for REFERENCE"
    ),
) -> None:
    """Hold-out benchmark: closed- and open-set accuracy, calibration, transfer ECE."""
    from flyconn.compare import build_atlas
    from flyconn.compare.calibrate import calibration_path
    from flyconn.compare.classify_bench import fit_calibration, run_benchmark, write_benchmark
    from flyconn.graph.matrix import ConnectivityMatrix

    atlas = build_atlas(Store.open(reference))
    results = []
    for q in query:
        store = Store.open(q)
        m = ConnectivityMatrix.from_store(store, min_weight=int(atlas.params["min_weight"]))
        results.append(
            run_benchmark(
                m,
                atlas,
                query_ref=store.ref,
                truth_column=truth_column,
                max_types=max_types,
                seed=seed,
            )
        )
        typer.echo(f"{q}: {results[-1].records['type'].nunique()} types benchmarked")
    cal = fit_calibration(results, atlas, n_folds=folds, seed=seed)
    path = write_benchmark(out, results, cal)
    for q, levels in cal.metrics["per_query"].items():
        for level, m in levels.items():
            typer.echo(
                f"{q} {level}: top-1 {m['top1'][0]:.3f} ({m['top1'][1]:.3f}-{m['top1'][2]:.3f}), "
                f"coverage {m['coverage']:.2f}, accuracy covered {m['accuracy_covered']:.3f}, "
                f"open-set false accept {m['false_accept_open']:.3f}"
            )
    if write_calibration:
        cal.to_json(calibration_path(atlas.reference))
        typer.echo(f"calibration {cal.id} -> {calibration_path(atlas.reference)}")
    typer.echo(f"wrote {path}")
```

In `flyconn/cli/main.py`, add after `from flyconn.cli.data import app as data_app`:

```python
from flyconn.cli.classify import app as classify_app
```

and after `app.add_typer(data_app, name="data")`:

```python
app.add_typer(classify_app, name="classify")
```

If `tests.unit.compare.conftest` is not importable as a module (check
`tests/unit/compare/__init__.py` exists; it does), keep the import. Otherwise move
`_make` to `tests/unit/compare/synthetic.py` and import it from there in both places.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_classify.py tests/unit/test_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
git add flyconn/cli/classify.py flyconn/cli/main.py tests/unit/test_cli_classify.py
git commit -m "feat(cli): flyconn classify run / bench"
```

---

### Task 8: Real-data benchmark, shipped calibration, LB3 golden

**Files:**
- Create: `tests/golden/test_classify.py`
- Generated: `benchmarks/classify_flywire_783.json`, `flyconn/compare/calibration/classify_flywire_783.json`

**Interfaces:**
- Consumes: everything above; `tests/fixtures/shiu_2024_neuron_sets.json` (`sets.sugar_grn`, `sets.water_grn`).
- Produces: the shipped calibration file (used by `classify(..., calibration="default")`).

- [ ] **Step 1: Write the golden tests**

`tests/golden/test_classify.py`:

```python
"""Zero-shot classification on real data (ADR-0010).

1. Hold-out benchmark: MaleCNS v1.0 and BANC v888 against FlyWire v783 (truth: the query's
   published fafb_783_cell_type, single names), closed and open set, type-level folds;
   fits and checks the shipped pooled calibration.
2. Secondary: BANC v888 against MANC v1.2.1 (nerve cord; reported only).
3. LB3a-d against FlyWire's sugar and water GRN sets (Shiu ids), the case that motivated
   this milestone (GOLDEN_RESULTS 6g).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from flyconn.compare import build_atlas, classify
from flyconn.compare.calibrate import Calibration, calibration_path
from flyconn.compare.classify_bench import fit_calibration, run_benchmark, write_benchmark
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix

pytestmark = pytest.mark.golden
OUT = Path("benchmarks") / "classify_flywire_783.json"
SETS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json").read_text()
)["sets"]
MAX_TYPES = 1500
SPIKE_GROUP_TOP1 = {"malecns@1.0": 0.92, "banc@888": 0.66}  # throwaway spike, 400 types
LB3_6G_SUGAR = {"LB3a": 0.44, "LB3b": 0.65, "LB3c": 0.92, "LB3d": 0.86}


def _update(key: str, value: object) -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")


def test_holdout_benchmark_against_flywire(tmp_path: Path):
    t0 = time.time()
    atlas = build_atlas(Store.open("flywire@783"))
    results = []
    for q in ("malecns@1.0", "banc@888"):
        store = Store.open(q)
        m = ConnectivityMatrix.from_store(store, min_weight=5)
        results.append(run_benchmark(m, atlas, query_ref=store.ref, max_types=MAX_TYPES, seed=0))
    cal = fit_calibration(results, atlas, n_folds=5, seed=0)
    write_benchmark(tmp_path, results, cal)
    elapsed = time.time() - t0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    metrics["elapsed_s"] = round(elapsed, 1)
    _update("flywire@783", metrics)

    target = calibration_path("flywire@783")
    if os.environ.get("FLYCONN_WRITE_CALIBRATION") == "1":
        cal.to_json(target)
    shipped = Calibration.from_json(target)
    for level in ("neuron", "group"):
        assert np.allclose(shipped.models[level].coef, cal.models[level].coef, atol=1e-6)
    assert shipped.atlas_params_hash == atlas.params_hash

    m = cal.metrics
    for level in ("neuron", "group"):
        assert m["variants"][cal.variant][level]["ece_oof"] <= 0.05  # spec success criterion
        assert m["variants"][cal.variant][level]["false_accept_open"] <= 0.10
    # loose floors from the throwaway spike (400 types, no masking); tightened after the run
    for q, spike in SPIKE_GROUP_TOP1.items():
        assert m["per_query"][q]["group"]["top1"][0] >= spike - 0.1
    assert elapsed < 20 * 60


def test_banc_against_manc_nerve_cord():
    atlas = build_atlas(Store.open("manc@1.2.1"))
    store = Store.open("banc@888")
    m = ConnectivityMatrix.from_store(store, min_weight=5)
    res = run_benchmark(
        m,
        atlas,
        query_ref=store.ref,
        truth_column="manc_121_cell_type",
        max_types=MAX_TYPES,
        seed=0,
    )
    cal = fit_calibration([res], atlas, n_folds=5, seed=0)
    _update(
        "manc@1.2.1",
        {
            "per_query": cal.metrics["per_query"],
            "variant": cal.variant,
            "variants": cal.metrics["variants"],
        },
    )
    assert cal.metrics["per_query"]["banc@888"]["group"]["n_types"] > 100


def test_lb3_sugar_water_against_shiu_sets():
    groups = {"sugar_grn": SETS["sugar_grn"], "water_grn": SETS["water_grn"]}
    atlas = build_atlas(
        Store.open("flywire@783"), groups=groups, direction="out", on_missing="drop"
    )
    assert atlas.provenance["group_ids_dropped"] == 1  # 20/21 sugar, 18/18 water in v783
    malecns = Store.open("malecns@1.0")
    found: dict[str, dict[str, object]] = {}
    for sub in ("LB3a", "LB3b", "LB3c", "LB3d"):
        res = classify(malecns, atlas, cell_type=sub, calibration=None, top_k=2)
        assert res.group is not None
        s = dict(zip(res.group.top_labels, res.group.top_s, strict=True))
        found[sub] = {
            "sugar": round(s["sugar_grn"], 3),
            "water": round(s["water_grn"], 3),
            "agreement": round(res.group.agreement, 3),
            "votes": res.group.votes,
        }
    _update("lb3_sugar_water", found)
    assert found["LB3c"]["sugar"] > found["LB3c"]["water"]
    assert found["LB3d"]["sugar"] > found["LB3d"]["water"]
    assert found["LB3a"]["water"] > found["LB3a"]["sugar"]
    margins = {k: abs(v["sugar"] - v["water"]) for k, v in found.items()}  # type: ignore[operator]
    assert min(margins, key=margins.__getitem__) == "LB3b"
    for sub, ref in LB3_6G_SUGAR.items():
        assert abs(found[sub]["sugar"] - ref) <= 0.05, (sub, found[sub], ref)  # type: ignore[operator]
```

- [ ] **Step 2: Run the benchmark once to create the calibration file**

Run:

```bash
FLYCONN_CACHE=$PWD/.cache/flyconn FLYCONN_WRITE_CALIBRATION=1 /usr/bin/time -l \
  uv run pytest -m golden tests/golden/test_classify.py -q -x 2>&1 | tail -30
```

Expected:
- the three tests pass;
- `flyconn/compare/calibration/classify_flywire_783.json` and
  `benchmarks/classify_flywire_783.json` are written;
- the time report shows wall time and peak RSS (the `maximum resident set size` line).

What to do if it doesn't:

- **ECE above 0.05 or open false-accept above 0.10:** the spec criterion failed. STOP and
  report the numbers to the user. Don't loosen the assertion.
- **A LB3 sugar cosine more than 0.05 from 6g:** check whether the difference comes from
  the two documented method differences (6g pooled synapses across the set and kept
  unmatched MaleCNS partner names in the vector; the atlas averages normalised member
  profiles and drops unnamed partners). Recompute one subtype with 6g's method on the same
  partner vocabulary in a scratch script. Record both numbers. Change the tolerance only
  with that explanation in the test's docstring.
- **Runtime above 20 min or RSS above 8 GB:** reduce `MAX_TYPES` to 1000 and record why.
- **`group_ids_dropped` not 1:** set it to the measured value and note it. 6g says 20/21
  and 18/18.

- [ ] **Step 3: Pin bands to the measured numbers**

Open `benchmarks/classify_flywire_783.json` and read the measured
`per_query.<q>.group.top1[0]` and `per_query.<q>.neuron.top1[0]` for both queries. Replace
`SPIKE_GROUP_TOP1` in the test with a dict `MEASURED_TOP1` holding the measured group and
neuron top-1 per query (rounded to 3 decimals). Change the floor assertion to `>= measured -
0.03` for both levels. Add a comment naming the run date and the JSON as the source.

- [ ] **Step 4: Re-run without the write flag to prove reproducibility**

Run: `FLYCONN_CACHE=$PWD/.cache/flyconn uv run pytest -m golden tests/golden/test_classify.py -q`
Expected: PASS. The shipped calibration's coefficients match a fresh fit to 1e-6.

- [ ] **Step 5: Check the calibration ships in the wheel**

Run: `d=$(mktemp -d) && uv build --wheel -o "$d" && unzip -l "$d"/*.whl | grep classify_flywire_783.json`
Expected: the JSON is listed (hatchling includes package data under `packages = ["flyconn"]`).
If it is missing, add `[tool.hatch.build.targets.wheel.force-include]` for the calibration
directory in `pyproject.toml`.

- [ ] **Step 6: Gate and commit**

```bash
scripts/check.sh
git add tests/golden/test_classify.py benchmarks/classify_flywire_783.json flyconn/compare/calibration/classify_flywire_783.json
git commit -m "test(golden): classification hold-out benchmark (MaleCNS, BANC vs FlyWire), shipped calibration, LB3"
```

---

### Task 9: Docs, ADR, changelog

**Files:**
- Create: `docs/adr/0010-zero-shot-classification.md`, `docs/tutorials/classify.md`
- Modify: `CHANGELOG.md`, `docs/PROGRESS.md`, `docs/GOLDEN_RESULTS.md`, `mkdocs.yml`,
  `docs/superpowers/specs/2026-10-10-zero-shot-classify-design.md`

Every number in this task is copied from `benchmarks/classify_flywire_783.json`
(Task 8), never typed from memory.

- [ ] **Step 1: ADR-0010**

`docs/adr/0010-zero-shot-classification.md`, using the repo template (Context → Decision
→ Alternatives considered → Consequences → Evidence), Status: Accepted (2026-10-10, owner
approved the design). Content:

- **Context:**
  - the LB3 sugar/water call (6g) worked by hand;
  - mode A (partners named) vs mode B (nothing named);
  - the cross-reference columns available as truth (MaleCNS 143k and BANC 120k neurons
    with `fafb_783_cell_type`).
- **Decision:**
  - reuse the `compare/types` vocabulary (`partner_labels`) and profile idea; out+in
    fraction profiles;
  - nearest centroid with yardstick adjustment;
  - scipy logistic calibration fitted on pooled pairs; abstention thresholds from the
    open set;
  - the selected variant and why (from `variant_selected` and `variants`);
  - no scikit-learn, no NBLAST.
- **Alternatives considered:**
  - nearest-neighbour voting;
  - a learned model;
  - morphology/NBLAST (GPL extra);
  - mode B (deferred, interface kept via `partner_names`).
- **Consequences:**
  - calibration is per reference and per atlas parameters; custom groups are always
    uncalibrated;
  - MaleCNS accuracy may be optimistic (its cross-references may partly derive from
    connectivity; unverified), so BANC is the conservative number;
  - the transfer ECE is quoted in every calibrated result.
- **Evidence:** the measured table from the benchmark JSON (per pair and level: top-1 with
  CI, coverage, accuracy among covered, ECE out-of-fold, transfer ECE, open-set false
  accept), runtime and RSS.

- [ ] **Step 2: GOLDEN_RESULTS section 6i**

Insert a new `### 6i. Zero-shot classification benchmark (ADR-0010; <run date>)` after
section 6h. It contains:
- the same table as the ADR evidence;
- the BANC -> MANC secondary numbers;
- the LB3 numbers next to the 6g numbers, with the method difference explained in one
  sentence;
- the command that reproduces it.

- [ ] **Step 3: Tutorial**

`docs/tutorials/classify.md`. Cover:

- the Python example (`build_atlas`, `classify` on LB3b; read `per_neuron`, `group`,
  `caveats`);
- the custom sugar/water groups example;
- the two CLI commands;
- how to read `type` / `ambiguous` / `unknown` / `uncalibrated`;
- what "zero-shot" assumes (partners named via cross-references);
- the scientific caveats.

Add to `mkdocs.yml` nav right after `- Tutorials: tutorials/index.md`:

```yaml
  - Classification: tutorials/classify.md
```

- [ ] **Step 4: CHANGELOG, PROGRESS, spec**

- `CHANGELOG.md`, under `## [Unreleased]`, add `### M12 zero-shot classification`. One
  bullet each for:
  - the API;
  - the CLI;
  - the benchmark with its measured headline numbers;
  - the shipped calibration;
  - the caveat that MaleCNS accuracy may be optimistic.
- `docs/PROGRESS.md`: add a dated entry at the top: what was built, the measured numbers,
  open items (mode B; verify how MaleCNS assigned its FlyWire cross-references; hemibrain
  not benchmarked; kNN alternative if the benchmark shows heterogeneous types fail).
- The spec, section 2.5: replace `partner_labels=None` with `partner_names=None` (renamed
  to avoid shadowing `compare.types.partner_labels`). Add one line under the status: "As
  built: see ADR-0010."

- [ ] **Step 5: Gate and commit**

```bash
scripts/check.sh
uv run mkdocs build --strict 2>&1 | tail -5
git add docs/ CHANGELOG.md mkdocs.yml
git commit -m "docs(m12): ADR-0010 zero-shot classification; benchmark results; tutorial"
```

If `mkdocs` is not in the dev group, skip that command and note it in the commit body.

---

## Self-review notes (done while writing)

- **Spec coverage:**
  - §2.1 → Task 2; §2.2 → Task 1; §2.3 → Task 3; §2.4 → Task 4 (calls, thresholds, file)
    and Task 6 (fitting); §2.5 → Tasks 5 and 7;
  - §3 caveats → Task 5; §4 benchmark → Tasks 6 and 8 (BANC→MANC in Task 8);
  - §5 errors → Task 5 tests; §6 → unit tests in every task, golden in Task 8;
  - §7 → `run_environment` (Task 1), provenance in Tasks 5 and 6, docs in Task 9.
- **Spec gaps found and resolved:**
  - the spec's `partner_labels=` argument name is renamed `partner_names=` (Task 9 updates
    the spec);
  - the spec does not say how a level is calibrated when all its benchmark examples are
    correct: Task 6 uses a constant model flagged `degenerate`.
- **Type consistency:**
  - `Scores.top1/s1/a1/margin` are used in Tasks 4-6;
  - `Calibration.models/thresholds` are keyed `"neuron"`/`"group"` everywhere;
  - `run_benchmark(..., query_ref=...)` is used in Tasks 6-8;
  - `atlas.masked(mask, drop=...)` is used in Tasks 5-6.
