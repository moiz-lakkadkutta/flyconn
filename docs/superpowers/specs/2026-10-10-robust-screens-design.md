# M13 design: robustness-audited silencing screens with driver-line handoff

Status: design approved in conversation 2026-10-10 (option A, one axis at a time).
Branch `m13-robust-screens` (worktree `../google-fly-m13`, off `main` at 53d1715).
ADR-0011. GOLDEN_RESULTS section 6j. Runs in parallel with M12 (zero-shot
classification, ADR-0010, GOLDEN_RESULTS 6i); M12 confirmed it does not touch
`flyconn/experiments`, `flyconn/report`, `flyconn/access` or `compare/types.py`.

## 1. Goal and scope

Turn an M10 silencing screen into a ranked hit list a wet lab can act on. Each hit says:

- how robust the predicted effect is across the main modelling uncertainties;
- whether it beats silencing random, size-matched neuron sets;
- which split-GAL4 line (Meissner 2025) reaches it, and what experiment would test it.

**Primary consumer:** a wet-lab collaborator. Success means they can plan a silencing
experiment from `report.html` / `handoff.csv` alone, and can see why each hit is (or is not)
trusted.

**Why:** GOLDEN_RESULTS 6h showed by hand that MaleCNS screen hits depend on modelling
choices. GNG232 -> MN9 survived the w_syn change from 0.275 to 0.188 mV. DNge080, the
inhibitory-route relays and the top-10 activity selection did not. M13 makes that check
automatic and covers the remaining uncertainty axes.

**In scope:** `silence_each` sweeps (both `rank_by: activity` and `rank_by: pathway`).

**Out of scope:**
- `rate_hz` sweeps;
- joint (factorial or Monte Carlo) sampling of axes (option B, rejected: loses per-axis
  attribution);
- graph-only prefiltering of candidates (option C, rejected: may drop hits that arise from
  network dynamics);
- calling NeuronBridge over the network;
- inferring behaviour from readouts.

## 2. Components

```
YAML spec + robustness: block
        |
        v
robustness.plan_variants(spec) -> [Variant(axis, label, ExperimentSpec)]
        |
        v
run_sweep(variant_spec)       (existing M10 code; one call per variant, out/variants/<label>/)
        |
        v
robustness.score(...)         -> per hit x axis labels, tiers, path-stability column
        |
        v
access.attach_lines(hits)     -> LineCatalog.lines_for_type
        |
        v
report.html, hits.parquet, handoff.csv, provenance.json
```

### 2.1 Spec additions (`flyconn/experiments/spec.py`)

**`params: {w_syn_mv: float}`** (optional)
- Overrides w_syn.
- `prepare` passes `ShiuParams(w_syn_mv=...)` to `LIFNetwork.from_matrix`; today it always
  uses `default_params(dataset)`.
- The calibration banner (`_calibration_note`) already reports a w_syn that differs from
  the calibrated value.

**`network.sign_seed: int`** (optional)
- Allowed only with `sign_policy: sampled`. Any other combination fails validation.
- Passed as `seed=` to `ConnectivityMatrix.from_store`.
- **Guard:** `neuron_signs` silently falls back to argmax when the store has no NT
  probabilities, so `prepare` raises if `sign_policy: sampled` is used on a store without
  them. MaleCNS needs `flyconn data pull malecns@1.0 --level nt-probs`.

**`robustness:` block** (optional)
- Present only for `silence_each` sweeps; otherwise validation fails.
- Fields and defaults:

```yaml
robustness:
  primary_readout: MN9          # required; must name a readout
  readout_proxy: null           # optional free text + citation, e.g. "MN9 drives proboscis
                                #   extension (Shiu et al. 2024)"; never inferred
  w_syn: auto                   # auto | off | [list of mV]
  nt_samples: 8                 # 0 = off
  min_weights: [3, 5]           # [] = off
  replicate: flywire@783        # null = off
  random_null: {n_sets: 20}     # n_sets 0 = off
```

- **`w_syn: auto`**
  - Uses the distinct values in `CALIBRATIONS[dataset].ci_mv` together with
    `sensitivity_mv`, excluding the reference value. For MaleCNS that is
    {0.185, 0.192, 0.209} mV: three variants, because 0.185 is both a CI end and a
    sensitivity end.
  - A dataset with no calibration entry needs an explicit list. Otherwise validation fails
    and the message names the option.

### 2.2 `flyconn/experiments/robustness.py` (new)

**`plan_variants(spec) -> list[Variant]`**
- Pure function; does no I/O.
- Builds the reference variant plus one variant per axis value:
  - `wsyn_0.185`, ...
  - `nt_s0` ... `nt_s7`, with sign seeds `spec.seed + 2000 + k`
  - `minw_3`, `minw_5`
  - `rep_flywire_783`
- **Fixed candidates:**
  - Every variant gets the reference run's candidate list as an explicit
    `silence_each.select` of ids grouped by the reference labels. `rank_by` and
    `max_items` are dropped, so no variant re-selects.
  - Requires a small hook in `sweep.py` to accept an explicit candidate table. This is the
    only change to existing sweep code.

**Selection stability**
- Each non-replicate variant also records which candidates its own selection would have
  picked. This costs no simulation, because selection uses the variant's stimulated run,
  which is computed anyway.
- Reported as the Jaccard overlap with the reference list.

**`run_robust_screen(spec, out_dir, store=None) -> RobustScreenResult`**
- Runs the variants in sequence. Each writes `out/variants/<label>/` through `run_sweep`.
- Any variant whose directory holds a complete result (marked by a `done.json` written
  last) is skipped, which is what lets an interrupted run resume.
- The top-level `out_dir` may already exist only when its `robustness_plan.json` matches
  the current plan hash. Otherwise the existing refuse-to-overwrite rule applies.

**Random-silencing null**
- For each distinct candidate size n (neurons in the group) and super_class:
  - draw `n_sets` random sets of n neurons from that super_class, excluding every
    candidate's neurons;
  - silence each set in the reference network;
  - seeds are `spec.seed + 3000 + i`.
- Sizes are binned to the nearest of {1, 2, 4, 8, 16, 32, ...} to bound cost. The bin is
  recorded.

**Cross-dataset replicate**
- Uses `compare.match_types(query, replicate)` to map the stimulus selection, the readouts
  and each candidate type.
- Anything unmapped makes that hit's replicate label `not testable` (reason recorded).
- Runs at the replicate's default params: Shiu's 0.275 mV for FlyWire 783, labelled
  UNCALIBRATED.
- If the replicate dataset is not pulled, the axis is `not run`, and the message shows the
  pull command.

### 2.3 Scoring (`robustness.score`)

**Reference hit:** a candidate x primary-readout row with q < `report.alpha` (default 0.05,
BH as in M10) in the reference run.

**Per-variant label** (95 % CI of the variant's effect vs the reference sign; seeds shared,
so the comparison is paired):

| label | rule |
|---|---|
| holds | same sign, CI excludes 0 |
| weakens | same sign, CI includes 0 |
| flips | opposite sign, CI excludes 0 |
| weakens | opposite sign, CI includes 0 (no evidence either way) |

**Axis verdicts**

| axis | robust when |
|---|---|
| w_syn | holds at every w_syn value |
| NT signs | holds in >= 6 of 8 samples (threshold = ceil(0.75 x nt_samples)); any flip is listed |
| threshold | holds at every min_weight |
| random null | the reference effect is more extreme, in its own direction, than the 95th percentile of random-set effects for its size bin and super_class |
| replicate | holds / weakens / flips / not testable / not run; reported, **never graded** (uncalibrated, possibly sexually dimorphic) |

**Tier**
- Strong: robust on every graded axis that ran.
- Conditional: fails exactly one graded axis; the card names it and states the condition,
  e.g. "holds unless w_syn >= 0.209".
- Fragile: fails two or more graded axes.
- Axes that are switched off or not run do not count as failures. The card lists them as
  untested, and a hit tested on fewer than two graded axes carries a "thinly tested"
  caveat.

**Path-stability column** (informational, not graded)
- `uncertainty.path_stability` between the stimulus selection and the primary readout,
  restricted to paths through the candidate.
- Reported as the mean of the paths' sign-stability x threshold-presence.

**No multiple-testing correction across variants.** Variants measure stability; they are
not new hypotheses (ADR-0011).

### 2.4 Driver lines (`flyconn/access/handoff.py`, new; `attach_lines`)

- **Lookup:** per hit, `LineCatalog.lines_for_type(<group label>)`, sorted by quality
  (1 best) and then `specific`.
- **Match method:** recorded as `exact`, `glob` or `none`. Meissner annotations are free
  text across naming schemes (`docs/caveats.md`).
- **No match:** "no annotated split-GAL4 in Meissner 2025". Lines are never invented.
- **Data not pulled:** "lines not looked up", with the pull command.
- **NeuronBridge:** off-target counts are shown only when already cached under
  `$FLYCONN_CACHE/neuronbridge/`. No network calls.
- **Import:** lazy import of `flyconn.access`; the access extra stays optional.

### 2.5 Report and handoff (`flyconn/report/html.py`)

**`report.html`**
- Existing banners (model prediction; the calibration note).
- Ranked hit table: tiers Strong > Conditional > Fragile, then |effect|.
- One evidence card per hit:
  - reference effect, CI and q;
  - a line per axis with its label and counts;
  - the replicate line;
  - path stability;
  - lines;
  - suggested test;
  - caveats.
- A selection-stability panel.
- A negative-results section: candidates that were not reference hits.

**Suggested test (templated)**
- Text: "silence <label> (e.g. Kir2.1, GtACR1) under <stimulus summary>; the model predicts
  <primary readout> goes <up|down> by <effect> Hz."
- `readout_proxy` is appended verbatim when the spec gives one.
- Effector names are examples, not recommendations.

**Output files**
- `hits.parquet`: one row per candidate x variant (axis, label, effect, CI, q).
- `handoff.csv`: one row per reference hit. Columns: label, tier, direction, effect, CI
  low/high, q, failed axes, untested axes, replicate label, best line, line quality, line
  specific, line match method, suggested test.
- `provenance.json`:
  - the spec SHA-256;
  - dataset refs and checksums;
  - every variant's w_syn, sign seed and min_weight;
  - random-null seeds and size bins;
  - the replicate mapping table;
  - the Meissner table checksum;
  - package versions and git SHA.

**CLI:** `flyconn run SPEC --out DIR` detects the `robustness:` block. There is no new
command.

## 3. Caveats attached by default

- Model predictions from wiring plus Shiu LIF dynamics; not measurements.
- MaleCNS w_syn calibrated by protocol, not validated (ADR-0009).
- NT signs are predicted; sampled-sign variants assume the published probabilities are
  calibrated.
- The replicate runs at an uncalibrated w_syn on a different sex. Disagreement may be
  biology or calibration.
- Line annotations are curated text and may not match EM type names one to one.
- "Strong" means robust within the tested axes, not true.

## 4. Testing

**Unit** (offline; `tests/fixtures/` synthetic connectome)
- `plan_variants`:
  - exact variant list for a full block;
  - `auto` values from a test calibration entry;
  - each axis can be turned off;
  - missing calibration plus `auto` gives a clear error;
  - the fixed-candidate rewrite drops `rank_by` and `max_items`.
- Spec:
  - `params.w_syn_mv` changes spike counts vs the default;
  - `sign_seed` is reproducible and rejected without `sampled`;
  - `sampled` on a probability-less store raises.
- Labels: a table test, plus a `hypothesis` property that negating every effect swaps
  holds and flips.
- Tiers: every axis-outcome combination maps to exactly one tier; off and not-run axes
  never count as failures.
- Random null: sets match size bin and super_class, exclude candidates, and are the same
  for the same seed.
- Resume: interrupting after variant k and rerunning gives an identical result and skips
  finished variants. A mismatched plan hash refuses.
- `attach_lines`: a fake `lines.parquet` covers exact, glob and none, plus missing data
  giving "not looked up".
- Report and handoff: the columns, the banners, and every card having caveats.

**Golden** (`tests/golden/test_robust_screen.py`, spec
`examples/specs/w2_robust_pathway_mn9.yaml`)
- The spec is the 6h pathway screen plus a full `robustness:` block.
- **Negative-control candidate:** one extra cell type with no LB3 -> MN9 route, chosen when
  writing the spec and recorded there with the path query that confirms it.
- Asserts:
  - the output files and schema;
  - the same seed gives identical output;
  - the negative control is never Strong and fails the random null;
  - the reference effects match 6h within their CIs.
- Does **not** assert the tiers of GNG232, DNge080 or other relays before the first run.
  Observed tiers are recorded in GOLDEN_RESULTS 6j and then pinned as regression values.
- Records wall time and peak RSS (16 GB budget; variants run one at a time with one
  network in memory) in `benchmarks/robust_screen_malecns.json`.

## 5. Done when

- `scripts/check.sh` is green, and `uv run ruff format docs/` has been run on the new
  docs.
- The golden test passes on the M4 Pro.
- GOLDEN_RESULTS 6j holds the hit table: tiers, axis labels and lines.
- ADR-0011 is written: the axes, option A over B and C, the label and tier rules, and
  the replicate not being graded.
- `docs/tutorials/robust-screens.md` exists and is in the `mkdocs.yml` nav.
- CHANGELOG has an M13 entry, and PROGRESS has a new top entry.

## 6. Risks

- **Cost:** 15 screen runs for MaleCNS (1 reference + 3 w_syn + 8 NT + 2 thresholds +
  1 replicate) plus the random-null runs. For 10 relays at 10 x 0.3 s
  this is extrapolated, not measured, at about 25 min on the M4 Pro. Resume makes
  interruptions cheap. If it is much slower, cut `n_sets` before cutting axes.
- **The NT axis needs the 2.65 GB `nt-probs` level** for MaleCNS. Without it the axis is
  `not run`, never silently argmax.
- **The replicate mapping of the stimulus is non-trivial:** FlyWire types sugar and water
  GRNs both as `LB3`. The replicate maps LB3* to FlyWire `LB3` and records that it is the
  sugar+water group, the same as the current MaleCNS spec. When M12's classifier lands, a
  follow-up can replace the mapping with classified groups.
- **Fixed candidates hide re-selection effects;** the selection-stability panel reports
  them instead.
