# ADR-0012: Hardening robustness screens: informative nulls and controls, paired statistics

Status: Accepted (2026-10-24). Plan: `docs/superpowers/plans/2026-10-24-m14-scientific-hardening.md`.
Amends ADR-0011.

## Context

The M13 golden run (GOLDEN_RESULTS 6j) recorded weak checks rather than tuning them away:

- The random null could not discriminate for GNG232. All 40 random 2-neuron central-brain
  sets changed MN9 by exactly 0, so any nonzero effect reached the minimum p (1/41), and the
  tier counted that as a pass. The cause: silencing a neuron that never fires cannot change
  a run, and most randomly drawn neurons are silent under LB3 drive.
- The negative control passed trivially. LC4 fires no spike under LB3 drive.
- Effects used an unpaired bootstrap CI and a Mann-Whitney p, although trial k of every
  condition uses Poisson seed stream k (`flyconn/sim/engine.py`). That is conservative and
  wastes power.
- Null-set seeds depended on the position of a key among all keys, so adding a candidate
  reshuffled every other key's sets.
- The plan hash that guards resumed output directories ignored the dataset store, so a rerun
  against a different store build could resume old variants.

## Decision

For robustness screens (specs with a `robustness:` block):

- **D1. An uninformative null is not a pass.** When at least 50 % of a key's random sets
  have an effect of exactly 0, the random-null axis scores `uninformative`. The tier counts
  it as neither a pass nor a fail, and it does not count as "ran" for the thinly-tested
  flag. Cards and `handoff.csv` (`uninformative_axes`) say so.
- **D2. Activity-matched null pool.** Random sets are drawn only from neurons of the same
  super_class that fire at least one spike in the reference stimulated run. Default
  `random_null: {pool: active}`; `pool: all` restores the M13 pool. A bin whose active pool
  is smaller than the set size is skipped with a reason; it never falls back silently.
- **D3. Paired statistics.** `report.statistics: paired` (the default when a robustness
  block is present): per-trial differences against the stimulated run, a trial-bootstrap
  95 % CI and a Wilcoxon signed-rank p. The difference itself is unchanged, because the mean
  of per-trial differences equals the difference of means. `cohens_d` keeps the unpaired
  pooled-s.d. definition. Plain experiments and sweeps stay unpaired; `paired` without a
  sweep is refused.
- **D4. An off-route active negative control.** `robustness.negative_controls: N` (default
  1) adds the N groups with the highest summed stimulated rate (≥ 1 Hz) that have no neuron
  on any stimulus -> primary-readout path of ≤ `max_hops` hops at min_weight 1. Every
  control, automatic or from `extra_candidates`, is reported in `controls.parquet` with
  `informative = fires under the stimulus`. A silent control is reported as uninformative.
- **D5. Null seeds keyed by the key text** (`crc32(key)`), not by its index among keys.
- **D6. The plan hash includes a fingerprint of each store** (dataset ref, level, sorted
  input SHA-256 from `provenance.json`), for the main and the replicate store.

## Consequences

- Robustness screens from 0.4.0 are not comparable number-for-number. p values and CIs
  change (paired), null sets change (D2, D5), and a control is added (D4). GOLDEN_RESULTS
  6k reports the M14 golden runs against 6j.
- M13 output directories do not resume under M14: the spec's canonical form and the plan
  hash both changed.
- D1 can make a hit thinly tested that was not before. D2 can leave a bin not run on
  datasets where few neurons of a super_class are active.
- D4 adds one `find_paths` call at min_weight 1 to the reference run (204,286 paths on
  MaleCNS for LB3* -> MN9 at 3 hops). "Off-route" means none within `max_hops`: a control can
  still act through longer routes, so a large control effect is a warning about the model,
  not a hit.
- "Strong" still means robust within the tested axes, not true.
