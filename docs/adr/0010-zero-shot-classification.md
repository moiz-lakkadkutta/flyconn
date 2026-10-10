# ADR-0010: Zero-shot cell-type classification by partner profile, calibrated per type

Status: Accepted (2026-10-10, owner approved the design and the group-only calibration).

## Context

GOLDEN_RESULTS 6g called MaleCNS LB3a-d sugar- or water-like by comparing their
output-partner profiles with FlyWire's sugar and water GRN sets. That was done by hand,
with no confidence and no "unknown" answer. MaleCNS (143,154 neurons) and BANC (119,868)
publish a `fafb_783_cell_type` cross-reference, and BANC also publishes
`manc_121_cell_type` (25,308). These give held-out truth for a benchmark.

There are two settings:
- **Mode A:** the query neurons are unlabelled but their partners are named through the
  cross-references.
- **Mode B:** nothing is named.

Mode B needs iterative propagation from anchors and is a separate research problem.

## Decision

- **Mode A only.** `classify(..., partner_names=...)` lets a later mode B feed its own
  predictions as partner names.
- **Profiles.** Each neuron gets the fractions of its output and input synapses to/from
  each partner label, each half L1-normalised. Labels come from the shared FlyWire-or-MANC
  vocabulary of `compare.types.partner_labels`; this is the reuse decision.
- **Atlas.** One centroid per reference label (mean of member profiles), plus two
  yardsticks: the neuron yardstick (median leave-one-out member-to-centroid cosine) and
  the group yardstick (left/right centroid cosine, the variability `compare_type` already
  uses). Custom groups (neuron-id lists) are supported because FlyWire v783 names sugar
  and water GRNs both `LB3`.
- **Scoring.** Nearest centroid by cosine. `a = s / yardstick` is offered to the
  calibrator.
- **Calibration.**
  - Model: logistic P(top-1 correct) on `(s1, a1, margin)`, fitted with `scipy.optimize`.
    The benchmark kept the adjusted variant over the raw one `(s1, margin)` by the
    pre-registered rule (out-of-fold ECE and accuracy-at-coverage not worse by > 0.005).
  - Thresholds: accept at P >= 0.8; `unknown` below the cosine that at most 10 % of
    open-set queries reach; `ambiguous` lists candidates within `delta` of the best.
  - Fit: MaleCNS and BANC pooled against FlyWire v783.
- **Only whole-type (group) calls are calibrated.** Per-neuron calls return the best label
  and raw cosine, labelled uncalibrated, with a caveat. In the benchmark, per-neuron
  confidence did not transfer between datasets: leave-one-dataset-out ECE was 0.235
  (MaleCNS) and 0.237 (BANC), and BANC per-neuron calls with P >= 0.8 were right 74.9 %
  of the time. The owner chose this over shipping per-neuron probabilities.
- **No scikit-learn and no NBLAST** (GPL extra, ADR-0006).

## Alternatives considered

- **Nearest-neighbour voting.** It handles heterogeneous types better but is noisier for
  the many 2-neuron types. Kept as a possible follow-up if heterogeneous types turn out to
  fail.
- **A learned model trained on the reference.** It risks learning one dataset's tracing
  quirks, and is harder to explain.
- **Morphology (NBLAST).** GPL; it would be an optional extra.
- **Ship per-neuron calibration with a caveat.** Rejected: the error is too large.
- **Add a dataset-difficulty feature to the calibrator.** Not tried; open follow-up.

## Consequences

- A calibration file applies only to one reference, store and set of atlas parameters
  (`params_hash` includes a hash of the store's provenance counts). Custom groups and
  other parameters are always uncalibrated.
- Every calibrated result quotes the leave-one-dataset-out ECE. For group calls it is
  0.084 (MaleCNS) and 0.128 (BANC), above the 0.05 in-sample target. A new dataset should
  expect group confidence to be off by about that much.
- MaleCNS accuracy may be optimistic if its FlyWire cross-references were partly assigned
  by comparing connectivity (unverified). BANC is the conservative estimate.
- Comma-separated cross-references (several FlyWire types) are excluded from accuracy:
  2,751 MaleCNS and 1,022 BANC neurons.

## Evidence

`benchmarks/classify_flywire_783.json` (calibration `da750126359a`),
`tests/golden/test_classify.py`, `docs/GOLDEN_RESULTS.md` §6i. 1,500 types per query,
closed and open set, 5 type-level folds, 1,000 bootstrap resamples over types. All three golden
tests run in 179 s on the M4 Pro; the FlyWire and MANC benchmarks together peaked at
3.1 GB RSS.

| query (vs FlyWire v783) | level | top-1 (95 % CI) | top-3 | coverage | accuracy covered | ECE (out of fold) | open-set false accept |
|---|---|---|---|---|---|---|---|
| MaleCNS v1.0 | group | 0.909 (0.895-0.922) | 0.961 | 0.649 | 0.990 | 0.047 | 0.059 |
| MaleCNS v1.0 | neuron | 0.813 (0.697-0.904) | 0.899 | uncalibrated | | 0.061 | |
| BANC v888 | group | 0.657 (0.633-0.680) | 0.799 | 0.306 | 0.941 | 0.050 | 0.044 |
| BANC v888 | neuron | 0.423 (0.358-0.489) | 0.557 | uncalibrated | | 0.161 | |

Pooled group: ECE 0.012, coverage 0.478, accuracy covered 0.974, open-set false accept
0.052.
