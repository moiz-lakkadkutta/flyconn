# ADR-0011: Robustness-audited silencing screens, one uncertainty axis at a time

Status: Accepted (2026-10-10). Design: `docs/superpowers/specs/2026-10-10-robust-screens-design.md`.

## Context

An M10 silencing screen (`sweep: silence_each`) reports, for each silenced candidate, a
difference of means with a bootstrap CI and a BH q value. Those numbers are conditional on
one set of modelling choices: one w_syn, argmax neurotransmitter signs, one edge threshold
and one dataset. GOLDEN_RESULTS 6h checked one of these choices by hand, after MaleCNS
moved from w_syn 0.275 mV to the protocol value 0.188 mV (ADR-0009):

- GNG232 -> MN9 survived (-11.7 Hz at 0.275 mV, -14.5 Hz at 0.188 mV, both q < 0.01);
- DNge080 did not (-9.2 Hz, q 0.008 -> -4.3 Hz, q 0.26);
- the inhibitory-route relays lost their effects, and one (GNG215) changed sign;
- the 10 most active GNG types selected by an activity screen changed.

A wet-lab collaborator reading a screen cannot see any of this. The check was manual,
covered only w_syn, and had no reference for how large an effect silencing an arbitrary
set of neurons of the same size produces.

## Decision

`flyconn run SPEC --out DIR` runs a **robustness-audited screen** when a `silence_each`
spec has a `robustness:` block (`flyconn.experiments.robustness.run_robust_screen`).

- **Option A: one axis at a time.** The reference run is the spec as written. Each
  variant changes exactly one modelling choice and reruns the same candidates:
  - w_syn: the calibration CI ends and sensitivity range (`w_syn: auto`; MaleCNS
    0.185, 0.192, 0.209 mV);
  - NT signs: `nt_samples` sign draws from the published probabilities
    (`sign_policy: sampled`, sign seeds `seed + 2000 + k`);
  - threshold: each `min_weights` value as `network.min_weight`;
  - replicate: the screen mapped onto another dataset with `match_types` (FlyWire 783).
- **Fixed candidates.** Every variant silences the reference run's candidate table, so
  effects are comparable. Whether a variant *would* have selected the same candidates is
  reported separately as selection stability (Jaccard overlap with the reference list).
- **Random-silencing null.** For each candidate's super_class and size bin (nearest of
  1, 2, 4, 8, ...), `n_sets` random sets are silenced in the reference network (stimulus,
  readout and candidate neurons excluded; seeds `seed + 3000 + i`).
- **Negative control.** `extra_candidates` are always simulated, labelled `control`,
  never reported as hits, and get a random-null p. A type with no stimulus -> readout
  route is the intended use.
- **Variant label** of a reference hit (q < 0.05) from the variant's 95 % CI and the
  reference sign: holds (same sign, CI excludes 0), flips (opposite sign, CI excludes 0),
  weakens (CI includes 0, either sign).
- **Axis verdicts.** w_syn and threshold are robust when every variant holds; NT signs
  when at least ceil(0.75 x nt_samples) samples hold (6 of 8); the random null when
  p = (1 + number of null effects at least as extreme in the hit's direction) /
  (n_sets + 1) is below 0.05.
- **Tiers.** Strong: robust on every graded axis that ran. Conditional: fails exactly
  one. Fragile: fails two or more. Axes that are switched off or not run are listed as
  untested and never count as failures; a hit graded on fewer than two axes is marked
  "thinly tested".
- **The replicate is reported, never graded.** It runs at Shiu's uncalibrated 0.275 mV
  on a female brain, so disagreement may be biology or calibration.
- **No multiple-testing correction across variants.** Variants measure the stability of
  an effect that was already tested once (BH over the reference sweep); they are not new
  hypotheses.
- **Outputs:** `report.html` (one evidence card per hit with caveats and a suggested
  test), `handoff.csv` (one row per hit, with the best Meissner 2025 split-GAL4 line
  and its match method), `hits.parquet`, `null.parquet`, `selection_stability.parquet`,
  `provenance.json` (every variant's w_syn, sign seed and threshold, null seeds and bins,
  replicate mapping, line-table checksum). Interrupted runs resume per variant.

## Alternatives considered

- **B: joint Monte Carlo over all axes.** Draws w_syn, signs and threshold together and
  reports one distribution per hit. Rejected: it loses per-axis attribution, so a card
  cannot say "holds unless w_syn >= 0.209 mV", which is what a collaborator needs to
  judge whether the failing condition is plausible.
- **C: graph prefilter.** Drop candidates whose path strength changes under sign or
  threshold variants before simulating. Rejected: it removes hits that arise from network
  dynamics rather than from direct routes, and it would not test the simulated effect.
- **Grading the replicate.** Rejected while the replicate runs at an uncalibrated w_syn
  and a different sex.
- **Correcting across variants (BH over every variant x candidate).** Rejected: it
  treats the stability check as new tests and penalises hits for being checked more.

## Consequences

- A screen costs one reference sweep plus one sweep per variant (14 for the default block
  on MaleCNS) plus the random null; variants run without shared controls to save 4 runs
  each. Measured on the MaleCNS golden run (15 sweeps plus 120 random-null runs, 7 candidates): 1,088 s and 12.6 GB peak RSS on an M4 Pro CPU (golden test; GOLDEN_RESULTS 6j), under the 30 min extrapolated in the design.
- "Strong" means robust within the tested axes, not true. The labels use the unpaired
  bootstrap CI although conditions share Poisson inputs, so they are conservative.
- The NT axis needs the `nt-probs` level for MaleCNS; without it the axis is not run,
  never silently argmax.
- Line suggestions are lookups in curated text (match method recorded); NeuronBridge
  off-target counts appear only when already cached.

## Evidence

GOLDEN_RESULTS 6h (the hand check) and 6j (the first audited run;
`tests/golden/test_robust_screen.py`, `examples/specs/w2_robust_pathway_mn9.yaml`).
First audited run: one reference hit, GNG232 -> MN9 (-14.5 Hz), Conditional: it holds at
every w_syn and threshold and in the FlyWire replicate, but in only 5 of 8 sign samples.
Selection stability is 1.0 for every variant. The same run showed weak checks:
random 2-neuron central-brain sets never change MN9, so the random null passes any
nonzero effect at its minimum p (1/41) — the card now flags such a degenerate null but
the tier still counts it; and the negative control LC4 is silent under the stimulus, so
it passes trivially (p 0.95), which is not yet fixed. The random pool excludes only the
stimulus and the primary readout, so a secondary readout defined by a whole super_class
no longer empties it.
