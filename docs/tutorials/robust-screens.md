# Robustness-audited silencing screens

A silencing screen (`sweep: silence_each`) predicts how a readout changes when each
candidate cell type is silenced. Those predictions depend on modelling choices: the
synaptic weight w_syn, the predicted neurotransmitter signs, the edge threshold and the
dataset. A `robustness:` block reruns the screen with each choice changed one at a time,
compares every effect with silencing random neurons, and writes a hit list a wet lab can
plan from (ADR-0011). Everything it reports is a model prediction from wiring, not a
measurement.

## Prerequisites

```bash
flyconn data pull malecns@1.0 --level nt-probs     # NT probabilities for the sign axis (2.65 GB)
flyconn data pull flywire@783 --level weights      # replicate dataset
flyconn data pull flylight_lines@meissner2025      # split-GAL4 lines for the handoff
```

A missing replicate or line table does not stop the run: the replicate axis is reported
as `not run` and the lines as "not looked up", each with the pull command. A MaleCNS
store without NT probabilities makes the sampled-sign variants refuse to run rather than
fall back to argmax signs.

## Writing a `robustness:` block

Add the block to any `silence_each` spec. The example
`examples/specs/w2_robust_pathway_mn9.yaml` is the pathway screen of GOLDEN_RESULTS 6h
(LB3* GRNs at 100 Hz, relays on LB3* -> MN9 routes) plus:

```yaml
robustness:
  primary_readout: MN9          # required; must name one of the spec's readouts
  readout_proxy: "MN9 motor neurons drive proboscis extension (Shiu et al. 2024, Nature 634:210)"
  w_syn: auto                   # auto | off | [list of mV]
  nt_samples: 8                 # sampled-sign variants; 0 = off
  min_weights: [3, 5]           # network.min_weight variants; [] = off
  replicate: flywire@783        # null = off
  random_null: {n_sets: 40}     # random sets per size bin and super_class; 0 = off
  extra_candidates:             # always simulated, labelled `control`, never hits
    - {cell_type: "LC4"}
```

- `w_syn: auto` uses the dataset's calibration CI ends and sensitivity range
  (`flyconn.sim.CALIBRATIONS`). A dataset without a calibration entry needs an explicit
  list.
- `readout_proxy` is free text you supply, with a citation. flyconn never infers
  behaviour from a readout; the text is copied into the suggested test.
- `extra_candidates` take the same selection grammar as `select`. Use them for a negative
  control: a type with no route from the stimulus to the primary readout. The example
  uses LC4; the path query that confirms it has no LB3* -> MN9 route is in the spec's
  header comment. Controls join the reference BH family, so q values differ slightly
  from the same screen without them; compare effects and CIs instead.
- Every variant silences the reference run's candidates. Whether a variant would have
  selected the same candidates is reported as selection stability.

## Running it

```bash
flyconn run examples/specs/w2_robust_pathway_mn9.yaml --out runs/robust_mn9
```

`flyconn run` detects the block; there is no separate command. Variants run one after
another, each into `runs/robust_mn9/variants/<label>/`. An interrupted run resumes when
started again with the same `--out`: finished variants are skipped and a half-written one
is moved to `variants/.incomplete/` and rerun. Changing the spec and reusing the
directory is refused. The MaleCNS example takes tens of minutes on a laptop CPU; see
GOLDEN_RESULTS 6j for the measured time and memory.

Outputs:

| File | Contents |
|---|---|
| `report.html` | banners, ranked hit table, one evidence card per hit, selection stability, negative results |
| `handoff.csv` | one row per reference hit, for planning experiments |
| `hits.parquet` | every candidate x variant on the primary readout, with its label |
| `null.parquet` | the random-silencing effects (key, seed, set size, difference) |
| `selection_stability.parquet` | Jaccard overlap of each variant's natural selection with the reference |
| `provenance.json` | spec hash, dataset checksums, every variant's w_syn / sign seed / threshold, null seeds and bins, replicate mapping, line-table checksum, package versions, git SHA |

## Reading `handoff.csv`

One row per **reference hit**: a candidate whose primary-readout effect has q < 0.05 in
the reference run. Columns:

- `label`, `direction`, `effect_hz`, `ci_low`, `ci_high`, `q_value`: the reference
  effect (difference of mean readout rates, silenced minus stimulated; bootstrap 95 % CI;
  BH q over the reference sweep).
- `tier`, `failed_axes`, `untested_axes`, `thinly_tested`: the robustness verdict (below).
- `replicate`: holds / weakens / flips / not testable / not run on the replicate dataset.
  Informational only.
- `path_stability`: mean sign-stability x threshold-presence of the stimulus -> readout
  paths through the candidate. Informational only.
- `best_line`, `line_quality`, `line_specific`, `line_match`, `lines`: split-GAL4 lines
  from Meissner et al. 2025 annotated with the candidate type, best quality first.
  `line_match` is `exact`, `glob` or `none`; annotations are free text, so check them.
  No match reads "no annotated split-GAL4 in Meissner 2025"; lines are never invented.
- `off_target_bodies`: NeuronBridge candidate counts, only when already cached.
- `suggested_test`: a templated sentence ("silence X (e.g. Kir2.1, GtACR1) under the
  stimulus; the model predicts the readout goes down by N Hz"). The effectors are
  examples, not recommendations.

Candidates that were not reference hits are listed in the report's negative-results
section, not in `handoff.csv`.

## What the tiers mean

Each variant gets a label against the reference sign: **holds** (same sign, CI excludes
0), **weakens** (CI includes 0) or **flips** (opposite sign, CI excludes 0). Axis
verdicts:

| Axis | Robust when |
|---|---|
| w_syn | holds at every w_syn value |
| NT signs | holds in at least 6 of 8 sign samples (any flip is listed) |
| threshold | holds at every `min_weights` value |
| random null | p < 0.05: silencing the candidate moves the readout further, in its direction, than silencing random sets of the same size bin and super_class |
| replicate | reported, never graded |

- **Strong:** robust on every graded axis that ran.
- **Conditional:** fails exactly one graded axis; the card names it.
- **Fragile:** fails two or more.

Axes that were off or not run are untested, not failed. No correction is applied across
variants; they test the stability of one effect, not new hypotheses.

## Caveats (on every evidence card)

- Model predictions from wiring plus Shiu LIF dynamics; not measurements.
- MaleCNS w_syn calibrated by protocol, not validated (ADR-0009). Other datasets without
  a calibration entry run at Shiu's published w_syn, uncalibrated.
- NT signs are predicted; sampled-sign variants assume the published probabilities are
  calibrated.
- The replicate runs at an uncalibrated w_syn on a different sex; disagreement may be
  biology or calibration.
- Line annotations are curated text and may not match EM type names one to one.
- "Strong" means robust within the tested axes, not true.

Two more limits of the random null (not yet printed on the cards):

- The random null can be degenerate. In the MaleCNS golden run nearly all random sets
  changed the readout by exactly 0 Hz (GOLDEN_RESULTS 6j), so p floors at 1 / (n_sets + 1)
  for any nonzero effect and "null p < 0.05" is then weak evidence. Check
  `null.parquet` before relying on that axis.
- A readout that is a whole super_class (e.g. `descending`) removes every neuron of that
  super_class from the random pool, so candidates of the same super_class get no null p
  and the axis is reported untested for them (`provenance.json`,
  `robustness.random_null.skipped`).
