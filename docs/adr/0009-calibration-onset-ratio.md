# ADR-0009: Re-calibrate w_syn by transferring v630's onset ratio, with a positive control

Status: Accepted (2026-10-10, owner approved the MaleCNS default). Refines ADR-0004's re-calibration protocol.

## Context

Shiu et al. 2024 chose w_syn = 0.275 mV so that 100 Hz sugar-GRN drive gives "roughly
80 % of maximal MN9 firing". M4 implemented this literally (readout at 100 Hz divided by
readout at 200 Hz, target 0.8) and found MaleCNS "unresolved". The protocol had never been
run on the network it came from. On Shiu's v630 network (21 sugar GRNs, contralateral MN9,
10 x 1 s trials, `tests/golden/test_shiu_calibration.py`):

- MN9 does not saturate: 66 / 95 / 111 Hz at 100 / 200 / 400 Hz drive (0.275 mV).
- The 100 / 200 Hz ratio is 0.69 at 0.275 mV and plateaus at 0.70-0.79 for every
  w_syn >= 0.25 mV. The literal target 0.8 is never reached: the literal rule is
  **unresolved on v630 itself**, and the 100 / 200 Hz ratio cannot identify w_syn above
  ~0.25 mV (transfer of v630's 0.689: ambiguous, crossing 0.247 mV, CI 0.244-0.267,
  bootstrap support 0.81).
- The 50 / 200 Hz ratio (onset ratio) rises steeply through 0.275 mV
  (0.05 / 0.22 / 0.38 at 0.25 / 0.275 / 0.3 mV).

The previous MaleCNS run also drove all 78 LB3 neurons (sugar and water GRNs), while Shiu
drove sugar GRNs only, and used 5 x 0.5 s trials.

## Decision

- `calibrate_w_syn` keeps per-trial readouts; the decision step `select_w_syn` bootstraps
  each grid point's ratio (trials resampled per condition), interpolates where the ratio
  crosses the target, bootstraps the crossing, and returns `calibrated` only for a single
  crossing present in >= 95 % of replicates (`ambiguous` for several or weakly supported
  crossings, `unresolved` when the target is never reached).
- The default transfer rule is the **onset ratio**: reference 50 Hz, maximum 200 Hz,
  target = the ratio v630 gives at 0.275 mV (0.205, seed 0). The rule is accepted only
  because it passes a **positive control**: on independent seeds it recovers
  0.273 mV (95 % CI 0.268-0.279; support 1.00) on v630.
- MaleCNS is calibrated with the **sugar-like LB3 subtypes LB3c + LB3d** (49 neurons):
  by output-partner type profile they resemble FlyWire's sugar GRN set (cosine 0.92 /
  0.86; LB3a resembles the water set, 0.87; LB3b is mixed). Per neuron, 41 of 49 LB3c/d
  neurons are closer to the sugar profile.
- A protocol outcome is labelled "calibrated by protocol, not validated against data".
  It becomes a dataset default only by an entry in `flyconn.sim.CALIBRATIONS`, which
  `LIFNetwork.from_matrix` applies when no parameters are passed. MaleCNS v1.0 has such an
  entry (0.188 mV) since 2026-10-10; passing `ShiuParams()` explicitly restores 0.275 mV and
  the UNCALIBRATED label.

## Alternatives considered

- Literal 80 % of the 200 Hz readout: fails the positive control (above).
- 80 % of a true maximum: MN9 has no plateau up to 400 Hz drive in the model.
- Matching absolute MN9 rates: depends on MN9's own input count, which differs between
  datasets (MaleCNS MN9 at 200 Hz drive is 49 Hz at 0.2 mV vs 95 Hz in v630 at 0.275 mV).
- Whole-curve least squares: more parameters to justify; the onset ratio already
  identifies w_syn and is scale-free.
- Global synapse-count scaling (0.275 mV divided by MaleCNS / FlyWire synapses per matched
  connection): kept as an independent cross-check, not as the rule, because the ratio
  depends on the connection-strength floor (1.67 at all matched type pairs, 1.47 for
  pairs with >= 5 synapses per neuron pair).

## Consequences

- MaleCNS onset-ratio calibration: **w_syn = 0.188 mV (95 % CI 0.185-0.192; support
  0.99)**; the synapse-scaling cross-check predicts 0.165-0.187 mV. With all LB3 as the
  stimulus the outcome is ambiguous, so the stimulus set matters.
- Sensitivity (`tests/golden/test_malecns_calibration_inputs.py`): probabilistic signs from
  the tbar-aggregated NT probabilities give 0.189 mV (0.187-0.192); the per-neuron sugar
  call (49 neurons: 8 LB3b, 17 LB3c, 24 LB3d) instead of the LB3c+d subtypes gives 0.204 mV
  (0.199-0.209). The recorded range is 0.185-0.209 mV; the stimulus-set choice dominates.
- Not covered: the 50 Hz choice of reference rate, the MN9 readout, and whether MN9 in a
  male CNS should behave like MN9 in FlyWire at all.
- MaleCNS defaults to 0.188 mV (2026-10-10). Every MaleCNS simulation number recorded before
  that date used 0.275 mV and is kept in the docs labelled as such.

## Evidence

`benchmarks/shiu_calibration.json`, `benchmarks/malecns_calibration.json`,
`benchmarks/malecns_calibration_inputs.json`;
`docs/GOLDEN_RESULTS.md` §6g.
