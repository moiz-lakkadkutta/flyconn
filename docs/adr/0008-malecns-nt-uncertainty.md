# ADR-0008: MaleCNS neurotransmitter uncertainty comes from the per-presynapse file, aggregated by us, with a confidence-only fallback

Status: Proposed (2026-09-26)

## Context

M3 samples NT identities from per-neuron probability vectors. Verified MaleCNS
v1.0 file contents (`docs/research/data_malecns.md` §4.2): the 42 MB body-level
file carries only `predicted_nt` (argmax), `predicted_nt_confidence`, a
type-level argmax/confidence, `ground_truth` and `consensus_nt`; **no per-
transmitter probabilities**. Full 7-class probabilities exist only in
`tbar-neurotransmitters-male-cns-v1.0.feather` (2.65 GB, 45,656,140 rows, one per
presynapse, columns `nt_{acetylcholine,dopamine,gaba,glutamate,histamine,
octopamine,serotonin}_prob`, rows sum to 1). FlyWire's Codex `neurons.csv`
carries per-neuron mean probabilities directly, so the two datasets are
asymmetric.

## Decision

- Default MaleCNS install pulls the small files only. NT sampling then uses a
  **confidence-only model**: P(argmax) = `predicted_nt_confidence`, remaining
  mass spread over the other six classes uniformly, and `unclear` treated as
  missing. Outputs are labelled "confidence-only NT model".
- `flyconn data pull malecns@1.0 --level nt-probs` (opt-in) streams the 2.65 GB
  tbar file with pyarrow record batches (never fully in memory), aggregates
  per-body mean probabilities (and per-body presynapse counts), and writes a
  ~170 k-row Parquet `nt_probs` table with the seven `nt_p_*` columns. The
  harmonized schema is then identical to FlyWire's. Peak memory target < 2 GB.
- Provenance records which model was used; stability scores computed under
  the confidence-only model are flagged as approximate.

## Alternatives considered

- Require the tbar download: rejected; violates "small files by default".
- Treat MaleCNS signs as certain: rejected; uncertainty is the differentiator.

## Consequences

- Aggregation choice (mean of probabilities vs. fraction of argmax votes) is a
  parameter; default = mean probability (matches how Codex's FlyWire per-neuron
  columns are described). Documented on the caveats page.

## Evidence

`docs/research/data_malecns.md` §4.2, §4.6.
