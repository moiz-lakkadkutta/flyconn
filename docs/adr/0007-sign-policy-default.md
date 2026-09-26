# ADR-0007: Default sign policy = Shiu/Eckstein rule (GABA, Glu inhibitory; ACh, DA, OA, 5-HT excitatory; unknown excluded), configurable

Status: Proposed (2026-09-26)

## Context

Signs are not measured; they are inferred from neurotransmitter predictions
(Eckstein et al. 2024 for FlyWire/hemibrain; the MaleCNS release's own
predictions). The literature default used by the whole-brain LIF model (Shiu et
al. 2024, Methods) and by Sapkal et al. 2024 is: GABAergic and glutamatergic
neurons inhibitory, all others excitatory, one sign per neuron (Dale's law
approximation), assigned by majority vote over presynapses with cleft score ≥ 50.
Glutamate is in reality mixed in the fly (GluCl inhibitory, but excitatory at
some synapses); Shiu et al. report that treating glutamate as excitatory raises
their screen's false-positive rate from 1 % to 16 %. Histamine (photoreceptors)
is inhibitory in the fly; MaleCNS predicts histamine for 5,910 consensus bodies.

## Decision

- `SignPolicy` enum with:
  - `argmax` (default): per-neuron consensus NT → sign via the table
    {GABA: −1, Glu: −1, His: −1, ACh: +1, DA: +1, OA: +1, 5-HT: +1};
    `unclear`/missing → edge excluded (weight 0) and counted in provenance.
  - `probabilistic`: per-neuron expected sign = Σ p(nt)·sign(nt), a float in
    [−1, 1], used for weighting.
  - `sampled`: one draw per replicate from p(nt) (used by `flyconn.uncertainty`).
  - Per-transmitter overrides (e.g. `glutamate: +1`) for sensitivity analyses.
- Histamine defaults to inhibitory. This differs from Shiu et al. (whose FlyWire
  predictions had no histamine class); the Shiu-parity engine configuration sets
  `histamine: +1` explicitly so golden tests match.
- Every output records the policy and the fraction of edges affected by
  `unclear`.

## Alternatives considered

- Unknown → excitatory (Shiu default for "other"): rejected as a default because
  MaleCNS has an explicit `unclear` class (14,365 Traced bodies); dropping is the
  honest choice, and users can override.

## Consequences

- Golden Shiu tests use the Shiu-exact configuration, not the flyconn default.
- The `histamine` default must be documented on the Scientific caveats page.

## Evidence

Shiu24 Methods (quoted in `docs/research/golden_results_raw.md` §A2);
`docs/research/data_malecns.md` §4.2.
