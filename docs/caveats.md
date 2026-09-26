# Scientific caveats

Grows with each milestone. Seeds:

- A connectome is wiring; it carries no measured dynamics.
- Neurotransmitter identities are classifier predictions (Eckstein et al. 2024 lineage; MaleCNS body-level file gives argmax + confidence only).
- Weights are synapse counts under a detection and confidence threshold that differs per dataset (see `DATA_SOURCES.md`).
- Simulation outputs are model predictions; the Shiu et al. constants were fitted to FlyWire, so MaleCNS runs are uncalibrated until stated otherwise.
- Bit-exact determinism holds on CPU float64 only; MPS/CUDA results are seeded but summation-order dependent.

## Path and cell-type caveats (M2)

- Labellar gustatory receptor neurons: FlyWire v783 types sugar and water GRNs together as `LB3` (bitter `LB1a/b`, Ir94e `LB2`); MaleCNS splits `LB3a–d`. Selecting "sugar GRNs" by type therefore includes water GRNs; use explicit ID lists (e.g. Shiu et al. 2024) where sugar specificity matters.
- Path "strength" is a product of per-hop input fractions; totals include synapses from unannotated fragments where the dataset provides them (MaleCNS), so fractions are smaller than in neuron-only graphs and are not comparable across datasets without care.
- The Shiu et al. export carries only a per-neuron sign, not a transmitter; flyconn stores `acetylcholine`/`gaba` placeholders so that sign policies reproduce the model, and labels the source `shiu2024_excitatory_flag`.

## Uncertainty caveats (M3)

- **Confidence-only NT model.** Where a dataset gives only an argmax label and a confidence (MaleCNS body-level file without the opt-in tbar aggregation), flyconn puts the confidence on the label and spreads the remainder uniformly over the other classes. This is a modelling convenience, not a calibrated posterior; results are labelled with the model used per neuron (`nt_probabilities` returns it).
- **Per-body mean of per-synapse probabilities** (MaleCNS `--level nt-probs`, FlyWire Codex `*_avg`) ignores within-neuron heterogeneity and synapse-level confidence; Dale's-law single sign per neuron is assumed throughout.
- **Degree-preserving rewiring** keeps each neuron's out-degree, its outgoing weight multiset and every neuron's in-degree, but not in-weight or spatial structure; it is a configuration-model null, not a biological alternative wiring.
- **Version drift by id.** `diff_versions` matches neurons by root/body id only. FlyWire root ids change with proofreading (only 106,785 of 127,978 v630 ids persist in v783), so absent ids are reported as unmatched rather than silently treated as removed neurons.
- **Empirical p-values** use the (k+1)/(n+1) correction and are two-sided around the null mean; with 100 nulls the smallest reportable p is about 0.01.
