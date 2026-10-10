# Scientific caveats

Grows with each milestone. Seeds:

- A connectome is wiring; it carries no measured dynamics.
- Neurotransmitter identities are classifier predictions (Eckstein et al. 2024 lineage; MaleCNS body-level file gives argmax + confidence only).
- Weights are synapse counts under a detection and confidence threshold that differs per dataset (see `DATA_SOURCES.md`).
- Simulation outputs are model predictions; the Shiu et al. constants were fitted to FlyWire, so other datasets are uncalibrated unless re-calibrated by protocol; MaleCNS v1.0 is (0.188 mV, ADR-0009), which is still not a validation against recordings.
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

## Simulation caveats (M4)

- The LIF model has one free parameter (w_syn = 0.275 mV) that Shiu et al. chose so that 100 Hz sugar-GRN drive gives about 80 % of MN9's maximal rate in FlyWire v630. Every other constant was fitted to FlyWire too; datasets without a protocol calibration (BANC, hemibrain, MANC) are **uncalibrated**.
- **Re-calibration (ADR-0009).** Shiu's literal rule (100 Hz drive gives 80 % of the 200 Hz readout) is never met on v630 itself; flyconn transfers v630's onset ratio (MN9 at 50 / 200 Hz drive) instead, which recovers 0.273 mV on v630. For MaleCNS with the sugar-like LB3c+LB3d GRNs this gives 0.188 mV (95 % CI 0.185-0.192). The interval covers trial noise only, not the sugar/water assignment of LB3 subtypes (by connectivity, not by receptor), the sign/NT model, or the choice of readout. Since 2026-10-10 MaleCNS v1.0 networks default to 0.188 mV and are labelled "CALIBRATED BY PROTOCOL, not validated against data"; the sensitivity range is 0.185-0.209 mV (choice of sugar GRN set). Pass `ShiuParams()` to use 0.275 mV (labelled UNCALIBRATED). MaleCNS numbers in the docs dated before 2026-10-10 used 0.275 mV. Calibration fixes one scale constant; it does not make MaleCNS spike counts comparable to recordings.
- Baseline firing is 0 Hz; there is no intrinsic activity, noise, neuromodulation, gap junctions, or synaptic plasticity. Outputs are model predictions of wiring-constrained excitability, not predictions of behaviour.
- Signs come from predicted transmitters under a single-sign-per-neuron rule; histamine is inhibitory by default in flyconn but Shiu's FlyWire predictions had no histamine class.
- Exactness: bit-identical spikes are guaranteed only for the CPU float64 path. float32 on CPU/MPS/CUDA matched the reference in every test so far, but summation order can move borderline spikes by a tick; compare distributions, not single runs.
- Tick kernel: large runs (and any run with `compile=True`) fuse the per-tick update with `torch.compile`; provenance records `tick_kernel` (`torch.compile` or `eager`). On CPU, inductor builds without FP contraction or unsafe math, and the compiled kernel reproduces the eager kernel and the v0.2.0 per-tick loop spike for spike (unit tests, plus the full v630 network in float32 and float64). MPS matched on the same full-network check but is not guaranteed; on CUDA, Triton may fuse multiply-adds, so compiled float32 can differ from eager at rounding level (the same class of difference as summation order). If compilation fails, the engine falls back to eager with a warning; `compile=False` forces eager.

## Experiment and report caveats (M5)

- Readout statistics compare per-trial mean rates of a readout group between conditions. Trials share the connectome and differ only in Poisson input seeds, so "n" is the number of trials, not of animals; effect sizes describe the model, not biology.
- Controls are simulated with the *same* seeds on degree-preserving rewired and sign-shuffled connectomes; a stimulated-vs-control difference shows that the specific wiring (or the specific signs) matters for the readout under this model, nothing more.
- Benjamini-Hochberg correction is applied across all readout x comparison rows of one experiment; with few rows the correction is mild.
- Sign shuffling as a null: with roughly two thirds of neurons excitatory, permuting signs across neurons breaks excitation/inhibition balance and typically produces runaway activity (seen in the W2 run: 59M spikes vs 0.6M). Treat it as a test of whether inhibitory identities matter *where they are*, not as a matched-activity null; compare readouts under the rewired control first.
- Perturbations are all-or-none (silencing zeroes outgoing synapses; stimulation is Poisson kicks); they do not model partial knockdown, developmental compensation or neuromodulation.

## Sweep caveats

- A sweep (`sweep: {silence_each: ...}` or `sweep: {rate_hz: [...]}`) runs the shared conditions (baseline, stimulated, controls) once and one simulation per variant. Every variant reuses the spec's seeds, so variant and reference trials share Poisson input; the Mann-Whitney test treats them as unpaired, which is conservative.
- n is the number of trials per variant. Benjamini-Hochberg runs across *all* variant x readout rows of the sweep (`sweep.parquet`); the shared stimulated-vs-baseline/control comparisons in `readouts.parquet` form a separate family.
- `silence_each` simulates only the `max_items` candidate groups with the highest summed firing rate in the stimulated condition (in the spirit of Shiu et al. 2024's screen of the most active neurons). Groups that never fire are listed as skipped: silencing zeroes outgoing synapses, so they cannot change that run. A type that is silent under this stimulation may still matter under another one. Ranking by summed rate favours types with many neurons.
- A screen ranks effects within one model and one stimulation; it is a list of candidates for experiments, not evidence that a type is necessary in the fly. MaleCNS magnitudes are only roughly comparable to FlyWire runs: w_syn is calibrated by protocol (onset ratio), not the other constants or absolute rates.
- To bound file size, `counts.parquet` stores all non-zero counts for the shared runs but only the readout neurons' counts for variant runs; per-variant totals are in `conditions.parquet` and `variants.parquet`.
- In a `rate_hz` sweep every `stimulate` entry is set to the same rate; the reference is the unstimulated baseline (or the stimulated condition when the baseline is disabled).

## Cross-dataset comparison caveats (M6)

- The null model is left/right variability *within* a brain. Schlegel et al. 2024 report that connectivity between two different female brains (FlyWire vs hemibrain) is less similar than left vs right within one brain (cosine effect size 0.045 ± 0.096). A male-vs-female difference must therefore be judged against that between-individual range, not only against the L/R permutation p-value; `compare_type` grades its verdict accordingly (no evidence / detectable but within the reported between-brain range / beyond it).
- Partner types without a cross-dataset match (e.g. VNC neurons, which FlyWire's brain volume cannot contain, or unmatched types) are excluded from the profiles and their fraction is reported; a large unmatched fraction means the comparison covers only part of the type's output or input.
- Datasets differ in synapse detection, confidence thresholds and proofreading completeness; use the same `min_weight` on both sides and expect residual technical differences.
- Sex-specific types by definition have no counterpart; `match_types` cannot represent them, and dimorphism annotations should be consulted separately (MaleCNS `dimorphism`, FlyWire `dimorphism`).

## Genetic access caveats (M7, exploratory)

- Line-to-type annotations (Meissner et al. 2025) are free text across nomenclatures (hemibrain, MANC, optic lobe, FlyWire) and carry a 1-4 quality grade; expression confidence lives in each line's source paper. EM body ids in the table have no dataset tag.
- NeuronBridge hits are colour-depth similarity candidates ranked by score; a hit is not an expression call, scores are not comparable across libraries, and absence of a hit is weak evidence. Off-target lists are long by construction; rank and inspect.
- Most FlyLight imagery is female and adult with specific reporters; expression can differ in males or with other effectors.

## BANC caveats (M8)

- The released `banc_888_edgelist_simple_v2.feather` (checksum-pinned, updated 2026-06-02) has 11,752,828 rows, not the 11,510,975 its documentation states, and contains 156,311 self-connections although the documentation says autapses were removed. flyconn drops them and reports the count.
- BANC's `malecns_cell_type` column refers to MaleCNS v0.9; flyconn stores it as `malecns_09_cell_type` and does not treat it as a v1.0 mapping.
- Comparing MaleCNS with BANC (both brain plus nerve cord) is the more complete male-vs-female test than MaleCNS vs FlyWire (brain only). For DNp01 the unmatched output fraction drops from 58 % (vs FlyWire) to 0 % (vs BANC). Two female datasets can also disagree: EPG was graded "beyond the between-brain range" against FlyWire (statistic 0.189) but only "within range" against BANC (0.088); when two female references disagree, the difference is not attributable to sex.
- Both antennal nerves were damaged in the BANC sample (Bates et al. 2026); antennal inputs are under-represented.

## Hemibrain and MANC caveats

- Hemibrain weights are neuPrint `weight`, which counts every postsynaptic density because the dataset's `postHighAccuracyThreshold` is 0; the high-confidence `weightHP` is not in the export. The store is the v1.2 export; neuPrint v1.2.1 differs by 342 edges / 4,077 synapses.
- The hemibrain volume is mostly the right hemisphere. Left-side neurons and left-side arbors are truncated, so the left/right null used by `compare_type` can be meaningless for a hemibrain type (PFL3: within-hemibrain left/right similarity 0.31 against 0.99 in MaleCNS). Check `within_similarity_b` before reading a verdict.
- Hemibrain side, ItoLee hemilineage and short `cell_class` come from Schlegel et al. 2024 Supplemental file 5 (411 traced bodies have no row); `super_class`, `flow` and `region` come from the sjcabs `hemibrain_121_meta`. No FlyWire type cross-reference is filled because none of these files publishes one, so hemibrain partners keep hemibrain names under every partner vocabulary.
- Hemibrain transmitter probabilities come from `hemibrain-v1.2-body-mean-neurotransmitters.feather`, uploaded without a README; its provenance is UNVERIFIED (the class set matches the Eckstein et al. 2024 classifier). All 1,927 typed Kenyon cells (`KC*`) are predicted dopaminergic, although Kenyon cells are cholinergic; Eckstein et al. 2024 report the same misprediction. Prefer per-type known transmitters for KCs.
- MANC has no flat v1.2.x export. flyconn uses the sjcabs compiled meta and edge list (built by flyconnectome/bancpipeline); their row count and synapse sum differ slightly from neuPrint, and the confidence threshold used for the build is not documented. Per-neuron input totals count traced partners only.
- MANC transmitter predictions are 3-class (acetylcholine, GABA, glutamate); a MANC neuron is never predicted monoaminergic. `nt_conf` is the Codex confidence and is filled only where Codex's label agrees with the sjcabs label (23,410 of 23,650).
- MANC cell types are neuPrint `type` names (e.g. `Ti flexor MN`); Codex's primary types differ for 420 neurons. `manc_121_cell_type` equals `cell_type`, so MANC partners share the FlyWire-or-MANC vocabulary with MaleCNS and BANC.


## Pathway-ranked screens

- Path strength is a product of per-hop input fractions under the chosen thresholds; it ranks *candidates for silencing*, it is not a prediction of the silencing effect. The simulation decides the effect, and its sign can disagree with the path sign when routes interact.
- Thresholds matter: with `min_edge_fraction: 0.01` only six types lie on LB3 -> MN9 routes in MaleCNS; looser thresholds admit many weak relays and stricter ones may admit none.
