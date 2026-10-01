# Changelog

All notable changes. Format: Keep a Changelog; versions follow SemVer once 0.1.0 ships.

## [Unreleased]

### M9 simulator speed-up
- Engine processes ticks in chunks of the 18-tick synaptic delay (one spike search and delivery per chunk), precomputes kicks per chunk from the unchanged RNG stream, and optionally fuses the per-tick update with `torch.compile` (`simulate(..., compile=None|True|False)`; auto above 5e8 neuron-updates; falls back to eager). Results identical to the previous engine spike for spike (new equivalence tests against the old loop; Brian2 parity and Shiu golden unchanged). Benchmark: 6-6.4x faster on CPU, 8-11x on MPS.

### M9 experiment sweeps
- `sweep:` section in experiment YAML: `silence_each` (rank candidate groups by activity in the shared stimulated run, silence the N most active one at a time) and `rate_hz` (readout vs stimulation rate). Shared conditions and controls run once; one `sweep.parquet` with effects, CIs, BH q across the whole sweep; one HTML report with ranked table and bar chart or rate curves. `flyconn run` dispatches sweeps automatically. Example `examples/specs/w2_sweep_silence_gng.yaml`.

### M9 hemibrain + MANC
- `flyconn data pull hemibrain@1.2.1` (21,739 traced neurons, 3,550,403 edges; side and hemilineage from Schlegel 2024 Supplemental file 5; transmitter file labelled UNVERIFIED provenance) and `manc@1.2.1` (23,650 neurons from the sjcabs compiled meta, 5,303,770 neuron edges; `manc_121_cell_type` set so the FlyWire-or-MANC vocabulary works). Checksums pinned; discrepancies vs neuPrint documented in GOLDEN_RESULTS and caveats.

## [0.2.0] - 2026-10-01

### M8 BANC
- `flyconn data pull banc@888 --level weights`: BANC v888 converter (meta feather + edgelist v2, both SHA-256 pinned). Neuron universe proofread or roughly proofread excluding glia, trachea and non-neurons (155,858); 156,311 autapses in the released edge list are dropped and counted; per-neuron totals from the release's `post_count`/`pre_count`; BANC's automatic hemibrain matches (`auto:` prefix) flagged in `hemibrain_121_match_auto`; MaleCNS matches kept as `malecns_09_cell_type` (they refer to v0.9).
- `compare_type(..., partner_vocabulary="fafb_or_manc")` and `partner_labels`: shared FlyWire-or-MANC partner names so MaleCNS and BANC comparisons include nerve-cord partners.

## [0.1.0] - 2026-09-29

First public release: milestones M0-M7 of docs/PLAN.md.

### M7 access (exploratory)
- Registry entry `flylight_lines@meissner2025` (eLife 13:RP98405 Figure 1-source data 1, CC BY 4.0, SHA-256 pinned) and converter to `lines.parquet` (token lists for cell types and EM body ids, raw text kept).
- `LineCatalog`: `lines_for_type` (glob, quality filter, specificity flag), `types_for_line`, `lines_for_body`, citation.
- NeuronBridge (CC BY 4.0): `aggregate_cds_matches` over the published JSON schema; `fetch_line_off_targets` (public S3, cached) returns best-scored candidate EM bodies per dataset for a line. Candidates, not expression calls.
- Golden W5: 4,433 lines / 2,667 adult with types / 320 parsed EM-id rows; DNp01 -> SS02299, SS00727; network test on SS02299.

### M6 compare
- `match_types`: cross-dataset type mapping from the published cross-reference columns (MaleCNS `fafb_783_cell_type` -> FlyWire `cell_type`, comma lists split); 6,836 MaleCNS types map to FlyWire v783.
- `type_profile`: partner-type fraction vectors per type and side; `compare_type`: male-vs-female comparison whose statistic is cross-dataset cosine dissimilarity minus within-brain left/right dissimilarity, with a permutation null (dataset labels shuffled within side), partner-level differences, unmatched-partner reporting and a graded verdict referenced to Schlegel et al. 2024's between-brain effect size.
- Golden W3 on MaleCNS v1.0 vs FlyWire v783 (PFL3, DNp01, EPG, PAM08).

### M5 experiments + report
- YAML experiment spec (`flyconn.experiments.spec`): dataset, network options, background stimulation, perturbation (silence / extra stimulation), readout groups by ids or attribute globs, trials, seed, device, controls (degree-preserving rewiring and sign shuffle, on by default), report options; spec SHA-256 in provenance.
- Runner (`run_experiment`): baseline / stimulated / perturbed conditions plus control connectomes with identical seeds; long-form `counts.parquet`, `readouts.parquet` (difference of means, bootstrap 95 % CI, Mann-Whitney p, Benjamini-Hochberg q, Cohen's d), `conditions.parquet`, `provenance.json`.
- Self-contained HTML report (`flyconn.report.html`): model-prediction and calibration banners, design, effects vs controls with figures, top neurons, provenance, citations.
- `flyconn run exp.yaml --out DIR`.
- Example spec `examples/specs/w2_malecns_lb3_silence_gng232.yaml` and golden W2 run.

### M4 sim
- `ShiuParams` (all constants from Shiu et al. 2024), `LIFNetwork.from_matrix`, batched event-driven `simulate` (CPU float64 reference; float32 CPU/MPS/CUDA), seeded per-trial Poisson drive, explicit input events, silencing, refractory rules of the reference code, spike events and rate tables, provenance labelled "model prediction".
- Validation ladder: rung 1 Brian2 spike-for-spike parity (`pytest -m brian`), rung 2 Shiu golden results (`tests/golden/test_shiu_golden.py`), rung 3 `benchmarks/bench_sim.py`.
- `calibrate_w_syn`: Shiu 80 %-of-maximum re-calibration protocol; MaleCNS run recorded as unresolved (uncalibrated).

### M3 uncertainty
- NT uncertainty models: real per-neuron probabilities where available, confidence-only model otherwise (ADR-0008); seeded sign sampling.
- MaleCNS `--level nt-probs`: streams the 2.65 GB per-presynapse file into per-body mean probabilities and joins them into `neurons` (`nt_source = malecns_v1.0_tbar_mean`).
- Null models: degree-preserving rewiring (out-degree, out-weights and in-degree preserved), sign shuffle; `null_distribution` with empirical z and two-sided p.
- `path_stability`: paths across synapse thresholds with `threshold_presence`, `modal_sign`, `sign_stability`.
- `diff_versions` (W4): neurons added/removed, attribute changes, edge changes for neurons of interest, unmatched-id caveat.
- Golden: W1 stability on FlyWire (Shiu v630) and MaleCNS; W4 v630→v783 drift for the sugar circuit; examples `examples/w1_pathways.py`, `examples/w4_version_drift.py`.

### M2 graph
- `ConnectivityMatrix` (pre-in-rows CSR) from a `Store` or frames; `SignPolicy` argmax / probabilistic / sampled / unsigned with per-transmitter overrides (ADR-0007); input-normalised weights using fragment-inclusive totals.
- `aggregate` / `aggregate_edges` by any label column; `find_paths` (1..k hops, reachability-pruned, per-hop fraction and strength filters, signs and strengths per path); native `effective_connectivity` and `effective_by_hops`.
- Optional `flyconn[interpret]` adapter to `connectome_interpreter` (pinned git SHA) with a parity test against the native implementation.
- Shiu et al. 2024 v630 model-input converter (`shiu@630`), unthresholded edges with pre-baked signs.
- W1 golden run: sugar GRN -> MN9 paths on FlyWire (Shiu export) and MaleCNS in 0.1 s each.

### M1 data
- Dataset registry (YAML, pinned URLs/sizes/SHA-256): malecns@1.0, flywire@630, flywire@783, shiu@630, hemibrain@1.2.1, manc@1.2.1, banc@888.
- Resumable, checksummed downloads (`flyconn.data.download`).
- Harmonized schema (SJCABS/BANC vocabulary) with NT, super_class and side normalisers.
- Converters: MaleCNS v1.0 (streamed 152M-edge weights, per-neuron totals incl. fragments), FlyWire Codex 630/783 (pair-aggregated edges + per-neuropil table).
- DuckDB `Store` with `neurons()`, `edges()`, `query()`, `citations()`.
- `flyconn data list|info|pull`.
- Golden Tier 0 tests reproduce the published counts (`pytest -m golden`).

### M0 scaffold
- Project skeleton: `uv`, `ruff`, `pyright` (strict), `pytest` + `hypothesis`, CI, mkdocs.
- `flyconn.testing.synthetic_connectome` deterministic fixture generator.
- `flyconn --version` CLI entry point.
