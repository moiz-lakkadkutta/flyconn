# Changelog

All notable changes. Format: Keep a Changelog; versions follow SemVer once 0.1.0 ships.

## [Unreleased]

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
