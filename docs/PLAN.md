# PLAN — scope, decisions, milestones, risks

Status: Phase 0 output, 2026-09-26; approved 2026-09-26 ("go with recommendations"). Build status as of 2026-09-27 (all on `main`):

| Milestone | Status | Acceptance evidence |
|---|---|---|
| M0 scaffold | done | CI config, gate script, synthetic fixture, CLI |
| M1 data | done | Tier 0 golden counts reproduce for MaleCNS v1.0, FlyWire v630/v783, Shiu v630; MaleCNS `--level nt-probs` |
| M2 graph | done | W1 single run on both datasets; connectome_interpreter parity |
| M3 uncertainty | done | W1 stability + W4 drift golden; null models with statistics |
| M4 sim | done | Brian2 spike-for-spike parity; Shiu golden (MN9 66.5 Hz at 100 Hz); benchmarks; MaleCNS calibration protocol run (unresolved) |
| M5 experiments + report | done | W2 from one YAML with controls and HTML report |
| M6 compare | done | W3 with graded verdicts (6,836 matched types) |
| M7 access | done (exploratory) | W5 on the Meissner table and NeuronBridge candidates |
| M9 sim speed-up | done | 6-11x faster, spike-for-spike identical; Shiu golden 30 trials in ~20 s |
| M9 sweeps | done | silencing screen over 10 GNG types on MaleCNS in 286 s, ranked report |
| M9 hemibrain + MANC | done | 21,739 / 23,650 neurons; counts match exports; compare_type runs vs MaleCNS |
| M11 calibration | done | positive control recovers 0.273 mV on v630; MaleCNS onset-ratio transfer 0.188 mV (range 0.185-0.209), now the MaleCNS default, labelled not validated (ADR-0009) |
| M8 BANC | done | 155,858 neurons, DN/AN counts match the paper; W3 MaleCNS vs BANC with nerve-cord partners |

**What is not done** (deferred, see PROGRESS.md entries): MaleCNS ROI-resolved edges; CUDA validation of the compiled simulator; PDF reports and sweeps over seeds or multiple YAML files; input-direction batch comparisons and morphology-based matching; VFB/FlyBase adapters. MaleCNS runs use a protocol calibration (0.188 mV, ADR-0009) that is not validated against data.
Inputs: `LANDSCAPE.md`, `DATA_SOURCES.md`, `GOLDEN_RESULTS.md`, ADRs 0001–0008.

## 1. Corrections to the brief (things the research showed to be wrong, different, or already done)

| # | Brief said | What we verified | Effect on plan |
|---|---|---|---|
| 1 | MaleCNS ≈125 M synapses | 125,024,863 is the neuPrint Neuron→Neuron connection weight sum (124,025,046 Traced→Traced in the flat file). Total PSDs are 311,833,243, presynapses 45,656,140 | Say "125 M neuron-to-neuron synaptic connections"; ingest counts both |
| 2 | NT prediction probabilities available for sampling | MaleCNS's 42 MB body file has only argmax + confidence; the 7-class probabilities live in the 2.65 GB per-presynapse file. FlyWire's Codex `neurons.csv` has per-neuron probabilities | ADR-0008: confidence-only default + opt-in `--level nt-probs` aggregation |
| 3 | Shiu model = FlyWire v630 | It is a specific unthresholded v630 export: 127,400 neurons, 14,687,178 edges, min 1 synapse, shipped MIT in the repo (with a v783 twin). Codex v630 has 127,978 neurons and a ≥5-synapse pair table | Sim golden input = the Shiu parquet (pinned by SHA-256), separate from the data layer's Codex v630 |
| 4 | Evaluate eonsystems fly-brain as backend | GPL-2.0 benchmark harness: forward Euler, CUDA/CPU only, unseeded, no API, no tests | ADR-0004: own engine; borrow its batching and comparison metrics |
| 5 | Build compare on cocoa; upstream MaleCNS v1.0/BANC | cocoa has zero tests, no CI, GPL, tokens + internal SeaTable; v1.0 reached by string-ordering accident; BANC only on an unmerged branch | ADR-0003: reimplement label-graph matching with tests; offer a small PR upstream |
| 6 | sjcabs harmonized files solve M1 | They are a workshop bundle on MaleCNS v0.9 with no checksums/DuckDB | ADR-0001: adopt their column vocabulary, not their files |
| 7 | FlyWire current public release v783 (verify) | Confirmed: v783 is still the latest public snapshot. **No login is needed for any Codex bulk file** (public GCS bucket); Codex web/CAVE need Google sign-in. Codex `connections.csv` = ≥5-synapse pairs split per neuropil; since July 2025 there are two synapse detections (Buhmann vs Princeton) | Registry pins v630 and v783 from the public bucket with per-file hashes (files are refreshed in place); `synapse_source` is explicit metadata |
| 8 | Apple MPS supported like CUDA | torch 2.14 on MPS: sparse COO matmul and `index_add_` work; CSR, float64 and deterministic `index_add_` do not | Engine is COO/edge-list based; exactness claimed on CPU float64 only |
| 9 | W5 gated on openly licensed data | Data exists: Meissner 2025 line table (CC BY 4.0, 409 KB, 4,433 lines with connectome-style type names) + NeuronBridge precomputed matches on public S3 (CC BY 4.0, v3_10_0) + VFB public Neo4j | W5 stays last, but it is a build, not a skip |
| 10 | neuPrint access via token | Tokens issued before 2026-08 are invalid (auth migration); the library refuses to start without one; MaleCNS bulk files need no token at all | Bulk-first design; neuPrint optional |
| 11 | Package name `flyconn` may collide | Free on PyPI and conda-forge; only two dormant GitHub namesakes | Keep (ADR-0005) |
| 12 | `Cell` paper numbers | Cell full text is Cloudflare-blocked from here; abstract numbers (166,700 neurons, 11,710 types, 138/289/71 dimorphic/male/female-specific) differ from the v1.0 file (11,751 types; different dimorphism tallies) | Tests use the file; paper numbers documented. If you have institutional access, the Methods would settle the NT thresholds and type-count delta |

## 2. Module decisions

| Module | Decision | Reasoning | ADR |
|---|---|---|---|
| M1 `data` | **Keep, build** (greenfield) | Nothing exists; clearest gap. Bulk-first, tokenless, Feather→Parquet+DuckDB, SJCABS vocabulary | 0001 |
| M2 `graph` | **Keep; wrap connectome_interpreter** for multi-hop/effective connectivity; native signed sparse matrices, aggregation, thresholding and 1–3-hop path enumeration | Their kernels are validated and dataset-agnostic; torch stays optional | 0002, 0007 |
| M3 `uncertainty` | **Keep, build** (core differentiator) | Absent everywhere. Sampling loop wraps M2/M4 pure functions; MaleCNS needs the opt-in probability aggregation | 0008 |
| M4 `sim` | **Keep, build own PyTorch engine**; Brian2 as oracle; copy the MLX port's validation design | eonsystems unusable; exact integrator required for parity | 0004 |
| M5 `experiments` + `report` | **Keep, build** | Nobody has it; flybench's pre-registered task list is a good example source | – |
| M6 `compare` | **Modify: build on our schema, not on cocoa**; thin because datasets already publish cross-type columns | cocoa untested/GPL; the L/R technical-noise null is small to implement | 0003 |
| M7 `access` | **Keep as exploratory, last** | Openly licensed data confirmed; scientific caveats large (candidate matches, free-text types) | – |
| CLI | typer, after Python API per module | – | – |
| REST/UI, LLM | **Cut** for now | brief; no workflow needs them | – |
| Morphology / template transforms | **Reuse** navis + flybrains as optional extra only | GPL, heavy | 0006 |
| Licence | **Apache-2.0**, GPL deps optional | needs owner confirmation | 0006 |

## 3. Harmonized schema (seed; finalised in M1 with DATA_SOURCES.md alignment table)

`neurons` (one row per neuron per dataset version): `dataset, version, neuron_id (int64), status, flow, super_class, cell_class, cell_sub_class, cell_type, hemilineage, side, soma_neuromere, nt_pred, nt_conf, nt_p_ach, nt_p_glut, nt_p_gaba, nt_p_da, nt_p_oa, nt_p_5ht, nt_p_his, nt_source, fafb_783_cell_type, hemibrain_121_cell_type, manc_121_cell_type, malecns_10_cell_type, *_match_id, vfb_id`.
`edges`: `dataset, version, pre, post, weight (int32), roi` (roi nullable; MaleCNS flat weights have no ROI column, FlyWire Codex is per-neuropil).
`provenance`: file name, URL, SHA-256, bytes, download time, converter version, thresholds.
Default neuron universe: MaleCNS `superclass IS NOT NULL` (166,700); FlyWire all proofread root IDs; full-segment graphs opt-in.

## 4. Memory budget (16 GB target; measured here)

| Object | Size | Strategy |
|---|---|---|
| MaleCNS full weights (151.9 M edges) | 3.64 GB as 3×int64 in memory; 1.05 GB Feather | Convert by record batch; store `pre,post` int64 + `weight` int32 Parquet sorted by `pre`; never materialise fully by default |
| MaleCNS neuron graph (25.6 M Traced→Traced edges) | ~0.5 GB | default working graph |
| FlyWire v783 Codex connections (3.87 M rows) | <0.5 GB | trivial |
| Shiu v630 parquet (14.7 M edges) | 86 MB file | sim input |
| LIF state, 139 k neurons × B trials × (v,g) float32 + delay ring (18 steps) | ~20 B×N×B bytes ≈ 0.6 GB for B=32 | batch cap by device memory |

## 5. Milestones (one branch each; tests green per commit)

| Milestone | Deliverable | Acceptance | Depends on |
|---|---|---|---|
| **M0 scaffold** | uv project, ruff, pyright strict on public API, pytest+hypothesis, CI (lint/types/unit on Linux+macOS; heavy manual/scheduled), mkdocs skeleton, CLAUDE.md, CHANGELOG, `ATTRIBUTION.md`, tiny synthetic fixture connectome | CI green on empty package | – |
| **M1 data** | registry YAML (MaleCNS 1.0, FlyWire 630/783, Shiu v630 sim inputs, hemibrain 1.2.1, MANC 1.2.1, BANC 888), resumable checksummed pull, Feather/CSV→Parquet, DuckDB queries, `citations()` | Tier 0 counts in GOLDEN_RESULTS.md reproduce; MaleCNS + FlyWire load < 16 GB; unit tests offline | M0 |
| **M2 graph** | signed sparse matrices (3 sign policies), aggregation by type/superclass/ROI, thresholds, 1–3 hop paths, connectome_interpreter adapter | W1 single-run on both datasets in minutes; parity test vs `find_paths_of_length` | M1 |
| **M3 uncertainty** | NT sampling (probabilities or confidence-only), threshold sweeps, degree-preserving rewiring, sign shuffle, version diff (v630↔v783, MaleCNS v0.9↔v1.0 if v0.9 files still public), stability wrapper | **W1 and W4 end-to-end** with stability + drift sections; null tests | M2 |
| **M4 sim** | engine (CPU f64 oracle, f32 batched CPU/MPS/CUDA), Brian2 gate, Shiu golden tests, benchmarks, MaleCNS calibration protocol + "uncalibrated" flag | validation rungs 1–3 pass (heavy job) | M1 |
| **M5 experiments + report** | YAML spec, runner with controls on by default, Parquet results, HTML report (effect sizes, multiple-comparison correction, provenance, citations) | **W2 from one YAML** | M3, M4 |
| **M6 compare** | label-graph type matching, cosine co-clustering, male vs female per type with L/R null (Schlegel 2024 parameters) | **W3 with verdict + caveats** | M3 |
| **M7 access** | Meissner table + NeuronBridge S3 + VFB adapters; on-target lines, off-target ranking | **W5** with caveats; or a documented skip | M1–M5 |
| Docs | one tutorial notebook per W1–W5, API reference, Scientific caveats page | – | each milestone |

Order: M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7. M4 can start in parallel with M3 once M1 lands, since its input is the Shiu parquet.

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| FlyWire Codex files are refreshed in place without version bumps | certain | registry pins SHA-256 + GCS `updated` time per file and refuses silently changed files; Zenodo 10676866 as the frozen v783 alternative |
| Brian2 parity fails on refractory/delay edge cases | medium | copy the MLX port's tick semantics; start with 20-neuron hand-checkable nets |
| MPS non-determinism confuses users | high | scope the determinism claim; report device in provenance |
| MaleCNS NT sampling under confidence-only model is crude | medium | label outputs; make `--level nt-probs` cheap (streamed aggregation, ~2 GB peak) |
| connectome_interpreter PyPI staleness | certain | pin git SHA; vendor nothing |
| Cell paper Methods unreachable | medium | ask owner for institutional PDF; tests use files |
| Session/token limits during long builds | observed in Phase 0 | ≤3 parallel agents; incremental file writes |
| BANC/FANC access terms | low (BANC) / high (FANC) | BANC v888 is fully public, CC BY 4.0, with cross-dataset match columns: include in M1. FANC live access is restricted and its public export has no cell types: adapter only |

## 7. Open questions for the owner (do not block M0–M2)

1. Licence: Apache-2.0 as proposed (ADR-0006)?
2. Register `flyconn` on PyPI now (placeholder release) or at M1?
3. Do you have institutional access to the MaleCNS Cell paper Methods (10.1016/j.cell.2026.08.015)? It would settle the NT "unclear" thresholds and the 11,710 vs 11,751 type delta.
4. Priority between M4 (sim) and M3 (uncertainty) if time is short: the plan puts M3 first because W1/W4 need it and it is the differentiator.
