# Progress log

Running log so a later session can resume. Newest entry first.

## 2026-09-29 — Published; M8 BANC done

- GitHub: https://github.com/moiz-lakkadkutta/flyconn (public; CI green on Linux/macOS x Python 3.11/3.12). Docs: https://moiz-lakkadkutta.github.io/flyconn/. Release v0.1.0 created with wheel and sdist attached; `release.yml` publishes to PyPI by trusted publishing. **PyPI upload pending**: the owner must register the pending publisher on pypi.org (project `flyconn`, owner `moiz-lakkadkutta`, repo `flyconn`, workflow `release.yml`, environment `pypi`), then re-run the failed `publish-pypi` job of the v0.1.0 release run.
- M8: BANC v888 converter and shared FlyWire-or-MANC partner vocabulary; golden `tests/golden/test_banc.py` (counts; W3 MaleCNS vs BANC for DNp01, PFL3, EPG). Results in GOLDEN_RESULTS 6d. Key finding: EPG's "beyond range" verdict against FlyWire drops to "within range" against BANC, so it is not a sex effect.
- Local note: `scripts/check.sh` uses `uv run`, which re-syncs the default environment and drops the `brian` group, so the Brian2 parity test is skipped locally unless run with `uv run --group brian pytest -m brian`; the heavy CI workflow runs it.

## 2026-09-27 — Final verification on main

`FLYCONN_CACHE=$PWD/.cache/flyconn uv run pytest -m "golden or brian"`: 33 passed in 11 min 28 s on the M4 Pro (MaleCNS reconverted under converter version 2 incl. tbar aggregation in 43 s; Shiu 100/200/10 Hz golden 57/56/47 s; W2 277 s; W3 3 s per type; MaleCNS calibration protocol 120 s). Unit gate: 177 passed. Peak RSS reported inside that single long pytest process reached 10.9 GB because simulations and conversions share the process; the MaleCNS conversion alone peaks at 5.7 GB.

## 2026-09-27 — M7 access done (exploratory; branch m7-access, merged)

- `flyconn data pull flylight_lines@meissner2025`: 4,433 lines; 2,667 adult lines with cell-type annotations; EM body ids parsed for 320 lines (323 non-empty cells; 3 contain non-numeric text, kept raw). `LineCatalog.lines_for_type("DNp01")` -> SS02299 (quality 1, specific), SS00727 (quality 2).
- NeuronBridge off-target candidates for SS02299 (10 of its images): 6,073 scored candidate EM bodies across datasets, cached under `$FLYCONN_CACHE/neuronbridge/`. Marked `network`; not part of CI.
- Not built: VFB and FlyBase adapters; the NeuronBridge curated Confident/Probable/Candidate matches (only behind an authenticated API as of Phase 0).
- Unit: 176 offline; golden: 34 (+1 network).

Project status after M7: all planned milestones M0–M7 delivered; see PLAN.md status table and the "What is not done" list there.

## 2026-09-27 — M6 compare done (branch m6-compare, merged)

- `match_types(malecns@1.0, flywire@783)`: 6,836 MaleCNS types mapped via the published `fafb_783_cell_type` column (comma lists split).
- `compare_type` (output partner-type profiles, min_weight 5, 500 permutations, ~3 s per type after the ~10 s matrix builds): PFL3 cross-similarity 0.929 vs within-brain L/R 0.988/0.981, statistic 0.056, p 0.002 -> "detectable but within Schlegel's between-brain range"; PAM08 0.931 / 0.986 / 0.986, statistic 0.055 -> same grade; EPG 0.795 / 1.000 / 0.969, statistic 0.189 -> "beyond the between-brain range" (largest partner shifts: Delta7 0.19 vs 0.06, EL 0.09 vs 0.17, ExR6 0.10 vs 0.17); DNp01 (n = 2 per dataset): 58 % of male output goes to VNC neurons without a FlyWire counterpart (excluded and reported), p 0.52.
- Interpretation guard: verdicts are graded against Schlegel et al. 2024's 0.045 +/- 0.096 between-female-brain effect size, and small-n and unmatched-partner caveats are attached; an EPG-sized difference may still reflect type splitting/merging between datasets rather than sex.
- Unit: 171 offline; golden: 31.

Deferred: input-direction and type-pair batch reports; BANC/hemibrain converters (registry entries exist); NBLAST/morphology-based matching (would need navis, GPL extra).

Next: M7 access (exploratory).

## 2026-09-27 — M5 experiments + report done (branch m5-experiments, merged)

- `flyconn run examples/specs/w2_malecns_lb3_silence_gng232.yaml --out DIR` runs W2 from one YAML: MaleCNS v1.0, 78 LB3 GRNs at 100 Hz, silence GNG232, readouts MN9 / descending / motor, 20 trials x 0.5 s, controls 2 rewired + 2 sign-shuffled connectomes + unstimulated baseline; 283 s on CPU float32 (7 simulated conditions). Report is self-contained HTML with model-prediction and UNCALIBRATED banners, effect table and figures, provenance, citations.
- Numbers (model predictions, uncalibrated): MN9 44.9 Hz stimulated vs 23.1 Hz with GNG232 silenced (difference −21.8 Hz, Cohen's d −3.9, q = 9e-8); descending group 3.36 → 2.65 Hz (d −1.2); motor 7.56 → 4.91 Hz (d −1.1). Degree-preserving rewiring abolishes the MN9 response (0.0 Hz). Sign shuffling drives the whole network into runaway firing (59M and 40M spikes, 96k / 68k active neurons vs 9,367) — with ~65 % excitatory neurons, shuffling signs breaks E/I balance, so this control mostly tests "are the inhibitory neurons where they are for a reason"; documented in caveats.
- Unit: 165 tests offline; golden: 26.

Deferred: multi-experiment sweeps (rates, seeds across YAML lists), FlyWire-side readouts by annotation (the Shiu store carries no cell types; use flywire@630 ids), report PDF export.

Next: M6 compare (male vs female per cell type with L/R null).

## 2026-09-27 — M4 sim done (branch m4-sim, merged)

Validation ladder (all run on this machine, M4 Pro):
- Rung 1: float64 CPU engine == Brian2 2.10.1 spike for spike on 60- and 200-neuron random nets with fixed input trains (`pytest -m brian`, 33 s).
- Rung 2: Shiu golden on FlyWire v630, 30 trials x 1 s, CPU float32: MN9 66.5 ± 4.1 Hz at 100 Hz sugar (published 65.7/67.0), 94.5 Hz at 200 Hz (93.2), 0 Hz at 10 Hz; active neurons 416/444/39 vs 404–410/455/45. ~70 s per 30-trial batch.
- Rung 3: `benchmarks/sim_throughput.json`: CPU 2.2 s per biological second at 30 batched trials (15 s single trial), MPS 2.7 s (float32 spike counts identical to CPU on the full network for 100 ms). The per-tick Python loop dominates; MPS gives no gain yet.
- MaleCNS re-calibration protocol (`calibrate_w_syn`, Shiu's 80 %-of-max rule) run with 78 LB3 GRNs -> MN9 over w_syn in {0.1, 0.2, 0.275, 0.4, 0.6} mV: readout at 100/200 Hz = 0/0.6, 31/55, 43/76, 66/103, 60/122 Hz; no grid point reaches 0.8 (best 0.4 mV at 0.63) -> **unresolved; MaleCNS simulations stay labelled uncalibrated** (`benchmarks/malecns_calibration.json`).

Design notes: event-driven delivery via CSR gather + `index_add_` (single code path for CPU/MPS/CUDA); per-trial Poisson draws precomputed from `SeedSequence(seed).spawn` so batched and single runs agree; externally driven neurons get refractory 0 as in the reference code; kicks and synaptic input are dropped during refractoriness (Brian2 `unless refractory` semantics as transcribed by the MLX port).

Known limits: spike-time recording is CPU-side per tick (slow for dense activity); no CUDA machine was available to run rung 3 on GPU; throughput could improve ~5–10x by moving the tick loop into TorchScript/compiled kernels (not done).

Next: M5 experiments + report (YAML spec -> runner with controls by default -> Parquet + HTML report; W2).

## 2026-09-26/27 — M3 uncertainty done (branch m3-uncertainty, merged)

What works (golden, real data):
- W1 stability, FlyWire (Shiu v630): 50 paths over thresholds 5/10/20 at ≥1 % per-hop fraction, mean threshold presence 0.81, sign stability 1.0 (Shiu signs are certain by construction), 0.5 s.
- W1 stability, MaleCNS (confidence-only NT model before the tbar aggregation): 74 paths, mean sign stability 0.89, threshold presence 0.77; example prints type-level paths such as LB3c → GNG232 → DNge080 → MN9 (net excitatory) and LB3c → GNG132 → GNG130 → MN9 (net inhibitory).
- W4 v630→v783 for the 24 sugar-circuit neurons: 2 ids unmatched (incl. ipsilateral MN9), 21 attribute changes (sugar GRNs gained cell type `LB3` in v783), 55 edges added / 179 removed / 3 changed, 0.8 s.
- MaleCNS `--level nt-probs` (2.65 GB tbar file, SHA-256 pinned): per-body mean probabilities for 165,665 neurons (842 fall back to the confidence model, 193 unknown); argmax of the means agrees with the release consensus label for 96.2 % of neurons. LB3→MN9 sign stability with real probabilities: mean 0.83 (vs 0.89 confidence-only), best paths 0.90–0.94.

Deferred: statistical tests around whole-pathway counts vs nulls are available via `null_distribution` but not yet wired into a report (M5).

Next: M4 sim.

## 2026-09-26 — M2 graph core done (branch m2-graph, merged)

What works:
- Signed sparse matrices with four sign policies; aggregation; reachability-pruned 1–3 hop path enumeration; native effective connectivity that matches `connectome_interpreter.compress_paths_signed` (excitatory minus inhibitory) to 1e-6 on a random graph.
- W1 single-run acceptance (golden): FlyWire (Shiu v630 export) 21 sugar GRNs → 2 MN9: 31 three-hop paths (29 net-excitatory, 2 net-inhibitory) at min_weight 5 and ≥1 % per-hop input fraction, 0.1 s. MaleCNS 78 LB3 GRNs → 2 MN9: 74 three-hop paths (44 / 30), 0.1 s. No direct or 2-hop paths survive the 1 % per-hop filter; with no fraction filter there are 10 two-hop paths in FlyWire.
- Shiu v630 converter reproduces 127,400 neurons / 14,687,178 edges / 52,793,639 synapses.

Honest discrepancies:
- JO-CE→aBN1 and JO-F→aBN1 from the repo notebook ID lists on the repo's own v630 parquet give 77 and 69 synapses, not the paper's 103 and 78. Pinned as regression values; cause UNVERIFIED (different JON list or export in the paper).
- "Sugar GRN" cannot be selected by cell type: v783 types both sugar and water labellar GRNs as `LB3` (MaleCNS: `LB3a–d`). FlyWire W1 uses the Shiu ID list; MaleCNS W1 uses the LB3 group, documented as sugar+water.

Deferred to M3+: threshold sweeps / NT sampling around these paths (M3), type-level path aggregation (fold into W1 report in M3).

Next: M3 uncertainty (NT sampling incl. MaleCNS `--level nt-probs`, threshold sweeps, degree-preserving and sign-shuffle nulls, version diff v630↔v783, stability wrapper; W1 + W4 end-to-end).

## 2026-09-26 — M1 data layer done (branch m1-data, merged)

What works:
- `flyconn data pull malecns@1.0 --level weights` and `flywire@{630,783}`: download (resumable, SHA-256/MD5 verified), convert to harmonized Parquet, query offline via `Store` (DuckDB). Golden Tier 0 tests pass: MaleCNS 166,700 neurons, 151,856,684 raw edges, 311,833,243 PSDs, 25,582,938 neuron→neuron edges; FlyWire v783 139,255 / 2,700,513; v630 127,978 / 2,613,129.
- MaleCNS conversion: 26 s, 5.7 GB peak RSS (memory-mapped record-batch streaming; the 16 GB target holds with margin).
- Registry also lists hemibrain@1.2.1, manc@1.2.1, banc@888 and shiu@630 with pinned checksums; their converters are deferred (BANC/hemibrain/MANC to M6, Shiu to M4) and `pull` raises NotImplementedError for them.

What doesn't / deferred:
- MaleCNS NT probabilities (`--level nt-probs`, ADR-0008) not yet implemented: neurons carry consensus label + confidence only.
- FlyWire per-neuron NT probabilities are Codex `*_avg` means (6 classes); histamine/tyramine columns null.
- No MaleCNS ROI edge table (flat weights have no ROI); needs syn-partners (opt-in) or neuPrint.
- Raw files for this machine were hard-linked from `.cache/data/` into `.cache/flyconn/raw/` (set `FLYCONN_CACHE=$PWD/.cache/flyconn`).

Learned / plan changes:
- Codex `connections.csv` per-neuropil rows must be summed per pair before thresholding; the converter does this and keeps the per-neuropil table separately.
- pandas-stubs under pyright strict need `cast("pd.Series", df[col])` for column access; keep helpers for this.

Next: M2 graph (signed sparse matrices, aggregation, 1–3 hop paths, connectome_interpreter adapter).

## 2026-09-26 — Phase 0 complete; waiting for go-ahead

Deliverables (all committed on `main`):
- `docs/LANDSCAPE.md` — tools, verdicts, gaps; substrate facts (MPS sparse status, Brian2).
- `docs/DATA_SOURCES.md` — per-dataset access, files, SHA-256, schema, licence, citations; harmonized-schema alignment table.
- `docs/GOLDEN_RESULTS.md` — Tier 0 counts (re-verified in-session), Lin24 graph stats, Shiu24 LIF parameters and outputs, Schlegel24 variability, CI25 targets, known inconsistencies.
- `docs/PLAN.md` — corrections to the brief, module decisions, schema seed, memory budget, milestones M0–M7, risks, open questions.
- ADRs 0001–0008 (all "Proposed"): data layer greenfield; wrap connectome_interpreter; don't depend on cocoa; own PyTorch sim engine; keep name `flyconn`; Apache-2.0 licence (needs owner confirmation); sign policy; MaleCNS NT uncertainty.
- Raw notes in `docs/research/` (six files). Pinned downloads in `.cache/data/` (~4 GB incl. MaleCNS weights and FlyWire 630/783 Codex CSVs; gitignored).
- Repo `CLAUDE.md`.

Process notes:
- Six parallel research agents exhausted the account session limit; four were cut off. Retries reused on-disk artifacts. Rule going forward: ≤3 parallel agents, incremental output files.
- Firecrawl credits ran out early; agents fell back to curl/GitHub API/Europe PMC.
- Research venv (not part of the repo): scratchpad `research-venv` with pyarrow/pandas/duckdb/torch.

Next (after go-ahead): M0 scaffold on branch `m0-scaffold`, then M1 data. First M1 target: MaleCNS v1.0 + FlyWire v630/v783 registry entries reproducing GOLDEN_RESULTS §1.1–1.3.

## 2026-09-26 — Phase 0 started

- Repo initialised (`git init`, main branch). No product code yet, per brief.
- Dev machine: MacBook Pro, Apple M4 Pro, 48 GB RAM, Python 3.13.7, uv 0.11.7. Target remains 16 GB laptops.
