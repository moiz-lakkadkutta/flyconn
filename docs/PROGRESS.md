# Progress log

Running log so a later session can resume. Newest entry first.

## 2026-09-26 — M3 uncertainty (branch m3-uncertainty)

What works (golden, real data):
- W1 stability, FlyWire (Shiu v630): 50 paths over thresholds 5/10/20 at ≥1 % per-hop fraction, mean threshold presence 0.81, sign stability 1.0 (Shiu signs are certain by construction), 0.5 s.
- W1 stability, MaleCNS (confidence-only NT model before the tbar aggregation): 74 paths, mean sign stability 0.89, threshold presence 0.77; example prints type-level paths such as LB3c → GNG232 → DNge080 → MN9 (net excitatory) and LB3c → GNG132 → GNG130 → MN9 (net inhibitory).
- W4 v630→v783 for the 24 sugar-circuit neurons: 2 ids unmatched (incl. ipsilateral MN9), 21 attribute changes (sugar GRNs gained cell type `LB3` in v783), 55 edges added / 179 removed / 3 changed, 0.8 s.
- MaleCNS `--level nt-probs` aggregation implemented and unit-tested; the 2.65 GB download was started on this machine (golden test `test_malecns_nt_probs.py` runs once it is cached).

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
