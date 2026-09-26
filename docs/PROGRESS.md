# Progress log

Running log so a later session can resume. Newest entry first.

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
