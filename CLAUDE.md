# flyconn — repo conventions for agents and humans

Research-grade toolkit over public Drosophila connectomes. Read `docs/PROGRESS.md`
first to resume; `docs/PLAN.md` is the scope; `docs/adr/` holds decisions.

## Principles (non-negotiable, from the project brief)
- Reuse the ecosystem; write an ADR for every build-vs-reuse choice.
- Every result carries provenance (dataset, version, checksums, thresholds, seeds, package versions, git SHA).
- Scientific honesty: wiring only; NT identities are predicted; simulation outputs are model predictions; controls on by default.
- Uncertainty is first class. Laptop budget: 16 GB RAM; CPU / Apple MPS / CUDA.
- Numbers in docs come from code run in this repo, never memory.

## Layout
`flyconn/{data,graph,uncertainty,sim,experiments,compare,report,cli}`, `docs/`, `tests/`, `benchmarks/`, `examples/`.
Raw research notes: `docs/research/`. Pinned downloads: `.cache/data/` (gitignored).

## Commands
- `scripts/check.sh` runs the full local gate (ruff, format, pyright, unit tests); use it before every commit
- `uv sync` · `uv run pytest` (unit, offline) · `uv run pytest -m golden` (needs cached data) · `uv run ruff check` · `uv run pyright`
- `flyconn data list` · `flyconn data info malecns@1.0` · `flyconn data pull malecns@1.0 --level weights`
- Golden tests on real cached data: `FLYCONN_CACHE=$PWD/.cache/flyconn uv run pytest -m golden`
- Cache layout: `$FLYCONN_CACHE/raw/<name>/<version>/` (downloads), `$FLYCONN_CACHE/store/<name>/<version>/` (Parquet + provenance.json)

## Conventions
- Python ≥3.11, `uv`, `ruff`, `pyright` strict on public API, `pytest` + `hypothesis`, `pytest-benchmark`.
- Unit tests never touch the network; use `tests/fixtures/` synthetic connectome.
- Conventional commits; one branch per milestone; CHANGELOG per milestone; each commit leaves tests green.
- GPL packages (navis, flybrains, fafbseg) only as optional extras, imported lazily (ADR-0006).
- Session hygiene: at most 3 parallel subagents; subagents write output files incrementally.
