# Changelog

All notable changes. Format: Keep a Changelog; versions follow SemVer once 0.1.0 ships.

## [Unreleased]

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
