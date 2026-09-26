# Changelog

All notable changes. Format: Keep a Changelog; versions follow SemVer once 0.1.0 ships.

## [Unreleased]

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
