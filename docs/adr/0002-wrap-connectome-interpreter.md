# ADR-0002: Wrap `connectome_interpreter` for multi-hop / effective connectivity instead of reimplementing

Status: Proposed (2026-09-26)

## Context

M2 needs k-hop paths, signed effective connectivity and (later) differentiable
activation analyses. `connectome_interpreter` (MIT, 6 contributors, ~530 test
functions, CI, last commit 2026-08-27) implements exactly this and is
dataset-agnostic: every entry point takes a `scipy.sparse` matrix plus
index-keyed dicts (`compress_paths(A, step_number, ...)`,
`compress_paths_signed(inprop, idx_to_sign, target_layer_number, ...)`,
`find_paths_of_length(edgelist, inidx, outidx, target_layer_number)`,
`MultilayeredNetwork(all_weights, sensory_indices, ..., idx_to_group)`).
Caveats: PyPI release 2.9.5 (2025-06) lags the repo by ~15 months and the README
recommends installing from git; `torch` is a hard dependency; CI tests Python
3.10 only; signs are a fixed dict (no uncertainty).

## Decision

- `flyconn.graph` produces the input-proportion-normalised signed sparse matrix
  and the index-aligned metadata dicts from the harmonized store, and calls
  `connectome_interpreter` for `compress_paths*`, path enumeration and the rate
  model. It is an optional extra `flyconn[interpret]`, pinned to a git commit
  SHA until a PyPI release catches up.
- `flyconn.uncertainty` treats these kernels as pure functions and wraps them in
  the sampling loop (NT draws, threshold sweeps, nulls). We do not add
  uncertainty features inside `connectome_interpreter`.
- Direct 1–3 hop path enumeration with per-edge weights and signs (W1) is
  implemented natively on our sparse matrices (it is ~50 lines and must not
  require torch); parity with `find_paths_of_length` is a test.
- We contribute upstream: a MaleCNS v1.0 / BANC 888 loader example, and any
  bug we hit.

## Alternatives considered

- Reimplement matrix-power effective connectivity: rejected; no scientific
  value added, and their chunked GPU implementation is already validated in a
  preprint.
- `coconat::effective_connectivity` (R): reference only for parity tests.

## Consequences

- Torch becomes a dependency of the `interpret` and `sim` extras only; the core
  data/graph layer stays numpy/scipy/duckdb.
- Convention mismatch to encapsulate: analysis functions want pre-in-rows;
  the torch model wants pre-in-columns (`add_sign` transposes).

## Evidence

`docs/research/landscape_access_analysis.md` §8 and §10.
