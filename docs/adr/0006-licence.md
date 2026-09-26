# ADR-0006: flyconn is licensed Apache-2.0; GPL packages are optional extras, never hard imports; data stays CC-BY with per-output citation lists

Status: Accepted (2026-09-26, owner approved)

## Context

Half the ecosystem is GPL-3.0 (navis, flybrains, fafbseg, cocoa, R natverse
packages, banc); the other half is permissive (neuprint-python BSD-3, caveclient
MIT, connectome_interpreter MIT, Shiu model MIT, drosophila-brain-mlx MIT). The
datasets are CC-BY 4.0 (MaleCNS, hemibrain, MANC; FlyWire per Codex terms, see
DATA_SOURCES.md). A research toolkit gains adoption if labs can embed it in their
own (often permissive) analysis code.

## Decision

- Code licence **Apache-2.0** (patent grant, permissive, compatible with MIT/BSD
  dependencies and with importing GPL packages *optionally at runtime* by the
  end user).
- GPL packages (navis, flybrains, fafbseg) are declared only in optional extras
  (`flyconn[morph]`, `flyconn[flywire-live]`) and imported lazily inside functions;
  the core (`data`, `graph`, `uncertainty`, `sim`, `experiments`, `report`,
  `compare`) must run without any GPL import.
- No GPL code is copied. Ideas borrowed from MIT projects (drosophila-brain-mlx's
  validation design, Shiu's model definition) are attributed in `ATTRIBUTION.md`.
- Every result object carries `citations()` returning the dataset and method
  citations actually used (Dorkenwald 2024, Schlegel 2024, Eckstein 2024, Berg
  2026, Shiu 2024, Yin 2025, etc.), satisfying CC-BY attribution.

## Alternatives considered

- GPL-3.0 for flyconn: would let us import navis/cocoa freely, but blocks
  embedding in permissive lab code and in commercial-friendly pipelines.
- MIT: fine, but Apache-2.0's explicit patent grant is preferable for a toolkit
  that may end up in grant deliverables.

## Consequences

- `compare` reimplements label-graph matching instead of importing cocoa
  (ADR-0003). `report` cannot use navis plotting in the core; matplotlib only.
