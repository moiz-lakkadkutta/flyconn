# ADR-0003: Do not depend on `cocoa`; reimplement label-graph type matching with tests, contribute upstream where cheap

Status: Proposed (2026-09-26)

## Context

The brief suggested building M6 on `cocoa` and upstreaming MaleCNS v1.0 and BANC
support. Verified state of `flyconnectome/cocoa` on 2026-09-26: GPL-3.0, not on
PyPI, one contributor, **zero tests, no CI**, last `main` commit 2026-01-09;
requires neuPrint and CAVE tokens plus lab-internal SeaTable for live
annotations; MaleCNS v1.0 is reached only by accident (`male-cns:latest` →
`max(versions)`), docs still say v0.9; BANC exists only on an unmerged branch
via internal SeaTable. Its valuable idea is `GraphMapper`: build a graph of
`id→label` and `label↔label` (synonym / coarser-label) edges, then maximise
connected components while preserving cross-dataset mappings.

## Decision

- `flyconn.compare` reimplements label-graph matching over our harmonized
  schema, using the cross-dataset type columns the datasets already publish
  (MaleCNS `flywireType/mancType/hemibrainType`; BANC `fafb_783_cell_type`,
  `manc_121_cell_type`, `*_match_id`; FlyWire `hemibrain_type`), with property
  tests (hypothesis) and parity tests against a few `coconatfly` outputs.
- Cosine-similarity co-clustering and the left/right technical-noise null
  (Schlegel 2024) are implemented natively; they are small.
- We offer `cocoa` a PR adding a public-feather BANC dataset and v1.0
  docstrings if the maintainer wants it, but flyconn does not import cocoa.

## Alternatives considered

- Depend on cocoa (GPL, untested, token-bound): rejected; a research-grade
  toolkit cannot rest on an untested single-maintainer package with online-only
  data paths.
- Use `coconatfly` via rpy2: rejected (R dependency, experimental lifecycle).

## Consequences

- More code in `compare` than "thin glue", but all of it tested and offline.
- Licence freedom: flyconn can stay permissive (see ADR-0006).

## Evidence

`docs/research/landscape_access_analysis.md` §6–7.
