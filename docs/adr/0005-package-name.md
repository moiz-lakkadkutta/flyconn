# ADR-0005: Package name stays `flyconn`

Status: Proposed (2026-09-26)

## Context

Checked 2026-09-26: `flyconn` returns 404 on PyPI and conda-forge. GitHub has
Akash-2176/flyconn (TypeScript, 0 stars, 2026-09-15) and k3t3n/FlyConn (R, 3
stars, 2022), neither released nor used. Alternatives: `flyconnectome` is the
Cambridge group's GitHub org (cocoa authors); `connectomix` is fMRI tooling;
`flywiring` implies FlyWire affiliation; `dmelconn`, `flycon`, `wirefly` are free
but ambiguous or awkward. `flybrain` on PyPI was taken on 2026-09-13.

## Decision

Keep `flyconn` (PyPI distribution and import name). Register the PyPI name at the
first milestone release. README carries a "not to be confused with
FlyConnectome/cocoa" line. Fallback: `dmelconn`.

## Consequences

None beyond the README note.
