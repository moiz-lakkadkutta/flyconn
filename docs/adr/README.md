# Architecture Decision Records

One file per decision, numbered, never deleted (superseded ones are marked so).
Status values: Proposed (awaiting Phase 0 go-ahead), Accepted, Superseded.
Template: Context → Decision → Alternatives considered → Consequences → Evidence.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-data-layer-greenfield.md) | Own harmonized Parquet/DuckDB data layer | Accepted |
| [0002](0002-wrap-connectome-interpreter.md) | Wrap `connectome_interpreter` for effective connectivity | Accepted |
| [0003](0003-do-not-depend-on-cocoa.md) | Do not depend on `cocoa` | Accepted |
| [0004](0004-sim-backend.md) | Own PyTorch LIF engine; Brian2 as parity oracle | Accepted |
| [0005](0005-package-name.md) | Package name stays `flyconn` | Accepted |
| [0006](0006-licence.md) | Apache-2.0; GPL packages optional extras | Accepted |
| [0007](0007-sign-policy-default.md) | Default sign policy (Shiu/Eckstein rule) | Accepted |
| [0008](0008-malecns-nt-uncertainty.md) | MaleCNS NT uncertainty from the per-presynapse file | Accepted |
| [0009](0009-calibration-onset-ratio.md) | Re-calibrate w_syn by the onset ratio, with a positive control | Accepted |
| [0011](0011-robustness-audited-screens.md) | Robustness-audited silencing screens, one axis at a time | Accepted |
