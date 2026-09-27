# flyconn

Research-grade Python toolkit over the public *Drosophila* connectomes
(MaleCNS v1.0, FlyWire v630/v783, hemibrain, MANC, BANC): a harmonized offline
data layer, signed sparse graphs, uncertainty propagation, a validated
cross-platform LIF simulator, declarative in-silico experiments with reports,
and cross-dataset comparison.

Status: pre-alpha; milestones M0–M7 of `docs/PLAN.md` are implemented and validated against
published results on real data (`docs/GOLDEN_RESULTS.md`, `docs/PROGRESS.md`).

## Quickstart

```bash
uv sync --extra sim --extra interpret --extra access
uv run flyconn data list
uv run flyconn data pull malecns@1.0 --level weights      # ~1.1 GB, checksummed, resumable
uv run flyconn data pull shiu@630 --level weights          # Shiu et al. 2024 model inputs (90 MB)
uv run flyconn run examples/specs/w2_malecns_lb3_silence_gng232.yaml --out runs/w2
```

Python:

```python
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, find_paths
from flyconn.uncertainty import path_stability

m = ConnectivityMatrix.from_store(Store.open("malecns@1.0"), min_weight=5)
lb3 = m.meta.index[m.meta["cell_type"].str.match(r"^LB3", na=False)]
mn9 = m.meta.index[m.meta["cell_type"] == "MN9"]
paths = find_paths(m, lb3, mn9, max_hops=3, min_edge_fraction=0.01, label="cell_type")
stab = path_stability(m, lb3, mn9, thresholds=[5, 10, 20], n_samples=100, seed=0)
```

Anchor workflows: `examples/w1_pathways.py`, `examples/w4_version_drift.py`,
`examples/specs/w2_*.yaml`, `examples/w3_male_vs_female.py`, `examples/w5_driver_lines.py`.
Scientific caveats: `docs/caveats.md`.

Not to be confused with the Cambridge FlyConnectome group's tools (`cocoa`,
`flywire_annotations`); flyconn is independent and builds on that ecosystem.

## Scientific stance

A connectome is wiring. Neurotransmitter identities are predicted, weights are
synapse counts, and every simulation output is a model prediction. Controls run
by default, uncertainty is propagated, and every result carries provenance and
the citations of the datasets it used. See `docs/caveats.md` once written.

## Licence and attribution

Code: Apache-2.0. Data: CC BY 4.0 by the respective consortia; every result
object emits the citation list it depends on. Design ideas borrowed from
MIT-licensed projects are listed in `ATTRIBUTION.md`.
