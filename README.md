# flyconn

Research-grade Python toolkit over the public *Drosophila* connectomes
(MaleCNS v1.0, FlyWire v630/v783, hemibrain, MANC, BANC): a harmonized offline
data layer, signed sparse graphs, uncertainty propagation, a validated
cross-platform LIF simulator, declarative in-silico experiments with reports,
and cross-dataset comparison.

Status: pre-alpha, under construction milestone by milestone. See `docs/PLAN.md`.

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
