# Zero-shot cell-type classification

Give `flyconn` neurons that have no type label in their own dataset, and it names the closest
reference type, or answers "ambiguous" or "unknown". It uses wiring only: each neuron is
described by the fractions of its synapses that go to and come from each partner type
(ADR-0010).

**What "zero-shot" assumes.** The query neurons are unlabelled, but most of their partners
are named in the reference vocabulary through the query dataset's published
cross-references (`fafb_783_cell_type`, then `manc_121_cell_type`; MaleCNS and BANC both
publish them). This fits new neurons in an updated release, and types a release left
coarse. A fully unlabelled connectome is out of scope for now.

## Python

```python
from flyconn.compare import build_atlas, classify
from flyconn.data.store import Store

atlas = build_atlas(Store.open("flywire@783"))  # one centroid per FlyWire type
res = classify(Store.open("malecns@1.0"), atlas, cell_type="LB3b")

res.per_neuron  # one row per neuron: call, label, p, s1, a1, margin, top_labels, ...
res.group  # the selection as a whole: call, label, agreement, votes, top_labels/top_s
res.caveats  # always read these
res.provenance  # datasets, atlas parameters, calibration id, package versions, git SHA
```

Select neurons by the query's own `cell_type` or by `neuron_ids=[...]`. By default the selection's own
cross-reference names are blanked as partner names (`mask_self=True`), so a neuron is not
identified by connections to its own type. The benchmark does the same, which is what the
calibration assumes.

### Custom reference groups

FlyWire v783 calls both sugar and water labellar GRNs `LB3`. To tell them apart, build the
atlas from your own neuron lists, e.g. the Shiu et al. sugar and water GRN ids:

```python
import json

sets = json.load(open("tests/fixtures/shiu_2024_neuron_sets.json"))["sets"]
atlas = build_atlas(
    Store.open("flywire@783"),
    groups={"sugar_grn": sets["sugar_grn"], "water_grn": sets["water_grn"]},
    direction="out",  # GRNs have few inputs; compare output partners only
    on_missing="drop",  # v630 ids absent from v783 are dropped and counted
)
res = classify(Store.open("malecns@1.0"), atlas, cell_type="LB3c", calibration=None)
```

Custom groups are never calibrated: results carry raw cosines only.

## Command line

```
flyconn classify run malecns@1.0 --reference flywire@783 --type LB3b --out lb3b.parquet
flyconn classify run malecns@1.0 --reference flywire@783 --ids ids.txt --groups groups.json --direction out --out x.parquet
flyconn classify bench --query malecns@1.0 --query banc@888 --reference flywire@783 --out bench/
```

`run` writes the per-neuron table as Parquet and a `.json` sidecar with the group call,
caveats and provenance. `bench` runs the hold-out benchmark described below.

## Reading the calls

Only whole-type (group) calls are calibrated. Per-neuron rows are always `uncalibrated`
with the best label and its raw cosine. In the benchmark, per-neuron confidence did not
carry over from one dataset to another (calibration error about 0.24), so `flyconn` does
not report a per-neuron probability. Group calls use:

| call | meaning |
|------|---------|
| `type` | calibrated P(the best label is right) >= 0.8 |
| `ambiguous` | the best match is not certain; `label` lists every candidate within `delta` of the best (`A\|B`) |
| `unknown` | the best cosine is below the floor that at most 10 % of neurons reach when their true type is missing from the reference; or the neuron has no named partners (`reason`) |
| `uncalibrated` | per-neuron rows always; group calls without a matching calibration (custom groups, other parameters, other reference). `label` is the best match, `s1` a raw cosine, `p` empty |

- `s1` is the cosine to the best reference centroid.
- `a1` divides it by the yardstick of that label: how alike the label's own members are
  (per neuron) or its left and right copies are (per group). `a1` near 1 means "as alike as
  the type is to itself".
- `margin` is the gap to the runner-up.

## Where the confidence comes from

`flyconn classify bench` hides labels between datasets that already cross-reference each
other (MaleCNS and BANC against FlyWire v783).

- Closed set: each evaluated type is blanked as a partner name, so a neuron cannot be
  identified by its connections to its own type.
- Open set: the type is also removed from the reference, so the right answer is
  "unknown".

Folds and bootstrap intervals are over types, not neurons. The shipped calibration
(`flyconn/compare/calibration/classify_flywire_783.json`, group calls only) pools both
pairs. Every calibrated result quotes its leave-one-dataset-out calibration error (0.08 for
MaleCNS and 0.13 for BANC at group level), because a new dataset was by definition not in
the fit. Treat group probabilities as approximate to about that degree. Measured numbers: `docs/GOLDEN_RESULTS.md` §6i.

## Caveats

- Wiring similarity only: no morphology, genetics or transmitter identity.
- Errors in the query's published cross-references propagate into every profile.
- A `type` call names the nearest reference label. Types split or merged between datasets
  are not resolved one to one.
- MaleCNS's FlyWire cross-references may partly have been made by comparing connectivity,
  which would flatter the MaleCNS benchmark. BANC is the conservative estimate.
- hemibrain is a truncated volume and was not benchmarked.
