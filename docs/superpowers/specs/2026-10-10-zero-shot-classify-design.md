# M12 design: zero-shot cell-type classification by partner profile

Status: design approved in conversation 2026-10-10 (option C, per-neuron + group,
nearest centroid with yardstick adjustment). Branch `m12-classify`.

## 1. Goal and scope

Given neurons in a query dataset with no type labels for those neurons, assign a
reference-dataset type with a calibrated confidence, or answer "ambiguous" or
"unknown". This generalises the LB3 sugar/water call (GOLDEN_RESULTS 6g) into an API
with a benchmark.

**In scope (mode A):** the query neurons are unlabelled, but their partners carry
reference names through the query dataset's cross-reference columns (the shared
FlyWire-or-MANC vocabulary, `partner_labels(..., "fafb_or_manc")`). Use cases: new
neurons in an updated release, types an annotated release left coarse (LB3a-d), and
checking a published cross-reference.

**Out of scope, but the design must allow it later (mode B):** fully unlabelled
datasets, where partner names come from the classifier's own earlier predictions
(iterative propagation from anchors). M12 only keeps the interface open (section 2.5).

**Not used:** morphology/NBLAST (GPL extra, ADR-0006), learned models (option 3), and
nearest-neighbour voting (option 2; possible later alternative if the benchmark shows
centroids fail on heterogeneous types).

**Success criteria:**
- the benchmark (section 4) runs as a golden test and reports per-pair accuracy, coverage,
  calibration error and open-set false-accept rate with bootstrap CIs;
- confidence is calibrated on held-out types: ECE <= 0.05 on the fitting pair's held-out
  folds; the transfer pair's ECE is reported as measured, not required;
- the LB3 example reproduces the GOLDEN_RESULTS 6g cosines through the public API.

## 2. Components (`flyconn/compare/classify.py`)

### 2.1 Reference atlas

`build_atlas(store, *, groups=None, min_weight=5, vocabulary="fafb_or_manc") -> Atlas`

- Reference labels are the store's `cell_type` by default, or user-supplied
  `groups: Mapping[str, Sequence[int]]` (label -> neuron ids). Custom groups are
  required because FlyWire v783 types sugar and water GRNs both as `LB3`.
- For each label it stores:
  - the centroid profile (mean of member profiles, section 2.2);
  - the neuron yardstick: median leave-one-out cosine of members to their own centroid;
  - the group yardstick: cosine between the left and right centroids.
- Labels with one member, or members on only one side, use the atlas-wide median
  yardstick and are flagged `yardstick_fallback`.
- Profiles and centroids are sparse (`scipy.sparse`). Measured in the throwaway spike:
  FlyWire 783 gives 8,452 types x 16,904 columns, built in seconds.
- Atlases are cached as Parquet plus `provenance.json` under
  `$FLYCONN_CACHE/atlas/<ref>/<hash of parameters>/`.

### 2.2 Profiler

`profiles(matrix, labels, vocab, *, mask=()) -> csr_matrix`

- Each row is the out-partner fractions concatenated with the in-partner fractions over
  the shared vocabulary. Each half is L1-normalised separately (synapse counts, edges >=
  `min_weight`).
- A neuron with no inputs (sensory) or no outputs (motor) has one zero half. Cosine then
  uses only the other half: that is the fallback.
- `mask` lists labels to blank as partner names. Benchmarks blank the held-out type so a
  neuron cannot be identified by its connections to its own type.
- Partners without a reference name are dropped and counted. `unlabelled_partner_fraction`
  is reported per neuron, and a caveat is attached above 0.3.

### 2.3 Scorer

- `s_T` = cosine(query profile, centroid_T) for every atlas label. Group queries use the
  mean profile of the group.
- `a_T = s_T / yardstick_T`: neuron yardstick for neuron queries, group yardstick for
  group queries. `a_T >= 1` means "as alike as the type's own members are to each other".
- Candidates are ranked by `s_T`. The top-k (default 5) candidates keep `s`, `a` and the
  margin to the next candidate.
- Queries are processed in chunks of 5,000 rows, so the dense score block stays under
  ~350 MB (16 GB budget).

### 2.4 Decider

Calibrated mode (a calibration file exists for this atlas and partner-vocabulary setup):

- features `(s1, a1, s1 - s2)` -> logistic model -> `p` = P(top-1 correct). The model is
  fitted with `scipy.optimize`; no new dependency.
- Calls, checked in this order:
  1. `unknown` if `s1 < s_floor`;
  2. `type` if `p >= accept` (default 0.8);
  3. otherwise `ambiguous`, listing every candidate with `s >= s1 - delta`. This may be
     a single candidate, which reads as "low confidence X".
  `s_floor` and `delta` come from the benchmark:
  - `s_floor`: lowest score at which the open-set false-accept rate is <= 10 %;
  - `delta`: largest margin bin where top-1 accuracy is < 50 %.
- The calibration file (`flyconn/compare/calibration/classify_<ref>.json`) holds the
  coefficients, thresholds, benchmark metrics and provenance. It is versioned in the repo
  and produced only by the benchmark.

Uncalibrated mode (custom groups, or a reference without a calibration file):

- returns the ranked `s`/`a` candidates with call `uncalibrated`;
- the result carries an UNCALIBRATED banner, as simulation results do.

Group result:

- the call on the mean profile;
- the per-neuron vote table (calls, counts);
- `agreement` = fraction of neurons whose top-1 label matches the group's top-1 label (works in both modes);
- a `heterogeneous` caveat when agreement is below 0.7 (the LB3b case).

### 2.5 Public API and CLI

```python
atlas = build_atlas(Store.open("flywire@783"))
res = classify(Store.open("malecns@1.0"), atlas, type="LB3b")   # or neuron_ids=[...]
res.per_neuron   # DataFrame: neuron_id, call, label, p, s1, a1, margin, top_k, unlabelled_partner_fraction
res.group        # GroupCall: call, label(s), p, agreement, votes
res.caveats; res.provenance
```

- `classify(..., partner_labels=None)`: an optional Series overriding the partner names
  (default: the query's cross-reference vocabulary). Mode B will pass the classifier's own
  predictions here. No other change is needed for it.
- CLI:
  - `flyconn classify QUERY --reference REF (--type T | --ids FILE) [--groups FILE.json] --out OUT.parquet`
  - `flyconn classify bench --query Q --reference REF --out DIR`

## 3. Caveats attached to results (defaults on)

- Assignments come from wiring similarity alone, not morphology or genetics.
- The partner vocabulary depends on the query dataset's published cross-references. Errors
  in those propagate.
- When a type is split or merged between datasets, the classifier returns the nearest
  reference split. A "type" call is not proof of one-to-one correspondence.
- Truncated volumes (hemibrain) lower similarities. Hemibrain is not benchmarked in M12.

## 4. Benchmark (`flyconn/compare/classify_bench.py`)

**Truth:** the query's `fafb_783_cell_type` where it is a single token that exists in the
reference. Comma lists (several FlyWire types) are excluded from accuracy and counted
separately.

**Pairs:**

| query | reference | role |
|-------|-----------|------|
| malecns@1.0 | flywire@783 | fit calibration (type-level 5-fold split) |
| banc@888 | flywire@783 | transfer test: calibration is not refitted here; per-pair refit reported too |
| banc@888 | manc@1.2.1 (via `manc_121_cell_type`) | secondary, nerve cord |

**Protocols:**

1. Closed set: every evaluated type is blanked as a partner label in both query and atlas;
   its centroid stays in the atlas. Scored at neuron and group level.
2. Open set: same, but the true type's centroid is also removed. The correct answer is
   `unknown` (or `ambiguous`). This measures the false-accept rate.
3. Splits are by type, never by neuron, so a type's neurons never sit in both the fit and
   the evaluation folds.

**Metrics:**

- top-1 and top-3 accuracy;
- coverage (fraction not `unknown`/`ambiguous`) and accuracy among covered;
- ECE (10 bins) and Brier score;
- open-set false-accept rate;
- bootstrap 95 % CIs over types (1,000 resamples, seeded).

**Ablation:** raw-score calibrator `(s1, s1 - s2)` vs the yardstick-adjusted one
`(s1, a1, s1 - s2)`. The adjusted model stays the default only if its held-out ECE and
accuracy-at-coverage are no worse. The result goes into the ADR either way.

**Outputs:** `benchmarks/classify_<query>_<ref>.json`, the calibration file (section 2.4),
and the GOLDEN_RESULTS entry.

**Known optimism:** MaleCNS's FlyWire cross-references may partly have been assigned
using connectivity to FlyWire (unverified; to be checked in the MaleCNS release paper).
That would inflate MaleCNS->FlyWire accuracy. BANC->FlyWire is the conservative
estimate. Both are always reported side by side.

**Spike evidence** (throwaway script, scratchpad; 400 random types per pair, out+in
profiles, min_weight 5, partner labels not blanked except in a 40-type check; not
reproducible numbers for docs):

- MaleCNS->FlyWire: group top-1 0.92, per-neuron 0.89. BANC->FlyWire: 0.66 / 0.57.
- Accuracy rises with score in both pairs (BANC: 0.35 at s <= 0.5, 0.97 at s > 0.9) and
  with margin (BANC: 0.24 at margin <= 0.02, 0.94 at > 0.2). Calibration and abstention
  are therefore feasible, but must be per pair.
- Blanking the true type as a partner label did not change accuracy on 40 types
  (0.875 / 0.775).

## 5. Error handling

- `KeyError`: unknown type or neuron ids.
- `ValueError`: query and reference vocabularies share less than 1 % of the profile mass
  (wrong cross-reference).
- Neurons with an empty profile get call `unknown`, reason `no_labelled_partners`.
- Calibration-file mismatches (other reference version, vocabulary or min_weight) fall
  back to uncalibrated mode with a caveat, never to a silently wrong calibration.

## 6. Testing

Unit tests (offline, synthetic, extending `tests/unit/compare/conftest.py`):

- the same dataset as query and reference gives 100 % top-1;
- a planted partner leak is removed by `mask`;
- open-set queries go to `unknown` once `s_floor` is set;
- the logistic calibrator recovers a known monotone relation (hypothesis property:
  `p` rises with `s1` and with the margin);
- `agreement` and the `heterogeneous` flag on a planted mixed group;
- results do not depend on neuron order;
- chunking gives identical scores;
- custom groups work;
- a calibration mismatch falls back to uncalibrated mode;
- the CLI smoke test.

Golden tests (`tests/golden/test_classify.py`, cached data):

- the MaleCNS->FlyWire and BANC->FlyWire benchmarks, with accuracy and ECE pinned in bands
  after the first real run;
- LB3 via custom sugar/water groups (Shiu ids) reproduces GOLDEN_RESULTS 6g:
  LB3c 0.92 / 0.62, LB3a water-like, LB3b flagged heterogeneous or ambiguous.

Benchmark runtime target: < 10 min per pair on the M4 Pro, peak RSS < 8 GB.

## 7. Provenance and docs

- Every result records:
  - dataset refs and store checksums;
  - atlas parameter hash;
  - min_weight, vocabulary, mask;
  - the calibration file id and its benchmark metrics;
  - seeds, package versions, git SHA.
- ADR-0010: reuse the `compare/types` profiles and vocabulary; centroid over kNN/learned
  models; scipy logistic, no sklearn; no NBLAST.
- CHANGELOG entry for M12; PROGRESS entry; GOLDEN_RESULTS section with the measured
  numbers; tutorial `docs/tutorials/classify.md`.
