# Landscape and gap analysis

Survey date 2026-09-26. Every figure (versions, dates, stars, issues, licences)
was read from the GitHub API, PyPI JSON, the package source or a cited page on
that date; items we could not confirm are marked UNVERIFIED. Full per-tool notes
with quotes and URLs: `docs/research/landscape_access_analysis.md` and
`docs/research/landscape_sim_and_longtail.md`.

Verdict legend: **depend-on** (import), **wrap** (adapter, optional extra),
**borrow-ideas** (reimplement with attribution), **contribute-upstream**,
**reference** (parity target only), **ignore**.

## 1. One-paragraph conclusion

The ecosystem has mature *data-access clients* (neuprint-python, caveclient,
fafbseg, navis/flybrains, banc) and exactly one mature *analysis library*
(connectome_interpreter). It has **no** harmonized offline multi-dataset store,
**no** uncertainty propagation anywhere (every tool uses point NT labels and a
fixed threshold), **no** validated cross-platform simulator (the reference is a
Brian2 script; the reimplementations are either Apple-only, CUDA-only, Euler-
integrated, unseeded or days old), and **no** declarative experiment/report
layer. Cross-dataset typing exists in `cocoa` (Python, untested, GPL, token-
bound) and `coconatfly` (R). Those four gaps are flyconn's scope; everything
else we wrap or cite.

## 2. Data access and analysis libraries

| Tool | Lang / licence | Datasets (verified) | Latest release | Last commit | Issues / contributors | CI | Gap it leaves | Verdict |
|---|---|---|---|---|---|---|---|---|
| neuprint-python | Py / BSD-3 | any neuPrint: `hemibrain:v1.2.1`, `male-cns:v0.9`, `male-cns:v1.0`, `manc:v1.0/1.2.1/1.2.3`, `optic-lobe` | 0.6.3, 2026-07-20 | 2026-07-20 | 18 / 7 | no test CI (pytest suite exists) | online only; token mandatory (client raises without one, though the public server answered anonymous Cypher for `male-cns:v1.0`); tokens issued before 2026-08 invalid; no type hints; slow for whole-CNS adjacency vs the 1.1 GB feather | **depend-on** (thin, optional `[neuprint]`) |
| caveclient | Py / MIT | CAVE datastacks: `flywire_fafb_public`, `brain_and_nerve_cord` (BANC), FANC | 8.2.1, 2026-07-10 | 2026-07-10 | 38 / 14 | yes | token required for every datastack; raw synapse tables, no harmonized schema; CAVE datasets only | **depend-on** (optional `[cave]`) |
| fafbseg-py | Py / GPL-3.0 | FlyWire production/sandbox/public, `flat_630`, `flat_783` | 3.2.2, 2026-02-20 | 2026-07-09 | 8 / 4 | yes | FlyWire only; heavy (cloud-volume, navis<2.0); GPL | **wrap** lazily (optional) |
| navis | Py / GPL-3.0 | morphology, dataset-agnostic; 0 code refs to MaleCNS | 1.12.0, 2026-07-13 | 2026-09-06 | 31 / 16 | 5 workflows | morphology-centric; GPL; heavy | **depend-on** optional `[morph]` |
| navis-flybrains | Py / GPL-3.0 | templates incl. `JRCFIB2022M` (=MaleCNS), MANC, FANC, BANC, FLYWIRE, FAFB14; bridging FLYWIRE↔MaleCNS, BANC↔MaleCNS | 0.6.3, 2025-11-17 | 2026-09-02 | 6 / 2 | publish only | h5py<=3.12.1 pin; transforms downloaded separately | **depend-on** optional |
| cocoa (flyconnectome) | Py / GPL-3.0 | FlyWire 783, hemibrain 1.2.1, MANC 1.2.1, MaleCNS via `male-cns:latest` (resolves to v1.0 by string ordering; docs say v0.9); BANC/FANC TODO (BANC only on unmerged `aedes` branch via internal SeaTable) | not on PyPI; git 0.2.1 | main 2026-01-09 | 1 / 1 | **none; 0 tests** | untested, single maintainer, tokens + internal SeaTable, GPL, no uncertainty | **borrow-ideas** (GraphMapper) + **contribute-upstream** where cheap; do not depend (ADR-0003) |
| coconatfly / coconat / malecns / bancr | R / GPL≥3 | flywire, malecns (v1.0 default), manc, fanc, hemibrain, opticlobe, banc, yakubavnc; multi-hop effective connectivity added 2026-08-19 | GitHub releases only | 2026-08-19 | 9 / 1 | R-CMD-check | R; "experimental" lifecycle | **reference** (parity targets); copy bancr's public feather path |
| connectome_interpreter (YijieYin) | Py / MIT | dataset-agnostic (`scipy.sparse` + index dicts); tutorials FlyWire 783, BANC, MaleCNS, hemibrain, MANC | PyPI 2.9.5, 2025-06-26; repo 2.10.0 unreleased (~15 months of commits) | 2026-08-27 | 0 / 6 | flake8, black, mypy(non-blocking), pytest+cov on py3.10 | torch mandatory; fixed `idx_to_sign` (no uncertainty); no data layer; PyPI stale | **depend-on** optional `[interpret]` pinned to a git SHA (ADR-0002) |
| sjcabs/fly_connectome_data_tutorial | Py+R / MIT | harmonized feather/parquet on GCS: BANC 888, FAFB 783, hemibrain 1.2.1, MANC 1.2.1, **MaleCNS v0.9** | not a package | 2026-06-07 | – / 4 | – | workshop bundle; v0.9; no DuckDB; assumes Google credentials | **adopt vocabulary**, **contribute** MaleCNS v1.0 (ADR-0001) |
| YijieYin/connectome_data_prep | Py | `.npz` inprop/outprop/syncount for MaleCNS, BANC, FAFB 783, hemibrain, MANC | – | 2026-08-12 | – / 4 | – | dataset versions unstated; **no licence file** | ignore as data source; contribute a v1.0 prep |
| schlegelp/connecto | Py / GPL-3.0 | unified CAVE/neuPrint query interface | – | 2026-09-22 | – | 2 workflows | live-query oriented | reference |
| flyconnectome/drosophila_neurotransmitters | data | versioned NT ground truth across datasets | – | – | – | – | – | **use** as ground-truth table for NT sanity tests (licence to verify in M3) |
| flywire_annotations (Schlegel 2024 supplements) | data | FlyWire 783 annotations, hemibrain meta | – | – | – | – | – | primary source for FlyWire types |

Direct answers the brief asked for:

- **connectome_interpreter takes an arbitrary scipy sparse matrix + metadata?** Yes: `compress_paths(A: spmatrix, step_number, ...)`, `compress_paths_signed(inprop, idx_to_sign: dict, target_layer_number, ...)`, `find_paths_of_length(edgelist: spmatrix | DataFrame, inidx, outidx, target_layer_number)`, `_NetworkBase(all_weights: Tensor | spmatrix, sensory_indices, ..., idx_to_group)`. Weights must be input-proportion normalised; analysis functions want pre-in-rows, the torch model pre-in-columns.
- **cocoa MaleCNS v1.0?** No explicit support; BANC still a README TODO.
- **neuprint-python against MaleCNS?** Yes, `Client("https://neuprint.janelia.org", dataset="male-cns:v1.0")`; a token is always required by the library.

## 3. Simulation codebases

| Project | What | Licence | Datasets | Last commit | Stars / issues / contributors | Tests | Integration / device | Gap | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| philshiu/Drosophila_brain_model | reference Brian2 LIF (Shiu 2024) | MIT | FlyWire v630 (+ v783 files shipped) | 2024-09-14 | 347 / 2 / 3 | none | Brian2 2.5.1 `method='linear'` (= exact), dt 0.1 ms, delay 18 steps, refractory 22 steps (0 for Poisson-driven), Cython runtime | one dataset, CPU, unseeded (joblib, per-trial rebuild), silencing zeroes only outgoing weights (issue #10) | **reference + parity oracle**; never imported at runtime |
| eonsystemspbc/fly-brain | six-backend *benchmark harness* (Brian2 C++/CUDA, PyTorch CUDA, NEST GPU, GeNN, Brian2GeNN) | **GPL-2.0-or-later** | v783 (v630 archive) | 2026-08-29 | 920 / 0 / 1 | none, no CI | PyTorch runner is **forward Euler** (`v += dt/tau*(g-(v-v_rest))`), CUDA-or-CPU (no MPS), **no `manual_seed`**, no library API; Brian2 comparison is Jaccard/rate-r only and its results JSON is not committed | cannot be spike-exact by construction; GPL | **not a backend** (ADR-0004); borrow trial batching, comparison metrics, parquet spike schema |
| Kisame76/drosophila-brain-mlx | Shiu LIF on Apple MLX/Metal | MIT | v630, MaleCNS v1.0 | 2026-09-26 | 2 / 0 / 1 | 17 pytest files, no CI | fixed-spike-train Brian2 gate on an 800-neuron subnet: float64 NumPy oracle **spike-for-spike identical** to Brian2 2.10.1; float32 Metal lane within 5 % of spikes; 0.29 s per biological second on M4 Pro vs Brian2 Cython 2.07 s; degree-preserving shuffle; manifest records "constants fitted to FlyWire" | Apple-only, single author | **borrow-ideas heavily**: validation design, ref64 oracle, pack manifest, shuffle |
| seohyunjun/mps-malecns-model | MaleCNS LIF on PyTorch MPS via gather + `index_add_` (no sparse tensors) | none | MaleCNS | 2026-09-12 | 3 / 0 / 1 | 6 | float32; CPU/MPS not bit-identical (summation order) | slower than real time; unlicensed | confirms MPS edge-list approach |
| eonfathom/FastFly | CUDA/CuPy push-model LIF | none stated | v783 | 2026-02-27 | 11 | none | event-driven | NVIDIA only | ignore (note push idea) |
| TuragaLab/flyvis | task-trained DMN of the optic lobe (Lappalainen 2024) | MIT | optic lobe | 2026-08-06 (1.2.0) | 190 / 4 / 5 | yes | non-spiking, trained | different problem | borrow packaging/ensemble ideas |
| brandoncho369/flybench | 31 pre-registered behavioural tasks + shuffled controls on v783/MaleCNS | MIT | v783, MaleCNS | 2026-09-24 | 2 | tests, 3 workflows | n/a | no engine | borrow task list for YAML experiments |
| Others (mehrantsi/flyBrain Rust/Metal, Pronexsteam/brainlab, annel0/flybrain, snedea/flybrain JS, erojasoficial-byte/fly-brain, ruvnet/Connectome-OS, abgnydn/webgpu-fly, ~60 game/desktop-pet repos on awesome-fly) | hobby sims, Sept-2026 wave | mostly MIT | v783/MaleCNS | days–weeks old | – | rarely | none validated against Brian2 | – | ignore |

No JAX, Norse, snnTorch, Nengo, Lava, BindsNET or BrainPy port of the Shiu model exists on GitHub (searched 2026-09-26).

Substrate facts that fix the engine design (verified on this machine, torch 2.14.0, macOS 26.2, M4 Pro):

| Capability | CPU | MPS | Notes |
|---|---|---|---|
| sparse COO `torch.sparse.mm` | OK | **OK** (0.20 s for 200 k × 200 k, 3 M nnz × 32 cols) | |
| sparse CSR matmul | OK (0.015 s) | **FAIL** `new_compressed_tensor` NotImplementedError (pytorch #140941 closed-not-planned) | CSR only on CPU/CUDA |
| gather + `index_add_` | OK | OK (0.10 s) but **non-deterministic** under `use_deterministic_algorithms(True)` | |
| float64 | OK | **not supported** | float64 oracle runs on CPU |
| `torch.mps.manual_seed` + `bernoulli` | – | OK | seedable Poisson drive |
| Brian2 | 2.10.1 (2025-12-05), Python ≥3.12, arm64 wheels, tested on Apple Silicon since 2.6.0 | – | Shiu pinned 2.5.1/py3.10 |

## 4. Long tail and naming

- **flybrain**: the PyPI name was taken on 2026-09-13 by a hobby monorepo (alextitonis/fly.ai); FlyBrainLab (BSD-3, Columbia) is dormant since 2025-09; `flybrains` is the navis transforms package. Nothing to build on.
- **connectome-lab**: four unrelated 0-star repos from mid-2026; PyPI name free. Nothing to build on.
- **awesome-fly** (cobanov, CC0, 617 stars, 97 repos): a good discovery index with honest "prototype / not independently reproduced" labels. Its solid entries are the data clients above, connectome_interpreter, cocoa, flywire_annotations, drosophila_neurotransmitters, flygym, flyvis and flybody.
- **PyPI 2025–26 sweep** (46 MB simple index grepped): every "LIF/spiking" fly package is ≤0.2.0 and ≤2 weeks old; `connectome-kg` is Elastic-2.0; the BBP `connectome-manipulator` (Apache-2.0) has transferable rewiring ideas.
- **Name check**: `flyconn` is free on PyPI and conda-forge; two GitHub namesakes (TypeScript, 0 stars, 2026-09-15; R, 3 stars, 2022) have no releases or users. `flyconnectome` is the Cambridge group's GitHub org, `connectomix` is fMRI tooling, `flywiring` implies FlyWire affiliation. **Keep `flyconn`** (ADR-0005); fallback `dmelconn`.

## 5. Licence map (matters for our own licence choice, ADR-0006)

| Permissive | Copyleft |
|---|---|
| neuprint-python (BSD-3), caveclient (MIT), connectome_interpreter (MIT), Shiu model (MIT), drosophila-brain-mlx (MIT), flyvis (MIT), sjcabs tutorial (MIT), flygym (Apache-2.0) | navis, navis-flybrains, fafbseg, cocoa, coconat*, malecns (R), bancr, connecto, banc: GPL-3.0; eonsystems fly-brain: GPL-2.0+; brian2cuda: GPL-3.0; Brian2: CeCILL-2.1 |

## 6. Gaps flyconn fills (and what it must not rebuild)

| Gap | Evidence | flyconn module |
|---|---|---|
| Offline, pinned, checksummed, harmonized Parquet/DuckDB store across MaleCNS v1.0 / FlyWire / hemibrain / MANC / BANC | no package exists; SJCABS is v0.9 and not a package; official files are Feather in three vocabularies | `data` (ADR-0001) |
| Uncertainty propagation: NT sampling, threshold sweeps, null models, version diffs | absent in every surveyed tool | `uncertainty` |
| Validated cross-platform (CPU/MPS/CUDA) seeded LIF engine with Brian2 parity | reference is a script; ports are Apple-only, CUDA-only, Euler, unseeded | `sim` (ADR-0004) |
| Declarative experiments with controls-by-default and standalone reports | flybench has the task-list idea; nobody has the runner/report | `experiments`, `report` |
| Tested cross-dataset type matching + L/R null | cocoa untested/GPL; coconatfly R | `compare` (ADR-0003) |
| Line → cell type with off-target evidence | no Python tool combines Meissner 2025 + NeuronBridge + connectome types | `access` (exploratory) |

Must not rebuild: neuPrint/CAVE clients, morphology and template transforms (navis/flybrains), effective-connectivity kernels and differentiable rate model (connectome_interpreter), Brian2 itself (used as the oracle).
