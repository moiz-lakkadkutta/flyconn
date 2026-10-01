# Data sources

Verified 2026-09-26 by listing buckets, downloading files, computing on them and
querying APIs. Every SHA-256 below is of a file on this machine under
`.cache/data/` and its base64 MD5 matched the GCS object hash. Anything not
directly verified is marked UNVERIFIED. Full notes with schemas, snippets and every
column: `docs/research/data_malecns.md`, `docs/research/data_flywire_and_others.md`.

## 0. Summary

| Dataset | Version(s) | Bulk access (no token) | Live API (token) | Neurons | Licence | Our registry status |
|---|---|---|---|---|---|---|
| MaleCNS | **v1.0** (2026-06-08; v0.9 2025-10-05) | `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` Feather | neuPrint `male-cns:v1.0` | 166,700 | CC BY 4.0 | **pin now** |
| FlyWire / FAFB | **v783** (latest, Oct 2023), v630 (June 2023) | `gs://flywire-data/codex/data/fafb/{630,783}/` CSV.gz, **public** (the Codex web page is only a login front for the same bucket); Zenodo 10676866 (v783 Feather) | CAVE `flywire_fafb_public` (token; even the info endpoint redirects to login now) | 139,255 / 127,978 | papers + Zenodo CC BY 4.0; Codex bucket has no LICENSE object; FlyWire ToS text UNVERIFIED | **pin now** (both) |
| Shiu 2024 model inputs | v630 export of 2023-03-23 (+ v783 twin) | GitHub philshiu/Drosophila_brain_model raw files | – | 127,400 | MIT | **pin now** (sim golden input) |
| hemibrain | v1.2.1 (neuPrint); v1.2 flat export | `gs://hemibrain/v1.2/` (public) | neuPrint `hemibrain:v1.2.1` | 21,739 traced non-cropped | CC-BY (version UNVERIFIED on page; paper CC BY 4.0) | pin in M1 |
| MANC | v1.0 (flat export), v1.2.1 (neuPrint, Codex, sjcabs), v1.2.3 | `gs://flyem-manc-exports/v1.0/` (public); **no v1.2.x flat export** except synapse partners in `gs://manc-seg-v1p2/`; Codex `manc/1.2.1/neurons.csv.gz` | neuPrint `manc:v1.2.1` | 23,759 / 23,665 traced | CC-BY | pin v1.2.1 via Codex neurons + neuPrint edges (or sjcabs) in M1 |
| BANC | **v888** (publication, 2026-04-16), v626 (preprint) | `gs://lee-lab_brain-and-nerve-cord-fly-connectome/` (public) and Harvard Dataverse DVN/7WTH1N v3.0 (CC BY 4.0); Codex `banc/888/` | CAVE `brain_and_nerve_cord_public` (token) | 156,012 proofread+rough (meta); 158,262 (Codex) | **CC BY 4.0** | pin in M1 (was "TODO" in the brief; it is fully public) |
| FANC | public exports v1237, v1444; BANC cross-refs v1116 | `gs://lee-lab_female-adult-nerve-cord/CAVE/v1444/` parquet (public) | CAVE `fanc_production_mar2021` **restricted** (PI must email the lab) | 21,978 cell IDs | **none stated** | adapter only; no cell types in public export |
| sjcabs compiled_data | BANC 888, FAFB 783, MANC 1.2.1, hemibrain 1.2.1, **MaleCNS v0.9** | `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/` (public) | – | – | code MIT; data "CC-BY by the respective creators" | vocabulary adopted; files not primary |

Two access facts that change the plan relative to the brief: FlyWire bulk data needs
**no login**, and BANC is **fully public under CC BY 4.0** with a compiled meta table
that already carries cross-dataset type and match-ID columns.

Token status (2026-09-26): neuPrint tokens issued before 2026-08 are invalid
(auth migration); the public neuPrint server answered anonymous Cypher for
`male-cns:v1.0`, `hemibrain:v1.2.1` and `manc:*`, but `neuprint-python` refuses to
construct a client without a token and this openness is undocumented. CAVE info
and materialization endpoints for FlyWire, BANC and FANC all redirected to login.

## 1. MaleCNS v1.0

- Site: https://male-cns.janelia.org (download/, release/). Release notes are two lines: v1.0 (June 8, 2026) "Minor proofreading changes", "Refinement of neuron annotations"; v0.9 (October 5, 2025) initial release. No counts or thresholds on the site.
- Paper: Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, … Jefferis GSXE (111 authors). *Sexual dimorphism in the complete Drosophila male central nervous system connectome.* Cell 189(18):5504–5526.e15, 2026. doi:10.1016/j.cell.2026.08.015. PMID 42691995. CC BY 4.0. Preprint bioRxiv 10.1101/2025.10.09.680999. Companion Cell papers: visual pathways 10.1016/j.cell.2026.08.014, gustatory connectome 10.1016/j.cell.2026.08.016. (Cell full text is Cloudflare-blocked from here.)
- Licence: "The Male CNS dataset is licensed under CC-BY" (footer links CC BY 4.0). No further terms.
- neuPrint: `https://neuprint.janelia.org`, dataset `male-cns:v1.0` (v0.9 still served). Also `neuprint-cns.janelia.org` (401 anonymous). Clio, Cell Type Explorer, Dimorphism Explorer, NeuronBridge.
- Bucket: `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`; HTTPS `https://storage.googleapis.com/flyem-male-cns/...`. Neo4j database directory (251.8 GB, Neo4j 4.4.16) at `v1.0/database/neo4j/`; neuPrint input CSV/Feather at `v1.0/database/neuprint-inputs/` (194 GB).

| File | Bytes | SHA-256 | Rows | Default? |
|---|---|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` | 211,577 × 36 | yes |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` | 1,835,518 × 10 | yes |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` | 151,856,684 × 3 | `--level weights` |
| `connectome-weights-…-traced-only.feather` (undocumented) | 508,025,642 | UNVERIFIED (not downloaded) | 25,563,197 × 5 (`type_pre/type_post` added) | candidate default neuron graph |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 778,062,826 | `ca5dc83a26382ae70c8d8f42fc09ce2dbc1af7c03f3a001a1936b5e142540647` | 88,384,522 × 11 | optional |
| `tbar-neurotransmitters-male-cns-v1.0.feather` | 2,651,680,218 | not downloaded | 45,656,140 (7 `nt_*_prob` columns, rows sum to 1) | `--level nt-probs` |
| `syn-partners-…-minconf-0.5.feather` | 6,777,179,098 | not downloaded | 311,833,243 | `--level synapses` |
| `syn-points-…-minconf-0.5.feather` | 13,061,489,098 | not downloaded | 357,489,383 | `--level synapses` |

Schema essentials (annotations): `bodyId`; `status` (Traced 165,122 / Orphan / Glia / Unimportant / Assign / Anchor) and ordered `statusLabel` (31 categories); `superclass` (27 values; non-null = **166,700 = the paper's neuron count**); `class`, `subclass`, `type` (11,751 distinct), `instance`, `group`, `supertype`; `somaSide` L/R/M, `rootSide`, `somaNeuromere`, `somaLocation` [x,y,z] 8 nm voxels; cross-refs `flywireType` (143,156 bodies), `hemibrainType` (32,919), `mancType/mancBodyid/mancGroup/mancSerial`; `itoleeHl`, `trumanHl`, `birthtime`; `dimorphism` (4 values), `fruDsx` (6), `synonyms`, `matchingNotes`; `entryNerve/exitNerve`, `receptorType`, `assignedOlHex1/2`, `vfbId`.
NT file: `body, cell_type, total_nt_predictions, predicted_nt, predicted_nt_confidence, ground_truth (85,484 rows), celltype_*, consensus_nt`. **No per-transmitter probabilities at body level.** Traced-body `predicted_nt`: ACh 94,946, Glu 28,055, GABA 20,218, unclear 14,365, DA 4,443, His 2,026, 5-HT 465, OA 102.
Weights: `body_pre, body_post, weight` int64; no ROI column; sum 311,833,243 = total PSDs; Traced→Traced 25,563,197 edges / 124,025,046; ≥5: 7,622,864 edges (all segments).
Thresholds: `minconf-0.5` = synapse predictions with confidence < 0.5 excluded before export (flyem-snapshot `min-confidence`); presynapses are already ≥0.70, so the filter bites on postsynaptic confidence only. neuPrint `Meta`: `postHighAccuracyThreshold` 0.5, `postHPThreshold` 0.7. Hence flat `weight` = neuPrint `weight`.
ROIs: 144 primary neuPrint ROIs (brain neuropils L/R, VNC neuropils, nerves), 5,619 total incl. optic-lobe column ROIs. The flat weights file has no ROI split; per-ROI weights need syn-partners (`primary_post`) or neuPrint.
Caveats: 58 % of PSDs sit on unannotated fragments; 1,991 superclass-bearing neurons are "Out of scope"; 516 Traced bodies lack a superclass; type count 11,751 vs paper's 11,710 (UNVERIFIED cause); dimorphism tallies in the file do not reproduce the abstract's counts.
Citation to emit: Berg et al. 2026 (Cell) plus, for NT predictions, the paper's Methods (the classifier lineage is Eckstein et al. 2024; UNVERIFIED whether a separate MaleCNS NT paper exists).

## 2. FlyWire / FAFB (v630, v783)

- Latest public snapshot **v783** (Codex FAQ "v783 – Oct 2023 [latest release]"; bucket has only `630/` and `783/`; flywire_annotations v3.1.0 of 2026-07-21 is still 783-based; Zenodo says 783 is latest). Whether CAVE has newer *materializations* is UNVERIFIED (token).
- Bulk, no login: `https://storage.googleapis.com/flywire-data/codex/data/fafb/{630|783}/<file>` (this is the URL template in Codex's own loader). Codex web downloads need Google sign-in + `api_token` but front the same bucket. Files are **refreshed in place** without version bumps (e.g. 783 `classification.csv.gz` 2026-02-24), so the registry pins MD5/SHA-256 per file and records the GCS `updated` time.
- Zenodo 10.5281/zenodo.10676866 (CC BY 4.0): `proofread_connections_783.feather` 852 MB (pre/post root, neuropil, syn_count, per-NT avg), `flywire_synapses_783.feather` 9.5 GB (has `cleft_score`), `proofread_root_ids_783.npy`.
- Annotations: `flyconnectome/flywire_annotations` (Schlegel 2024 supplements; v3.1.0 2026-07-21): `Supplemental_file1_neuron_annotations.tsv` 139,248 rows × 31 (`cell_type`, `hemibrain_type`, `supertype`, `top_nt`, `top_nt_conf`, `known_nt`, `dimorphism`, `fru_dsx`, `vfb_id`, `fbbt_id`, …); `Supplemental_file5_hemibrain_meta.csv` 25,397 hemibrain bodies.
- CAVE: datastack `flywire_fafb_public`; token from `https://global.daf-apis.com/auth/api/v1/create_token` after accepting FlyWire terms.

Key files on disk (SHA-256 in the research note for all 40):

| File | Bytes | SHA-256 | Rows |
|---|---|---|---|
| `630/neurons.csv.gz` | 1,572,661 | `a0be4670993d86cea0a4724ba24f874c9d867a24f4e248ff30dcf9b6565d81bb` | 127,978 |
| `630/classification.csv.gz` | 1,011,550 | `b2cf34ef76f5750e647334b56c73ae7f49c96f0fee3371e8a52d26c835a3b499` | 127,979 (one id not in neurons.csv) |
| `630/connections.csv.gz` | 26,124,953 | `68144490b8585fb6b1bf5979d923a3b1081b435cd52b7f82daed1beb6b84781a` | 3,794,615 rows; 2,613,129 pairs |
| `783/neurons.csv.gz` | 1,679,884 | `6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719` | 139,255 |
| `783/classification.csv.gz` | 934,402 | `e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663` | 139,255 |
| `783/fw_and_hemibrain_types.csv.gz` | 1,043,857 | `193f266279f20cfd816e3631a9225cf887d3424d337454a20f09d10916472bd0` | 139,255 |
| `783/consolidated_cell_types.csv.gz` | 901,707 | `8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d` | 138,327 |
| `783/cell_stats.csv.gz` | 2,526,548 | `bd5879e1b5df964bea2f3ca5316348d4276ce2ccaac283f0e36583c04fbd3d8e` | 139,246 (9 neurons lack stats; not a neuron-count source) |
| `783/connections.csv.gz` (= `connections_buhmann.csv.gz`, identical MD5) | 50,289,304 | `d49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9` | 3,869,878 rows; 2,700,513 pairs |
| `783/connections_princeton.csv.gz` | 68,456,801 | `445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b` | 5,342,446 rows; 3,732,460 pairs |
| `783/connections_no_threshold.csv.gz` | 212,093,967 | not downloaded | all pairs |
| `630/no_threshold_connections.csv.gz` | 116,904,594 | not downloaded (**no header row**) | all pairs |

Schema: `neurons.csv` = `root_id, group, nt_type ∈ {ACH,GABA,GLUT,DA,SER,OCT}, nt_type_score, da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg` (mean per-presynapse probabilities, Eckstein 2024, 6 classes; 9,906 all-zero rows in 783; no confidence floor applied; DA calls drop from 3,189 (630) to 584 (783), reason UNVERIFIED). `classification.csv` 783 = `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve` (types live in `fw_and_hemibrain_types.csv`); 630 also has `cell_type, hemibrain_type`. `super_class` vocabulary: optic, central, sensory, visual_projection, ascending, descending, sensory_ascending, visual_centrifugal, motor, endocrine.
Connections: `pre_root_id, post_root_id, neuropil (79 values incl. UNASGD), syn_count, nt_type` (presynaptic neuron's call). **One row per (pre, post, neuropil); the ≥5 filter is on the pair total** (per-row counts go to 1). Summing rows recovers exactly 2,613,129 (v630, Lin24) and 2,700,513 (v783, Dork24) pairs. Two synapse detections since July 2025: Buhmann 2021 (`connections.csv`, `connections_buhmann*`) and Princeton/Yu 2025 (`connections_princeton*`, ~48 % more synapses; Codex's live UI count 3,732,460 comes from it). Codex CSVs expose no per-synapse cleft scores; Shiu and Lin used cleft score ≥ 50 on Buhmann synapses.
Shiu model inputs vs Codex v630: 127,319 IDs in common; 659 Codex-only neurons have no ≥5 connection anywhere; 81 Shiu-only IDs are pre-freeze root IDs (UNVERIFIED); every Codex v630 pair appears in the Shiu file with identical weight.
Citations (from Codex's own citation table): always co-cite Dorkenwald et al. 2024 (10.1038/s41586-024-07558-y) and Schlegel et al. 2024 (10.1038/s41586-024-07686-5); plus Zheng 2018 (EM volume), Buhmann 2021 + Heinrich 2018 (synapses before July 2025) or Yu 2025 bioRxiv 10.1101/2025.07.11.664377 (Princeton synapses), Eckstein et al. 2024 (NT), Matsliah et al. 2024 (optic-lobe types), Lin et al. 2024 (connectivity tags), Deutsch 2025 (dsx/fru). For annotation versions ≥3.0.0 also Berg et al. 2026.
Licence: Nature papers and Zenodo deposits CC BY 4.0; the Codex bucket carries no LICENSE object; FlyWire ToS applies to accounts (text UNVERIFIED).

## 3. Shiu et al. 2024 model inputs (sim golden data)

Repo philshiu/Drosophila_brain_model, MIT. Raw URLs `https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/<file>`.

| File | Bytes | SHA-256 |
|---|---|---|
| `2023_03_23_connectivity_630_final.parquet` | 86,630,944 | `94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3` |
| `2023_03_23_completeness_630_final.csv` | 3,057,611 | `e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb` |
| `Connectivity_783.parquet` / `Completeness_783.csv` | 100,804,642 / 3,327,347 | not downloaded |
| `results/example/sugarR{,_100Hz,-<id>}.parquet` | 0.5–0.9 MB each | copied to `.cache/data/shiu/results/` |

Columns: `Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index, Connectivity, Excitatory (±1), Excitatory x Connectivity`. 127,400 neurons, 14,687,178 edges, min 1 synapse. Cite Shiu et al. 2024 (10.1038/s41586-024-07763-9). Raw model outputs (4.5 GB) on Edmond doi:10.17617/3.CZODIW.

## 4. hemibrain v1.2 / v1.2.1

- neuPrint `hemibrain:v1.2.1` only; `Meta`: totalPreCount 9,496,606, totalPostCount 64,139,744, **`postHighAccuracyThreshold` 0** (weights include low-confidence PSDs; `weightHP` uses 0.7), no NT properties on Neuron nodes.
- Bucket `gs://hemibrain/v1.2/` (public): `exported-traced-adjacencies-v1.2.tar.gz` 45,872,577 B, SHA-256 `07d8946eb0c4e3a5cb23d5769c9817847494f9fcadbc0ca239eed7bbd5555cf7` → `traced-neurons.csv` (21,739 non-cropped Traced; `bodyId, type, instance`), `traced-total-connections.csv` (3,550,403 edges, Σ 14,329,229, ≥5: 662,578), `traced-roi-connections.csv` (62 ROIs incl. `NotPrimary`); `hemibrain-v1.2-body-mean-neurotransmitters.feather` 45,591,786 B, SHA-256 `aab49d858415f559f469a9293adfb4d58e423db83a5debb24272ee4d66e059ad` (837,710 bodies; `gaba, acetylcholine, glutamate, serotonin, octopamine, dopamine, neither` probabilities summing to 1; `predicted_nt`; uploaded 2026-05-15, **provenance UNVERIFIED**, no README); also per-T-bar NT feather (443 MB bz2), all-traced adjacencies, synapse partners (859 MB), skeletons.
- v1.2 export vs v1.2.1 DB: 342 edges / 4,077 synapses differ ("minor fixes").
- Side/hemilineage are not in neuPrint; use Schlegel Supp. 5 or sjcabs `hemibrain_121_meta`.
- Licence CC-BY (Janelia page; version UNVERIFIED). Cite Scheffer et al. 2020, eLife 9:e57443 (10.7554/eLife.57443).
- Converter `flyconn.data.convert.hemibrain` (`flyconn data pull hemibrain@1.2.1`): universe and edges from the export tarball; `Supplemental_file5_hemibrain_meta.csv` pinned at `https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v3.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv` (2,634,367 B, SHA-256 `c9ca8421a5b6a5382348e06a7f62ef452bc78562983deff9ff479e30be2bd2a2`; same bytes on `main`); sjcabs `hemibrain_121_meta.feather` (2,023,858 B, SHA-256 `a2775c31450d3a060875bd5690adcb9800d3c9a580a9a919e10829984c79a1d1`, MD5 matches the bucket listing). The tarball is registered at level `meta` because it carries the neuron list. Results in GOLDEN_RESULTS 6e.

## 5. MANC v1.0 / v1.2.1

- neuPrint `manc:v1.0` (23,759 Traced today), `manc:v1.2.1` (23,665 Traced; 5,305,638 Traced→Traced edges, Σ 30,934,610 = Codex's unthresholded count), `manc:v1.2.3` (property update). `postHighAccuracyThreshold` 0.4.
- Flat export only for v1.0: `gs://flyem-manc-exports/v1.0/` (`manc-v1.0-neuron-properties.feather` 17,188,218 B, SHA-256 `0c4476528906bb0a20e05e1f01e83fc2e5a582761536171ede5463669ca0b891`, 102,369 bodies × 56 cols incl. `ntGabaProb, ntAcetylcholineProb, ntGlutamateProb, ntUnknownProb, predictedNt, predictedNtProb`; `traced-neurons.csv` 23,188; `traced-connections.csv`; neuPrint CSV/Feather dumps). v1.2 synapse partners: `gs://manc-seg-v1p2/manc-v1.2-synapse-partners-minconf-0.0.feather` (1.94 GB). Codex `gs://flywire-data/codex/data/manc/1.2.1/neurons.csv.gz` (23,665 rows, SHA-256 `ddeea1ff2859941d60a8f24933eb963d1f65d28d2399768c589a74bb90ac4a80`).
- NT classifier is **3-class** (ACh/GABA/Glu) + unknown/unclear; no monoamines. Sides `RHS/LHS`; hemilineages Truman `0A…27X`.
- Licence CC-BY. Cite Takemura et al. 2024 eLife 13:RP97769; Marin et al. 2024 eLife 13:RP97766; Cheong et al. 2025 eLife 13:RP96084.
- Converter `flyconn.data.convert.manc` (`flyconn data pull manc@1.2.1`) reads the sjcabs compiled files: `compiled_data/manc_121/manc_121_meta.feather` (1,446,722 B, SHA-256 `067c88e687ade67bd1e69b88765b4b21596280618cefde0fa43117d4af4a9439`; 23,650 rows = 23,665 Traced minus 15 glia) and `manc_121_simple_edgelist.feather` (87,386,906 B, SHA-256 `c3f9ca6dc9d100e72299b41136d373ffcd60b328624cbfd6ac145f071e0b28ac`, MD5 `7ko0TfSZiZ1EGPF2O8OaMA==` as listed; 5,305,354 rows `pre, post, count, norm, total_input`, string ids, Σ count 30,943,884; `total_input` = Σ count per post over traced partners), plus Codex `neurons.csv.gz` for NT confidence and `vfbId`. sjcabs data are CC BY by the dataset creators; cite the MANC papers. Results in GOLDEN_RESULTS 6e.

## 6. BANC v888

- Public bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome/` (README, CHANGELOG, `compiled_data/`, `neuron_annotations/v888/`, `neuron_connectivity/v888/`, `synapses/`, `nblast/`, meshes, skeletons); Harvard Dataverse doi:10.7910/DVN/7WTH1N v3.0 (2026-07-01, **CC BY 4.0**, 536 GB). CAVE `brain_and_nerve_cord_public` (token). Codex `banc/888/`.
- Paper: Bates AS, Phelps JS, Kim M, Yang HH, … Lee W-CA, Wilson RI. *Distributed control circuits across a brain-and-cord connectome.* Nature 656:957–970 (2026). doi:10.1038/s41586-026-10735-w. CC BY 4.0.
- `banc_888_meta.feather` 57,503,026 B, SHA-256 `86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098`, 188,508 × 81 (patched in place after publication: `root_890` present; pin MD5). Identifier columns are strings. `proofread` TRUE 150,952 + `roughly_proofread` 5,060 = 156,012 (paper: 155,916). `super_class` BANC vocabulary; `neurotransmitter_predicted` 8 classes (adds histamine, tyramine) with `neurotransmitter_score`; cross-refs `fafb_cell_type` (131,240), `manc_cell_type`, `malecns_cell_type` (43,490; v0.9), `hemibrain_cell_type`, `fanc_cell_type`, `*_match` IDs, `*_nblast_match`.
- `neuron_annotations/v888/codex_annotations.parquet` 74,350,281 B, SHA-256 `8a52f5f84f439603881bb360ea990601fd593072261c8cfb9afb0b3842f89eda`: long-form, 1,841,078 rows, 31 classification systems incl. `fafb_783_cell_type`, `fafb_783_match_id`, `manc_121_*`, `hemibrain_121_*`, `malecns_09_*`, `fanc_1116_*`.
- Edgelists (not downloaded): `compiled_data/banc_888/banc_888_edgelist_simple_v2.feather` 305 MB (11,510,975 rows; synapse detector v2, size ≥5, **no pair threshold**), `_v3` 359 MB (13,507,098; new detector, size ≥10). Columns `pre, post, count, norm, pre_count, post_count`. Per-synapse 8-NT probabilities: `synapses/v3.0/banc_nt_prediction_v3_w_sizethresh_10_05042026.parquet` 5.79 GB.
- Codex `banc/888/neurons.csv.gz` 2,881,728 B, SHA-256 `40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f`, 158,262 rows; Codex BANC default threshold 3.
- Caveats: both antennal nerves damaged; two synapse versions (paper v2, recommended v3); 22 known problem regions; left optic lobe less typed.

## 7. FANC

- Public exports `gs://lee-lab_female-adult-nerve-cord/CAVE/{v1237,v1444}/` (parquet with CAVE metadata; datastack `fanc_production_mar2021`): `cell_ids_v2.parquet` (21,978; `user_id` is the stable cell ID), `somas_dec2022`, `neck_connective`, `peripheral_nerves`, `proofread_first_pass` (7,246), `proofread_second_pass` (1,771), `synapses_nov2022_…connectioncounts_countthresh3.parquet` (1,878,659 pairs ≥3, Σ 11,801,582; most postsynaptic roots are unproofread fragments), per-neuropil variant (43 neuropils/tracts). SHA-256s in the research note.
- Live CAVE is restricted to community members (PI emails the Lee lab; no public applications). **No cell-type table in the public export**; FANC types reach us only via BANC (`fanc_1116_cell_type`, 2,905 neurons).
- Licence: none stated (UNVERIFIED). Cite Phelps et al. 2021 Cell; Azevedo et al. 2024 Nature 631:360 (10.1038/s41586-024-07389-x); Lesser et al. 2024 Nature 631:369 (10.1038/s41586-024-07600-z).

## 8. sjcabs compiled_data (harmonized vocabulary source)

Bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/{banc_888,fafb_783,manc_121,hemibrain_121,malecns_09,fanc_1116}/`; built by `flyconnectome/bancpipeline`; code MIT; data "CC-BY by the respective dataset creators"; asks to cite original papers + Eckstein 2024. Meta feathers on disk (SHA-256 in the note): fafb_783 144,837 × 24; hemibrain_121 25,397 × 19; manc_121 23,650 × 15; malecns_09 165,114 × 33; banc_888 188,508 × 81. Thirteen columns are common to all five: `region, hemilineage, nerve, flow, super_class, cell_class, cell_sub_class, cell_type, neurotransmitter_predicted, cell_function, cell_function_detailed, body_part_sensory, body_part_effector`. Edgelists carry `pre, post, count, norm[, total_input]` with **no pair threshold**. BANC-space files for other datasets are morphology only.

## 9. Harmonized schema alignment (seed for M1)

| flyconn field | sjcabs/BANC | Codex FAFB | MaleCNS v1.0 | hemibrain | MANC | Notes |
|---|---|---|---|---|---|---|
| `neuron_id` | `<ds>_id` (string) | `root_id` int64 | `bodyId` | `bodyId` | `bodyId` | store as uint64/string; keep source dtype in provenance |
| `flow` | `flow` | `flow` | derive from `superclass` | – | derive from `class` | afferent/intrinsic/efferent |
| `super_class` | BANC vocab | `central`, `optic`, … | `cb_/ol_/vnc_intrinsic`, … | Schlegel Supp. 5 / sjcabs | `class` | map to BANC vocab; keep `super_class_raw` |
| `cell_class`, `cell_sub_class` | long snake_case | `class`, `sub_class` | `class`, `subclass` | – | `subclass` | keep raw |
| `cell_type` | `cell_type` | `cell_type` + `hemibrain_type` (783: separate file) | `type` | `type` | `type`, `systematicType` | |
| `hemilineage` | ItoLee / Truman mixed | ItoLee | `itoleeHl`, `trumanHl` | Supp. 5 | Truman | two columns |
| `side` | left/right/center | `side` | `somaSide` L/R/M | Supp. 5 | `somaSide` LHS/RHS | normalise to left/right/center |
| `nt_pred`, `nt_conf`, `nt_p_*` | `neurotransmitter_predicted/_score` (string dtype in F/H!) | `nt_type`, `nt_type_score`, six `*_avg` | `predicted_nt`, `predicted_nt_confidence`, `consensus_nt` (no probs) | feather 6+neither probs | 3-class probs | record class set per source (3/6/7/8) |
| cross-refs | `fafb_cell_type`, `manc_cell_type`, `hemibrain_cell_type`, `malecns_cell_type`, `fanc_cell_type`, `*_match` | `hemibrain_type` | `flywireType`, `hemibrainType`, `mancType`, `mancBodyid` | – | `synonyms` | plus `vfb_id`/`fbbt_id` as neutral keys |
| edges | `pre, post, count, norm` (no threshold) | `pre_root_id, post_root_id, neuropil, syn_count` (pair ≥5, per-neuropil rows) | `body_pre, body_post, weight` (no ROI) | `bodyId_pre, bodyId_post, weight` (+ROI file) | neuPrint `weight` | store `threshold` and `synapse_source` metadata per edge table |

## 10. Open items

CAVE materialization lists (token); hemibrain NT feather provenance; FlyWire ToS text and the 630→783 DA-call drop; the 81/659 Shiu–Codex ID differences; FANC licence; MANC v1.2.x flat exports; MaleCNS NT "unclear" thresholds and the 11,751 vs 11,710 type delta (Cell Methods).
