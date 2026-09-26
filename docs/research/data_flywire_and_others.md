# FlyWire/FAFB, hemibrain, MANC, FANC, BANC and the sjcabs harmonized files — dataset verification notes

Scope: everything except MaleCNS (see `data_malecns.md`; MaleCNS appears here only where the sjcabs/BANC harmonized files carry it).
Verified on 2026-09-26. Rule: every number below was either computed on a file that is on disk under
`/Users/moizp/google-fly/.cache/data/` (the previous session downloaded them; nothing was re-downloaded except two small
Codex `neurons.csv.gz` files noted in §5.4/§3.5), read from a document that is on disk, or fetched live with `curl`/`requests`
(GCS JSON listings, `HEAD` requests, tokenless neuPrint Cypher, Zenodo/Dataverse/GitHub APIs). Anything not verified is marked
**UNVERIFIED**. Python: `/private/tmp/claude-502/-Users-moizp-google-fly/bdaccf09-812b-4c61-9717-b57ad8a87724/scratchpad/research-venv`
(pandas 3.0.6, pyarrow). `gsutil` is not installed; all bucket access used `https://storage.googleapis.com/storage/v1/b/<bucket>/o?prefix=…`
(listing) and `https://storage.googleapis.com/<bucket>/<object>` (download). Inspection script:
`scratchpad/inspect_all.py` (outputs `scratchpad/out_{fanc,banc,sjcabs,hemibrain,manc,flywire}.txt`); listing helper `scratchpad/gcs_ls2.py`;
checksums `scratchpad/sha256_all.txt`. SHA-256 values below are from `shasum -a 256`; MD5 (base64) values were compared with the
`md5Hash` returned by the GCS listing and **all matched** for every file checked (hemibrain, MANC, BANC, sjcabs, FANC, Codex).

Contents: §0 executive answers · §1 FlyWire/FAFB (Codex + Schlegel annotations + Shiu) · §2 hemibrain v1.2/v1.2.1 · §3 MANC · §4 FANC ·
§5 BANC · §6 sjcabs harmonized files + schema alignment table · §7 cross-dataset cell-type vocabulary · §8 open items.

---

## 0. Executive answers to the specific questions

1. **FlyWire latest public snapshot = v783, still, as of 2026-09-26.** Evidence: (a) Codex FAQ on disk (`codex_faq.html`, fetched
   this session): "FlyWire FAFB … default snapshot 783; available in Codex: v783 - Oct 2023 [latest release]"; (b) the public bucket
   `gs://flywire-data/codex/data/fafb/` has exactly two version prefixes, `630/` and `783/` (live listing today and the 2 GB listing on
   disk); the newest object under `783/` is dated 2026-02-24 (annotation refreshes, not a new materialization); (c) `flywire_annotations`
   v3.1.0 (2026-07-21) is "still based on the 783 materialization"; (d) Zenodo 10676866 (FlyWire Consortium): "Currently, the latest
   release is version 783". CAVE's materialization list could not be read without a token (see §1.6) — **UNVERIFIED** whether
   `flywire_fafb_public` has newer *materializations* beyond 783 (fafbseg-py has historically exposed 630 and 783 as the public ones).
2. **No login is needed for the bulk files.** All Codex CSVs live in a public GCS bucket and every URL tested returns HTTP 200 with no
   credentials: `https://storage.googleapis.com/flywire-data/codex/data/fafb/{630|783}/<file>` (§1.2). The Codex *web* download page and
   its `api/download_resource` endpoint require Google sign-in + an `api_token`, but they are just a front for the same bucket. Login
   is needed for: CAVE (`flywire_fafb_public` — token from https://global.daf-apis.com/auth/api/v1/create_token, terms acceptance on first
   FlyWire login), neuPrint (token; hemibrain/MANC) — but tokenless raw Cypher `POST /api/custom/custom` currently works (verified again
   today for hemibrain:v1.2.1, manc:v1.0, manc:v1.2.1, manc:v1.2.3), FANC CAVE (community membership; PI must email W.-C. Lee), BANC
   CAVE (`brain_and_nerve_cord_public` datastack exists but the CAVE info endpoint redirected to login today).
3. **Key counts** (computed): FlyWire v630 127,978 neurons / 3,794,615 connection rows / 2,613,129 (pre,post) pairs ≥5 syn;
   v783 139,255 / 3,869,878 rows / 2,700,513 pairs (Buhmann synapses) and 5,342,446 rows / 3,732,460 pairs (Princeton synapses);
   hemibrain v1.2 export 21,739 traced non-cropped neurons / 3,550,403 edges / 14,329,229 synapses; MANC v1.0 23,188 traced neurons
   (neuPrint today: v1.0 23,759 Traced, v1.2.1 23,665 Traced, 5,305,638 Traced→Traced edges); FANC v1444 public parquet 21,978 cell IDs,
   1,878,659 edges ≥3 syn; BANC v888 meta 188,508 rows (150,952 `proofread`, 5,060 `roughly_proofread`), Codex BANC 158,262 neurons.
4. **Harmonized schema insight (§6.4):** the sjcabs/BANC "compiled_data" files already define a working cross-dataset vocabulary
   (`root_id | region | side | hemilineage | nerve | flow | super_class | cell_class | cell_sub_class | cell_type | neurotransmitter_predicted
   | neurotransmitter_score | cell_function | body_part_*`). Codex's `classification.csv` uses the same four-level hierarchy under
   different names (`super_class/class/sub_class/cell_type`) but a *different* `super_class` vocabulary (`optic`, `central` vs
   `optic_lobe_intrinsic`, `central_brain_intrinsic`), and MaleCNS uses `superclass/class/subclass/type` with yet another vocabulary
   (`ol_intrinsic`, `cb_intrinsic`, `vnc_intrinsic`). The cleanest seed for flyconn is the sjcabs column set plus explicit `*_cell_type` /
   `*_match_id` cross-reference columns and a `dataset` column.
5. **Surprises:** (i) Codex's `783/connections.csv.gz` is byte-identical (MD5 `QSBkQDGMd0GL/Hz/HLPg+g==`) to `connections_buhmann.csv.gz`
   — since July 2025 Codex ships two synapse detections (Buhmann 2021 vs "Princeton"/Yu 2025) and the neutral file name currently means
   *Buhmann*; (ii) the Shiu et al. model's 127,400 neurons are a strict subset (minus 81 IDs) of Codex v630's 127,978 and every one of
   Codex's 2,613,129 ≥5-synapse pairs is present with identical weight in the Shiu file (§1.7); (iii) `gs://hemibrain/v1.2/` now hosts
   several 2025–2026 additions (body-mean and per-T-bar NT predictions, all-traced adjacencies, skeletons) that are not linked from the
   Janelia page; (iv) the on-disk `banc_888_meta.feather` (188,508 rows × 81 cols, 2026-08-21) is newer than its own documentation
   (188,162 × 79) and gained `root_890`; (v) MANC v1.2.1 has **no** flat-file export bucket — `gs://flyem-manc-exports/` only has `v1.0/`;
   v1.2 synapse partners live in `gs://manc-seg-v1p2/`; (vi) Codex now hosts public data directories for `banc/{305…888}`, `manc/1.2.1`,
   `maol/1.1`, `mcns/{0.9,1.0}` in the same bucket, and Codex's default connection thresholds differ per dataset (FAFB 5, BANC 3, MANC 1,
   MAOL 1, MCNS 5).

---

## 1. FlyWire / FAFB (Codex snapshots 630 and 783; flywire_annotations; Shiu model files)

### 1.1 Versions and release dates

| Item | Value | Source |
|---|---|---|
| Public snapshots | **630** (June 2023 public release; basis of the 2023 bioRxiv papers, Lin et al. 2024 and the Shiu model) and **783** (Oct 2023; basis of Dorkenwald 2024 / Schlegel 2024 Nature papers; current default) | Codex FAQ: "default snapshot 783; available in Codex: v783 - Oct 2023 [latest release]"; `flywire_annotations` README changelog (1.x = 630, 2.x/3.x = 783) |
| Latest as of 2026-09-26 | v783. Bucket `gs://flywire-data/codex/data/fafb/` prefixes: `630/`, `783/` only (live). | live GCS listing |
| Annotation releases on top of 783 | `flywire_annotations` v1.0.0 2023-06-27, v1.1.0 2023-07-16 (630-based); v2.0.0 2024-01-10, v2.1.0 2024-07-30 (Nature papers); v3.0.0 2025-10-09 (MaleCNS v0.9 cross-validation, adds `dimorphism`, `fru_dsx`, `synonyms`, `supertype`, drops `morphology_group`); **v3.1.0 2026-07-21** (MaleCNS v1.0 / Berg et al. Cell; JO neuron retyping). Latest commit 8587524c 2026-07-21. | GitHub API today |
| Codex file refresh dates (783) | `classification.csv.gz`, `consolidated_cell_types.csv.gz`, `visual_neuron_types.csv.gz`, `column_assignment.csv.gz` 2026-02-24; `processed_labels.csv.gz` 2026-01-07; `labels.csv.gz` 2025-06-23; `connections_princeton*.csv.gz`, `connections_buhmann*.csv.gz`, `connections_princeton_no_threshold`, `connections_buhmann_no_threshold` 2025-07-08; `fafb_v783_princeton_synapse_table.csv.gz` 2025-07-24; `neurons.csv.gz` 2024-12-05 | bucket listing (`updated`) |
| Codex sister datasets in the same bucket | `codex/data/banc/{305,470,484,514,537,554,587,610,626,850,888}/`, `codex/data/manc/1.2.1/`, `codex/data/maol/1.1/`, `codex/data/mcns/{0.9,1.0}/`; plus `codex/synapses/fafb/630/individual/<root>_{in,out}.csv.gz` (1,097,565 objects) and `codex/skeletons/{fafb,banc,mcns}/…` (280,559 / 263,190 / 211,760 objects) | 2 GB listing on disk + live listing |

### 1.2 Access

- **Public GCS bucket (no auth):** `https://storage.googleapis.com/flywire-data/codex/data/fafb/<version>/<file>` — this is exactly the
  template hard-coded in Codex's own loader (`codex/data/local_data_loader.py`, copy in scratchpad:
  `GCS_RAW_DATA_URL_TEMPLATE = "https://storage.googleapis.com/flywire-data/codex/data/fafb/{version}/{filename}"`). The previous session
  downloaded every file this way (its log lines read `DOWNLOADED 783/processed_labels.csv.gz HTTP 200 …`). Verified today:

  ```bash
  curl -sI https://storage.googleapis.com/flywire-data/codex/data/fafb/783/neurons.csv.gz          # HTTP/2 200, 1,679,884 bytes
  curl -sI https://storage.googleapis.com/flywire-data/codex/data/fafb/630/connections.csv.gz      # 200, 26,124,953 bytes
  curl -sI https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_no_threshold.csv.gz            # 200, 212,093,967
  curl -sI https://storage.googleapis.com/flywire-data/codex/data/fafb/630/no_threshold_connections.csv.gz            # 200, 116,904,594
  curl -sI https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections_princeton_no_threshold.csv.gz  # 200, 275,679,780
  curl -sI https://storage.googleapis.com/flywire-data/codex/data/fafb/783/fafb_v783_princeton_synapse_table.csv.gz   # 200, 2,695,106,039
  ```
- **Codex web download page** (`https://codex.flywire.ai/api/download?dataset=fafb`, page saved as `codex_download_page.html`): requires
  Google sign-in ("By signing in, you agree to the FlyWire Terms of Service & Privacy Notice"; "access to its apps requires signing in").
  FAQ: programmatic pattern is `https://codex.flywire.ai/api/download_resource?data_product=...&dataset=fafb&api_token=...` with a token
  from the Account page; "Codex intentionally does not provide a general programmatic live-query API". `GET api/download?dataset=fafb`
  without a token returned HTTP 200 with a 2-byte body today.
- **CAVE (`flywire_fafb_public`)**, for live queries/materializations: token from https://global.daf-apis.com/auth/api/v1/create_token
  (Google account that has accepted FlyWire terms); `caveclient.CAVEclient("flywire_fafb_public", auth_token=...)` or
  `client.auth.save_token(token=..., overwrite=True)` (stored in `~/.cloudvolume/secrets/cave-secret.json`). Today
  `https://global.daf-apis.com/info/api/v2/datastack/full/flywire_fafb_public` and
  `https://prod.flywire-daf.com/materialize/api/v2/datastack/flywire_fafb_public/versions` both **302-redirect to
  `sticky_auth/api/v1/authorize`**, i.e. even the info service needs a token now. CAVEclient latest release v8.2.1 (2026-07-16). **UNVERIFIED**
  (not tested with a token): the exact list of public materialization versions.
- **Zenodo (CC BY 4.0, no login):** 10.5281/zenodo.10676866 "FlyWire Whole-brain Connectome Connectivity Data", version 783.0,
  2024-06-02, creator "FlyWire Consortium": `flywire_synapses_783.feather` 9,492,998,242 B (md5 f8f1b97c…), `proofread_connections_783.feather`
  852,022,274 B, `proofread_root_ids_783.npy` 1,114,168 B, `per_neuron_neuropil_count_{pre,post}_783.feather` (16.9 MB / 233.8 MB).
  10.5281/zenodo.10877326 (Schlegel & Jefferis, 2024-03-27, CC BY 4.0): `sk_lod1_783_healed_ds2.parquet` 5.36 GB skeletons,
  `nblast_flywire_all_right_aba_comp.feather` 809 MB, `nblast_flywire_hemibrain_min_comp.feather` 212 MB, `nblast_flywire_mirrored_hemibrain_min_comp.feather` 223 MB.
- **GitHub:** `flyconnectome/flywire_annotations` (supplemental TSVs; fafbseg-py reads `Supplemental_file1_neuron_annotations.tsv` from a
  pinned commit — `FLYWIRE_ANNOT_URL` in `fafbseg/flywire/annotations.py`). fafbseg-py latest v3.2.2 (2026-02-20).

### 1.3 Files on disk (`.cache/data/flywire/`), sizes and SHA-256

| File | Bytes | SHA-256 | Rows (computed) |
|---|---|---|---|
| `630/neurons.csv.gz` | 1,572,661 | a0be4670993d86cea0a4724ba24f874c9d867a24f4e248ff30dcf9b6565d81bb | 127,978 |
| `630/classification.csv.gz` | 1,011,550 | b2cf34ef76f5750e647334b56c73ae7f49c96f0fee3371e8a52d26c835a3b499 | 127,979 (127,979 unique; 1 id not in neurons.csv) |
| `630/cell_stats.csv.gz` | 2,415,308 | 58d2697f9760ecb40cd73f9cbd81b8640d7fafb91ed71aa1e3a2a4880ff4b897 | 127,978 |
| `630/connections.csv.gz` | 26,124,953 | 68144490b8585fb6b1bf5979d923a3b1081b435cd52b7f82daed1beb6b84781a | 3,794,615 |
| `630/names.csv.gz` | 1,080,880 | bd4aa2bedbc72db90550de6019f5208b2bbe1ad49c1db503f20abc9b19e38611 | 127,978 |
| `630/labels.csv.gz` | 4,401,980 | 8caac8ad7d4152a467690d6c7e12cef039af50d59ef06f1853077bf5f319279d | 150,182 (100,343 root_ids) |
| `630/coordinates.csv.gz` | 4,584,155 | d2c874401be8a4c68b23b461107ede9d3ae3e32f4e993ed2fa06b04cc7996038 | 207,700 (131,459 root_ids) |
| `630/connectivity_tags.csv.gz` | 653,687 | ddfd1290d5ba39b7480f14e64ab7bf97d884483d3e54d9603e8e96f1355a1b03 | 117,334 |
| `630/connectivity_clusters.csv.gz` | 522,912 | 12a92e154491a83f6670a96cacac517a98f8e26c389f53381b4bb6f2cf89d6b1 | 87,635 |
| `630/morphology_clusters.csv.gz` | 736,744 | 6def5fa17b2bea2bffabe21a7d6e0eb521aa7bc939f7278485b6b16f8ff96cfc | 124,988 |
| `630/lr_matching.csv.gz` | 265,040 | 126d2467b256676b0ac195e8db8191254b308b4e96bee1c8fb5f80ee9c307752 | 27,011 |
| `630/dsx_fru_types.csv.gz` | 16,625 | e7b54aa082611f3ab874ee2ead8d582f65a6956655bcc2ac6674f80ed10662f2 | 2,712 |
| `630/visual_neuron_types.csv.gz` | 191,976 | d194ef491421d7fa147fd6e42d3da2b3de2d34b9539d426231641d4555512708 | 37,345 |
| `630/neuropil_synapse_table.csv.gz` | 4,490,002 | 050fbbd3adee366039ca1814367716b402adbc04b139fd2456ab58154be797d7 | 124,891 |
| `783/neurons.csv.gz` | 1,679,884 | 6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719 | 139,255 |
| `783/classification.csv.gz` | 934,402 | e946b552f4056dfc977707be0674609832c3f64332a22d69dc0d9615e7aae663 | 139,255 |
| `783/classification_with_fw_and_hemibrain_types.csv.gz` | 1,362,939 | e2e8e892ca4b8498e25e22c7625f3e8486751a530d0cbdd34203df2f8cbb00ff | 139,255 |
| `783/fw_and_hemibrain_types.csv.gz` | 1,043,857 | 193f266279f20cfd816e3631a9225cf887d3424d337454a20f09d10916472bd0 | 139,255 |
| `783/consolidated_cell_types.csv.gz` | 901,707 | 8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d | 138,327 |
| `783/cell_stats.csv.gz` | 2,526,548 | bd5879e1b5df964bea2f3ca5316348d4276ce2ccaac283f0e36583c04fbd3d8e | 139,246 |
| `783/connections.csv.gz` | 50,289,304 | d49dd692e59e153aa3c83f5257bfc0eff51247b86d7bb183386c6d1622c70fc9 | 3,869,878 |
| `783/connections_princeton.csv.gz` | 68,456,801 | 445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b | 5,342,446 |
| `783/names.csv.gz` | 1,181,576 | e541ef9ef4b9e62d798f165ae76853d15b84174cf1dce95392f313a491055332 | 139,255 |
| `783/labels.csv.gz` | 4,771,292 | bdd4eafab2bfe30540256c84ea1513e4b1877c0c4cf03f919204b4eafae5868e | 160,045 (110,038 root_ids) |
| `783/processed_labels.csv.gz` | 1,017,658 | 9feab030bd7f0f9f9909a56f4ed37f019fc19cc076180b494f3bff55bf1480d2 | 100,091 |
| `783/coordinates.csv.gz` | 5,314,546 | 14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05 | 238,909 (139,255 root_ids) |
| `783/connectivity_tags.csv.gz` | 637,719 | 68c69cec13810fa543c600a5b9973d10718b449740e010834662be1dc1b8696c | 134,437 |
| `783/column_assignment.csv.gz` | 462,838 | bdf4ce7f62cc63493d53eefad3816ff2dfd08b190e97b35a492e0e453df2f0f6 | 45,528 |
| `783/visual_neuron_types.csv.gz` | 631,701 | 4bcc6a2f98b86e6c3fb7eaddb49736f3d81ab65bda35da8f740641201a1e379f | 95,079 |
| `783/dsx_fru_types.csv.gz` | 8,981 | 2c338f126523620c713f4d47881b23ba4ef02786eea6b2e71baebbb02782a45e | 1,407 |
| `783/neuropil_synapse_table.csv.gz` | 4,674,663 | e525bdea7bc2fe585cf8ab8f9fc76bea5e29f2c3e5240588b6add518bcda5ab1 | 134,181 |
| `783/neuropil_stats.csv` | 7,884 | 0c23d57e1f255ed33df1e0d6e4ef4122b174541b89e357d7ee4568b1670bb661 | 78 |
| `783/synapse_attachment_rates.csv.gz` | 3,257 | 2e412c5714c21aa4f6ebff74801f620d161594740aaee25c73f5bc18b0110447 | 162 |
| `783/cross_version_consistent_names.csv` | 4,158,624 | 8a81f8dd234369056ca1ced16ac6d80721d098dfa37d466564c2941a8ce16d25 | 139,254 + header-less first row (no header: cols are `root_id,name`) |
| `783/extended_columns.csv` | 88,433 | 2ab9e75ab0a7ab26cea414a81f1f495db0a2335dc545fc82a1ec1e2041b68e7b | 3,705 (+1; no header) |
| `flywire_annotations/Supplemental_file1_neuron_annotations.tsv` | 31,718,505 | 9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be | 139,248 |
| `flywire_annotations/Supplemental_file5_hemibrain_meta.csv` | 2,634,367 | c9ca8421a5b6a5382348e06a7f62ef452bc78562983deff9ff479e30be2bd2a2 | 25,397 |
| `flywire_annotations/main_README.md` / `supplemental_README.md` | 16,464 / 7,525 | c19f7697… / 49e884a1… | — |
| `flywire_citation_table.csv` | 1,367 | bab718dcb26d2eafb14a9f266acc0064cbdfe4bc6cc97f146b44aedb8799bbf8 | — |
| `flywire-data_bucket_listing.json` | 2,046,065,874 | (not hashed) | 1,853,229 objects, all under `codex/` |

Not downloaded but present in the bucket (sizes from the listing/HEAD): `783/connections_buhmann.csv.gz` 50,289,304 (MD5 identical to
`783/connections.csv.gz`), `783/connections_buhmann_5_ol_2.csv.gz` 86,393,577, `783/connections_buhmann_no_threshold.csv.gz` 212,093,967
(MD5 identical to `783/connections_no_threshold.csv.gz`), `783/connections_princeton_5_ol_2.csv.gz` 113,182,013,
`783/connections_princeton_no_threshold.csv.gz` 275,679,780, `783/connections_new_syns_5{,_ol_2}.csv.gz` (older names of the Princeton
tables, 2025-02-20), `783/fafb_v783_princeton_synapse_table.csv.gz` 2,695,106,039, `783/synapse_coordinates.csv.gz`, `783/nblast.csv.gz`,
`783/flywire_neurons_VFB.csv` 14,704,621, `630/no_threshold_connections.csv.gz` 116,904,594, `630/synapse_coordinates.csv.gz`,
`630/nblast.csv.gz`, `630/svd*.csv.gz`, and dated `neuron_db_schema_*.pickle.gz` Codex pickles (not for external use).

### 1.4 Schemas (from `schema_inspection*.txt` + re-checks)

**`neurons.csv.gz` (630 and 783, identical layout):** `root_id` int64 · `group` str (auto-name group, e.g. `ME.LO`) · `nt_type` str ∈
{ACH, GABA, GLUT, DA, SER, OCT} or empty · `nt_type_score` float · `da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg` float =
mean per-transmitter probability over the neuron's presynapses (Eckstein et al. 2024 classifier, 6 classes). Checks: 783 — rows whose six
averages sum to 1 ± 0.02: 127,617; all-zero rows 9,906; `nt_type_score == max(avg)` for 129,503 rows; `nt_type` empty for 19,658
(630: 14,867); minimum non-null score 0.28 (630: 0.29), i.e. no confidence floor is applied by Codex. 783 `nt_type` counts: ACH 82,298,
GLUT 19,605, GABA 16,017, SER 1,021, DA 584, OCT 72. 630: ACH 74,927, GLUT 18,476, GABA 15,502, DA 3,189, SER 957, OCT 60 (note DA drops
from 3,189 to 584 between snapshots — Codex re-derived NT calls for 783; **UNVERIFIED** why).

**`classification.csv.gz`:** 630 = `root_id, flow, super_class, class, sub_class, cell_type, hemibrain_type, hemilineage, side, nerve`;
783 = `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve` (types moved to `fw_and_hemibrain_types.csv.gz` /
`consolidated_cell_types.csv.gz`; `classification_with_fw_and_hemibrain_types.csv.gz` is the 10-column join, refreshed 2025-05-15).
Vocabularies (783): `flow` intrinsic 118,464 / afferent 19,300 / efferent 1,491; `super_class` optic 77,873, central 32,381, sensory
16,938, visual_projection 7,684, ascending 1,750, descending 1,305, sensory_ascending 612, visual_centrifugal 522, motor 110, endocrine 80;
`side` left 69,959 / right 69,093 / center 173 / empty 30; `nerve` ∈ {CV, AN, MxLbN, OCN, PhN, aPhN, NCC, ON}; `class` 30 values (optic_lobe_intrinsic
77,382, visual 11,426, Kenyon_Cell 5,177, CX 2,878, mechanosensory 2,674, olfactory 2,281, AN 2,276, ALPN 685, …); `sub_class` values are
Codex-specific (e.g. `t4_neuron`, `distal_medulla`, `SA_DMT_DMetaN`, `KCg`). 630 `super_class` used `sensory` 9,708 / `ascending` 2,364 and
no `sensory_ascending`.

**`fw_and_hemibrain_types.csv.gz` (783):** `cell_type` (FlyWire/Schlegel type) non-null 111,090 (5,634 distinct), `hemibrain_type`
non-null 29,764 (3,551 distinct), both 9,043, neither 7,444. **`consolidated_cell_types.csv.gz` (783):** `root_id, primary_type,
additional_type(s)` — 138,327 rows, 8,772 distinct `primary_type`, 13,956 rows with additional types; joined to Supplemental_file1,
`primary_type == coalesce(cell_type, hemibrain_type)` for 133,339 of 138,318 (Codex "consolidates" from several sources, per FAQ).
**`visual_neuron_types.csv.gz` (783):** `root_id, type, family, subsystem, category, side` (Matsliah 2024; 95,079 rows).
**`column_assignment.csv.gz`:** `root_id, hemisphere, type, column_id, x, y, p, q` (hex column coordinates). **`dsx_fru_types.csv.gz`:**
`root_id, type_name, gene` (Deutsch 2025). **`cell_stats.csv.gz`:** `length_nm, area_nm, size_nm` (int). **`coordinates.csv.gz`:**
`position` "[x y z]" in 4×4×40 nm voxels + `supervoxel_id`. **`labels.csv.gz`:** community labels with `user_id, user_name,
user_affiliation, date_created, label_id`. **`connectivity_tags.csv.gz`:** comma list (rich_club, reciprocal, 3_cycle_participant,
feedforward_loop_participant, nsrn, attractor, repeller, …; Lin et al. 2024). **`neuropil_synapse_table.csv.gz`:** per-neuron
`input/output synapses|partners in <NEUROPIL>` for 78 neuropils (+`UNASGD`). **`neuropil_stats.csv` (783):** `neuropil, Brain compartment,
syn_count_total, syn_count_proof_post, syn_count_proof_pre, syn_proof_ratio_post, syn_proof_ratio_pre, vol_µm3, side` — e.g. ME_R
16,718,114 synapses, post-attachment ratio 0.565, pre 0.927; GNG 7,307,494 / 0.384 / 0.964.

**Connections tables** (`connections.csv.gz`, `connections_princeton.csv.gz`, `connections*_no_threshold.csv.gz`): columns
`pre_root_id, post_root_id, neuropil, syn_count, nt_type`; one row per (pre, post, **neuropil**) — a pair appears in several rows if its
synapses span neuropils, so `weight(pair) = sum(syn_count)` over rows. `neuropil` ∈ 79 values (78 L/R/central neuropils + `UNASGD`).
`nt_type` is the *presynaptic neuron's* Codex NT call repeated on every row (values ACH/GABA/GLUT/DA/SER/OCT; the `connections_new.csv.gz`
head shows lowercase `ach` — an older variant). **Threshold semantics verified:** in all three thresholded files the *pair total* is ≥5 while
individual rows go down to 1 (630: 1,174,020 rows < 5; min pair total 5 in 630, 783 and princeton) — i.e. the ≥5 filter is applied per
(pre,post) pair *before* splitting by neuropil, matching Dorkenwald 2024 ("we use a threshold of five synapses to determine a connection").
`*_no_threshold*` files keep every pair (630 file has **no header row**; 783 files have one). `*_5_ol_2*` = ≥5 in central brain but ≥2 inside
optic lobes (Matsliah 2024's optic-lobe threshold). Codex FAQ default minimums: FAFB 5, BANC 3, MANC 1, MAOL 1, MCNS 5.

| Table | Rows | Pairs | Σ syn_count | unique pre / post | max syn_count | NT rows ACH/GABA/GLUT/DA/SER/OCT |
|---|---|---|---|---|---|---|
| 630 `connections` | 3,794,615 | 2,613,129 | 32,970,606 | 121,327 / 119,687 | 2,358 | 2,178,231 / 850,741 / 641,306 / 70,042 / 37,380 / 16,915 |
| 783 `connections` (= Buhmann) | 3,869,878 | 2,700,513 | 34,153,566 | 129,351 / 125,192 | 2,405 | 2,258,155 / 865,318 / 654,183 / 37,705 / 37,450 / 17,067 |
| 783 `connections_princeton` | 5,342,446 | 3,732,460 | 50,666,648 | 137,518 / 130,183 | 2,633 | 3,210,049 / 1,172,932 / 826,380 / 63,704 / 40,396 / 28,985 |

2,700,513 matches Dorkenwald 2024's count; 3,732,460 is the number Codex's home page now shows for "FAFB v783 … 3,732,460 connections",
i.e. **Codex's live UI uses the Princeton synapse table** while the neutral CSV name still points to Buhmann. Princeton synapses
(Yu et al. 2025, bioRxiv 10.1101/2025.07.11.664377) cover ~48% more synapses than Buhmann; `fafb_v783_princeton_synapse_table.csv.gz`
columns: `pre_x,pre_y,pre_z,ctr_x,ctr_y,ctr_z,post_x,post_y,post_z,size,pre_root_id_720575940,post_root_id_720575940,neuropil` (root IDs
stored *minus the common prefix 720575940*). The Buhmann synapse table (Zenodo `flywire_synapses_783.feather`) carries the
`cleft_score`; Dorkenwald 2024 Methods: synapses were filtered with the Heinrich cleft segmentation, and Shiu/Lin used **cleft score ≥ 50**
(Shiu Methods: "cleft score cutoff of 50"). Codex CSVs do not expose per-synapse scores.

```python
import pandas as pd

c = pd.read_csv(".cache/data/flywire/783/connections.csv.gz")  # 3,869,878 rows
g = c.groupby(["pre_root_id", "post_root_id"]).syn_count.sum()  # 2,700,513 pairs, min 5
assert g.min() == 5 and (c.syn_count < 5).sum() > 0  # per-row counts can be < 5
```

### 1.5 Neuron counts, computed vs stated

| Snapshot | Stated | Computed | Notes |
|---|---|---|---|
| v630 | 127,978 (Lin et al. 2024 Methods) | `neurons.csv.gz` 127,978; `cell_stats` 127,978; `classification` 127,979 (one extra id); `names` 127,978; 124,891 neurons take part in a ≥5 connection; 3,087 have none | exact |
| v783 | 139,255 (Dorkenwald 2024; Codex tile "139,255 neurons") | `neurons.csv.gz` 139,255; `classification` 139,255; `cell_stats` 139,246; `consolidated_cell_types` 138,327; Supplemental_file1 139,248 (139,241 in common with Codex 783) | exact; 106,785 root_ids shared between 630 and 783 |

### 1.6 Neurotransmitter, threshold and citation summary for FlyWire

- NT prediction = Eckstein, Bates et al. 2024 (6 classes, per synapse; Codex averages per neuron → `*_avg`, `nt_type`, `nt_type_score`; Supplemental_file1 gives `top_nt`, `top_nt_conf`, plus `known_nt`/`known_nt_source` literature ground truth for 87,837 neurons).
- Codex citation table (`flywire_citation_table.csv`, verbatim header): "When using the FlyWire resource, please co-cite the Dorkenwald et al. and Schlegel et al. manuscripts. To give credit for specific aspects of data creation (=columns) please select the appropriate citations":
  Dorkenwald 2024 10.1038/s41586-024-07558-y (reconstruction, community/hierarchical annotations, nerves, hemilineages, cell_type, hemibrain matching, synapses & connectivity, NT); Schlegel 2024 10.1038/s41586-024-07686-5 (reconstruction, hierarchical annotations, nerves, hemilineages, cell_type, hemibrain matching); Zheng 2018 10.1016/j.cell.2018.06.019 (EM volume); **Yu 2025** bioRxiv 10.1101/2025.07.11.664377 (synapses & connectivity "from Jul. 2025"); Buhmann 2021 10.1038/s41592-021-01183-7 and Heinrich 2018 10.1007/978-3-030-00934-2_36 (synapses "before Jul 2025"); Eckstein, Bates 2024 10.1016/j.cell.2024.03.016 (NT); Matsliah, Yu 2024 10.1038/s41586-024-07981-1 (optic-lobe cell_type); Lin 2024 10.1038/s41586-024-07968-y (connectivity tags); Deutsch 2025 bioRxiv 10.1101/2025.06.10.658788 (gene/dsx-fru annotations).
- `flywire_annotations` README: for annotation versions ≥3.0.0 cite Berg et al. (bioRxiv 10.1101/2025.10.09.680999; now Cell 10.1016/j.cell.2026.08.015), Schlegel 2024, Matsliah 2024, Dorkenwald 2024.
- License: both Nature papers are CC BY 4.0; the Zenodo connectivity/skeleton deposits are CC BY 4.0. The Codex bucket carries no LICENSE object; FlyWire's site-level "Terms of Service & Privacy Notice" applies to Codex/CAVE accounts. **UNVERIFIED**: the exact FlyWire ToS text (Cloudflare/JS page not fetched).
- Full references: Dorkenwald S, Matsliah A, Sterling AR, Schlegel P, Yu S-C, McKellar CE, … Murthy M, Seung HS & the FlyWire Consortium. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024). doi:10.1038/s41586-024-07558-y. — Schlegel P, Yin Y, Bates AS, Dorkenwald S, Eichler K, … Bock DD, Jefferis GSXE. Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152 (2024). doi:10.1038/s41586-024-07686-5. — Lin A, Yang R, Dorkenwald S, … Murthy M. Network statistics of the whole-brain connectome of *Drosophila*. *Nature* 634, 153–165 (2024). doi:10.1038/s41586-024-07968-y. — Matsliah A, Yu S-C, Kruk K, … Jefferis GSXE. Neuronal parts list and wiring diagram for a visual system. *Nature* 634, 166–180 (2024). doi:10.1038/s41586-024-07981-1. — Eckstein N, Bates AS, Champion A, … Funke J. Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187, 2574–2594 (2024). doi:10.1016/j.cell.2024.03.016. — Buhmann J et al. Automatic detection of synaptic partners in a whole-brain *Drosophila* electron microscopy data set. *Nat Methods* 18, 771–774 (2021). doi:10.1038/s41592-021-01183-7. — Zheng Z et al. A complete electron microscopy volume of the brain of adult *Drosophila melanogaster*. *Cell* 174, 730–743 (2018). doi:10.1016/j.cell.2018.06.019.

### 1.7 What the Shiu et al. 2024 model consumed, and the 127,400 vs 127,978 mismatch

Files (copies in `scratchpad/shiu_repo/`): `comp630.csv` (= `2023_03_23_completeness_630_final.csv`; 127,400 rows, columns
`Unnamed: 0` (root_id), `Completed` all True) and `conn630.parquet` (= `2023_03_23_connectivity_630_final.parquet`; 14,687,178 rows,
columns `Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index, Connectivity, Excitatory, Excitatory x Connectivity`;
`Connectivity` min 1, Σ 52,793,639). Comparison with Codex 630 (computed this session):

- Shiu IDs 127,400 ∩ Codex 630 `neurons.csv` 127,978 = **127,319**; Shiu-only 81; Codex-only **659** (super_class: sensory 563, ascending 45,
  optic 36, central 7, endocrine 4, descending 4; flow: afferent 608). **None of the 659 Codex-only neurons has any ≥5-synapse connection in
  Codex 630** and none appears in the Shiu edge list — they are small (mostly sensory) segments that were added to the proofread list
  between the Shiu export (dated 2023-03-23 in the file name) and the Codex 630 freeze. The 81 Shiu-only IDs are presumably root IDs that
  were edited before the freeze (**UNVERIFIED**, would need CAVE lineage lookups).
- Edge weights: aggregating Shiu edges to pairs gives 14,687,178 pairs, of which 2,614,028 have ≥5 synapses; **all 2,613,129 Codex pairs are
  present with identical weights**; the 899 Shiu-only ≥5 pairs involve the 81 old IDs. Hence the Shiu file is the unthresholded v630 graph
  (Buhmann synapses, cleft score ≥50, no pair threshold) restricted to a March-2023 proofread list; Codex's `630/no_threshold_connections.csv.gz`
  (116.9 MB, no header) should reproduce it up to the 81/659 ID differences (**UNVERIFIED**, that file was not downloaded).
- Sign rule used by Shiu: neuron is inhibitory if >50% of its presynapses are predicted GABA or glutamate; DA/OA/5-HT treated as excitatory.

```python
comp = pd.read_csv("shiu_repo/comp630.csv")
con = pd.read_parquet("shiu_repo/conn630.parquet")
n630 = pd.read_csv(".cache/data/flywire/630/neurons.csv.gz")
c630 = pd.read_csv(".cache/data/flywire/630/connections.csv.gz")
ids, cids = set(comp.iloc[:, 0]), set(n630.root_id)  # 127,400 vs 127,978 -> 127,319 common
g = con.groupby(["Presynaptic_ID", "Postsynaptic_ID"]).Connectivity.sum()
cg = c630.groupby(["pre_root_id", "post_root_id"]).syn_count.sum()
m = pd.concat(
    [g[g >= 5].rename("shiu"), cg.rename("codex")], axis=1
)  # 2,613,129 both (all equal), 899 shiu-only, 0 codex-only
```

### 1.8 Schlegel/flyconnectome annotation files (v3.1.0 content, on disk)

`Supplemental_file1_neuron_annotations.tsv` — 139,248 rows × 31 columns: `supervoxel_id, root_id, pos_x/y/z, soma_x/y/z, nucleus_id,
flow, super_class, cell_class, cell_sub_class, supertype, cell_type, hemibrain_type, ito_lee_hemilineage, hartenstein_hemilineage, top_nt,
top_nt_conf, known_nt, known_nt_source, side, nerve, vfb_id, fbbt_id, status, dimorphism, matching_notes, fru_dsx, synonyms`. Non-null:
`cell_type` 137,720 (8,840 distinct), `hemibrain_type` 33,271 (4,217), `ito_lee_hemilineage` 37,542 (214), `hartenstein_hemilineage` 34,803
(194), `supertype` 33,845 (1,898 — MaleCNS neuPrint supertype ids), `vfb_id` 139,245, `fbbt_id` 28,905, `nucleus_id` 107,080, soma coords
118,104. Vocabularies: `super_class` identical to Codex 783 but with slightly different counts (optic 77,541, central 32,383, sensory
16,907, visual_projection 8,038, ascending 1,750, descending 1,303, sensory_ascending 612, visual_centrifugal 524, motor 110, endocrine 80);
`cell_class` 49 values including optic-lobe flow classes (`ME>LO`, `LA>ME`, …) that Codex does not use; `side` left/right/center/na;
`top_nt` acetylcholine 86,193 / glutamate 24,875 / gaba 19,171 / dopamine 5,909 / serotonin 2,282 / octopamine 216; `dimorphism`
isomorphic 138,072, sexually dimorphic 652, female-specific 270, potentially sexually dimorphic 161, potentially female-specific 93;
`fru_dsx` fru 3,174 / coexpress 80 / dsx 54; `status` outlier_seg 344 / outlier_bio 314. Coordinates are 4×4×40 nm voxels; `side` for
sensory/ascending is nerve-entry side; FlyWire imagery is L/R-inverted relative to the fly (Codex FAQ) but annotations are already
corrected.

`Supplemental_file5_hemibrain_meta.csv` — 25,397 hemibrain v1.2.1 bodies × 16 columns: `bodyId, instance, type, morphology_type,
cell_class, ito_lee_hemilineage, morphology_group, notes, cellBodyFiber, side, somaLocation, pre, post, pre_con2, cropped, fbbt_id`;
`type` non-null 22,715 (5,629 distinct), `morphology_type` 5,244 distinct, `cropped` True 4,039, `side` right 18,459 / left 3,691 / none 541 / na 145 / center 8.

Caveats: Codex and flyconnectome annotations diverge ("Codex presents a mix of annotations from different sources which likely diverge
from the systematic and cross-checked annotations presented here" — main README); Codex 783 files are refreshed in place (e.g.
`classification.csv.gz` 2026-02-24) without a version bump, so pin MD5s (bucket `md5Hash`) when reproducing.

---

## 2. Hemibrain v1.2 / v1.2.1

### 2.1 Versions, neuPrint names, access

| Item | Value | Source |
|---|---|---|
| Releases | v1.0 2020-01-22; v1.1 2020-06; **v1.2 2020-12-23** (Janelia page news); neuPrint serves **`hemibrain:v1.2.1`** only (uuid `31597d95bd844060b0ccc928a1a8a0a4`, `lastDatabaseEdit` 2020-12-05 21:53:59, `Meta.tag = "v1.2.1"`). The flat "exported-traced-adjacencies-**v1.2**" tarball predates the v1.2.1 DB fix-ups (see 2.4). | `hemibrain.txt`, `neuprint_datasets.json`, Cypher today |
| neuPrint | `https://neuprint.janelia.org`, dataset `hemibrain:v1.2.1`; `Meta`: totalPreCount 9,496,606, totalPostCount 64,139,744, `postHighAccuracyThreshold` 0 (!), `postHPThreshold` 0.7, `preHPThreshold` 0, voxelSize [8,8,8] nm. Neuron nodes carry **no** NT properties (`count(n.predictedNt)=0`). Token normally required (neuprint-python raises without one; tokens issued before 2026-08 are invalid after the auth migration), but `POST /api/custom/custom` answered without a token today. | Cypher today |
| Bulk bucket (public, no auth) | `gs://hemibrain/` prefixes `v1.0/`, `v1.1/`, `v1.2/`. `v1.2/` objects (2026-09-26 listing): `exported-traced-adjacencies-v1.2.tar.gz` 45,872,577 B (2020-12-22, md5 `7B4HbRVELzaPpZeLAPSmHA==`); `hemibrain-v1.2-all-traced-adjacencies.tar.gz` 80,815,554 (2026-05-15); **`hemibrain-v1.2-body-mean-neurotransmitters.feather` 45,591,786 (2026-05-15, md5 `711dUR0Z7ZvKXY3FyMymkA==`)**; `hemibrain-v1.2-tbar-neurotransmitters.feather.bz2` 442,710,628 (2025-09-07); `hemibrain-v1.2-synapse-partners.feather` 859,399,658; `hemibrain-v1.2-synapse-points-with-body-roi-type-status.feather` 1,838,033,050; `hemibrain-v1.2-ftr.tar.gz` 6,330,212,651 (2026-08-18; neo4j feather dump); `hemibrain-v1.2-skeletons.tar.gz` 1,881,216,175; `hemibrain-v1.2-neuprint-mito-stats-filtered-20201207.feather` 603,428,186; `hemibrain-v1.2.1-traced-synapse-to-nearest-mito-distances.feather` 632,698,434; `roi-meshes.tar.gz` 193,047,019; `traced-uncropped-contact-sizes.csv.gz` 161,259,596; plus `v1.2/tmp/` scratch files. Download: `https://storage.googleapis.com/hemibrain/v1.2/<name>`. | live listing |
| Janelia page links | "Compact connection matrix summary v1.2 release (v1.1) (v1.0): CSV files containing over 20,000 neurons, connections between them, decomposition of connections by brain region, and cell type information"; DVID/neuroglancer; "Hemibrain is licensed under CC-BY". | `hemibrain.txt` |

### 2.2 Files on disk (`.cache/data/hemibrain/`)

| File | Bytes | SHA-256 | Rows |
|---|---|---|---|
| `exported-traced-adjacencies-v1.2.tar.gz` | 45,872,577 | 07d8946eb0c4e3a5cb23d5769c9817847494f9fcadbc0ca239eed7bbd5555cf7 | tar |
| `extracted/…/README` | 1,420 | 7f4fa5ed02dbbea7679855aaee02b1bcf1f242a9504956e5313d58f68729fb0f | — |
| `extracted/…/traced-neurons.csv` | 582,598 | d07c1cac648c12d8135786996ac55de5a2b63d605cd6a24c5ce366224172c825 | 21,739 |
| `extracted/…/traced-total-connections.csv` | 82,191,321 | 1b954eb5b73e8ca8a650323e2a41a079b12d6fb2e92e3b5127f727785741742e | 3,550,403 |
| `extracted/…/traced-roi-connections.csv` | 126,023,187 | d4f4ecaa5a328a29c550f16aaf725cce80193b09a9bc503561a664fae0dcfe12 | 4,259,624 |
| `hemibrain-v1.2-body-mean-neurotransmitters.feather` | 45,591,786 | aab49d858415f559f469a9293adfb4d58e423db83a5debb24272ee4d66e059ad | 837,710 |

### 2.3 Schemas and computed counts

README (verbatim gist): tables cover "all non-cropped Traced Neurons in the hemibrain v1.2 dataset"; produced with neuprint-python
v0.4.12 `fetch_traced_adjacencies('exported-traced-adjacencies-v1.2')` against `Client('neuprint.janelia.org', 'hemibrain:v1.2')`.
`traced-roi-connections.csv` splits each pair by primary ROI; synapses outside primary ROIs are represented by roi `NotPrimary` in this
export (the README says they are dropped, but the file contains a `NotPrimary` roi and the per-ROI weights sum exactly to the totals).

- `traced-neurons.csv`: `bodyId, type, instance` — 21,739 bodies; `type` non-null 19,723 (5,554 distinct types), `instance` 21,323 (7,498).
- `traced-total-connections.csv`: `bodyId_pre, bodyId_post, weight` — 3,550,403 edges, Σ weight 14,329,229, min 1, max 4,299, no
  duplicates, no autapses, 21,692 unique pre / 21,738 unique post, all IDs in traced-neurons. Edges ≥5: 662,578; ≥10: 305,058.
  `weight` = number of postsynaptic densities (neuPrint `ConnectsTo.weight`, all confidences, because `postHighAccuracyThreshold` is 0
  in this DB); neuPrint `weightHP` (post confidence ≥0.7) is *not* in the export.
- `traced-roi-connections.csv`: `bodyId_pre, bodyId_post, roi, weight` — 62 ROI values (61 primary ROIs + `NotPrimary`); Σ = 14,329,229,
  3,550,403 pairs. ROI names use neuPrint style `SLP(R)`, `aL(L)`, `EB`, `NotPrimary`.
- neuPrint today for `hemibrain:v1.2.1`: 97,900 `Traced` Neuron nodes (65,231 Assign, 22,250 Orphan, 586 Unimportant, 94 null),
  **21,739 Traced ∧ cropped=false** (= the export), 22,656 Traced with a type (5,620 distinct); Traced/non-cropped ConnectsTo edges
  3,550,061 / Σ 14,325,152 — i.e. the v1.2 export differs from the v1.2.1 DB by 342 edges / 4,077 synapses (v1.2.1 = "minor fixes").
- `hemibrain-v1.2-body-mean-neurotransmitters.feather`: pandas index **`body`** (int64, unique) + 11 columns: `type, instance,
  statusLabel` (ordered dictionary, 31 categories; non-null 149,701), per-transmitter mean probabilities `gaba, acetylcholine, glutamate,
  serotonin, octopamine, dopamine, neither` (rows sum to 1.0; `predicted_nt` = argmax for 100% of rows), `predicted_nt` ∈ 7 values.
  837,710 bodies (every body with a T-bar); 21,709 of the 21,739 traced bodies covered. All-body `predicted_nt`: acetylcholine 328,287,
  glutamate 142,048, gaba 112,244, dopamine 101,273, serotonin 93,230, octopamine 31,170, neither 29,458; restricted to the 21,739 traced
  bodies: acetylcholine 9,577, glutamate 5,342, dopamine 3,183, gaba 3,009, serotonin 282, octopamine 191, neither 125. Provenance: file
  sits in `gs://hemibrain/v1.2/` next to `hemibrain-v1.2-tbar-neurotransmitters.feather.bz2` (per-T-bar probabilities) and was uploaded
  2026-05-15; the Arrow metadata carries only a pandas blob. The 6+"neither" class set matches the Eckstein et al. 2024 classifier as
  applied to hemibrain in that paper (their Fig. 4 compares FlyWire and hemibrain predictions), but **UNVERIFIED** that this feather *is*
  the Eckstein output rather than a FlyEM re-run; the high dopamine share (3,183 traced bodies, cf. Kenyon cells being mispredicted as
  dopaminergic in Eckstein 2024) is consistent with Eckstein-style predictions. Note the sjcabs `hemibrain_121_meta.feather` carries
  `neurotransmitter_predicted` with 8 values incl. `neither` and `unknown` (§6).

```python
import pandas as pd, pyarrow.feather as pf

tn = pd.read_csv(
    ".cache/data/hemibrain/extracted/exported-traced-adjacencies-v1.2/traced-neurons.csv"
)
tc = pd.read_csv(
    ".cache/data/hemibrain/extracted/exported-traced-adjacencies-v1.2/traced-total-connections.csv"
)
nt = pf.read_table(
    ".cache/data/hemibrain/hemibrain-v1.2-body-mean-neurotransmitters.feather"
).to_pandas()  # index name 'body'
print(
    len(tn), len(tc), tc.weight.sum(), tn.bodyId.isin(nt.index).sum()
)  # 21739 3550403 14329229 21709
```

### 2.4 License, citations, caveats

- License: "Hemibrain is licensed under CC-BY" (Janelia project page; no version number given on the page — **UNVERIFIED** whether 4.0 is
  stated elsewhere; the eLife paper is CC BY 4.0).
- Citation: Scheffer LK, Xu CS, Januszewski M, Lu Z, Takemura S-y, Hayworth KJ, … Plaza SM. A connectome and analysis of the adult
  *Drosophila* central brain. *eLife* 9:e57443 (2020). doi:10.7554/eLife.57443 (the neuPrint dataset description names this as the citation).
  neuPrint itself: Plaza SM et al. neuPrint: an open access tool for EM connectomics. *Front Neuroinform* 16:896292 (2022). doi:10.3389/fninf.2022.896292 (**UNVERIFIED** DOI, from memory).
- Caveats: only ~21.7k of ~25k annotated neurons are non-cropped/traced; the volume is one hemisphere plus parts of the other (ROIs mostly
  `(R)`); the neuPrint `weight` includes low-confidence PSDs (threshold 0) — use `weightHP` or the synapse-partner feather with confidence
  columns for a conservative graph; hemilineage/side annotations are *not* in neuPrint — Schlegel's `Supplemental_file5_hemibrain_meta.csv`
  (25,397 bodies, `ito_lee_hemilineage`, `side`, `cell_class`, `morphology_type`) and the sjcabs `hemibrain_121_meta.feather` supply them;
  `type` names in hemibrain have connectivity suffixes (`_a`, `_b`) that FlyWire/MaleCNS collapse (`morphology_type`).

---

## 3. MANC v1.0 / v1.2.1 (male adult nerve cord)

### 3.1 Versions and neuPrint names

| Item | Value | Source |
|---|---|---|
| Releases | **v1.0** 2023-06-06; **v1.2** 2024-03-11 (Janelia page). neuPrint datasets: `manc:v1.0` (uuid `59b37970bc7a4341b9a3a965a0d6b402`, lastDatabaseEdit 2023-05-31), `manc:v1.2.1` (uuid `7b5e8f7f805c4314bee37b75b4ff9292`, 2024-02-01) and `manc:v1.2.3` (same uuid, "2024-08-31 segment property update"). All three still served today. | `neuprint_datasets.json`, Cypher today |
| Meta thresholds | v1.0 and v1.2.1: totalPreCount 10,343,391; totalPostCount 74,456,993 (v1.0) / 74,457,060 (v1.2.1); `postHighAccuracyThreshold` **0.4**, `postHPThreshold` 0.7, `preHPThreshold` 0. | Cypher today |
| Neuron counts | v1.0: 23,759 `Traced` Neuron nodes today (23,200 `Traced` in the flat neuron-properties feather; 23,188 in `traced-neurons.csv`); v1.2.1: **23,665 Traced** (23,546 typed, 4,076 distinct types, 23,637 with `predictedNt`, 46 distinct hemilineages), Traced→Traced ConnectsTo 5,305,638 edges / Σ weight 30,934,610. Codex "MANC v1.2.1 … 23,665 neurons, 5,305,638 connections" therefore equals the *unthresholded* traced graph (FAQ: MANC default minimum 1 synapse). | Cypher today; Codex home |
| Bulk flat files | **`gs://flyem-manc-exports/`** — contains only `v1.0/` (19 objects, 18,546,268,160 B): `manc-synapse-partners-2023-05-03-215e08-minconf-0.0.feather.bz2` 1,117,661,989; `manc-traced-adjacencies-v1.0/{README 1,195; traced-neurons.csv 629,548; traced-connections.csv 75,262,163; traced-connections2.csv (identical md5); traced-connections-per-roi.csv 159,195,790}`; `manc-v1.0-neuron-properties.feather` 17,188,218; `neuprint_manc_v1.0/{README 287; neuprint_manc_v1.0_csv.tar.gz 5,291,966,339; neuprint_manc_v1.0_ftr/Neuprint_*_manc_v1.ftr (Meta 28,146; Neurons 917,907,106; Neuron_Connections 736,271,930; Synapses 3,885,378,106; Synapse_Connections 619,827,626; SynapseSet 1,305,057,754; SynapseSet_to_SynapseSet 1,026,393,722; SynapseSet_to_Synapses 2,234,058,778; Neuron_to_SynapseSet 1,084,176,706); all_ROIs.txt 594}`. **No v1.2/v1.2.1 flat export exists in this bucket.** v1.2 synapse partners: `gs://manc-seg-v1p2/manc-v1.2-synapse-partners-minconf-0.0.feather` 1,937,212,010 B (2026-08-20) + `manc-v1.2-synapse-partners-minconf-0.0.precomputed/`; segmentation `gs://manc-seg-v1p2/manc-seg-v1.2/`; neuroglancer states `manc-v1.2.{1,3}-neuprint-layers.json`. Download via `https://storage.googleapis.com/flyem-manc-exports/v1.0/...` (public, verified listing). | live listings; `manc_exports_listing.txt` |
| Codex mirror | `gs://flywire-data/codex/data/manc/1.2.1/`: `neurons.csv.gz` 1,019,361 B (2026-08-21; 23,665 rows), `connections_princeton.csv.gz` 26,747,152 (2026-01-10), `neuron_attributes.pickle.gz`, `neuron_db*.pickle.gz`. FAQ: for MANC "please use the respective project homepages or archives for canonical downloads". | live listing |
| Other access | neuprint-python / neuprintr, natverse `malevnc`, Clio (`clio.janelia.org`), NeuronBridge, DVID/TensorStore for subvolumes. | `manc-connectome.txt` |

### 3.2 Files on disk (`.cache/data/manc/`)

| File | Bytes | SHA-256 | Rows |
|---|---|---|---|
| `manc-v1.0-neuron-properties.feather` | 17,188,218 | 0c4476528906bb0a20e05e1f01e83fc2e5a582761536171ede5463669ca0b891 | 102,369 × 56 |
| `traced-neurons.csv` | 629,548 | 6b6e1dfdba71f0cef9870d8414a5cbdaa2edf25dcf6d56de0467c79debac3a5a | 23,188 |
| `manc-traced-adjacencies-v1.0-README` | 1,195 | fdb7226dd60a1d0794c78ef3050057d79b222fc973e0f70ee80270e3be1c2fa3 | — |
| `neuprint_manc_v1.0-README` | 287 | 57e71297acb743c8c7195b575308e4b97a8ef2b2e418eefa621bf4014bed690b | — |
| `all_ROIs.txt` | 594 | b068f65d729bd0131b42e49091da415f0647d488e92178092012dc63edea9220 | 60 ROI names |
| `codex_1.2.1/neurons.csv.gz` (fetched today) | 1,019,361 | ddeea1ff2859941d60a8f24933eb963d1f65d28d2399768c589a74bb90ac4a80 | 23,665 |

### 3.3 Schemas and computed counts

- `traced-neurons.csv` (v1.0): `bodyId, type, instance` — 23,188 rows; `type` 21,657 non-null (2,440 distinct); 3,562 instances are `TBD`.
  README: traced-connections.csv `weight` = "total synapse count in the connections between each neuron pair"; per-ROI file uses roi
  `NotPrimary` for synapses outside primary ROIs; produced with neuprint-python v0.4.25 on `manc:v1.0`.
- `manc-v1.0-neuron-properties.feather` — 102,369 bodies × 56 columns (all neuPrint Neuron properties incl. non-traced bodies):
  `bodyId, instance, type, systematicType, pre, post, downstream, upstream, size, synweight, status, statusLabel, cropped, class, subclass,
  group, serial, serialMotif, hemilineage, somaSide, somaNeuromere, rootSide, entryNerve, exitNerve, longTract, birthtime, modality,
  transmission, receptorType, origin, target, prefix, subcluster, positionType, position, somaLocation, rootPosition, tosomaPosition,
  avgLocation, somaRadius, cellBodyFiber(null), synonyms, tag, roiInfo (dict), inputRois, outputRois, neuropilsAxonal/Dendritic (null),
  namingUser (null), source, ntGabaProb, ntAcetylcholineProb, ntGlutamateProb, ntUnknownProb, predictedNt, predictedNtProb`.
  Key counts: `status` Traced 23,200 / Unimportant 1,530 / Assign 889 / RT Orphan 314 / Orphan 251 / PRT Orphan 245 / Anchor 196 / null
  75,744; `statusLabel` Roughly traced 18,305, Prelim Roughly traced 4,895, …; `class` intrinsic neuron 13,059, sensory neuron 5,938,
  ascending neuron 1,866, descending neuron 1,328, motor neuron 737, sensory ascending 539, Glia 340, Interneuron TBD 113, efferent neuron
  94, Sensory TBD 89, TBD 61, efferent ascending 8; `somaSide` RHS 7,808 / LHS 7,794 / Midline 317; `rootSide` RHS/LHS/BIL/MID/Midline;
  `somaNeuromere` T2 5,071, T1 4,536, T3 3,961, A1 787, A2–A10 (23–284), …; `hemilineage` 49 distinct (17,232 non-null; Truman
  nomenclature `0A`…`27X`, dotted merges like `20A.22A`); `birthtime` secondary 10,380 / primary 3,124 / early secondary 2,272;
  `transmission` putative electrical 189 / putative neurosecretory 176 / neurosecretory 59 / electrical 52; `modality` tactile 2,764 /
  proprioceptive 1,428 / chemosensory 1,217 / unknown 1,125.
  **NT columns:** `ntGabaProb, ntAcetylcholineProb, ntGlutamateProb, ntUnknownProb` (96,418 bodies; the four sum to 1 for all of them —
  MANC's classifier has only 3 transmitter classes + unknown) and `predictedNt` ∈ {acetylcholine 11,701, glutamate 8,037, gaba 6,147,
  unknown 502} with `predictedNtProb` (0.275–1.0) for 26,387 bodies; among the 23,200 Traced: acetylcholine 11,454, glutamate 5,748,
  gaba 5,740, unknown 100, null 158. v1.2.1 neuPrint adds `unclear` (39) and `totalNtPredictions`, `celltypePredictedNt`,
  `celltypeTotalNtPredictions` (per `meta_manc_v1.2.1.json`), and Traced `predictedNt`: acetylcholine 11,518, glutamate 6,283, gaba 5,738.
- ROIs (`all_ROIs.txt`, 60): neuropils `LegNp(T1..3)(L/R)`, `mVAC(T1..3)(L/R)`, `NTct(UTct-T1)`, `WTct(UTct-T2)`, `HTct(UTct-T3)`, `IntTct`,
  `LTct`, `ANm`, `Ov(L/R)`; nerves `CV, CvN, ADMN, PDMN, PrN, DProN, VProN, ProAN, ProCN, ProLN, MesoLN, MesoAN, DMetaN, MetaLN, AbN1-4,
  AbNT`; `GF(L/R)` (giant fiber ROIs). v1.2.1 `roiHierarchy` also lists `IntNp(T1..3)`, `AMNp` (v1.0 meta) — **UNVERIFIED** differences.
- Codex `codex_1.2.1/neurons.csv.gz` (fetched today; same 21-column export used for BANC): `Root ID, Top in/out region, Community labels,
  Predicted NT type (ACH 11,503 / GLUT 6,191 / GABA 5,730 / empty 241), Predicted NT confidence, Verified NT type, Verified Neuropeptide,
  Body Part, Function, Flow (intrinsic 13,177 / afferent 7,253 / efferent 3,235), Super Class (intrinsic_neuron 13,058, sensory 5,925,
  ascending 1,862, descending 1,328, motor 737, sensory_ascending 535, efferent 93, intrinsic 71, interneuron 33, glia 15,
  efferent_ascending 8), Class (35), Sub Class (2,295 — Codex packs the MANC `subclass`+nerve/tract strings here), Hemilineage (43),
  Nerve (94), Soma side, Primary Cell Type (4,074), Alternative Cell Type(s), Cable length (nm) (empty), Surface area (empty), Volume (nm^3)`.
  `Community labels` is a `key::value` bag of the remaining neuPrint fields (`description::`, `group::`, `instance::`, `rootLocation::`,
  `synonyms::`, `transmission::`, `vfbId::`, `tag::`).

```python
import pyarrow.feather as pf, pandas as pd

npf = pf.read_table(".cache/data/manc/manc-v1.0-neuron-properties.feather").to_pandas()
print(
    (npf.status == "Traced").sum(), npf.predictedNt.value_counts().to_dict()
)  # 23200 {'acetylcholine': 11701, 'glutamate': 8037, 'gaba': 6147, 'unknown': 502}
assert (
    npf[["ntGabaProb", "ntAcetylcholineProb", "ntGlutamateProb", "ntUnknownProb"]]
    .sum(axis=1)
    .round(3)
    .dropna()
    == 1
).all()
```

### 3.4 License, citations, caveats

- License: "The MANC is licensed under CC-BY" (Janelia page). Papers are eLife (CC BY 4.0).
- Citations (as listed in the neuPrint `Meta.description` for both versions): Takemura S-y, Hayworth KJ, Huang GB, Januszewski M, Lu Z,
  Marin EC, … Berg S. A Connectome of the Male *Drosophila* Ventral Nerve Cord. *eLife* 13:RP97769 (2024). doi:10.7554/eLife.97769 —
  Marin EC, Morris BJ, Stürner T, Champion AS, Krzeminski D, Badalamente G, … Jefferis GSXE. Systematic annotation of a complete adult
  male *Drosophila* nerve cord connectome reveals principles of functional organisation. *eLife* 13:RP97766 (2024). doi:10.7554/eLife.97766 —
  Cheong HSJ, Boone KN, Bennett MM, Salman F, Ralston JD, Hatch K, … Card GM. Transforming descending input into behavior: The
  organization of premotor circuits in the *Drosophila* Male Adult Nerve Cord connectome. *eLife* 13:RP96084 (2025). doi:10.7554/eLife.96084.
  (neuPrint links the `.1`/`.2` reviewed-preprint versions: 97769.1, 97766.1, 96084.2.)
- Caveats: three neuPrint versions with the same "v1.2" segmentation but different annotations (v1.2.1 vs v1.2.3 = 2024-08 property
  update; Codex and sjcabs use **v1.2.1**); v1.0 vs v1.2.1 differ in traced count (23,759 vs 23,665 today) and typing (2,448 vs 4,076
  distinct types, `systematicType` filled in v1.2.1); NT classifier is 3-class (ACh/GABA/Glu) — no monoamines; sides are `RHS/LHS` in
  neuPrint but `right/left` in Codex/sjcabs; hemilineage uses Truman VNC nomenclature (`0A`…`27X`), not ItoLee; the v1.0 flat neuron
  table includes 79k non-neuron bodies (filter `status == 'Traced'`, or better `class` not null and not `Glia`).

---

## 4. FANC (Female Adult Nerve Cord)

### 4.1 What the `fanc/v1444/*.parquet` files are

They are byte-identical copies (MD5 verified against the bucket) of the CAVE table exports published in the public bucket
**`gs://lee-lab_female-adult-nerve-cord/CAVE/v1444/`** (prefixes in that bucket: `CAVE/`, `VNC_templates/`, `alignmentV4/`, `colormips/`,
`meshes/`; no README object at the root). `CAVE/` holds two materialization snapshots: `v1237/` (uploaded 2025-08-15, 8 files) and
**`v1444/`** (uploaded 2025-10-09, 10 files). Each parquet carries CAVE `PANDAS_ATTRS` metadata (`datastack_name: fanc_production_mar2021`,
`table_aligned_volume: fanc_v4`, `table_schema`, `table_description`). Public read requires no auth
(`https://storage.googleapis.com/lee-lab_female-adult-nerve-cord/CAVE/v1444/<file>`). The materialization timestamp of v1444 is not in
the files (**UNVERIFIED**; the CAVE materialization endpoint needs a token).

| File (bucket = disk) | Bytes | SHA-256 | Rows | Content |
|---|---|---|---|---|
| `cell_ids_v2.parquet` | 1,037,439 | a6ca467c29c0c273acdcde2b885bd912b17eabb34c951b5ca48f3cc2caca28f4 | 21,978 | schema `bound_tag_user`; one point per cell; `user_id` **is the stable FANC cell ID** ("Motor neurons and efferent non-motor neurons have 3-digit cell IDs (100-999), sensory neurons 4-digit (1000-9999), descending central neurons 10000-11999, ascending …"); `tag` soma 16,729 / peripheral nerve 3,721 / neck connective 1,437 / cut-off soma 54 / backbone 35 / orphan 2; `pt_root_id` unique (21,978) |
| `somas_dec2022.parquet` | 1,218,840 | 8376a28222beccec474e6bb2f62d48e52f8f5bb657ca1ba8a3a6ca67213f416d | 16,784 | schema `nucleus_detection`; `id` (17-digit nucleus id), `volume` (µm³), `pt_position`, `bb_start/end_position`; 16,774 unique roots |
| `neck_connective.parquet` | 146,623 | 859c3754073528e313e0d3dfc72ffac95f1b4cfe9e56e170a58195c665080600 | 3,850 | `bound_tag`; points at plane y=75200; tag `neck connective (right)` 1,922 / `(left)` 1,909 / `(middle)` 19; 3,761 unique roots |
| `peripheral_nerves.parquet` | 166,458 | 048b5e029d0d1252dc45ece3ab245ca213e22ea5082e54f3a2aa321ca5ba97df | 4,043 | `bound_tag`; 21 nerve tags (`left T1 leg nerve` 903, `right T2 leg nerve` 884, `right ADMN` 558, `left ADMN` 546, haltere/PDMN/dorsal/chordotonal/accessory/ventral/prosternal/cervical nerves) |
| `proofread_first_pass.parquet` | 371,800 | a5ea372576a887a3753ae16e6b476309861e0077cd422569490806a843784528 | 7,246 | `proofreading_boolstatus_user`; "first pass" = ≥10 min, no large merges, backbone complete; `valid_id` = root at time of marking; 7,001 unique current roots |
| `proofread_second_pass.parquet` | 92,795 | c83d8c6e9bb01704042b9731b0f278212bb76e9e9a0127d2e118457cc2f100bf | 1,771 | "second pass" = anatomy & connectivity accurate; 1,766 unique roots |
| `synapses_nov2022_human_readable_connectioncounts_countthresh3.parquet` | 12,498,976 | 618a9698fd446a7118f5cae30e87e4168803923001e60c1f19aec5de3c24a552 | 1,878,659 | `pre_root_id, post_root_id, num_synapses` (pairs with ≥3 synapses); Σ 11,801,582; min 3, max 870; ≥5: 630,322; ≥10: 219,339; 102,136 unique pre / 728,951 unique post (most posts are unproofread fragments); no duplicate pairs |
| `synapses_nov2022_human_readable_connectioncountsperneuropil_countthresh3.parquet` | 15,524,381 | 4f081003384cda4e15af0acf7edb5e8ee1cf2a90568b1f01adcc71d5ca3458d0 | 2,454,146 | + `neuropil` (43 values: `ProNm/MesoNm/MetaNm_{L,R}`, `NTct/WTct/HTct/IntTct/LTct_{L,R}`, `mVAC_{L,R}`, `ANm_{L,R}`, `AMNp_{L,R}`, tracts `CFF, DLT, DLV, DMT, ITD, ITD_HC, ITD_HT, MDT, VLT, VTV` ×L/R, `unassigned`); same 1,878,659 pairs and Σ 11,801,582 |

Not downloaded (in the same prefix): `synapses_nov2022_human_readable_connectioncountsperneuropil.parquet` 192,564,883 B (no count
threshold) and `synapses_nov2022_human_readable_id_size_prerootid_postrootid_prex_prey_prez_neuropil.parquet` 768,822,583 B (per-synapse
table: id, size, pre/post root, presynaptic xyz, neuropil). `v1237/` has the same set minus the `countthresh3` files.

Computed joins: 17,136 of the 102,136 presynaptic roots are in `cell_ids_v2`; restricting both ends to `cell_ids_v2` leaves 598,122 edges /
6,098,104 synapses; 6,866 presynaptic roots are in `proofread_first_pass`. "num_synapses" = count of predicted synaptic partner pairs
(Nov-2022 synapse prediction; confidence/size thresholds **UNVERIFIED** — the per-synapse table has a `size` column).

```python
import pyarrow.parquet as pq

cc = pq.read_table(
    ".cache/data/fanc/v1444/synapses_nov2022_human_readable_connectioncounts_countthresh3.parquet"
).to_pandas()
print(
    len(cc), cc.num_synapses.sum(), cc.num_synapses.min(), (cc.num_synapses >= 5).sum()
)  # 1878659 11801582 3 630322
print(pq.read_schema(".cache/data/fanc/v1444/cell_ids_v2.parquet").metadata[b"PANDAS_ATTRS"][:300])
```

### 4.2 Access model, version, license, citations

- **CAVE datastack `fanc_production_mar2021`** (server `https://cave.fanc-fly.com`, aligned volume `fanc_v4`) is **restricted**: "Access to the
  latest reconstruction of FANC is restricted to authorized users … generate an API key by visiting
  https://global.daf-apis.com/auth/api/v1/create_token and logging in with your FANC-authorized google account"; to join, "have their
  PI/group leader contact Wei-Chung Lee"; "we currently do not accept applications from members of the general public" (wiki
  `Joining-the-community.md`). The Nature paper's data availability: "The electron microscopy dataset and reconstructions are freely
  available. The segmentation of the FANC dataset is available by joining the FANC community". No data-use agreement text is published;
  new groups "read and agree to follow the Community Code of Conduct". Secret file layout: `~/.cloudvolume/secrets/cave-secret.json`
  with keys `token` and `fanc_production_mar2021` (python package `fanc-fly`, `fanc.save_cave_credentials`).
- **Public without auth:** the `gs://lee-lab_female-adult-nerve-cord` bucket (CAVE exports above; EM `alignmentV4/em/rechunked`;
  segmentation `alignmentV4/segmentation/seg_mar2021`; published-neuron meshes under `meshes/FANC/` and `meshes/JRC2018_VNC_FEMALE/`;
  neuroglancer viewers `https://ng.fanc.community/published-neurons-viewer`). Manual CATMAID reconstructions (2021–22) at
  `https://catmaid.virtualflybrain.org/`.
- **Versions:** the wiki lists CAVE materialization **v840** (timestamp 1705479001.179 = 2024-01-17) as the state used by Azevedo et al.
  2024; the public exports are **v1237** and **v1444**; BANC/sjcabs cross-reference **FANC v1116** (`fanc_1116_cell_type`, `fanc_1116_match_id`,
  `nblast/banc_fanc_1116_nblast.feather`). No FANC neuron-annotation table (`neuron_information`, the CAVE table with the hierarchical
  vocabulary: primary class → sensory/central/motor/efferent non-motor/glia; hemilineage `0A…27X`; fast NT cholinergic/GABAergic/
  glutamatergic; publication tags) is in the public export — cell types for FANC reach us only via BANC's `fanc_cell_type` (2,905
  BANC neurons) and the Stürner/Brooks 2025 DN/AN comparison.
- **License:** none stated in the bucket, wiki or README (**UNVERIFIED** — treat as "public but no explicit license"; the Nature paper is
  not open access).
- **Citations:** Phelps JS, Hildebrand DGC, Graham BJ, Kuan AT, Thomas LA, Nguyen TM, … Lee W-CA. Reconstruction of motor control circuits
  in adult *Drosophila* using automated transmission electron microscopy. *Cell* 184, 759–774.e18 (2021). doi:10.1016/j.cell.2020.12.013
  (**UNVERIFIED** page range/DOI, from memory) — Azevedo A, Lesser E, Phelps JS, Mark B, Elabbady L, … Lee W-CA, Tuthill JC. Connectomic
  reconstruction of a female *Drosophila* ventral nerve cord. *Nature* 631, 360–368 (2024). doi:10.1038/s41586-024-07389-x (HTML on disk;
  pages **UNVERIFIED**) — Lesser E, Azevedo AW, Phelps JS, … Tuthill JC. Synaptic architecture of leg and wing premotor control networks in
  *Drosophila*. *Nature* 631, 369–377 (2024). doi:10.1038/s41586-024-07600-z (HTML on disk). Software: `htem/FANC_auto_recon`.
- Caveats: FANC root IDs (`6485183464…`) and BANC root IDs (`7205759414…`) live in different chunked graphs; only motor/DN/AN/sensory
  subsets are curated; the public "human_readable" synapse counts include all segments, so degree statistics require a proofread mask
  (`proofread_first_pass` ∪ `cell_ids_v2`).

---

## 5. BANC (Brain And Nerve Cord) v888

### 5.1 Access status as of 2026-09-26

| Channel | Status | Evidence |
|---|---|---|
| Public GCS bucket `gs://lee-lab_brain-and-nerve-cord-fly-connectome/` | **Public, no auth** (verified listing + downloads today). Root: `README.txt` 12,505 B (2026-06-23), `CHANGELOG.md` 3,737, `BANC-to-BANC_top_NBLAST_scores.csv` 1.17 GB, `neuron_skeletons.zip` 215.6 MB, `neuron_colormips_v{626,888}.zip`; prefixes `_archive/ behavior/ compiled_data/ documentation/ (97 md files) imported_meshes/ light_level/ microCT/ nblast/ neuron_annotations/ neuron_colormips/ neuron_connectivity/ neuron_meshes/ neuron_skeletons/ nuclei/ region_outlines/ registrations/ synapses/ templates/`. | live listing |
| Harvard Dataverse | doi:10.7910/DVN/7WTH1N "Publication version" — version 3.0 released 2026-07-01, **license CC BY 4.0**, 379 files, 536,062,963,020 B (mirrors the bucket incl. influence chunks, verified-match PNG zips 45 GB etc.); doi:10.7910/DVN/8TFGGB "Preprint version" (v626) — version 8.1, 2026-05-15, CC BY 4.0, 489 files, 270.5 GB. | Dataverse API today |
| CAVE | Public datastack **`brain_and_nerve_cord_public`** (documentation: `https://global.daf-apis.com/info/datastack/brain_and_nerve_cord_public`); production datastack `brain_and_nerve_cord` restricted to community members (token at `global.daf-apis.com/auth/api/v1/create_token`, secret key `brain_and_nerve_cord`). Today the info endpoint 302-redirected to login for both, so a Google-authenticated token is needed even for the public stack. Segmentation `precomputed://gs://brain-and-nerve-cord-public/zetta_lee_fly_cns_001_segmentation/v888/v1_sharded/` (doc: "public and no-auth", ~452 GiB). | docs on disk; live probe |
| Codex | `https://codex.flywire.ai/banc` — BANC v888 (2026-05-20, default) and v626 (2025-07-20); tile "158,262 neurons, 3,037,361 connections"; default threshold 3 synapses. Public data dir `gs://flywire-data/codex/data/banc/888/`: `neurons.csv.gz` 2,881,728 B (2026-08-18), `connections_princeton.csv.gz` 29,164,674 (2026-06-29), `neuron_attributes.pickle.gz`, `neuron_db_schema_version_20251007.pickle.gz`, `influence/`; older `banc/{305,470,484,514,537,554,587,610,626,850}/`. | Codex FAQ/home; live listing |
| Data-use agreement | None for the published files (CC BY 4.0 on Dataverse; bucket README asks to cite the Dataverse DOI and the paper). Community membership (proofreading/live CAVE) goes through FlyWire (`flywire.ai/banc`). | README, wiki |

### 5.2 Version history

- CAVE materialization **626** (timestamp 2025-07-21) = preprint release (2025-07-31; left optic lobe incomplete).
- **850** = interim materialization (root IDs kept as `root_850` column).
- **888** (timestamp **2026-04-16**) = publication version (bucket release 2026-05-18; `compiled_data/` added 2026-05-21; README/docs 2026-06-15).
  The paper: Bates AS*, Phelps JS*, Kim M*, Yang HH*, … Lee W-CA, Wilson RI. Distributed control circuits across a brain-and-cord
  connectome. *Nature* 656, 957–970 (June 2026; PMC13518251; CC BY 4.0). doi:10.1038/s41586-026-10735-w.
- **890**: the on-disk meta feather has a `root_890` column (188,234 non-null) — a newer materialization used for post-publication
  patches (meta re-uploaded 2026-06-26/29, 2026-08-12 "shoreup", 2026-08-21). Codex/bucket still call the release v888. Newer than 888 there
  is no separate public release as of today.

### 5.3 Files on disk (`.cache/data/banc/` and the identical copies in `sjcabs/`)

| File | Bytes | SHA-256 | GCS md5 match | Rows × cols |
|---|---|---|---|---|
| `banc_888_meta.feather` (= `sjcabs/banc_888_meta.feather`) | 57,503,026 | 86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098 | yes (`jCuTpgjHFj7J2U1o4HVv9A==`, 2026-08-21) | 188,508 × 81 |
| `sjcabs/banc_888_metrics.feather` | 12,285,378 | 3ee647d402510b6b1c1bc3b0249db83f799ba94a1094c8ec530fcb09fac1ce60 | yes (2026-08-28) | 188,508 × 19 |
| `neuron_annotations/v888/codex_annotations.parquet` | 74,350,281 | 8a52f5f84f439603881bb360ea990601fd593072261c8cfb9afb0b3842f89eda | yes | 1,841,078 × 12 |
| `neuron_annotations/v888/backbone_proofread.parquet` | 10,316,976 | 9b553956c11558b9d5c7c9d3e66cea8af08f358ee897ab6e0e8ce91faa5c9a2d | yes | 211,947 × 10 |
| `README.txt`, `CHANGELOG.md`, `documentation/*.md` (4 + 7 fetched today), `sjcabs_banc_data.md` | 12,505 / 3,737 / … | see `sha256_all.txt` | — | — |
| `codex_888/neurons.csv.gz` (fetched today) | 2,881,728 | 40a2201554a8c34d2c4b07c8322543a07c1b3faf5acafbac363fd1a3d0fa617f | yes | 158,262 × 21 |

Bucket `compiled_data/banc_888/` (24 objects, 40.5 GB): `banc_888_edgelist_simple_v2.feather` 305,250,378 (11,510,975 rows per doc);
`banc_888_edgelist_simple_v3.feather` 359,161,658 (13,507,098); `banc_888_edgelist_split_v{2,3}.feather` 847.5 MB / 940.6 MB;
`banc_888_synapses_v2_enriched.parquet` 17,146,564,854 (168,951,110 rows × 21); `banc_888_synapses_v3_enriched.parquet` 19,733,123,829
(198,741,886 × 10); `banc_888_neurotransmitter_prediction_v2.csv` 21,107,592 (188,199 rows); betweenness/spectral CSVs (v2 and v3);
`banc_888_proofread_influence.sqlite` 723.7 MB; `influence/all_to_all/` (~287 GB parquet); several `*.backup_*`/`*.pre_*` versions of meta
and metrics (provenance). `neuron_annotations/v888/`: `cell_info.parquet` 12.3 MB, `cell_representative_point.parquet` 5.5 MB (158,265 rows),
`codex_annotations_flat_table_260520.csv` 39.3 MB, `mitochondria_v1_*`, `neck_connective_y{92500,121000}.parquet`, `peripheral_nerves.parquet`,
`proofreading_notes.parquet`, `somas_v1.parquet`. `neuron_connectivity/v888/`: `synapses_v{1,2,3}_human_readable.csv.gz` 12.0 / 12.3 / 15.6 GB,
parquet variants, `banc_888_edgelist_neuropil_v{2,3}.feather` 4.5 / 5.1 GB, `*_sizethresh3_connectioncounts_countthresh3.parquet`.
`synapses/v2.0/banc_nt_prediction_w_sizethresh_5_11102025.parquet` 3.89 GB and `synapses/v3.0/banc_nt_prediction_v3_w_sizethresh_10_05042026.parquet`
5.79 GB (259,391,803 rows × 11 = per-synapse 8-NT probabilities). `nblast/` (14 files, 5.48 GB): `banc_{fafb_783,manc_v1.2.1,hemibrain_v1.2.1,
malecns_v0.9,fanc_1116}_nblast.feather`, `banc_native_nblast.feather` 2.1 GB, `banc_mirror_nblast.feather` 1.4 GB, `banc_*_reviewed_matches.csv`.

### 5.4 Schemas and computed counts

**`banc_888_meta.feather`** (81 columns; all identifier columns are *strings*; `proofread`/`roughly_proofread` are the strings "TRUE"/"FALSE"):
identity `banc_888_id, root_id, supervoxel_id, position ("x, y, z" in 4×4×45 nm voxels), root_626, root_850, root_888, root_890, nucleus_id,
proofread, roughly_proofread, status (3,012 distinct comma-joined flags), side, root_position, root_position_nm, root_region (109 values,
e.g. ITO_midbrain_AL_R, MANC_vnc_NTct_UTct_T1_L)`; anatomy `region, hemilineage (259), nerve (75), tract (5), neuromere (13), flow`;
hierarchy `super_class (16), cell_class (119), cell_sub_class (183), cell_type (11,504 distinct)`; cross-dataset `fafb_cell_type (9,898),
fafb_alignment_cell_type, manc_cell_type (4,721), malecns_cell_type (10,445), hemibrain_cell_type (5,940), fanc_cell_type (1,036),
{fafb,manc,malecns,hemibrain,fanc}_match (= matched neuron ID in the other dataset), {…}_nblast_match`; groupings `sexually_dimorphic,
cluster, manual_cluster, super_cluster (34), cns_network (13), body_part_sensory (47), body_part_effector (24), peripheral_target_type (103),
cell_function (51), cell_function_detailed (90)`; neurochemistry `neurotransmitter_predicted (8), neurotransmitter_score (float),
neurotransmitter_verified (35), neuropeptide_verified (73)`; metrics `l2_nodes, l2_cable_length_um, volume_nm3, input_connections,
output_connections, input_side_index, output_side_index, mitochondria, mitochondria_volume, pd_width, segregation_index`; provenance
`seed_01…seed_14`.

Computed: rows 188,508 (unique `banc_888_id` = `root_888`; `root_id` has 188,340 distinct — 168 duplicates); `proofread` TRUE 150,952,
`roughly_proofread` TRUE 5,060, disjoint (both = 0), either = **156,012** (paper Fig. 1f: "proofread and roughly proofread neurons totalling
155,916"; Supplementary Data 2 says 187,590 rows); `super_class` non-null 159,876: optic_lobe_intrinsic 72,947, central_brain_intrinsic
31,879, sensory 16,563, ventral_nerve_cord_intrinsic 12,866, glia 12,750, visual_projection 7,316, ascending 1,849, descending 1,316, motor
805, sensory_ascending 517, visual_centrifugal 472, visceral_circulatory 221, not_a_neuron 195, trachea 162, sensory_descending 13,
ascending_visceral_circulatory 5, empty 28,632; `flow` intrinsic 141,708 / afferent 17,089 / efferent 1,033; `region` optic_lobe 108,764 /
central_brain 47,688 / ventral_nerve_cord 31,688 (no `neck_connective` value — retired in v888; ANs/DNs are found via `super_class`);
`side` left 87,160 / right 86,447 / empty 14,901; `neurotransmitter_predicted` acetylcholine 87,059, glutamate 25,099, gaba 21,686,
dopamine 8,344, histamine 7,411, octopamine 2,303, serotonin 1,936, tyramine 215, empty 34,455 (8-class Eckstein-style classifier
retrained for BANC; scores 0–1); `sexually_dimorphic` isomorphic 114,571 / dimorphic 2,862 / female-specific 946 / male-specific 8;
`cell_type` non-null 118,748; `fafb_cell_type` 131,240 (more than `cell_type`, because FAFB matches exist for un-typed optic-lobe cells);
`manc_cell_type` 26,613; `malecns_cell_type` 43,490; `hemibrain_cell_type` 30,910; `fanc_cell_type` 2,905; `*_match` (IDs): fafb 64,789,
manc 24,277, malecns 23,608, hemibrain 6,768, fanc 2,451. Proofread × super_class: glia 144 of 12,750 proofread; 11,344 proofread rows have
no super_class.

**`banc_888_metrics.feather`** — 19 columns (doc says 12; file adds `side, region, branchpoints, endpoints, axon_length, dend_length,
projection_score`), 188,508 rows.

**`codex_annotations.parquet`** — long-form CAVE table (`cell_type_reference` schema): `id_ref, created_ref, valid_ref, pt_supervoxel_id,
pt_root_id, id, created, valid, target_id, classification_system, cell_type (value column!), pt_position`. 1,841,078 rows, all `valid`;
158,230 distinct `target_id` / 158,227 `pt_root_id` (= Codex's "158,262 neurons" set up to 35 ids). 31 `classification_system` values with
row counts: neurotransmitter_score 158,230, region 158,163, neurotransmitter_predicted 151,274, side 143,095, user_id 125,025,
**fafb_783_cell_type 121,479**, super_class 105,718, flow 105,336, cell_type 104,902 (11,458 distinct types), sexually_dimorphic 102,316,
**malecns_09_cell_type 94,764**, cell_class 75,802, neurotransmitter_verified 60,074, **fafb_783_match_id 57,596**, hemilineage 55,189,
cell_sub_class 37,807, **hemibrain_121_cell_type 30,389**, **manc_121_cell_type 25,630**, **manc_121_match_id 24,294**, cell_function 22,186,
body_part_sensory 17,336, peripheral_target_type 16,793, nerve 15,339, cell_function_detailed 9,766, neuropeptide_verified 7,311,
**hemibrain_121_match_id 6,757**, **fanc_1116_cell_type 2,886**, **fanc_1116_match_id 2,445**, **malecns_09_match_id 1,229**, body_part_effector
1,040, other_names 907. Controlled vocabularies observed: `super_class` (18 incl. typos `venral_nerve_cord_intrinsic`, `intrinsic`), `flow`
{intrinsic 87,254, afferent 17,048, efferent 1,034}, `region` {optic_lobe 92,168, central_brain 39,489, ventral_nerve_cord 22,795,
neck_connective 3,688, rind 22, brain 1}, `side` {right 73,184, left 69,911}, `nerve` 60 values (`left_antennal_nerve`,
`right_metathoracic_leg_nerve`, `abdominal_nerve_trunk`, …). Pivot recipe in `documentation/codex_annotations.md`; `cell_representative_point`
(158,265 rows) is the join target. The BANC taxonomy reference is Supplementary Data 1 (`banc_supplemental_data.zip`, 10 CSVs, 87 MB).

**`backbone_proofread.parquet`** — 211,947 rows (`proofread` all True, `valid` all True), 150,507 distinct `pt_root_id` (a root can carry
several marks); `valid_id` = root at marking time.

**Codex `codex_888/neurons.csv.gz`** — 158,262 rows × 21: `Root ID, Top in/out region, Community labels, Predicted NT type (ACH 85,006 /
GLUT 24,382 / GABA 20,604 / DA 8,072 / HIST 7,694 / OCT 1,903 / SER 1,612 / TYR 209 / empty 8,780), Predicted NT confidence (0.2–1.0),
Verified NT type (30), Verified Neuropeptide (68), Body Part (57), Function (133), Flow (intrinsic 128,776 / afferent 17,082 / efferent
1,033 / empty 11,371), Super Class (16 — same vocabulary as the meta feather), Class (117), Sub Class (182), Hemilineage (259), Nerve (71),
Soma side (right 82,502 / left 75,760), Primary Cell Type (11,546 distinct; 118,564 filled), Alternative Cell Type(s), Cable length /
Surface area / Volume (all empty for BANC)`. So Codex-BANC ≈ `codex_annotations` pivoted, restricted to 158,262 neurons.

**Edgelists / synapses (from docs on disk; not downloaded):** `banc_888_edgelist_simple_v{2,3}.feather` columns `pre, post, count, norm
(= count/post_count), pre_count, post_count`; v2 = synapse detector v2 with `size ≥ 5` voxels (paper), v3 = new detector with `size ≥ 10`
(~18% more synapses); autapses removed; **no pair-count threshold** ("apply `count >= 5` at read time for the Codex-style threshold";
Codex-BANC itself uses ≥3). `banc_888_synapses_v2_enriched.parquet`: `id, size, pre_root_id, post_root_id, X, Y, Z (nm), neuropil
(FlyWire-style codes e.g. MB_CA_R, ITO_optic_LO_R), region, side, acetylcholine…tyramine (8 probs), syn_top_nt, syn_top_p, label
(axon/dendrite/soma/unknown)`; upstream filter `cleft_score > 50` per `sjcabs_banc_data.md`. Raw `synapses_v2_human_readable.csv.gz`
columns `id, pre_x,pre_y,pre_z, post_x,post_y,post_z, ctr_x,ctr_y,ctr_z (nm), size (voxels), pre_root_id, post_root_id, …` (coordinates in nm,
unlike other BANC CAVE tables which are 4×4×45 nm voxels).

```python
import pyarrow.feather as pf, pyarrow.parquet as pq

m = pf.read_table(".cache/data/banc/banc_888_meta.feather").to_pandas()
pr = m.proofread.eq("TRUE")
rp = m.roughly_proofread.eq("TRUE")
print(len(m), pr.sum(), rp.sum(), (pr | rp).sum())  # 188508 150952 5060 156012
ca = pq.read_table(".cache/data/banc/neuron_annotations/v888/codex_annotations.parquet").to_pandas()
print(
    ca.classification_system.value_counts().head(8).to_dict(), ca.target_id.nunique()
)  # ... 158230
```

### 5.5 License, citations, caveats

- License: **CC BY 4.0** (Dataverse deposit metadata for both the publication and preprint versions; paper is CC BY 4.0). Bucket README:
  "Please cite that DOI [10.7910/DVN/7WTH1N] when referring to the dataset as a whole; cite the relevant Zenodo DOI … for a specific piece of
  analysis software" (bancr 10.5281/zenodo.20350647, bancpipeline 10.5281/zenodo.20350571, BANC-project 10.5281/zenodo.20350641,
  influencer 10.5281/zenodo.20350563, ConnectomeInfluenceCalculator 10.5281/zenodo.15999929, nat.ggplot 10.5281/zenodo.20350565,
  synister_banc 10.5281/zenodo.20350569, drosophila_neurotransmitters 10.5281/zenodo.20818141, drosophila_neuropeptides 10.5281/zenodo.20818143).
- Citation: Bates AS, Phelps JS, Kim M, Yang HH, et al. Distributed control circuits across a brain-and-cord connectome. *Nature* 656,
  957–970 (2026). doi:10.1038/s41586-026-10735-w. Preprint bioRxiv 10.1101/2025.07.31.667571.
- Caveats: dissection damaged both antennal nerves (JO and nearby neurons under-represented); left optic lobe less thoroughly typed than
  right; 22 known problem regions (Supplementary Data 10 / `banc_problem_regions.csv`); two synapse versions in circulation (paper = v2,
  recommended for new work = v3); `region` value `neck_connective` exists in `codex_annotations` (3,688 rows) but not in the meta feather;
  meta feather is patched in place after publication (`root_890`, `pre_*` backups) — pin the GCS `md5Hash`; identifiers are strings.

---

## 6. sjcabs / fly_connectome_data_tutorial harmonized files ("compiled_data")

### 6.1 What it is, where it lives, who made it

- Repo: `https://github.com/sjcabs/fly_connectome_data_tutorial` (clone in scratchpad; HEAD 85c66244 2026-06-07; code **MIT**; instructors
  Sven Dorkenwald & Alexander Bates, contributions Philipp Schlegel, Greg Jefferis). README: "All datasets have been harmonized to use the
  unified metadata schema we used in the BANC project"; "The Google Bucket contains a curation of connectome data by Alexander Bates … You
  are welcome to use this data curation in your own work! Just let Alex know!"; "**Data:** Licensed under CC-BY by the respective dataset
  creators; **Code:** MIT". A ZIP snapshot of the repo is in the BANC Dataverse deposit (`doc_fly_connectome_data_tutorial_archive.md`:
  pinned commit 5416119 / 2026-05-22; "does not yet have a minted Zenodo DOI … cite the GitHub URL plus a commit SHA").
- Data: **`gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/`** (public, no auth) with sub-prefixes `banc_888/`, `fafb_783/`,
  `manc_121/`, `hemibrain_121/`, `malecns_09/`, `fanc_1116/` (the latter contains only `fanc_banc_space_swc/` — **no FANC meta/edgelist**).
  Built by `flyconnectome/bancpipeline` (`banc/meta/banc-data.R`, `banc/share/banc-sjcabs.R`; Zenodo 10.5281/zenodo.20350571). The per-file
  markdown notes in the bucket's `documentation/` folder (97 files, 2026-06-24) are the authoritative schema docs; copies on disk under
  `.cache/data/sjcabs/doc_*.md` and `.cache/data/banc/documentation/`.
- Datasets/versions covered: BANC **888**, FAFB-FlyWire **783**, MANC **v1.2.1** ("manc_121"), hemibrain **v1.2.1** ("hemibrain_121"),
  MaleCNS **v0.9** ("malecns_09" — *not* v1.0), FANC **v1116** (skeletons + match columns only).
- Redistribution stance: the derived tables are redistributed openly in a public bucket and in the CC BY 4.0 Dataverse deposit; the README
  asks users to cite the *original* dataset papers plus Eckstein 2024 for NT predictions (and to tell Alex Bates). No separate license file
  for the compiled data ("CC-BY by the respective dataset creators"). **UNVERIFIED**: whether the MaleCNS v0.9 derived tables carry any
  Janelia-specific restriction (MaleCNS itself is CC BY 4.0).

### 6.2 Bucket contents and sizes (live listing 2026-09-26)

| Prefix | Files (bytes) |
|---|---|
| `banc_888/` (24 objects, 40,468,298,635 B) | see §5.3; plus `banc_banc_space_swc/`, `banc_banc_space_split_swc/`, `_prel2refresh_bak/` prefixes |
| `fafb_783/` (11 objects, 22,985,111,794 B) | `fafb_783_meta.feather` 13,539,866 (2026-08-28); `fafb_783_simple_edgelist.feather` 302,625,658 (15,023,799 rows: `pre, post, count, norm, total_input`); `fafb_783_split_edgelist.feather` 531,734,234 (15,867,088 rows; compartment-level, `pre_label/post_label/compartment_input`); `fafb_783_synapses.parquet` 2,043,922,885 / `.feather` 4,612,862,010 (per-synapse: `pre, post, x, y, z (nm), prepost, syn_top_nt, syn_top_nt_p, cleft_scores, connector_id, neuropil, label`); DCV (dense-core-vesicle) tables `fafb_783_cell_dcv_detection.feather` 10.4 GB, `fafb_783_soma_dcv_detection.feather` 3.98 GB, `fafb_783_cell_dcv_column_assignment.parquet` 621.7 MB, `fafb_783_synapse_column_assignment.parquet` 461.0 MB, `fafb_dcv_scores_metadata_ya_3_5_26.csv`; prefixes `fafb_banc_space_swc/` (6.95 GB zip on Dataverse), `fafb_fafb_space_swc/` |
| `manc_121/` (5 objects, 6,837,682,657 B) | `manc_121_meta.feather` 1,446,722; `manc_121_simple_edgelist.feather` 87,386,906 (5,305,354 rows); `manc_121_split_edgelist.feather` 336,072,226 (6,761,806 rows × 14 incl. `pre_top_nt/post_top_nt`); `manc_121_synapses.parquet` 2,541,292,945 / `.feather` 3,871,483,858; prefixes `manc_banc_space_split_swc/`, `manc_manc_space_split_swc/` |
| `hemibrain_121/` (5 objects, 1,150,695,317 B) | `hemibrain_121_meta.feather` 2,023,858; `hemibrain_121_simple_edgelist.feather` 92,125,914 (4,679,482 rows × 4: `pre, post, count, norm` — **no `total_input`**); `hemibrain_121_split_edgelist.feather` 152,548,258; `hemibrain_121_synapses.parquet` 903,983,829 (`.feather` is a 13 KB stub); prefixes `hemibrain_banc_space_swc/`, `hemibrain_hemibrain_space_swc/` |
| `malecns_09/` (4 objects, 16,856,448,362 B) | `malecns_09_meta.feather` 16,593,866; `malecns_09_simple_edgelist.feather` 3,404,193,362 (142,881,142 rows); `malecns_09_split_edgelist.feather` 4,951,221,930 (145,879,828 rows); `malecns_09_synapses.parquet` 8,484,439,204 (301,131,971 rows × 22); prefixes `JRC2018U/`, `malecns_banc_space_swc/`, `malecns_malecns_space_swc/` |
| `fanc_1116/` | only `fanc_banc_space_swc/` (Dataverse zip 203,119,978 B) |

Edgelist semantics (docs): `count` = synapses pre→post; `norm = count / total_input` (BANC: `count / post_count`); no pair threshold in
any sjcabs edgelist — FAFB's 15.0 M rows is the unthresholded v783 graph (Schlegel 2024: "139,255 nodes and around 15.1 million weighted
edges"), MANC's 5,305,354 ≈ neuPrint's 5,305,638 Traced→Traced edges, hemibrain's 4,679,482 > the 3,550,403 non-cropped-traced export because
it includes cropped/all 25,397 bodies (**UNVERIFIED** exact filter). Synapse tables carry `cleft_scores` (FAFB, MANC) and per-synapse
`syn_top_nt`/`syn_top_nt_p`.

### 6.3 The harmonized metadata schema (from the five `*_meta.feather` on disk)

Files on disk (`.cache/data/sjcabs/`), all MD5-identical to the bucket:

| File | Bytes | SHA-256 | Rows × cols |
|---|---|---|---|
| `fafb_783_meta.feather` | 13,539,866 | 5e248762d5308bf31e77b0ec5a6acd91ad11a22213b8cd985229c5917acd1823 | 144,837 × 24 |
| `hemibrain_121_meta.feather` | 2,023,858 | a2775c31450d3a060875bd5690adcb9800d3c9a580a9a919e10829984c79a1d1 | 25,397 × 19 |
| `manc_121_meta.feather` | 1,446,722 | 067c88e687ade67bd1e69b88765b4b21596280618cefde0fa43117d4af4a9439 | 23,650 × 15 |
| `malecns_09_meta.feather` | 16,593,866 | 1a75ae60fca67f76a97b95bab6c384f65b4f9115afd4834648146d16d46c75a4 | 165,114 × 33 |
| `banc_888_meta.feather` | 57,503,026 | 86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098 | 188,508 × 81 |
| `banc_888_metrics.feather` | 12,285,378 | 3ee647d402510b6b1c1bc3b0249db83f799ba94a1094c8ec530fcb09fac1ce60 | 188,508 × 19 |

Column presence matrix (computed; F = fafb_783, H = hemibrain_121, M = manc_121, C = malecns_09, B = banc_888):

| Column | F | H | M | C | B | Notes |
|---|---|---|---|---|---|---|
| `<dataset>_id` (`fafb_783_id`, `hemibrain_121_id`, `manc_121_id`, `malecns_09_id`, `banc_888_id`) | ✓ | ✓ | ✓ | ✓ | ✓ | **string** in all files; BANC also has `root_id`, `root_626/850/888/890`, `supervoxel_id` |
| `region` | ✓ | ✓ | ✓ | ✓ | ✓ | F: optic_lobe/central_brain/neck_connective; H: central_brain/optic_lobe/neck_connective; M: ventral_nerve_cord/neck_connective; C: +unknown; B: no neck_connective |
| `side` | ✓ | – | ✓ | ✓ | ✓ | left/right/center(/na); H lacks `side` (available in Schlegel Supp. 5) |
| `hemilineage` | ✓ | ✓ | ✓ | ✓ | ✓ | F/H/C ItoLee names (`ALad1`, `putative_primary`); M Truman (`19A`, `20A.22A`); B mixes both (259 values) |
| `nerve` | ✓ | ✓ | ✓ | ✓ | ✓ | long names `left_antennal_nerve`, `right_metathoracic_leg_nerve` (F 24, H 7, M 40, C 67, B 75 values) |
| `flow` | ✓ | ✓ | ✓ | ✓ | ✓ | afferent / intrinsic / efferent (C has one stray `c`) |
| `super_class` | ✓ | ✓ | ✓ | ✓ | ✓ | vocabulary: `optic_lobe_intrinsic, central_brain_intrinsic, ventral_nerve_cord_intrinsic, sensory, sensory_ascending, sensory_descending, ascending, descending, motor, visual_projection, visual_centrifugal, visceral_circulatory, ascending_visceral_circulatory, endocrine (H only), glia, trachea, not_a_neuron, unknown (C)` |
| `cell_class` | ✓ | ✓ | ✓ | ✓ | ✓ | snake_case long names (F 96, H 49, M 26, C 111, B 119 values), e.g. `olfactory_receptor_neuron`, `kenyon_cell`, `transmedullary`, `single_leg_neuromere` |
| `cell_sub_class` | ✓ | ✓ | ✓ | ✓ | ✓ | e.g. `antenna_olfactory_receptor_neuron`, `hind_leg_claw_chordotonal_organ_neuron`, `ventral_nerve_cord_ipsilateral_restricted` |
| `cell_type` | ✓ | ✓ | ✓ | ✓ | ✓ | F 8,811 distinct / H 5,686 / M 4,075 / C 11,844 / B 11,504 |
| `neurotransmitter_predicted` | ✓ | ✓ | ✓ | ✓ | ✓ | full names; F 6 classes (acetylcholine, glutamate, gaba, dopamine, serotonin, octopamine); H 8 (+`unknown`, `neither`); M 5 (acetylcholine, glutamate, gaba, unknown, unclear); C 8; B 8 (+histamine, tyramine) |
| `neurotransmitter_score` | ✓ | ✓ | – | ✓ | ✓ | **string dtype in F and H** (float in C, B) — cast on read |
| `cell_function`, `cell_function_detailed`, `body_part_sensory`, `body_part_effector` | ✓ | ✓ | ✓ | ✓ | ✓ | controlled vocab in `data/meta_data_entries.csv` (248 rows) |
| `status` | ✓ | ✓ | – | ✓ | ✓ | F: outlier_seg/outlier_bio/not_a_neuron/tiny/duplicate/…; H: Traced/Orphan; C: 7 neuPrint statuses; B: comma-joined flags |
| `sexually_dimorphic` | ✓ | – | – | ✓ | ✓ | isomorphic / dimorphic / female-specific / male-specific (+ "potentially …" in C) |
| `instance`, `cropped`, `root` | – | ✓ | – | – | – | hemibrain-only |
| `neuromere`, `tract` | – | – | – | ✓ | ✓ | (C `neuromere` all null) |
| `fafb_cell_type`, `hemibrain_cell_type`, `manc_cell_type` | – | – | – | ✓ | ✓ | cross-dataset type matches; B adds `malecns_cell_type`, `fanc_cell_type`, `fafb_alignment_cell_type`, `*_match` (IDs) and `*_nblast_match` |
| `nucleus_id`, `pd_width`, `volume_nm3`, `soma_dcv_density`, `cell_dcv_density_um(3)` | ✓ | – | – | – | (B: nucleus_id, pd_width, volume_nm3) | FAFB extras |
| `optic_lobe_hex_1/2`, SeaTable `_id/_ctime/_mtime/_creator/_archived/_locked*` | – | – | – | ✓ | – | MaleCNS extras (leaked SeaTable bookkeeping columns) |
| `proofread`, `roughly_proofread`, `position`, `root_position(_nm)`, `root_region`, clusters, `cns_network`, `peripheral_target_type`, `neurotransmitter_verified`, `neuropeptide_verified`, metrics, `seed_01..14` | – | – | – | – | ✓ | BANC-only |

Row-set semantics differ: `fafb_783_meta` has 144,837 rows (139,255 neurons + 5,429 glia + 72 trachea + other non-neurons; `region`/`flow`
null for 5,501); `hemibrain_121_meta` 25,397 (all annotated bodies incl. 49 Orphan and 4,039 cropped); `manc_121_meta` 23,650 (≈ v1.2.1
Traced 23,665 minus 15 glia); `malecns_09_meta` 165,114 (all v0.9 neurons); `banc_888_meta` 188,508 (all segments touched by annotation,
incl. 12,750 glia and 28,632 without super_class). Counts per `super_class` are in `scratchpad/out_sjcabs.txt`.

```python
import pyarrow.feather as pf

metas = {
    n: pf.read_table(f".cache/data/sjcabs/{n}_meta.feather").to_pandas()
    for n in ["fafb_783", "hemibrain_121", "manc_121", "malecns_09", "banc_888"]
}
common = set.intersection(*[set(m.columns) for m in metas.values()])
# {'region','hemilineage','nerve','flow','super_class','cell_class','cell_sub_class','cell_type','neurotransmitter_predicted',
#  'cell_function','cell_function_detailed','body_part_sensory','body_part_effector'}
```

### 6.4 Alignment table — seed of the flyconn harmonized schema

Sources: sjcabs/BANC compiled meta (above); Codex FAFB `classification.csv.gz`/`neurons.csv.gz`/`fw_and_hemibrain_types.csv.gz` (§1.4);
Codex BANC/MANC `neurons.csv.gz` (§3.3/§5.4); Schlegel `Supplemental_file1` (§1.8); MaleCNS `body-annotations-…feather` (data_malecns.md §4.1)
and `body-neurotransmitters-…feather`; hemibrain neuPrint/`traced-neurons.csv`; MANC neuPrint properties (§3.3); FANC CAVE tables (§4.1).

| flyconn field (proposed) | sjcabs / BANC compiled | Codex FAFB CSVs | Codex BANC/MANC `neurons.csv` | Schlegel Supp. 1 (FlyWire) | MaleCNS v1.0 annotations | hemibrain neuPrint | MANC neuPrint | FANC CAVE |
|---|---|---|---|---|---|---|---|---|
| `neuron_id` (string) | `<ds>_id` / `root_id` | `root_id` (int64) | `Root ID` | `root_id` (+`supervoxel_id`) | `bodyId` | `bodyId` | `bodyId` | `pt_root_id` (root) / `user_id` in `cell_ids_v2` (stable cell id) |
| `dataset`, `version` | folder name (`fafb_783`, `manc_121`, …) | path (`fafb/783`) | path (`banc/888`, `manc/1.2.1`) | `783` (README) | `v1.0` | `hemibrain:v1.2.1` | `manc:v1.2.1` | `fanc_production_mar2021` + materialization |
| `flow` | `flow` (afferent/intrinsic/efferent) | `flow` | `Flow` | `flow` | *not present* (derive from superclass) | – | – (derive from `class`) | – |
| `super_class` | `super_class` (BANC vocab: `central_brain_intrinsic`, `optic_lobe_intrinsic`, `ventral_nerve_cord_intrinsic`, `sensory`, `ascending`, `descending`, `motor`, `visual_projection`, `visual_centrifugal`, `visceral_circulatory`, `sensory_ascending`, …) | `super_class` (`central`, `optic`, `sensory`, `sensory_ascending`, `ascending`, `descending`, `motor`, `endocrine`, `visual_projection`, `visual_centrifugal`) | `Super Class` (BANC vocab; MANC: `intrinsic_neuron`, `sensory`, `ascending`, `descending`, `motor`, `efferent`, …) | `super_class` (= Codex FAFB vocab) | `superclass` (`cb_intrinsic`, `ol_intrinsic`, `vnc_intrinsic`, `cb_sensory`, `ol_sensory`, `vnc_sensory`, `ascending_neuron`, `descending_neuron`, `vnc_motor`, `cb_motor`, `visual_projection`, `visual_centrifugal`, `sensory_ascending`, `cb_endocrine`, `vnc_efferent`, …) | – (`instance` prefixes only) | `class` (`intrinsic neuron`, `sensory neuron`, `motor neuron`, `descending neuron`, `ascending neuron`, `efferent neuron`, `sensory ascending`, `efferent ascending`) | `neuron_information` primary class (`sensory neuron`, `central neuron`, `motor neuron`, `efferent non-motor neuron`, `glia`) — not in public export |
| `cell_class` | `cell_class` (long snake_case) | `class` (short: `Kenyon_Cell`, `CX`, `olfactory`, `ALPN`, `DN`, `optic_lobe_intrinsic`, …) | `Class` | `cell_class` (short, + `ME>LO`-style optic classes) | `class` (`visual`, `Kenyon_Cell`, `CX`, `olfactory`, `mechanosensory_tactile`, …) | – | – | sub-class tags |
| `cell_sub_class` | `cell_sub_class` | `sub_class` | `Sub Class` | `cell_sub_class` | `subclass` | – | `subclass` | – |
| `cell_type` | `cell_type` | `cell_type` (FlyWire) + `hemibrain_type`; Codex `primary_type` = consolidated | `Primary Cell Type` + `Alternative Cell Type(s)` | `cell_type` + `hemibrain_type` | `type` (+ `instance`, `group`, `supertype`) | `type` (+ `instance`) | `type`, `systematicType`, `group`, `serial` | `cell_info`/`neuron_information` tags (restricted) |
| `hemilineage` | `hemilineage` | `hemilineage` (ItoLee) | `Hemilineage` | `ito_lee_hemilineage`, `hartenstein_hemilineage` | `itoleeHl`, `trumanHl` | – (Schlegel Supp. 5 `ito_lee_hemilineage`) | `hemilineage` (Truman) | hemilineage tag `0A…27X` |
| `side` | `side` (left/right/center) | `side` | `Soma side` | `side` | `somaSide` (L/R/M), `rootSide` | – (Supp. 5 `side`; `instance` suffix `_L/_R`) | `somaSide` (LHS/RHS/Midline), `rootSide` | tags `neck connective (left/right)` |
| `region` | `region` (central_brain / optic_lobe / ventral_nerve_cord / neck_connective) | – (derive) | `Top in/out region` (neuropil-based name) | – | derive from `superclass` prefix / `somaNeuromere` | ROI booleans | ROI booleans, `somaNeuromere` | neuropil column in synapse tables |
| `nerve` | `nerve` (long names) | `nerve` (abbrev. `AN`, `CV`, `MxLbN`, `OCN`, `PhN`, `aPhN`, `NCC`, `ON`) | `Nerve` | `nerve` | `entryNerve`, `exitNerve` | – | `entryNerve`, `exitNerve` | `peripheral_nerves.tag` |
| `neuromere` | `neuromere` (BANC/MaleCNS) | – | – | – | `somaNeuromere` | – | `somaNeuromere` | – |
| `nt_predicted` (full names) | `neurotransmitter_predicted` | `nt_type` (ACH/GABA/GLUT/DA/SER/OCT) + `*_avg` | `Predicted NT type` (adds HIST, TYR) | `top_nt` | `predicted_nt` / `consensus_nt` (7 classes incl. histamine) | none in DB; feather `predicted_nt` (6 + neither) | `predictedNt` (3 + unknown/unclear) | fast-NT tags (restricted) |
| `nt_score` | `neurotransmitter_score` | `nt_type_score` | `Predicted NT confidence` | `top_nt_conf` | `predicted_nt_confidence` | feather max prob | `predictedNtProb` | – |
| `nt_verified` | `neurotransmitter_verified` (BANC) | – | `Verified NT type` | `known_nt`, `known_nt_source` | `ground_truth` (NT file) | – | `transmission` (electrical/neurosecretory) | – |
| `proofread_status` | `proofread`/`roughly_proofread` (BANC), `status` | (all rows are proofread) | – | `status` outliers | `status`/`statusLabel` | `status`, `cropped` | `status`, `statusLabel` | `proofread_first/second_pass` |
| cross-dataset types | `fafb_cell_type`, `manc_cell_type`, `hemibrain_cell_type`, `malecns_cell_type`, `fanc_cell_type`, `*_match` ids | `hemibrain_type` | `Alternative Cell Type(s)` | `hemibrain_type`, `supertype` (MaleCNS) | `flywireType`, `hemibrainType`, `mancType`, `mancBodyid` | – | `synonyms` | BANC `fanc_match` |
| `dimorphism` | `sexually_dimorphic` | `dsx_fru_types.csv` (gene) | – | `dimorphism`, `fru_dsx`, `matching_notes` | `dimorphism`, `fruDsx` | – | – | – |
| morphology metrics | `l2_cable_length_um`, `volume_nm3` (B/F) | `cell_stats.csv` `length_nm, area_nm, size_nm` | `Cable length/Surface area/Volume` (mostly empty) | – | – | `size`, `pre`, `post` | `size`, `pre`, `post`, `synweight` | `somas.volume` |
| soma position | `position` (voxel string, B), `root_position_nm` | `coordinates.csv` `position` (4×4×40 nm voxels) | – | `soma_x/y/z`, `pos_x/y/z` (voxels) | `somaLocation` (8 nm voxels) | `somaLocation` | `somaLocation` (8 nm) | `pt_position` (4.3×4.3×45 nm voxels) |

Design implications: (1) store IDs as strings/uint64 (BANC/sjcabs strings vs Codex int64 vs FANC/BANC 64-bit roots); (2) keep the source
vocabulary in a `*_raw` column and map to the sjcabs/BANC `super_class` vocabulary (mapping Codex `central→central_brain_intrinsic`,
`optic→optic_lobe_intrinsic`, MaleCNS `cb_intrinsic→central_brain_intrinsic`, `ol_intrinsic→optic_lobe_intrinsic`, `vnc_intrinsic→
ventral_nerve_cord_intrinsic`, `cb/ol/vnc_sensory→sensory`, `ascending_neuron→ascending`, `descending_neuron→descending`,
`vnc_motor/cb_motor→motor`, MANC `intrinsic neuron→ventral_nerve_cord_intrinsic`, hemibrain: derive from Schlegel Supp. 5 / sjcabs);
(3) normalise NT labels to full lower-case names and record the class set per source (3/6/7/8 classes); (4) carry both a per-dataset
`cell_type` and typed cross-reference columns (`fafb_783_cell_type`, `manc_121_cell_type`, `hemibrain_121_cell_type`, `malecns_*_cell_type`,
`fanc_1116_cell_type`) plus `*_match_id`, exactly as BANC's `codex_annotations` does; (5) edge tables: `pre, post, count[, neuropil]` with
explicit `threshold` metadata (FAFB Codex ≥5 pair, Codex-BANC ≥3, MANC/MAOL ≥1, FANC export ≥3, MaleCNS Codex ≥5, sjcabs none) and
`synapse_source` (Buhmann/Princeton/v2/v3/neuPrint-weight vs weightHP).

---

## 7. Cross-dataset cell-type vocabulary — where the matches live

| Source | Columns carrying cross-dataset type matches | Coverage (computed) |
|---|---|---|
| FlyWire Codex 783 | `fw_and_hemibrain_types.csv.gz`: `cell_type` (FlyWire) / `hemibrain_type` (hemibrain v1.2.1 type name); `consolidated_cell_types.csv.gz`: `primary_type`, `additional_type(s)`; `classification_with_fw_and_hemibrain_types.csv.gz` | hemibrain_type 29,764 neurons (3,551 types); primary_type 138,327 |
| Schlegel `Supplemental_file1` (v3.1.0) | `cell_type`, `hemibrain_type` (33,271 neurons / 4,217 types), `supertype` (MaleCNS neuPrint supertype id, 33,845), `synonyms`, `fbbt_id` (FlyBase ontology, 28,905), `vfb_id` | + `Supplemental_file5_hemibrain_meta.csv`: `type`, `morphology_type` (suffix-collapsed), `fbbt_id`, `ito_lee_hemilineage`, `side`, `cell_class` for 25,397 hemibrain bodies |
| MaleCNS v1.0 annotations (data_malecns.md §4.1) | `flywireType` (143,156 bodies / 8,199 types), `hemibrainType` (32,919 / 4,495), `mancType` (22,744), `mancBodyid` (18,715), `mancGroup`, `mancSerial`, `synonyms`, `vfbId`, `supertype`, `group` | sjcabs `malecns_09_meta`: `fafb_cell_type` 140,421 (8,134), `hemibrain_cell_type` 32,798 (4,458), `manc_cell_type` 22,745 (3,848) |
| BANC v888 (`banc_888_meta.feather`) | `fafb_cell_type` 131,240 / `fafb_match` 64,789 / `fafb_nblast_match` 127,298 / `fafb_alignment_cell_type` 124,537; `manc_cell_type` 26,613 / `manc_match` 24,277; `malecns_cell_type` 43,490 / `malecns_match` 23,608; `hemibrain_cell_type` 30,910 / `hemibrain_match` 6,768; `fanc_cell_type` 2,905 / `fanc_match` 2,451 | CAVE `codex_annotations` classification systems `fafb_783_cell_type`, `fafb_783_match_id`, `manc_121_cell_type`, `manc_121_match_id`, `hemibrain_121_cell_type`, `hemibrain_121_match_id`, `malecns_09_cell_type`, `malecns_09_match_id`, `fanc_1116_cell_type`, `fanc_1116_match_id`, `other_names` |
| BANC `nblast/` + reviewed matches | `banc_{fafb_783,manc_v1.2.1,hemibrain_v1.2.1,malecns_v0.9,fanc_1116}_nblast.feather` (long tables: BANC `root_*` query, `match_id`, `match_cell_type` frozen snapshot, `score`, `validation`), `banc_*_reviewed_matches.csv`, `banc_to_*_verified_png_matches.zip` (Dataverse) | 5.48 GB total |
| Hemibrain | none in neuPrint; FlyWire↔hemibrain mapping comes from Schlegel 2024 (Supp. 1 `hemibrain_type`, Zenodo NBLAST `nblast_flywire_hemibrain_min_comp.feather`) | — |
| MANC | `synonyms`, `systematicType`, `group`, `serial` (serial homology); MaleCNS `mancType/mancBodyid` provide MANC↔MaleCNS; BANC provides MANC↔BANC | — |
| FANC | only via BANC (`fanc_cell_type`, `fanc_match`, `banc_fanc_1116_nblast.feather`) and Stürner/Brooks 2025 (DN/AN FANC↔MANC↔FAFB) | — |

Known mapping files: Schlegel 2024 Supplementary files (GitHub `flyconnectome/flywire_annotations`, Zenodo 10877326 NBLAST scores);
Codex `fw_and_hemibrain_types.csv.gz` / `consolidated_cell_types.csv.gz`; MaleCNS `body-annotations` (`flywireType`, `hemibrainType`,
`mancType`); BANC `codex_annotations.parquet` (+ Supplementary Data 2–5 = per-neuron tables for BANC, FAFB, MANC, MaleCNS with a common
`dataset` column and `*_match` ids); BANC `nblast/*_reviewed_matches.csv`; VFB ids (`vfb_id`/`vfbId`) and FBbt ids as the neutral
ontology key across FlyWire, hemibrain (Supp. 5), MaleCNS (`vfbId`) and MANC (`vfbId` in v1.2.1; Codex `Community labels vfbId::`).
Nomenclature caveats: hemibrain types carry connectivity suffixes (`_a/_b`) that FlyWire collapses; MANC `systematicType` differs from
`type`; BANC `cell_type` inherits FAFB names for brain/DN and MANC names for VNC/AN neurons; MaleCNS `type` (11,751 distinct) vs
`flywireType` (8,199) — the male-specific and split types have no FAFB counterpart; `fanc_1116` vs the public FANC exports `v1237/v1444` —
root IDs differ between materializations, so FANC matches must be re-mapped through the chunked graph (token required).

---

## 8. Open items / not verified

1. CAVE materialization lists for `flywire_fafb_public`, `brain_and_nerve_cord_public`, `fanc_production_mar2021` (all endpoints now
   require a token); FANC v1444 timestamp; whether newer BANC materialization 890 is exposed publicly.
2. Provenance of `hemibrain-v1.2-body-mean-neurotransmitters.feather` (classifier, thresholds) — no README in `gs://hemibrain/v1.2/`.
3. FlyWire Terms of Service text; whether Codex's `nt_type` re-derivation between 630 and 783 changed the DA calls (3,189 → 584).
4. Exact filter behind sjcabs `hemibrain_121_simple_edgelist` (4,679,482 rows vs 3,550,403 non-cropped) and `fafb_783_meta` non-neuron rows.
5. FANC license (none stated), FANC `size`/confidence thresholds in the Nov-2022 synapse table, FANC cell-type table (restricted).
6. The 81 Shiu-only / 659 Codex-only v630 root IDs (lineage lookup needs CAVE token); reproduction of the Shiu edge list from
   `630/no_threshold_connections.csv.gz` (not downloaded).
7. Whether MANC v1.2.x flat exports exist anywhere besides `gs://manc-seg-v1p2/manc-v1.2-synapse-partners-minconf-0.0.feather`; MANC v1.2.1
   vs v1.2.3 annotation differences; DOIs quoted from memory (Phelps 2021 page range, neuPrint paper DOI).
