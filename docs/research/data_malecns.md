# MaleCNS v1.0 — dataset verification notes

Scope: the MaleCNS v1.0 connectome release only (Janelia FlyEM + Cambridge Drosophila Connectomics Group + MRC LMB + Google Research).
Verified on 2026-09-26 by actually fetching pages, downloading files, and computing on them. Anything not directly
verified is marked **UNVERIFIED**. All commands were run from the venv at
`/private/tmp/claude-502/-Users-moizp-google-fly/bdaccf09-812b-4c61-9717-b57ad8a87724/scratchpad/research-venv`
(pyarrow, pandas 3.0.6, numpy, requests). `gsutil`/`gcloud` are **not installed** on this machine; every download
used the public HTTPS mirror `https://storage.googleapis.com/flyem-male-cns/...` with `curl -C -` (resumable).
Local data directory: `/Users/moizp/google-fly/.cache/data/malecns/`.

---

## 1. Release metadata (from https://male-cns.janelia.org/)

Sources fetched raw with `curl -sSL` and converted to text (FireCrawl had no credits; the pages are static MkDocs HTML):
`https://male-cns.janelia.org/download/`, `https://male-cns.janelia.org/release/`, `https://male-cns.janelia.org/`,
`https://male-cns.janelia.org/explore/`, `https://male-cns.janelia.org/build/dimorphism_overview/`,
`https://www.janelia.org/project-team/flyem/male-cns-connectome`.

| Item | Value | Where verified |
|---|---|---|
| Release version string | **`v1.0`** (release notes heading "v1.0 (June 8, 2026)"); neuPrint dataset id **`male-cns:v1.0`**; neuPrint `Meta.tag = "v1.0"`; GCS prefix `gs://flyem-male-cns/v1.0/` | release page; neuPrint Meta node |
| Release notes text (complete) | `v1.0 (June 8, 2026)` — "Minor proofreading changes", "Refinement of neuron annotations". `v0.9 (October 5, 2025)` — "Initial release of male CNS connectome". That is the entire release-notes page; it contains **no counts, no citation, no threshold definitions**. | https://male-cns.janelia.org/release/ |
| News timeline (home page) | 2026-09-03 MaleCNS paper published; 2026-06-08 v1.0 released; 2025-11-07 NeuronBridge matches; 2025-10-30 preprint v2 on bioRxiv; 2025-10-03 v0.9 released (note: release page says Oct 5 for v0.9, home page says Oct 3) | https://male-cns.janelia.org/ |
| License | "The Male CNS is licensed under CC-BY" linking to https://creativecommons.org/licenses/by/4.0/ (footer on every page: "The Male CNS dataset is licensed under CC-BY."). No separate terms-of-use page exists on the site. The Cell paper is also CC BY 4.0 (Crossref license list includes `http://creativecommons.org/licenses/by/4.0/`; Europe PMC `license: cc by`). | download page footer; Crossref; Europe PMC |
| Terms of use | None beyond CC-BY 4.0. **UNVERIFIED** whether neuPrint account creation imposes additional terms (not tested; no account created). | — |
| neuPrint server | `https://neuprint.janelia.org`, dataset `male-cns:v1.0` (also `male-cns:v0.9` still served). Deep link: `https://neuprint.janelia.org/?dataset=male-cns%3Av1.0&qt=findneurons` | download page; `/api/dbmeta/datasets` |
| Clio | `https://clio.janelia.org/ws/annotate?dataset=male-cns:v1.0-v1.0&tab=bodies` (note the odd dataset id `male-cns:v1.0-v1.0`) | home page |
| Other portals | Cell Type Explorer https://reiserlab.github.io/celltype-explorer-drosophila-male-cns ; Dimorphism Explorer https://male-cns.janelia.org/build/dimorphism_overview/ ; NeuronBridge; neuroglancer scene `gs://flyem-male-cns/v1.0/male-cns-v1.0.json` | home/explore pages |
| Stated counts on the site | Site itself states none on download/release pages. neuPrint dataset description: v0.9 "166k neurons", v1.0 "167k neurons". Janelia overview page: "262 sex-specific and 114 sexually dimorphic cell types, comprising 4.8% of the central brain" (this is preprint-era wording; the Cell abstract gives 289 male-specific / 71 female-specific / 138 dimorphic). | neuPrint `/api/dbmeta/datasets`; Janelia page |
| Stated confidence thresholds | Site: none stated (file names only carry `minconf-0.5`). neuPrint `Meta`: `postHighAccuracyThreshold = 0.5`, `postHPThreshold = 0.7`, `preHPThreshold = 0`. See §8 for the verified meaning of `minconf`. | neuPrint Cypher |

### Publication (verified via Crossref `https://api.crossref.org/works/10.1016/j.cell.2026.08.015` and Europe PMC)

- **Title:** Sexual dimorphism in the complete *Drosophila* male central nervous system connectome
- **Journal:** Cell, volume 189, issue 18, pages 5504–5526.e15, September 2026 (Crossref `created` 2026-09-03T15:01:57Z; Europe PMC `firstPublicationDate` 2026-09-01; site news item says published 2026-09-03). PMID 42691995.
- **DOI:** `10.1016/j.cell.2026.08.015` (PII S0092-8674(26)00942-6, URL https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6 — the Cell page itself is behind a Cloudflare challenge and returned 403/“Just a moment” to curl and WebFetch)
- **Authors:** 111 authors. First 10: Stuart Berg; Isabella R. Beckett; Marta Costa; Philipp Schlegel; Michał Januszewski; Elizabeth C. Marin; Aljoscha Nern; Stephan Preibisch; Wei Qiu; Shin-ya Takemura. Last 5: Michael B. Reiser; Harald F. Hess; Gerald M. Rubin; Gregory S.X.E. Jefferis (preceded by Carlos Ribeiro, Gwyneth M. Card, Scott Waddell, Louis K. Scheffer, Stephan Saalfeld ...).
- **License:** CC BY 4.0.
- **Preprint:** bioRxiv `10.1101/2025.10.09.680999` (v2 posted 2025-10-30). bioRxiv API/WebFetch returned empty/429, so preprint details beyond the DOI are from the site's links only.
- **Abstract (Europe PMC, verbatim numbers):** "This contains **166,700 neurons** spanning the brain and nerve cord, fully proofread and annotated, including fruitless/doublesex expression and **11,710 neuron types**. ... finding **8,069 isomorphic, 138 dimorphic, 289 male-specific, and 71 female-specific types**."
- The abstract does **not** state a synapse count. The "~125M synapses" figure in our brief is reproduced exactly by neuPrint's Neuron→Neuron `ConnectsTo` weight sum (125,024,863) and approximately by the flat file (124,025,046 for Traced→Traced), see §6. Wikipedia's "312M synapses" corresponds to total postsynaptic partners (311,833,243).

### Files listed on the download page (all under `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`)

HTTPS form: `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<name>`.
Sizes/MD5 below come from `curl -I` headers and the GCS JSON listing
(`https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o?prefix=v1.0/connectome-data/`), not from the page.

| File | Page-stated size | Exact bytes | GCS md5 (base64) | GCS last-modified | Page description (verbatim, abridged) |
|---|---|---|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | 14,483,314 | `UKdxh3DFciDxYLpPQxq4ng==` | 2026-06-03T13:54:38Z | "Curated neuron annotations (classes, types, sides, etc.), excluding neurotransmitter properties." |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | 43,282,834 | `PYQrEv5cSe763lKNfdJKHw==` | 2026-06-08T05:01:39Z | "Aggregate neurotransmitter predictions for each neuron. See manuscript methods section for details." |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 780 MB | 778,062,826 | `QEwzScKFgBSOFoFeuZ84Kg==` | 2026-06-03T13:54:48Z | "summary statistics (synapse counts) of all segments in the dataset (excluding those with no synapses)" |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.1 GB | 1,051,241,946 | `8w6dzKJc/QIb8eez2XVZng==` | 2026-06-03T13:54:47Z | "segment-to-segment connection strengths for all segments in the dataset (excluding those with no synapses). This is the full connection graph." |
| `syn-points-male-cns-v1.0-minconf-0.5.feather` | 12.7 GB | 13,061,489,098 | `xp0IdY3gdYIDXMiENXRJOg==` | 2026-06-03T13:56:31Z | "Pre-synapse and post-synapse locations, body (segment) ID, and encompassing ROIs. ... 'kind' column indicates PreSyn or PostSyn. ... uniquely identified by their x,y,z locations (voxel units, i.e. 8nm). ... each unique pre-synaptic location is listed only once." |
| `syn-partners-male-cns-v1.0-minconf-0.5.feather` | 6.8 GB | 6,777,179,098 | `WO/PcS+MTU3l8q1R6X3vdg==` | 2026-06-03T13:55:42Z | "Synaptic partner pairs, along with their associated body IDs and primary neuropil. Columns: 'x_pre','y_pre','z_pre','body_pre','conf_pre','x_post','y_post','z_post','body_post','conf_post','primary_post'" |
| `tbar-neurotransmitters-male-cns-v1.0.feather` | 2.7 GB | 2,651,680,218 | `UbAsEWkGYq7e8o+G05T/DQ==` | 2026-06-08T05:02:07Z | "Neurotransmitter prediction probabilities for each pre-synapse" |

**Undocumented extra files present in the bucket (not listed on the page):**

| File | Exact bytes | GCS md5 | Rows (from footer, see §4.6) |
|---|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | 508,025,642 | `ZgHUrQr6mf0D6wh5Ze8kIw==` | 25,563,197 |
| `connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather` | 502,169,298 | `CfL4M/cWGkatM81vnJBy9g==` | 25,568,639 |
| `syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather` | 2,965,367,002 | `9bwcXONKAbaJVkFLUw7dqA==` | not computed |
| `syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather` | 2,965,702,122 | `pbz0bYqLglq4yjJzxHeV4w==` | not computed |

Other bulk assets named on the download page (not downloaded): EM volumes `gs://flyem_cns_z0720_07m_dvidcoords_n5` (N5, 8 nm, uint8) and
`gs://flyem-male-cns/em/em-clahe-jpeg`; segmentation `gs://flyem-male-cns/v1.0/segmentation` (8 nm, uint64 with only low 32 bits used);
nuclei seg `gs://flyem-male-cns/v1.0/malecns-v1.0-nuclei-seg-16nm`; ROIs `gs://flyem-male-cns/rois/fullbrain-roi-v4` (256 nm, max label 96) and
`gs://flyem-male-cns/rois/malecns-vnc-neuropil-roi-v0` (256 nm, max label 27); skeletons in SWC (8 nm units) and precomputed (1 nm) under
`gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/`, mirrored variants under `skeletons-malecns-mirrored/`, and JRC2018-unisex-space
skeletons (µm) under `skeletons-unisex-template/`. Top-level `v1.0/` prefixes seen in the bucket: `connectome-data/`, `database/`,
`male-cns-meshes-transformed-to-fafb-flywire/`, `male-cns-v1.0-synapses-precomputed/`, `malecns-v1.0-nuclei-seg-16nm/`,
`malecns-v1.0-optic-lobe-column-pins/`, `malecns-v1.0-soma-points/`, `nblasts/`, `segmentation/`, `supervoxels/`, `synapse-ground-truth/`.

---

## 2–3. Downloads and checksums

```bash
cd /Users/moizp/google-fly/.cache/data/malecns
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather \
         body-neurotransmitters-male-cns-v1.0.feather \
         body-stats-male-cns-v1.0-minconf-0.5.feather \
         connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  curl -sS -C - --retry 5 --retry-delay 3 -o "$f" "$B/$f"
done
for f in *.feather; do
  echo "$f $(stat -f %z "$f") sha256=$(shasum -a 256 "$f" | cut -d' ' -f1) md5b64=$(openssl dgst -md5 -binary "$f" | base64)"
done
```

All four downloads succeeded on the first attempt (the 1.05 GB weights file took ~4 min). Local MD5s match the GCS `x-goog-hash` MD5s exactly.

| File | Bytes | SHA-256 | MD5 matches GCS |
|---|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` | yes |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` | yes |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` (extra, not requested) | 778,062,826 | `ca5dc83a26382ae70c8d8f42fc09ce2dbc1af7c03f3a001a1936b5e142540647` | yes |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` | yes |

Not downloaded (per brief): syn-points (13.06 GB), syn-partners (6.78 GB), tbar-neurotransmitters (2.65 GB). Their schemas were read
remotely via HTTP Range requests (§4.6).

---

## 4. Schemas and contents

Common snippet:

```python
import pyarrow.feather as pf, pandas as pd, time
t0 = time.time(); t = pf.read_table(path); print(t.num_rows, t.num_columns, t.nbytes, time.time() - t0)
print(t.schema.to_string(show_schema_metadata=False))   # all files carry only a b'pandas' metadata blob
df = t.to_pandas(); print(df.memory_usage(deep=True).sum()); print(df.head())
```

Note: with pandas 3.x, Arrow `string` columns load as the new `str` dtype (not `object`); code that tests `dtype == object` will miss them.

### 4.1 `body-annotations-male-cns-v1.0-minconf-0.5.feather`

- **Rows 211,577** (one per body; `bodyId` unique; min 10001, max 1,571,825,087). 36 columns. Arrow 50.9 MB, pandas 90.6 MB, load 0.014 s.
- Index: RangeIndex. Only `bodyId` is fully populated.

| Column | Arrow type | Non-null | Notes / distinct values |
|---|---|---|---|
| `bodyId` | int64 | 211,577 | unique key |
| `status` | string | 206,105 | Traced 165,122; Orphan 15,925; Glia 11,864; Unimportant 10,751; Assign 1,832; Anchor 611; null 5,472 |
| `statusLabel` | dictionary<string, int8, ordered> | 210,690 | 31 categories in dictionary (ordered), 20 used: Roughly traced 71,979; Reviewed 54,066; Prelim Roughly traced 36,387; Orphan 12,075; Glia 11,864; Unimportant 10,751; Out of scope 4,585; Orphan-artifact 2,318; RT Hard to trace 2,019; 0.5assign 1,832; Orphan hotknife 1,428; Leaves 528; Anchor 283; Soma Anchor 201; PRT Orphan 140; Hard to trace 104; Sensory Anchor 101; Partially traced 26; RT Orphan 3; null 887. Unused categories: Putative Leaves, Not examined, "", Cleaved Anchor, Will be merged, Cervical Anchor, Examined Soma Anchor, Primary Anchor, Traced in ROI, Traced, Finalized. |
| `superclass` | string | **166,700** | ol_intrinsic 89,403; cb_intrinsic 32,164; vnc_intrinsic 13,161; visual_projection 9,201; vnc_sensory 6,370; ol_sensory 6,098; cb_sensory 4,868; ascending_neuron 1,846; descending_neuron 1,314; vnc_motor 708; visual_centrifugal 563; sensory_ascending 537; cb_motor 107; vnc_efferent 94; cb_endocrine 72; ENS 50; vnc_tbc 38; vnc_sensory_tbc 36; vnc_endocrine 22; cb_sensory_tbc 14; sensory_descending 12; efferent_ascending 8; efferent_descending 4; cb_efferent 4; visual_projection_tbc 2; sensory_ascending_tbc 2; descending_neuron_tbc 2 |
| `class` | string | 26,513 | 21 values: visual 6,091; Kenyon_Cell 4,064; CX 2,950; olfactory 2,639; mechanosensory_tactile 2,558; mechanosensory 1,733; unknown_sensory 1,712; mechanosensory_proprioceptive 1,454; gustatory 1,428; ALPN 686; ALLN 420; DAN 340; ol_bilateral 116; MBON 97; hygrosensory 66; chemosensory 58; SEZPN 27; thermosensory 25; ALIN 24; ALON 14; mechanosensory_tbc 11 |
| `subclass` | string | 21,930 | 49 values, mostly sensory sub-modalities / VNC tracts (BI 3,336; IR 3,048; mechanosensory bristle 2,206; BR 1,883; BA 1,336; abdomen 1,145; CR 1,039; leg 873; ...) |
| `type` | string | 164,506 | **11,751 distinct** (top: R1-R6 3,377; Tm3 2,054; T3 1,940; T2a 1,872; L5 1,787) |
| `instance` | string | 161,506 | 23,932 distinct; convention `<type>_<L|R|M>` (e.g. `DNp01(GF)_R`) |
| `group` | double | 147,514 | 13,746 distinct among neurons; numeric group id (body id of a representative) |
| `supertype` | string | 34,096 | 1,925 distinct; numeric-string ids |
| `somaSide` | string | 150,726 | L 75,215; R 75,119; M 392; null 60,851 (18,003 null among the 166,700 neurons) |
| `rootSide` | string | 17,939 | R 9,723; L 7,802; unknown 414 (sensory/VNC root side) |
| `somaNeuromere` | string | 21,820 | T2 5,076; T1 4,296; T3 3,976; CG 2,401; LB 1,313; A1 933; TC 729; DC 568; MX 500; MD 372; A2–A10 (39–273 each); GNG 89; NA 2 |
| `somaLocation` | list<int64> | 141,781 | `[x, y, z]` in 8 nm voxels |
| `tosomaLocation` | list<int64> | 995 | point toward soma for bodies lacking soma |
| `flywireType` | string | 143,156 | 8,199 distinct (matching FlyWire/FAFB type) |
| `hemibrainType` | string | 32,919 | 4,495 distinct |
| `mancType` / `mancBodyid` / `mancGroup` / `mancSerial` | string / double / double / double | 22,744 / 18,715 / 14,554 / 5,422 | cross-refs to MANC v1.x |
| `mcnsSerial` | double | 3,945 | serial-homology id |
| `serialMotif` | string | 902 | cns convergent 341; cns tiling 240; vnc convergent 180; independent 44; brain convergent 42; sequential 38; complex 11; centripetal 6 |
| `itoleeHl` | string | 37,754 | 203 distinct hemilineages (Ito/Lee nomenclature); top `putative_primary` 3,915 |
| `trumanHl` | string | 19,753 | 76 distinct (Truman VNC hemilineage nomenclature, e.g. 07B, 20A.22A, 19A; `_putN` suffix = putative; TBD 1,604) |
| `birthtime` | string | 7,904 | early 7,891; late 13 |
| `dimorphism` | string | 2,368 | male-specific 1,258; sexually dimorphic 771; potentially sexually dimorphic 177; potentially male-specific 162 |
| `fruDsx` | string | 5,012 | fru_high 2,611; fru_low 1,989; coexpress_high 193; dsx_high 138; coexpress_low 65; dsx_low 16 |
| `synonyms` | string | 3,955 | 280 distinct literature synonyms (e.g. "Cachero 2010: aSP-a; Yu 2010: aSP2") |
| `matchingNotes` | string | 3,425 | 277 distinct free-text notes on male/female matching |
| `entryNerve` / `exitNerve` | string | 11,835 / 1,005 | nerve names (AN, MetaLN, MxLbN, ...) |
| `receptorType` | string | 752 | putative_ppk23 269; putative_ppk25 257; putative_IR52b 226 |
| `assignedOlHex1` / `assignedOlHex2` | double | 23,720 each | optic-lobe hex column coordinates |
| `vfbId` | string | 166,686 | Virtual Fly Brain ids, all unique |

Example rows (first 5, selected columns):

```
bodyId  status  statusLabel            superclass          type     instance      somaSide  flywireType  somaLocation
10001   Traced  Roughly traced         descending_neuron   DNp01    DNp01(GF)_R   R         DNp01        [37124, 22258, 36274]
10002   Traced  Roughly traced         visual_projection   OCG01d   OCG01d_L      L         OCG01d       [50214, 10279, 27497]
10003   Traced  Prelim Roughly traced  visual_centrifugal  VCH      VCH_R         R         VCH          [46716, 21029, 15714]
10005   Traced  Roughly traced         cb_intrinsic        AOTU019  AOTU019_R     R         AOTU019      [41190, 18491, 12885]
10006   Traced  Prelim Roughly traced  visual_projection   VS       VS_L          L         VS1,...,VS8  [69604, 28368, 42459]
```

Status structure (cross-tab `status` x `statusLabel`): `status` is a coarse roll-up of `statusLabel`: Traced = {Roughly traced, Reviewed,
Prelim Roughly traced, RT Hard to trace, Leaves, PRT Orphan, RT Orphan}; Orphan = {Orphan, Orphan-artifact, Orphan hotknife, Hard to trace};
Anchor = {Anchor, Soma Anchor, Sensory Anchor, Partially traced}; Assign = {0.5assign}; Glia; Unimportant; `status` null ⇔ statusLabel ∈
{Out of scope, null}.

Neuron definition: `superclass IS NOT NULL` gives **exactly 166,700 rows = the paper's neuron count**. Of those, status is Traced 164,606,
null 2,002 (1,991 "Out of scope"), Anchor 60, Orphan 32. Conversely 516 `Traced` bodies lack a superclass. Distinct `type` among neurons =
11,751 vs. paper's 11,710 (43 of them contain `tbc`/`putative`/`unclear`; the residual difference is UNVERIFIED — probably annotation
refinement between paper freeze and v1.0 export). Distinct types per `dimorphism` value: male-specific 266, sexually dimorphic 102,
potentially male-specific 47, potentially sexually dimorphic 65 (paper: 289 male-specific, 138 dimorphic — the paper's counts presumably
merge the "potentially" categories: 266+47 = 313, 102+65 = 167, so they do not match exactly either; UNVERIFIED how the paper counted).
Female-specific types cannot appear in this table by construction.

### 4.2 `body-neurotransmitters-male-cns-v1.0.feather`

- **Rows 1,835,518** (one per body with ≥1 presynapse, `body` unique), 10 columns. Arrow 138.3 MB, pandas 174.7 MB, load 0.02 s.
- **No per-transmitter probability columns at the body level.** Per-transmitter probabilities exist only in the per-presynapse file
  `tbar-neurotransmitters-male-cns-v1.0.feather` (columns `nt_acetylcholine_prob, nt_dopamine_prob, nt_gaba_prob, nt_glutamate_prob,
  nt_histamine_prob, nt_octopamine_prob, nt_serotonin_prob`, rows sum to 1.0; 7 transmitters). The body file gives argmax + confidence.

| Column | Arrow type | Non-null | Meaning |
|---|---|---|---|
| `body` | int64 | 1,835,518 | body id |
| `cell_type` | string | 164,446 | type at time of NT aggregation (11,751 distinct) |
| `total_nt_predictions` | int32 | 1,835,518 | number of presynapses aggregated (median 1; 857 bodies have 0) |
| `predicted_nt` | string | 1,835,518 | **body-level argmax** (with "unclear" when thresholds not met) |
| `predicted_nt_confidence` | double | 1,834,661 | confidence of body-level prediction (0.187–0.975) |
| `ground_truth` | string | 85,484 | experimental/literature NT for the type, when known |
| `celltype_total_nt_predictions` | int32 | 1,835,518 | presynapses aggregated over the whole type |
| `celltype_predicted_nt` | string | 1,835,518 | type-level argmax |
| `celltype_predicted_nt_confidence` | double | 164,445 | type-level confidence |
| `consensus_nt` | string | 1,835,518 | final consensus (ground truth > type-level > body-level; per flyem-snapshot `inputs/neurotransmitters.py`) |

Distribution of `predicted_nt` (all 1.84M bodies): unclear 1,684,226; acetylcholine 95,420; glutamate 28,199; gaba 20,358; dopamine 4,447;
histamine 2,259; serotonin 483; octopamine 126. Restricted to the 164,620 `Traced` bodies present: acetylcholine 94,946; glutamate 28,055;
gaba 20,218; unclear 14,365; dopamine 4,443; histamine 2,026; serotonin 465; octopamine 102. `consensus_nt` for Traced bodies: acetylcholine
103,718; glutamate 29,296; gaba 22,055; histamine 5,910; unclear 3,100; dopamine 392; octopamine 101; serotonin 48. `predicted_nt ==
consensus_nt` for 88.9% of Traced bodies; ground-truth agreement with body-level argmax 88.6% (n = 85,484 rows with GT).
Join: 187,016 of the 211,577 annotation bodies appear here; 164,620 / 165,122 Traced bodies (the rest have no presynapses).

flyem-snapshot doc-string (verified in `flyem_snapshot/inputs/neurotransmitters.py`): body-level prediction is set to `unclear` unless the
body has enough presynapses **and** confidence ≥ 0.5 (`min-body-confidence`); same for type-level (`min-celltype-confidence`). Exact
minimum-synapse-count parameters used for MaleCNS: **UNVERIFIED** (config not published with the data; "see manuscript methods").

Example rows:

```
body   cell_type  total_nt_predictions  predicted_nt_confidence  predicted_nt   ground_truth   celltype_total_nt_predictions  celltype_predicted_nt  celltype_predicted_nt_confidence  consensus_nt
10001  DNp01      1015                  0.500496                 acetylcholine  NaN            1991                           acetylcholine          0.528263                          acetylcholine
10002  OCG01d     1443                  0.955895                 acetylcholine  NaN            2818                           acetylcholine          0.956476                          acetylcholine
10003  VCH        4425                  0.879325                 gaba           gaba           8721                           gaba                   0.874799                          gaba
10005  AOTU019    2837                  0.835076                 gaba           gaba           5673                           gaba                   0.830504                          gaba
10006  VS          426                  0.904134                 acetylcholine  acetylcholine  5260                           acetylcholine          0.820038                          acetylcholine
```

### 4.3 `body-stats-male-cns-v1.0-minconf-0.5.feather` (extra)

- **Rows 88,384,522** (every segment with ≥1 synapse), 11 columns; Arrow 5.09 GB; load 0.92 s. Sorted by `synweight` desc (`rank` column 1..N).

| Column | Type | Notes |
|---|---|---|
| `body` | int64 | segment id |
| `pre` | int32 | presynapses (T-bars) in body; total **45,656,140** |
| `post` | int32 | postsynaptic sites in body; total **311,833,243** |
| `status_fine` | dictionary<string,int8,ordered> | = `statusLabel`; 88,192,967 rows have "" |
| `superclass`, `class`, `type`, `instance` | string | copied from annotations (166,576 with superclass) |
| `downstream` | int64 | number of postsynaptic partners of this body's T-bars; total 311,833,243 |
| `synweight` | int64 | = `post + downstream` (verified: 623,666,486 = 2 × 311,833,243) |
| `rank` | int64 | rank by synweight |

Every body in body-stats appears in the weights file and vice versa (88,384,522 both). Per-body `sum(weight)` grouped by `body_pre` equals
`downstream` for all bodies, and grouped by `body_post` equals `post` for all 87,576,984 postsynaptic bodies (0 mismatches).

### 4.4 `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

```python
import pyarrow.feather as pf, numpy as np, pandas as pd, time, resource
t0=time.time(); t=pf.read_table('connectome-weights-male-cns-v1.0-minconf-0.5.feather'); t_arrow=time.time()-t0
df=t.to_pandas(); w=df.weight.to_numpy()
```

- **Columns: `body_pre` int64, `body_post` int64, `weight` int64** — no per-ROI breakdown, no type columns (the `-traced-only` /
  `-significant-only` variants add `type_pre`, `type_post` string columns).
- **Edges 151,856,684**; **sum(weight) = 311,833,243** (= total postsynaptic sites = neuPrint `Meta.totalPostCount`).
- Unique `body_pre` 1,834,661; unique `body_post` 87,576,984; union 88,384,522. Self-edges 123. No duplicate (pre, post) pairs.
  Sorted by weight descending (first row 10352→10351 weight 2591).
- Weight: min 1, median 1, mean 2.05, max 2,591. Edges with weight ≥2: 57,670,765 (37.98%); ≥3: 23,014,406 (15.16%);
  **≥5: 7,622,864 (5.02%, 31.4% of weight mass)**; **≥10: 2,799,910 (1.84%, 21.8% of mass)**; ≥20: 1,068,256; ≥50: 228,233; ≥100: 55,520.
- Memory/time: file 1.05 GB (compressed IPC); Arrow table 3,644,560,416 bytes; pandas 3,644,560,548 bytes (3 × int64, zero-copy);
  `read_table` 1.0 s, `to_pandas` 1.25 s; peak RSS 4.86 GB.
- Join coverage: annotated bodies present 191,696 / 211,577; **Traced bodies present 164,789 / 165,122** (333 Traced bodies have no
  synapses at minconf 0.5); bodies with superclass present 166,576 / 166,700.
- Weight mass by presynaptic status: Traced 294,794,753 (138.1M edges); not-in-annotations 15,431,521; Orphan 1,257,239; Assign 124,860;
  Anchor 117,381; Glia 62,272; Unimportant 45,217. By postsynaptic status: Traced 130,413,767 (30.4M edges); not-in-annotations
  180,893,750 (121.2M edges, i.e. most PSDs sit on unannotated fragments); Orphan 414,354; ...
- **Traced→Traced subgraph: 25,563,197 edges, weight 124,025,046**, 163,539 unique pre / 164,463 unique post bodies. With weight ≥5:
  6,235,682 edges / 89,731,551; ≥10: 2,749,407 / 67,173,142. Superclass→superclass: 25,582,938 edges / 124,177,617.

Example rows:

```
   body_pre  body_post  weight
0     10352      10351    2591
1     13612      16076    2443
2     10663      10051    2248
3     10289      10611    2230
4     10893      10066    1929
```

### 4.5 neuPrint cross-check (Cypher via `POST https://neuprint.janelia.org/api/custom/custom`, dataset `male-cns:v1.0`)

| Query | Result |
|---|---|
| `Meta.totalPreCount / totalPostCount` | 45,656,140 / 311,833,243 (identical to flat files) |
| `Meta.tag, uuid, voxelSize, lastDatabaseEdit` | v1.0, `4b2087c0fbe046bfaf0d60bc970e3e5d`, [8,8,8] nanometers, "2026-03-28 11:56:30 -04:00 / 2026-06-08T01:31:39-04:00 (segment property update)" |
| `Meta.postHighAccuracyThreshold / postHPThreshold / preHPThreshold` | 0.5 / 0.7 / 0 |
| `count(:Segment)` | 88,404,403 (flat body-stats: 88,384,522 — 19,881 more Segment nodes in neuPrint, presumably synapse-less annotated bodies) |
| `count(:Neuron)` | 176,422 (Neuron label = Segment meeting neuPrint's size/status criteria); by status: Traced 165,122, Orphan 6,464, null 4,214, Anchor 611, Assign 11; `min(size)` 62,913 voxels |
| `count(:Neuron) WHERE superclass IS NOT NULL` | 166,700 |
| `count(:Neuron) WHERE type IS NOT NULL`, distinct types | 164,506 / 11,751 |
| `MATCH (:Neuron)-[e:ConnectsTo]->(:Neuron)` count / sum(weight) | **25,862,574 / 125,024,863** |
| `sum(n.pre), sum(n.post)` over Neuron | 42,957,584 / 130,889,681 |
| primary ROIs | 144 (brain neuropils L/R, VNC neuropils `LegNp(T1)(L)` …, nerves `AbN1(L)` …, plus `CentralBrain-unspecified`, `Optic-unspecified(L/R)`, `CV-unspecified`, `VNC-unspecified`); 5,619 total ROIs incl. optic-lobe column ROIs like `ME_R_col_06_08` |
| Neuron property keys (body 10001) | pre, post, downstream, upstream, synweight, bodyId, flywireType, group, instance, somaSide, statusLabel, superclass, type, vfbId, hemibrainType, itoleeHl, birthtime, mancBodyid, mancGroup, mancType, subclass, synonyms, somaLocation, status, totalNtPredictions, predictedNtConfidence, predictedNt, celltypeTotalNtPredictions, celltypePredictedNt, celltypePredictedNtConfidence, consensusNt, roiInfo, size, + one boolean per ROI |

### 4.6 Remote schema reads of the large files (no download)

Using a minimal HTTP-Range file object wrapped in `pa.PythonFile` and `pyarrow.ipc.open_file`, reading only the footer and selected
65,536-row batches (≈1–4 MB each):

```python
import requests, io, pyarrow as pa, pyarrow.ipc as ipc
class HttpFile(io.RawIOBase):
    def __init__(s,url): s.url=url; s.pos=0; s.size=int(requests.head(url).headers['content-length'])
    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.pos
    def seek(s,off,whence=0): s.pos={0:off,1:s.pos+off,2:s.size+off}[whence]; return s.pos
    def read(s,n=-1):
        if n<0: n=s.size-s.pos
        if n==0: return b''
        r=requests.get(s.url,headers={'Range':f'bytes={s.pos}-{s.pos+n-1}'}); r.raise_for_status(); s.pos+=len(r.content); return r.content
    def readinto(s,b): d=s.read(len(b)); b[:len(d)]=d; return len(d)
rd = ipc.open_file(pa.PythonFile(HttpFile(url), mode='r'))
rows = (rd.num_record_batches-1)*65536 + rd.get_batch(rd.num_record_batches-1).num_rows
```

| File | Batches | Rows (exact) | Schema |
|---|---|---|---|
| `syn-partners-…-minconf-0.5.feather` | 4,759 | **311,833,243** (= sum of weights) | `x_pre,y_pre,z_pre:int32; body_pre:int64; conf_pre:float; x_post,y_post,z_post:int32; body_post:int64; conf_post:float; primary_post:dictionary<string,int16,ordered>` |
| `syn-points-…-minconf-0.5.feather` | 5,455 | **357,489,383** (= 45,656,140 pre + 311,833,243 post) | `z,y,x:int32; kind:dict{PreSyn,PostSyn}; conf:float; sv:int64; body:int64; compartment:dict; major_label:int8; major:dict; primary_label:uint8; primary:dict; superprimary(_label); subprimary(_label); medulla/lobula/lobula_plate {r,l} column & layer (+_label) dictionaries; point_id:uint64` (43 columns; `point_id` is the pandas index) |
| `tbar-neurotransmitters-…-v1.0.feather` | 697 | **45,656,140** (one per presynapse) | `point_id:uint64; x,y,z:int32; conf:float; sv,body:int64; major,primary:dict; nt_{acetylcholine,dopamine,gaba,glutamate,histamine,octopamine,serotonin}_prob:float; split:dict{null,train,validation}` |
| `connectome-weights-…-traced-only.feather` | 391 | **25,563,197** (= Traced→Traced edges computed from full file) | `body_pre,body_post,weight:int64; type_pre,type_post:string` |
| `connectome-weights-…-significant-only.feather` | 391 | 25,568,639 | same as traced-only |

Sampled confidence floors (6 batches spread across syn-partners): `conf_pre` min 0.700, `conf_post` min 0.500; syn-points batch 0 `conf` min
0.500004; tbar file `conf` min 0.70. So the release's presynapses were already thresholded at ≥0.7 upstream and the `minconf-0.5` filter
bites only on postsynaptic confidence.

---

## 5. Computed vs. stated counts

| Quantity | Stated | Computed here | Comment |
|---|---|---|---|
| Neurons | 166,700 (Cell abstract); "167k" (neuPrint v1.0 description); "more than 166,000" (Janelia news) | **166,700** rows with non-null `superclass` in annotations (= neuPrint Neurons with superclass) | Exact match. Note `status == "Traced"` gives 165,122 — 516 Traced bodies have no superclass (fragments/“Leaves”/PRT Orphan), while 2,094 superclass-bearing bodies are not `Traced` (1,991 "Out of scope", 60 Anchor, 32 Orphan). Annotation file has 211,577 rows total because it also lists Glia (11,864), Unimportant (10,751), Orphan (15,925), Assign (1,832), Anchor (611) and Out-of-scope bodies. |
| Neuron types | 11,710 (abstract) | 11,751 distinct `type` | +41; UNVERIFIED cause (likely post-freeze re-typing; 43 types contain tbc/putative/unclear). |
| Male-specific / dimorphic types | 289 / 138 (abstract); 262 / 114 (Janelia overview page, older) | `dimorphism` column: 266 male-specific + 47 potentially; 102 dimorphic + 65 potentially (type-level) | Does not reproduce either stated figure exactly; paper's per-type classification presumably used additional criteria. UNVERIFIED. |
| "Synapses" ≈125M | 125M in our brief (source unknown) | Traced→Traced weight 124,025,046 (flat); neuPrint Neuron→Neuron ConnectsTo 125,024,863 | The ~125M figure is the number of *neuron-to-neuron postsynaptic connections* among (Traced) neurons, not total synapses. |
| Total synaptic connections (all segments) | "312M synapses" (Wikipedia) | 311,833,243 PSDs / partner pairs; 45,656,140 presynapses (T-bars) | Total weight of the full graph equals `Meta.totalPostCount`. |
| Neuron-level presyn/post | — | Traced bodies: pre 42,658,913, post 130,413,767, downstream 294,794,753 | i.e. 93% of T-bars but only 42% of PSDs are on Traced bodies; 58% of PSDs sit on ~87M small fragments. |
| Segments with synapses | — | 88,384,522 (body-stats = weights union); neuPrint Segments 88,404,403 | |

---

## 6. neuPrint access

- `GET https://neuprint.janelia.org/api/dbmeta/datasets` works **without a token** and lists: `hemibrain:v1.2.1`, `male-cns:v0.9`,
  `male-cns:v1.0`, `manc:v1.0`, `manc:v1.2.1`, `manc:v1.2.3`, `mushroombody`, `optic-lobe:v1.0.1`, `optic-lobe:v1.1`.
- `GET /api/serverinfo` → `{"IsPublic":true,"Version":"1.9.3","announcement":"neuPrint has moved to a new authorization system. Existing API
  tokens no longer work. Log in and visit your Account page to generate a new token.","announcement-id":"auth-migration-2026-08"}`.
  Important for flyconn users: **tokens issued before August 2026 are invalid**.
- Surprise: `POST /api/custom/custom` (raw Cypher) answered **without any Authorization header** for `male-cns:v1.0` (all §4.5 queries).
  `GET /api/npexplorer/roiconnectivity` returned 401 "authentication required", and `/api/dbmeta/dataset?dataset=…` 404. So read access to
  public datasets is currently partially open server-side, but this is undocumented and may be an artifact of the auth migration — do not
  rely on it. The download page instructs users to "go to neuPrint and create an account" (Google login) and obtain an API token.
- `neuprint-python` (PyPI 0.6.3, 2026-07-20): `Client(server, dataset=None, token=None)` **raises `RuntimeError("No token provided…")`
  unless a token is passed or `NEUPRINT_APPLICATION_CREDENTIALS` is set** (verified in `neuprint/client.py`). Its README, quickstart, client and
  queries docs and its GitHub releases feed contain **no mention of MaleCNS/male-cns** (grep count 0); the only documented MaleCNS usage is
  the snippet on the MaleCNS download page: `Client("https://neuprint.janelia.org", dataset='male-cns:v1.0', token=token)`,
  `fetch_neurons("DNge104")`, `fetch_adjacencies(...)`. R users: `natverse/neuprintr` and the dedicated `natverse/malecns` package
  (README verified; exposes flywireType, mancType, itoleeHl columns).
- No account was created.

## 7. Neo4j dump

- Location (download page + verified GCS listing): `gs://flyem-male-cns/v1.0/database/neo4j/` — "The complete neo4j database backing the
  male-cns:v1.0 neuprint dataset hosted on neuprint.janelia.org", "Built for neo4j v4.4.16", "Constructed using flyem-snapshot".
  **1,028 objects, 251,752,665,322 bytes (251.8 GB)**; `data/` 251.72 GB (largest: `neostore.propertystore.db` 95.3 GB,
  `neostore.relationshipstore.db` 52.9 GB, `neostore.propertystore.db.strings` 40.0 GB, `neostore.nodestore.db` 11.6 GB), plus `plugins/`
  (21.6 MB, 1 file), `conf/`, `logs/`, `scripts/`. It is an uncompressed database directory, not a `.dump` file. Uploaded 2026-06-08.
- Input CSVs: `gs://flyem-male-cns/v1.0/database/neuprint-inputs/` — 101,086 objects, 193,978,239,316 bytes (194.0 GB): root-level
  `Neuprint_SynapseSet.csv` 26.5 GB, `Neuprint_SynapseSet_to_Synapses.csv` 24.0 GB, `Neuprint_Neuron_Connections.csv` 16.9 GB,
  `Neuprint_Synapse_Connections.csv` 11.7 GB, `Neuprint_Neuron_to_SynapseSet.csv` 8.7 GB, `Neuprint_SynapseSet_to_SynapseSet.csv` 6.3 GB,
  `Neuprint_Neurons.feather` 4.6 GB, `roi_elements.feather` 3.7 GB, `Neuprint_Neuron_Connections.feather` 3.5 GB; directories
  `Neuprint_Synapses/` (36,692 CSVs, 58.8 GB), `Neuprint_Neurons/` (26,465 files, 28.2 GB), `Neuprint_Elements/`, `Neuprint_ElementSets/`.
- Also `gs://flyem-male-cns/v1.0/database/dvid-exports/` (DVID API key-value exports, e.g. `roisSmoothedDecimated` meshes; >1,000 objects, ≥110 MB).

Listing snippet:

```python
import requests
url="https://storage.googleapis.com/storage/v1/b/flyem-male-cns/o"
params={'prefix':'v1.0/database/neo4j/','fields':'items(name,size),nextPageToken','maxResults':1000}
items=[]
while True:
    d=requests.get(url,params=params).json(); items+=d['items']
    if 'nextPageToken' not in d: break
    params['pageToken']=d['nextPageToken']
print(len(items), sum(int(i['size']) for i in items))
```

## 8. Caveats and definitions (verified against `janelia-flyem/flyem-snapshot`, branch `master`)

- **`minconf-0.5` meaning.** `flyem_snapshot/inputs/synapses.py` config key `min-confidence`: "Before generating any results, exclude synapse
  predictions which fall below this confidence level. Note: This pre-filters the synapse table, so it is effectively a minimum bound on the
  confidence threshold for neuprint exports." Implementation `_filter_for_confidence`: `point_df = point_df[conf >= min_conf]` and
  `partner_df = partner_df[(conf_pre >= min_conf) & (conf_post >= min_conf)]`. The output file tag is `f"{snapshot_tag}-minconf-{min_conf}"`.
  For MaleCNS v1.0 `min_conf = 0.5`; empirically presynapses are all ≥ 0.70 already. neuPrint's standard `weight` uses
  `postHighAccuracyThreshold = 0.5` (`conf_post >= 0.5`), and `weightHP` uses `postHPThreshold = 0.7`. Hence the flat `weight` equals neuPrint
  `weight` for the same body pair.
- **`-significant-only` / `-traced-only`** (`flyem_snapshot/outputs/flat.py`): both endpoints must have ordered `status >= MIN_SIGNIFICANT_STATUS
  ("Sensory Anchor")` resp. `>= MIN_TRACED_STATUS`; the traced-only row count (25,563,197) equals exactly the edges whose two bodies both have
  `status == "Traced"` in the annotations file, so "traced" = `status ∈ {Traced}` in the coarse status column. These variants add `type_pre` /
  `type_post`. They are not documented on the download page (README of flyem-snapshot mentions "connectome-weights (all and significant only)").
- **Proofreading status categories**: see §4.1. "Prelim Roughly traced" (36,387 neurons) is a lower-completeness tier than "Roughly traced"
  (71,979) and "Reviewed" (54,066). "Leaves"/"PRT Orphan"/"RT Orphan" are counted as Traced in the coarse status. 1,991 neurons with a
  superclass are labelled "Out of scope" (status null) — include them if you follow the paper's 166,700 definition.
- **Orphans/fragments**: 87M+ segments carry synapses but are unannotated; 58% of all PSDs and 6.6% of T-bars lie on non-Traced bodies. Any
  "fraction of inputs accounted for" metric must be computed against `body-stats.post`, not the Traced-only graph.
- **Sexual dimorphism annotations**: `dimorphism` (4 values incl. "potentially …"), `fruDsx` (fru/dsx expression levels, 6 values),
  `matchingNotes`, `synonyms`, `flywireType` (female FAFB match) support male↔female comparison; female-specific types are absent by definition.
- **VNC coverage**: full CNS incl. VNC and intact neck connective. VNC neuromeres in `somaNeuromere` (T1–T3, A1–A10, plus brain CG/LB/TC/DC/MX/MD/GNG);
  VNC neuropil ROIs (`LegNp(T1)(L)`, `mVAC(T2)(R)`, `ANm`, ...) and nerves are neuPrint primary ROIs. Trumans hemilineages `trumanHl`, MANC
  cross-references `mancType/mancBodyid/mancGroup/mancSerial`.
- **Coordinate/template spaces**: native EM space at 8 nm isotropic voxels (`somaLocation`, `x/y/z` in syn tables are voxel units);
  precomputed skeletons in 1 nm; mirrored skeletons via a navis-flybrains transform; skeletons also provided in JRC2018 unisex template space
  (µm). Brain ROIs were initialized by transfer from JRC2018M then refined manually. FlyWire v783 and hemibrain v1.2.1 meshes are provided
  transformed into MaleCNS space (`v1.0/male-cns-meshes-transformed-to-fafb-flywire/` also exists for the reverse direction).
- **Known defects**: the neuroglancer scene contains "brain-/vnc-defects" layers "show[ing] areas with known data defects/artefacts";
  `statusLabel` values "Orphan hotknife" (1,428) and "Orphan-artifact" (2,318) flag fragments caused by hot-knife section boundaries/artefacts.
- **Nuclei segmentation** is "not filtered to exclude glia, noise, etc."; segmentation voxels are uint64 but only the low 32 bits are used.
- **Auth migration**: neuPrint tokens created before 2026-08 no longer work (server announcement).
- **Annotations vs. neuPrint**: neuPrint has 176,422 `Neuron` nodes vs. 211,577 annotation rows — the flat annotation file includes Glia,
  Unimportant and small Orphan bodies that neuPrint keeps only as `Segment`s. Use `superclass IS NOT NULL` (166,700) as the canonical neuron set.

## 9. Things that could not be verified

- Cell full text / methods (Cloudflare-blocked); bioRxiv abstract (429). Publication metadata came from Crossref + Europe PMC only.
- Exact minimum-synapse thresholds used for NT "unclear" assignments (config not published).
- Why the type count is 11,751 vs. 11,710 and why per-type dimorphism counts differ from the abstract.
- Row counts of the two `syn-partners-…-{traced,significant}-only` files (not sampled).
- Whether tokenless Cypher access is intentional.
