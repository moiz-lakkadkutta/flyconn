# W5 "genetic access": data availability assessment

Date: 2026-09-26. Question: can flyconn answer "which published driver lines (split-GAL4 / GAL4 / LexA) target cell type X, and what else do they hit?" from openly licensed, machine-readable data, without accounts?

Everything below was verified by fetching the live resource unless marked UNVERIFIED. Downloaded samples live in `/Users/moizp/google-fly/.cache/data/lines/`.

## TL;DR

Yes, W5 is buildable. Two openly licensed sources carry the load:

1. **Meissner et al. 2025 (eLife 98405), Figure 1-source data 1** - a single 409 KB xlsx (CC BY 4.0) with 4,433 split-GAL4/GAL4/LexA lines, a `Cell types` column (2,667 adult lines annotated, connectome-style type names) and an `EM Body IDs` column (323 lines). This is the curated "on-target" table.
2. **NeuronBridge precomputed matches** on public S3 (`s3://janelia-neuronbridge-data-prod`, v3_10_0, 2026-05-21, CC BY 4.0 per the NeuronBridge footer and the FlyLight AWS Open Data entry) - per-line and per-body JSON with scored EM<->LM color-depth matches against hemibrain 1.2.1, MANC 1.2.1, FlyWire FAFB v783, BANC v626 and Male CNS v0.9. This is the "off-target" evidence.

VFB (public Neo4j, no login) adds FBbt-anchored line->type links for older GAL4/LexA lines; FlyBase (CC BY 4.0) adds split->hemidriver->stock resolution. neuPrint itself carries no line information.

---

## 1. Janelia FlyLight Split-GAL4 collection + Meissner et al. 2025

### 1a. splitgal4.janelia.org (website)

- URL: https://splitgal4.janelia.org/cgi-bin/splitgal4.cgi (verified 2026-09-26)
- License: footer states CC BY 4.0 ("(c) 2012-2026 HHMI Janelia ... licensed under a Creative Commons Attribution 4.0 International License").
- Content: 5,038 lines in the line dropdown, 46 named releases (Aso 2021, Descending Neurons 2018/2025, Lateral Horn 2019, MB Paper 2014, Nern et al 2024, Wolff et al 2024, SEZ 2021, Split-GAL4 Omnibus Rescreen, ...). A fourth search mode "EM body ID/cell type" queries an internal view `em_annotation_vw`: 831 distinct EM cell-type terms and 405 EM body IDs are in its dropdowns.
- Machine access: no CSV/JSON export. The search is a multipart POST to `splitgal4_summary.cgi` (`_search_toggle=em_annotation`, repeated `emc=<type>`, `_esearch=Search`) that returns HTML with columns `Line | Expressed in | Genotype | AD | DBD`; e.g. `emc=aIPg1,DNp01` returned SS32237 (aIPg1, aIPg2, aIPg3, aIPg), SS67129 (DNp01, DNp74), SS86900 (aIPg_m4, aIPg_m2, aIPg1). Term matching is exact (`term = 'PAM01'` returned 0 lines; PAM types are apparently named differently here).
- Verdict: **partial** - scrape-able but no bulk file; the same annotations are in the eLife spreadsheet below, which is the better source.

### 1b. Meissner et al. 2025, "A split-GAL4 driver line resource for Drosophila neuron types", eLife 13:RP98405

- Article license: CC BY 4.0 (eLife API `copyright.license: CC-BY-4.0`; PMC11759409 states "distributed under the terms of the Creative Commons Attribution License").
- Data availability statement: "All the primary adult data ... freely available under a CC BY 4.0 license through https://doi.org/10.25378/janelia.21266625.v1 and ... https://splitgal4.janelia.org and https://flylight-raw.janelia.org". The figshare deposit (api.figshare.com/v2/articles/21266625) is titled "Fly Brain Anatomy: FlyLight Gen1 and Split-GAL4 Imagery", license CC BY 4.0, and only points at the S3 imagery buckets (no table).
- **Figure 1-source data 1** - `https://cdn.elifesciences.org/articles/98405/elife-98405-fig1-data1-v2.xlsx` (418,803 bytes, no login). Cached at `.cache/data/lines/elife-98405-fig1-data1-v2.xlsx`.
  - One sheet "Cell type lines", 4,433 data rows x 17 columns:
    `Line | Adult/Larval | Main annotator | Lab | Quality | Sex difference | VNC only | Brain only | DOI to cite | First author(s) & year | Janelia Robot ID | Cell types | EM Body IDs | Alias | Genotype`
  - Rows: 3,060 Adult, 982 Larval split, 353 Larval-LexA, 31 Larval-Gen1, 7 Larval-dup.
  - `Cell types` filled for 2,668 rows (2,667 adult). Free text, comma-separated, mostly connectome type names (e.g. `PFNa`, `EPG`, `DNp54`, `L2`, `Tm3`, `LC10b`, `FB6A`, `OA-VPM4`, `07B` hemilineage, `MBON` names), 2,059 distinct tokens; some are informal ("TuBu subset", "WN_LN_mixed", "or similar").
  - `EM Body IDs` filled for 323 rows; hemibrain-style (`1912789965, 2166133204`, `5813057864`) and MANC-style (`15292`, `25908, 159206`) IDs, **without a dataset tag** - the dataset must be inferred from the DOI/lab (VNC-only lines -> MANC, brain -> hemibrain).
  - `DOI to cite` covers 50+ source papers: 2,137 rows cite the Omnibus preprint, 320 Nern et al. 2024/2025 optic lobe, 240 Wolff et al. 2024, 164 Dorsal VNC 2023, 138 Namiki 2018, 67 Sterne 2021, 69 Aso 2014, 4 Dolan 2019, etc. Line prefixes: SS 3,875, R 215, VT 169, MB 127, OL 47.
  - `Quality` 1-4 for adult lines (1: 1,767; 2: 1,232; 3: 26; 4: 34) - "For details of confidence levels and expression in multiple cell types, refer to the original publication listed for the line."
- **Figure 1-source data 2** (`elife-98405-fig1-data2-v2.xlsx`, 730 KB): 6,276 imaging-sample rows for rescreened lines (`Publishing Name, Slide Code, ..., Line, Genotype`) - useful to join to NeuronBridge `slideCode`.
- Verdict: **yes**. Best single on-target table. Caveats: type names are not normalised to a specific connectome/version (hemibrain, MANC, optic-lobe and FlyWire names mixed; "Cell types" can list several); no FBbt IDs; annotation confidence lives in the per-line source paper.

## 2. NeuronBridge precomputed EM<->LM matches (AWS Open Data)

- Bucket: `https://janelia-neuronbridge-data-prod.s3.amazonaws.com/` - anonymous listing works. `current.txt` = `v3_10_0` (data released 2026-05-21; older versions v2_1_0 ... v3_9_0 are retained).
- Layout (`v3_10_0/`): `DATA_NOTES.md`, `config.json`, `references.json`, `collections.json`, `schemas/{DataConfig,ImageLookup,PrecomputedMatches,CustomMatches}.json`, `metadata/by_line/<LINE>.json`, `metadata/by_body/<bodyId>.json`, `metadata/cdsresults/<imageId>.json`, `metadata/pppmresults/<imageId>.json`.
- Counts: `by_line/` = **52,449** files (IS 34,888; SS 11,827; VT 2,832; R 2,320; MB 307; LH 114; SL 83; OL 74). EM libraries in `config.json`: FlyEM_Hemibrain_v1.2.1 44,477 bodies; FlyEM_Male_CNS_Brain_v0.9 129,399 + VNC 29,648; FlyWire_FAFB_v783_realign 144,005; FlyWire_BANC_v626 75,633 brain + 23,179 VNC; FlyEM_MANC_v1.2.1 24,350. LM libraries: Split-GAL4 Drivers (96,582 brain + 30,145 VNC CDMs), Split-GAL4 Omnibus Broad (425,488 + 236,384), Gen1 MCFO v1.1 (346,342), Annotator Gen1 MCFO v1.1 (311,222).
- Format: JSON validated by the published JSON schema. `by_body/<id>.json` -> `{results:[EMImage...]}` with `libraryName, publishedName ("hemibrain:v1.2.1:1912789965"), neuronType, neuronInstance, annotations[], files.CDSResults, files.PPPMResults, files.AlignedBodySWC`. Note the same numeric id can appear for several datasets in one file (e.g. `10000.json` holds vnc:v0.5 and manc:v1.2.1 bodies). `by_line/<line>.json` -> `{results:[LMImage...]}` with `slideCode, objective, gender, anatomicalArea, channel, files.CDSResults`. `cdsresults/<imageId>.json` -> `{inputImage, results:[CDSMatch{normalizedScore, matchingPixels, mirrored, image:{...publishedName, neuronType, annotations}}]}`. `pppmresults` -> `PPPMatch{pppmRank, pppmScore}` (EM->LM only, MCFO libraries).
- Scoring: color-depth MIP search (CDS): `matchingPixels` = raw pixel overlap, `normalizedScore` = gradient-area-gap-normalised score used for ranking. Parameters (DATA_NOTES): maskThreshold 20, targetThreshold 20, xyShift 2, mirrorMask on, pixColorFluctuation 1.0, pctPositivePixels 1.0; for EM->LM the top 300 lines get gradient scoring; FlyWire matches with pixel score < 3.5% were dropped; LM MIPs are segmented into "searchable neuron" fragments. PPPM = PatchPerPixMatch (Kainmueller lab), hemibrain 1.2.1/MANC vs Gen1 MCFO only.
- Sample sizes: hemibrain body 1912789965 (VL1_ilPN): CDS file 3.75 MB, 2,477 matches over 840 distinct lines (915 Gen1 MCFO, 858 Split-GAL4 Drivers, 485 Annotator MCFO, 219 Omnibus Broad); top hits SS02049 normalizedScore 50,000/356 px; PPPM file 232 KB, 138 matches. LM image files range 2 KB-3.7 MB.
- License: NeuronBridge web app footer (bundle `main.894a4d62.js`, app v3.5.5): "The data displayed here are licensed under a Creative Commons Attribution 4.0 International License". The FlyLight AWS Open Data registry entry (`datasets/janelia-flylight.yaml`) lists License CC BY 4.0 for `janelia-flylight-imagery`, `-templates`, `-color-depth`. The match-results bucket itself has no LICENSE file and no registry entry of its own (UNVERIFIED that the JSON is separately licensed; strong presumption of CC BY 4.0). Upstream EM skeletons carry their own terms: hemibrain/MANC/MaleCNS CC BY 4.0; FlyWire FAFB v783 CC BY-NC 4.0 (VFB records "FlyWire connectome neurons CC-BY-NC_4.0"); BANC v626 UNVERIFIED.
- **Curated matches (new)**: NeuronBridge web v3.5.0 (2026-04-23) added "Curated Matches of Split-GAL4 Lines to Cell Types": expert annotations with confidence `Confident (>95%) / Probable (70-95%) / Candidate (30-70%)`, region and annotator; sources are (i) annotations shipped with splitgal4.janelia.org releases and (ii) scored PPPM results for a subset of Meissner 2025 lines ("may be published in a forthcoming standalone paper or addendum"). Delivery: `GET https://nan47vkv68.execute-api.us-east-1.amazonaws.com/{stage}/curated_matches?q=<line or type>` through AWS Amplify with Cognito credentials; anonymous curl returns 401/404. Response items: `itemType: line_name|cell_type`, `matches[{cell_type, dataset, body_id, annotation, region, annotator}]`. **Not in the S3 bucket as of v3_10_0** - no bulk download today.
- Verdict: **yes for off-target evidence** (bulk, machine-usable, versioned, no login). Matches are candidates ranked by score, not ground truth; a "line" file lists per-sample per-channel images, each with its own match list.

## 3. neuPrint annotations

- neuPrint user guide (`neuprint.janelia.org/public/neuprintuserguide.pdf`, 18 pp) Neuron properties: `bodyId, cropped, instance, post, pre, roiInfo, size, somaLocation, status, statusLabel, type` (+ dataset-specific). No GAL4/LexA/line/LM fields.
- neuprint-python `NeuronCriteria` documented parameters (flywireType, hemibrainType, mancType, hemilineage, vfbId, synonyms, ...) - none relate to driver lines. (`vfbId` exists on some datasets, giving a hook into VFB.)
- Verdict: **no** (use neuPrint only for body->type and type->body).

## 4. Virtual Fly Brain

- Public Neo4j: `http://pdb.virtualflybrain.org/db/neo4j/tx/commit` with the public read-only credentials baked into `vfb_connect` (user `neo4j`, password `vfb`); no account needed. Also SOLR, Owlery and the `https://v3-cached.virtualflybrain.org/` query service (`/run_query`, `/resolve_combination?query=<hemidriver>`, `/find_stocks`).
- Content (queried): 75,951 `Expression_pattern` nodes; 7,815 split expression-pattern **classes** (`Class:Expression_pattern:Split`, e.g. `VFBexp_FBtp0099455FBtp0119244` "P{R12C11-GAL4.DBD} ∩ P{R85F12-p65.AD} expression pattern", synonyms `SS00856`); 5,037 have xrefs to `FlyLightSplit`, 5,080 to `neuronbridge`.
- Line->type graph pattern (VFB's "Targeting splits" panel): `(n:Class:Neuron)<-[:INSTANCEOF]-(a:Individual)-[:part_of]->(ep:Class:Expression_pattern)`. Coverage: 1,232 split EPs -> 667 neuron classes (4,703 edges); 2,720 GAL4/LexA EPs -> 1,400 neuron classes (32,710 edges). Example: PAM01 classes (FBbt_00049930/1/2) are linked to `P{GMR66C08-GAL4}`, `P{GMR48H11-lexA}`, `PBac{IT.GAL4}0104`.
- FBbt <-> connectome: individuals carry `database_cross_reference` to Sites `neuprint_JRC_Hemibrain_1point2point1` (22,705), `neuprint_JRC_Manc_1_2_1` (23,665), `neuprint_JRC_OpticLobe_v1_0_1` (53,402), `flywire783` (139,255), `BANC626`, `BANC888`, `catmaid_fafb/fanc`. Caveat: many hemibrain individuals are typed only to coarse classes ("adult cholinergic neuron", "mushroom body dopaminergic neuron"), so FBbt granularity varies.
- Licenses: Drosophila Anatomy Ontology CC BY 4.0. Per-dataset `License` nodes: "Split-GAL4 lines from Meissner et al., 2024" CC-BY 4.0, Dolan 2019 splits CC-BY 4.0, "FlyLight - GMR GAL4 collection (Jenett2012)" and "Gen1 GAL4/LexA collection" **CC-BY-NC-SA 4.0** (images), "FlyWire connectome neurons" CC-BY-NC 4.0. The curated statements themselves are ontology/FlyBase-derived (CC BY 4.0). `vfb_connect` is GPLv3 (software).
- Verdict: **partial/yes** - the best source for GAL4/LexA (non-split) line->type statements and for FBbt-anchored joins; coverage of new split lines lags Meissner.

## 5. FlyBase

- License: "All annotations and data produced by FlyBase that are accessible from flybase.org are distributed under a CC BY 4.0 license" (wiki.flybase.org/wiki/FlyBase:About).
- Bulk files (`https://s3ftp.flybase.org/releases/current/precomputed_files/`, FB2026_03; note `ftp.flybase.org` did not respond):
  - `alleles/split_system_combinations_fb_2026_03.tsv.gz` (94 KB, 2,456 rows): `FB_id (FBco...) | Symbol | Component_Alleles (FBal ids) | Stocks (FBst + genotype) | Synonyms (SS numbers) | References`. Resolves SS line -> hemidrivers -> Bloomington stock.
  - `insertions/fu_gal4_table_fb_2026_03.json.gz` (937 KB, 390 "frequently used GAL4" drivers with image/publication metadata, tissue-level).
  - No precomputed driver->cell-type expression table; transgene expression curation is loaded into VFB (see 4).
- Verdict: **partial** (stock/hemidriver resolution only).

## 6. Paper supplementary tables (all downloadable without login)

| Paper | File | Rows / columns | Maps | License |
|---|---|---|---|---|
| Aso et al. 2014 eLife 04577 | `elife-04577-supp1-v1.xlsx` (174 KB) | 875 lines x 64 columns | MB split line -> AD/DBD + expression score (1-5) per MB cell type (KC subtypes, MBONs, DANs) | CC BY 4.0 |
| Namiki et al. 2018 eLife 34272 | `supp3` (also supp4/5) | 138 rows: `LineName, Genotype, DN type, sample, stochastic, bg` | SS line -> DN type(s) | CC BY 4.0 |
| Dolan et al. 2019 eLife 43079 | `supp2` | 226 rows: `LineCode, AD, DBD, Cell-Type, Quality` | LH line -> LH cell type | CC BY 4.0 |
| Sterne et al. 2021 eLife 71679 | `supp1` csv | 277 rows: `Cell type, AD, DBD, SS number, SEZ code, Quality, notes, imagery` | SEZ line -> named SEZ type (no EM ids) | CC BY 4.0 |
| Cheong et al. 2024 eLife 96084 (MANC DN/MN) | `supp1` csv (1,328 rows: `bodyid, type, systematic_type, ..., LM_match_slide_code`), `supp3` csv (834 rows: `bodyid, type, systematic_type, match_certainty(1-5), line_match`) | MANC body -> split line(s) with certainty | CC BY 4.0 |
| Ehrhardt et al. 2025 DN catalogue (eLife RP 107450 / bioRxiv 10.1101/2025.02.22.639679) | tables UNVERIFIED (eLife page needs JS) | ~500 DN lines, matched to MANC/FAFB/FANC | preprint CC BY-NC 4.0 (Crossref) |
| Nern et al. 2025 Nature (optic lobe) | lines already in Meissner table (320 rows, DOI 10.1101/2024.04.16.589741) | | CC BY 4.0 (Crossref) |
| Shiu et al. 2024 Nature | CC BY 4.0, but no line->type table found (UNVERIFIED) | | |

Most of these are already folded into Meissner Fig 1-source data 1 (its `DOI to cite` column), so they matter mainly for per-line confidence/quality fields.

## 7. Existing software

- `neuronbridge-python` (JaneliaSciComp, PyPI, MIT-style repo; picks latest S3 version): `Client.get_lm_images(line_id)`, `get_em_image(s)(body_id)`, `get_cds_matches(image)`, `get_ppp_matches(em_image)`, image helpers. No type-level aggregation, no curated matches.
- `neuronbridger` (natverse, R): `neuronbridge_search`, `neuronbridge_hits`, `neuronbridge_line_contents`, `neuronbridge_predict_split`, `neuronbridge_avoid`, `colormip_search`. Closest existing implementation of "what is in this line" and "predict a split" - R only.
- `vfb_connect` (GPLv3; Python): `term/terms/search`, Neo4j and Owlery tools; VFB web "Targeting splits" panel per neuron type.
- `hemibrainr::lm_matches` (R) - LM matches for hemibrain neurons (UNVERIFIED detail).
- FlyLight "Color Depth MIP mask search" Fiji plugin; NBLAST/`flybrains` are for morphology, not line lookup.
- Nothing in Python today combines curated tables + NeuronBridge + connectome annotations into type->lines / line->off-targets, so W5 is not a duplicate.

---

## Recommendation

**Buildable: yes.** Default to Meissner 2025 Fig 1-source data 1 (CC BY 4.0, 409 KB, vendorable) as the on-target table; parse `Cell types` into tokens and match them against flyconn's type vocabularies (hemibrain, MANC, optic lobe, FlyWire), falling back to the 323 explicit `EM Body IDs`. Ship VFB (public Neo4j) as the source for GAL4/LexA lines and FBbt joins, and FlyBase `split_system_combinations` for stock/hemidriver resolution. Both are tiny downloads.

**Off-target analysis** = NeuronBridge, LM->EM direction: `by_line/<LINE>.json` -> each LMImage's `cdsresults` -> EM matches (`publishedName` = `dataset:version:bodyId`, `neuronType`, `normalizedScore`, `matchingPixels`) -> aggregate per (dataset, type) taking the best score per body across the line's samples/channels -> rank; join `neuronType` (or your own body->type table) to name the types. Fetch on demand and cache (per-image files are 2 KB-4 MB; 52k line files total is ~100-200 MB). Optionally add EM->LM (`by_body`) for "which lines light up body X". The NeuronBridge **curated matches** (Confident/Probable/Candidate) would be the ideal off-target ground truth but are only reachable through the authenticated web API today - watch for them landing in the S3 bucket or a paper supplement.

**Caveats to document in the feature:**
- NeuronBridge hits are similarity candidates, not expression calls; scores are not comparable across libraries (MCFO single-neuron vs full split pattern) or across datasets; mirrored matches are flagged; VNC and brain are separate alignment spaces.
- Split-GAL4 images are full patterns segmented into fragments; a strong hit means the shape is present, not that the line is "clean". Absence of a hit is weak evidence (thresholding at 3.5% pixel score, top-300-line gradient scoring).
- Type names in Meissner are free text, dataset-agnostic and sometimes informal; EM body IDs lack dataset tags; confidence lives in the cited paper.
- Sex/age/effector matter: most FlyLight imagery is female, adult, specific reporters; expression may differ in males or with other effectors (see Meissner `Sex difference`, Fig 1-data 2 `Effector`).
- Licensing mix: line tables/NeuronBridge/hemibrain/MANC/FlyBase are CC BY 4.0; FlyWire skeletons CC BY-NC 4.0; FlyLight Gen1 raw images CC BY-NC-SA 4.0 in VFB - keep flyconn to metadata/scores and link out for imagery.
