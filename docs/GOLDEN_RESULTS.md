# Golden results: published numbers we use as regression tests

Compiled 2026-09-26. Every value below was quoted from a fetched primary source
(paper full text, supplementary table, authors' repository, official data file or
API) on that date. Values *computed by us from an authors' file* are labelled
"computed". Nothing is from memory. The unabridged extraction with per-cell source
locations is `docs/research/golden_results_raw.md` and `docs/research/data_malecns.md`.

How this file is used:

- Each row becomes a test in `tests/golden/` once the corresponding module exists.
  Tier 0 runs in normal CI (fixture-sized or cached-data tests); Tiers 1–3 are
  marked `golden` and run in the scheduled/manual heavy workflow.
- "Tolerance" is our proposal, with the reasoning in the Notes column. Exact
  means byte/integer equality.
- Known inconsistencies *between* sources are listed in §7 so nobody "fixes" a
  test to the wrong number.

Source keys: **Shiu24** Shiu et al., Nature 634:210 (2024) doi:10.1038/s41586-024-07763-9;
**Shiu-repo** github.com/philshiu/Drosophila_brain_model (MIT); **Shiu-ST** its
Supplementary Tables xlsx (`41586_2024_7763_MOESM2_ESM.xlsx`); **Dork24** Dorkenwald
et al., Nature 634:124 (2024) doi:10.1038/s41586-024-07558-y; **Schl24** Schlegel et
al., Nature 634:139 (2024) doi:10.1038/s41586-024-07686-5; **Lin24** Lin et al.,
Nature 634:153 (2024) doi:10.1038/s41586-024-07968-y; **Eck24** Eckstein et al., Cell
187:2574 (2024) doi:10.1016/j.cell.2024.03.016; **Sche20** Scheffer et al., eLife
9:e57443 (2020); **Take24** Takemura et al., eLife 13:RP97769 (2024); **Cheong25**
Cheong et al., eLife doi:10.7554/eLife.96084; **MCNS26** Berg et al., Cell
189:5504 (2026) doi:10.1016/j.cell.2026.08.015 (numbers from the Europe PMC abstract
and the bioRxiv v1 preprint 10.1101/2025.10.09.680999, because the Cell full text
returned HTTP 403); **BANC26** Bates et al., Nature (2026)
doi:10.1038/s41586-026-10735-w; **CI25** Yin et al., bioRxiv 10.1101/2025.09.29.679410.

---

## 1. Tier 0 — dataset counts (exact, checked on ingest)

### 1.1 MaleCNS v1.0 (computed from the official flat files, SHA-256 in DATA_SOURCES.md)

| Quantity | Value | Source | Tolerance |
|---|---|---|---|
| Neurons (rows with non-null `superclass` in body-annotations) | 166,700 | computed; equals MCNS26 abstract "166,700 neurons" and neuPrint `count(:Neuron) WHERE superclass IS NOT NULL` | exact |
| Rows in body-annotations file | 211,577 | computed | exact |
| Bodies with `status == "Traced"` | 165,122 | computed; equals neuPrint | exact |
| Distinct `type` among neurons | 11,751 | computed (abstract says 11,710; see §7) | exact vs file |
| Rows in body-neurotransmitters file | 1,835,518 | computed | exact |
| Edges in connectome-weights (all segments) | 151,856,684 | computed | exact |
| Sum of weights (= total PSDs = neuPrint `Meta.totalPostCount`) | 311,833,243 | computed; neuPrint | exact |
| Presynapses (T-bars), sum of body-stats `pre` = neuPrint `Meta.totalPreCount` | 45,656,140 | computed; neuPrint | exact |
| Traced→Traced edges / weight | 25,563,197 / 124,025,046 | computed; equals row count of the `-traced-only` file | exact |
| neuPrint Neuron→Neuron `ConnectsTo` edges / weight | 25,862,574 / 125,024,863 | neuPrint Cypher on `male-cns:v1.0` | exact (this is the "~125M synapses" figure) |
| Edges with weight ≥5 / ≥10 (all segments) | 7,622,864 / 2,799,910 | computed | exact |
| Traced→Traced edges with weight ≥5 / ≥10 | 6,235,682 / 2,749,407 | computed | exact |
| Superclass counts | ol_intrinsic 89,403; cb_intrinsic 32,164; vnc_intrinsic 13,161; visual_projection 9,201; descending_neuron 1,314; ascending_neuron 1,846; vnc_motor 708; cb_motor 107 (full table in research note) | computed | exact |
| Body-level `predicted_nt` among Traced bodies | acetylcholine 94,946; glutamate 28,055; gaba 20,218; unclear 14,365; dopamine 4,443; histamine 2,026; serotonin 465; octopamine 102 | computed | exact |
| Minimum `conf_pre` / `conf_post` in syn-partners (sampled batches) | 0.700 / 0.500 | computed via HTTP range reads | informational |

### 1.2 FlyWire / FAFB

| Quantity | Value | Dataset | Source | Tolerance |
|---|---|---|---|---|
| Proofread neurons | 139,255 | v783 | Dork24, Schl24, Lin24, Codex header | exact |
| Synapses between proofread neurons | 54.5 million | v783 | Dork24 | ±0.1 M |
| Connections ≥5 synapses / neurons involved | 2,700,513 / 134,181 | v783 | Dork24 (Lin24 says 2,701,601; see §7) | ±0.05 % |
| Unthresholded weighted edges | ≈15.1 million | v783 | Schl24 | ±0.1 M |
| Connections >100 / >1,000 synapses | 15,837 / 27 | v783 | Dork24 | exact |
| Intrinsic neurons | 118,501 | v783 | Dork24 | exact |
| Central-brain intrinsic / optic-lobe intrinsic | 32,388 / 77,536 | v783 | Dork24, Schl24 | exact |
| VPNs / VCNs | 8,053 / 524 | v783 | Dork24, Schl24 | exact |
| Sensory / ascending / descending | 5,512 / 2,362 / 1,303 | v783 | Schl24, Dork24 | exact |
| Motor / endocrine | 106 / 80 | v783 | Dork24 | exact |
| Cell types annotated | 8,453 (covers 96.4 % of neurons) | v783 | Schl24 | exact |
| Median in / out degree, intrinsic, ≥5 syn | 11 / 13 | v783 | Dork24 | exact |
| Neurons in v630 snapshot | 127,978 | v630 | Lin24 Methods; matches Codex v630 `cell_stats.csv.gz` row count (computed) | exact |
| Connections ≥5 synapses (v630) | 2,613,129 | v630 | Lin24 | exact |
| Unthresholded connections (v630) | ≈14.7 million | v630 | Lin24 | ±0.1 M |
| Kenyon cells | 2,597 (R) / 2,580 (L) | v783 | Schl24 | exact |
| Photoreceptors: compound eye / ocelli / eyelets | 11,118 / 273 / 8 | v783 | Dork24 | exact |
| Whole-volume NT fractions (as quoted by Shiu24 from Eck24) | ≈55 % ACh, 24 % Glu, 14 % GABA, 7 % DA+OA+5-HT | v630 | Shiu24 Methods | ±2 pp |

### 1.3 Shiu et al. model input file (the LIF golden input)

The model does **not** use the Codex `connections.csv` (≥5 threshold). It uses an
unthresholded v630 export shipped in the MIT-licensed repo:
`2023_03_23_connectivity_630_final.parquet` (86,630,944 bytes; local copy
SHA-256 `94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3`) and
`2023_03_23_completeness_630_final.csv` (3,057,611 bytes; SHA-256
`e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb`). The repo also
ships `Connectivity_783.parquet` (100,804,642 bytes) and `Completeness_783.csv`.

| Quantity | Value | Source | Tolerance |
|---|---|---|---|
| Neurons in model | 127,400 | Shiu24 Methods; completeness csv rows (computed) | exact |
| Edges | 14,687,178 | computed from parquet | exact |
| Minimum synapse count per edge | 1 (no threshold) | computed; edges with 1/2/3/4/5 synapses: 7,305,126 / 2,611,152 / 1,342,881 / 813,991 / 542,616 | exact |
| Total synapses | 52,793,639 | computed (`sum(Connectivity)`) | exact |
| Excitatory / inhibitory edges | 8,800,532 / 5,886,646 | computed | exact |
| Max weight | +1,801 (exc) / −2,358 (inh) | computed | exact |
| Columns | `Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index, Connectivity, Excitatory, Excitatory x Connectivity` | computed | exact |

### 1.4 Other datasets

| Quantity | Value | Dataset | Source | Tolerance |
|---|---|---|---|---|
| MaleCNS neurons / types (preprint) | 166,691 / 11,691 | v0.9-era preprint | MCNS26 bioRxiv v1 abstract | informational (v1.0 file gives 166,700 / 11,751) |
| MaleCNS cross-matched central-brain types | 7,319; 114 dimorphic, 262 male-specific, 69 female-specific | v1.0 vs FlyWire | MCNS26 preprint (Cell abstract: 8,069 isomorphic, 138 dimorphic, 289 male-specific, 71 female-specific) | see §7 |
| MaleCNS neurons matched to FAFB/hemibrain/MANC | 97.5 % (CB 96.4 %, OL 98.8 %, VNC 93.1 %) | v1.0 | MCNS26 preprint | ±0.5 pp |
| Hemibrain traced neurons | 21,663 (v1.1; "no updates to the connectome" in v1.2.1) | hemibrain | dvid.io release blog; FlyEM | exact on `traced-neurons.csv` row count |
| Hemibrain "well-reconstructed" neurons used by Eck24 | 24,666 | v1.2.1 | Eck24 | informational |
| Hemibrain synapses | "about 20 million" between traced neurons; 64 M PSDs, 9.5 M T-bars in volume | v1.1 | Sche20 | approximate |
| MANC neurons / T-bars / PSDs | ~23,000 / 10 M / 74 M | v1.0 | Take24 | approximate |
| MANC class counts | IN 13,066; DN 1,328; AN 1,862; MN 733; EN 92; EA 9; SN 5,927; SA 535 | v1.2.3 | Cheong25 | exact per class |
| BANC proofread neurons / incl. rough | 114,518 / 155,916 | v626 (paper) | BANC26 | exact per version; Codex v888 header says 158,262 |
| BANC synaptic links | 218,460,852 | v626 | BANC26 | exact |
| BANC DNs / ANs | 1,316 / 1,849 | v626 | BANC26 | exact |
| Eck24 classifier accuracy | 87 % per synapse (FAFB), 94 % per neuron, 91 % per cell type | FAFB/FlyWire | Eck24 | exact |
| Eck24 per-type accuracy | ACh 91 %, Glu 91 %, GABA 96 %, DA 90 %, OA 85 %, 5-HT 33 % (FAFB) | FAFB | Eck24 | exact |

---

## 2. Tier 1 — graph statistics (Lin24, FlyWire v630, ≥5-synapse graph)

These test `flyconn.graph` against an independent published analysis of the same
graph. Requires the v630 Codex `connections.csv.gz` (≥5 synapses) restricted to
the 127,978 v630 neurons.

| Statistic | Value | Tolerance | Notes |
|---|---|---|---|
| Nodes / edges | 127,978 / 2,613,129 | exact | |
| Connection probability | 0.000160 (Table 2) / 0.000161 (text) | 3 s.f. | |
| Reciprocity | 0.138 | ±0.001 | |
| Clustering coefficient | 0.0463 (Table 2) vs 0.0477 (text) | accept [0.046, 0.048] | paper inconsistent |
| Mean connection strength | 12.61 synapses (range 5–2,358) | ±0.05 | |
| Mean in/out degree (intrinsic) | 20.5; in–out Pearson R = 0.76 | ±0.1 | |
| Giant SCC / WCC | 93.3 % / 98.8 % of neurons | ±0.1 pp | |
| Mean shortest path (directed, SCC) | 4.42 hops, max 13 | ±0.02 | undirected 3.91, max 11 |
| Rich-club onset | total degree > 37; 40,218 neurons; in-club density 0.000870 | exact / ±1 % | |
| Neurons in ≥1 reciprocal connection | 77,607 | ±0.5 % | |
| Motif participants | FFL 113,978; 3-unicycle 66,835 | exact | Table 1 |

---

## 3. Tier 2 — Shiu et al. LIF model (FlyWire v630, `flyconn.sim`)

### 3.1 Model parameters (must match exactly; from Shiu24 Methods and `model.py`)

| Parameter | Value | `model.py` |
|---|---|---|
| V_rest = V_reset | −52 mV | `v_0`, `v_rst` (lines 22–23) |
| V_threshold | −45 mV, spike when `v > v_th` (strict) | `v_th` (24), `eq_th` (50) |
| Membrane time constant | 20 ms (C 2 µF/cm² × R 10 kΩ·cm²) | `t_mbr` (25) |
| Refractory | 2.2 ms; **0 ms for Poisson-driven neurons** | `t_rfc` (31); lines 92/103 |
| Synaptic decay τ | 5 ms | `tau` (28) |
| Synaptic delay | 1.8 ms | `t_dly` (34) |
| w_syn (single free parameter) | 0.275 mV | `w_syn` (37) |
| Edge weight | `Excitatory × Connectivity × w_syn` | line 183 |
| Equations | dv/dt = (v_0 − v + g)/t_mbr; dg/dt = −g/τ; on pre-spike g += w | lines 44–48, 175 |
| Reset | v = v_rst; g = 0 (also w = 0 in code) | `eq_rst` (52) |
| Integration | Brian2 `method='linear'` (exact) | line 163 |
| Poisson drive | `PoissonInput(N=1, rate=r_poi, weight=w_syn*f_poi)`, f_poi = 250 ⇒ 68.75 mV jump onto `v` | lines 85–91 |
| Default r_poi | 150 Hz (paper sweeps 10–200 Hz) | line 39 |
| Trials / duration | 30 × 1,000 ms | lines 17–18 |
| Sign rule | GABA, Glu ⇒ inhibitory; ACh, DA, OA, 5-HT ⇒ excitatory; per-neuron majority vote over presynapses with cleft score ≥ 50 | Shiu24 Methods |
| "Activated" | rate > 0 Hz in any of 30 trials | Shiu24 text |
| Rate | spikes per trial / 1 s, mean and s.d. over 30 trials (non-firing trials count 0) | `utils.get_rate` |
| Silencing | zero all *outgoing* weights of the neuron | `model.silence` |
| Brian2 | 2.5.1, Python 3.10, numpy 1.24 | `environment.yml` |

### 3.2 Stimulus fixtures (root IDs from `figures.ipynb`; full lists in the raw note)

| Set | n | Notes |
|---|---|---|
| Labellar sugar GRNs | 21 | canonical list; side labels inconsistent between paper and notebook — ignore them |
| Bitter GRNs / Ir94e GRNs / water GRNs | 21 / 18 / 18 | |
| JONs (JO-CE 70, JO-F 60, JO-D/m 16) | 146 IDs (paper says 147) | notebook has an undefined-name bug in the `neu_JON_all` cell |
| MN9 | 720575940660219265 (contralateral, "MN9_r"), 720575940645521262 (ipsilateral) | |
| aBN1 / aDN1 / aDN2 | 720575940630907434 / 720575940616185531 / 720575940629806974 | |
| SEZ split-GAL4 types | 106 types, 372 neurons (`sez_neurons.pickle`) | |

### 3.3 Golden simulation outputs

Rung 1 of the validation ladder (spike-for-spike parity with Brian2 on small
networks) has no published number; it is a self-consistency test. Rungs 2–3 use:

| Test | Published value | Tolerance | Source |
|---|---|---|---|
| **Primary:** 21 sugar GRNs @ 100 Hz → MN9 (…219265) mean rate | 65.7 Hz (s.d. 3.31, Shiu-ST 1A); 67.03 ± 6.60 Hz (computed from repo `sugarR_100Hz.parquet`) | mean in **[58, 76] Hz** (≈ ±1.5 s.d.); and MN9 contralateral > ipsilateral | Shiu-ST 1A; Shiu-repo |
| Same @ 200 Hz | 93.23 Hz (s.d. 5.18); 93.27 ± 3.15 (repo) | [83, 103] Hz | same |
| MN9 ipsilateral (…521262) @ 100 / 200 Hz | 49.67 / 62.9 Hz | ±2 s.d. (≈ ±9 / ±7 Hz) | Shiu-ST 1A |
| MN9 full frequency series @ 10/50/100/150/200 Hz | 0 / 19.43 / 65.7 / 83.67 / 93.23 Hz | ±2 s.d. per point; monotone non-decreasing | Shiu-ST 1A |
| Neurons activated (>0 Hz) by sugar @ 10 / 100 / 200 Hz | 45 / 410 / 455 (incl. the 21 GRNs) | ±5 % | Shiu24 text; Shiu-ST 1A (computed) |
| Neurons spiking in repo example @ 100 / 200 Hz; total spikes | 404 / 448; 289,073 / 511,566 | ±5 % | computed from repo parquet |
| Shuffled-connectome control (100 shuffles, 100 Hz) | MN9 > 0 Hz in 1 of 100 shuffles (correct connectome: 68.0 Hz) | ≤ 2/100 | Shiu-ST 1D |
| MN11 (…165019 / …868793) @ 100 Hz | 88.97 / 85.87 Hz | ±15 % | Shiu-ST 1A |
| MN8 (…352063) @ 100 Hz | 68.73 Hz | ±15 % | Shiu-ST 1A |
| Zorro (…888530) @ 100 / 200 Hz | 102.23 / 146.13 Hz | ±15 % | Shiu-ST 1A |
| Sugar-responsive & sufficient for MN9 / also required | 47 / 14 | exact set size ±2 | Shiu24 text (needs the top-200 protocol) |
| SEZ screen @ 50 Hz: types activating MN9 | 11 (roundup 79.5, diatom 29.13, sink_sync 22.17, G2N-1 12.37, clavicle 10.5, Fdg 22.53, bract 33.67, rattle 1.87, FMIn 0.37, TH-VUM 0.03, kitty 7.3 Hz) | set membership ±1 | Shiu-ST 3 |
| Overall accuracy vs experiment | 150/164 = 91.46 %; excl. Fig 2: 49/58 = 84.48 % | documentation only | Shiu-ST 10 |
| w_syn −30 % / +30 % → MN9 @ 100 Hz | 33.46 / 96.1 Hz | ±15 % | Shiu-ST 11A |
| JO-CE vs JO-F @ 150 Hz → aBN1 | 50.77 (s.d. 1.36) vs 1.23 Hz (s.d. 0.92) | 50.77 ± 15 %; JO-F < 3 Hz | Shiu-ST 8 |
| 147 JONs @ 140 Hz → aBN1 / aDN1 / aDN2 | 45.27 / 16.67 / 17.13 Hz | ±2 s.d. (≈ ±5 Hz) | Shiu-ST 7A |
| JONs @ 20…220 Hz: neurons >0 Hz | 227 / 367 / 503 / 628 / 720 / 823 at 20/60/100/140/180/220 Hz | ±5 % | computed from Shiu-ST 7A |
| Sugar vs water overlap @ 40 Hz MN9 | sugar 377, water 391, shared 250 (paper; naive recount of ST 4 gives 280) | 377/391 ±5 %; overlap documented only | Shiu24 Fig 3f; Shiu-ST 4 |

Independent cross-check: the third-party MLX port reports 67.30 Hz for the
100 Hz example, inside the proposed band.

### 3.4 Pure-connectivity checks derived from the same fixtures

| Check | Value | Tolerance |
|---|---|---|
| Synapses JO-CE → aBN1 / JO-F → aBN1 | 103 / 78 | exact (Shiu24 Fig 5g; computable from the v630 parquet with the ID lists) |
| NT split of the 613 taste-responsive neurons | 52 % ACh, 25.9 % GABA, 17 % Glu, 2.9 % 5-HT, 2.0 % DA, 0.2 % OA | exact if recomputed from ST 4 |

---

## 4. Tier 3 — cross-dataset variability (Schl24; null model for `flyconn.compare`)

Edges are cell-type → cell-type, unthresholded, FlyWire v783 left vs right and
FlyWire vs hemibrain v1.2.1.

| Quantity | Value | Tolerance |
|---|---|---|
| Pre/post-synapse counts per matched type: within brain / across brains | Pearson R 0.99 / 0.92 (pre), 0.76 (post) | ±0.01 |
| Edge-weight correlation within / across brains | R 0.97 / 0.8 | ±0.01 |
| Cosine-similarity effect size across vs within | 0.045 ± 0.096 | exact (absolute cosine values only in Fig 4d image) |
| Edge persistence | 53 % hemibrain→FlyWire; L→R 61 %, R→L 59 %; 572,980 edges in ≥1 hemisphere | ±1 pp |
| 1-synapse hemibrain edge present in one / both FlyWire hemispheres | 42 % / 16 % | ±1 pp |
| >90 % persistence rule | edges >10 synapses or ≥0.9 % of target input | exact |
| 99 % persistence rule | >2.6 % of input or 31 synapses | exact |
| 30-synapse edge regression | hemibrain 30 → FlyWire mean 29 (25 % <13, 5 % 1–2); FlyWire L 30 → R mean 31 (25 % ≤21, 5 % 1–8) | ±1 synapse |
| Technical-noise model | 65 % of L/R edge-weight variability within 5–95 % noise range; ≤30 % weight differences may be pure noise | key null-model parameter |
| Cell-count variability | KCs 2,597 R / 2,580 L / 1,917 hemibrain; average per-type variation 5 ± 12 %; hemilineage L/R 3 ± 4 % | exact |
| Type matching | 56 % (2,920/5,235) hemibrain types unambiguous; 664 merged/split; 1,651 not reidentified; 3,584 → 3,643 consensus types | exact |
| NT prediction L/R agreement (Eck24) | 1,586 L/R pairs; 95 % of 2,626 FlyWire/hemibrain types agree | exact |
| BANC vs FAFB / MANC matched type connections | 483,957 / 434,357 | exact (BANC26) |

---

## 5. Effective connectivity (CI25; parity targets for the `connectome_interpreter` wrapper)

| Quantity | Value | Tolerance |
|---|---|---|
| Random cell-type pairs connected within 2 / 5 hops, threshold 0 | ~70 % / 100 % | ±5 pp (100×100 random sample) |
| Same at 1 % normalized-input threshold | ~2 % / ~84 % | ±5 pp |
| Central-brain in-degree | mean ~130 partners (median ~90), ~80 types (median ~55) | ±10 % |
| Worked monosynaptic examples (edge exists at 1 % input) | HP5 → ipsilateral DNb05; JO-D → contralateral CB0916; JO-A/B → Giant Fiber, DNp02, DNp11; LPLC1 → DNa05 (2-hop) | existence |
| DNa10 direct VPN inputs >1 % | LLPC3, LPLC4, LC10d, LC10c, LTe64, LC22 | set membership |

No specific effective-connectivity *values* for named pathways appear in the text
(only in figures), so parity with `connectome_interpreter` itself is the practical
test: identical input matrix ⇒ identical `compress_paths` output.

---

## 6. Circuit facts checkable from connectivity alone (FlyWire v783 unless noted)

| Fact | Value | Source |
|---|---|---|
| Descending neurons | 1,303 (FlyWire); 1,316 (BANC); 1,328 (MANC) | Dork24; BANC26; Cheong25 |
| Ascending neurons | 2,362 (FlyWire); 1,849 (BANC); 1,862 (MANC) | same |
| Head motor neurons / endocrine | 106 / 80 | Dork24 |
| ALPNs / canonical types | ~130 / 58 | Schl24 |
| FC1–3 / FB1–9 neurons | 357 / 897 | Schl24 |
| Ocellar ganglion | 63 neurons; 15 DNs each receive >200 synapses from OCG01 | Dork24 |
| Hemilineages | 183 hemilineages, 88 % (30,233) of central-brain neurons | Schl24 |
| SEZ share of central-brain neuropil | 17.8 %; DNs get 52 % of inputs in SEZ | Dork24 |
| Optic-lobe types (right OL) | 156 types for 35,567 of 38,461 neurons | Schl24 |

---

## 6b. Re-verified in this repo on 2026-09-26 (research venv, files in `.cache/data/`)

| Check | Result |
|---|---|
| MaleCNS annotations rows / neurons / types | 211,577 / 166,700 / 11,751 ✓ |
| MaleCNS NT rows | 1,835,518 ✓ |
| MaleCNS weights edges / sum / ≥5 / ≥10 | 151,856,684 / 311,833,243 / 7,622,864 / 2,799,910 ✓ (Arrow load 1.4 s) |
| Shiu v630 parquet edges / synapses / neurons / exc / inh / min | 14,687,178 / 52,793,639 / 127,400 / 8,800,532 / 5,886,646 / 1 ✓ |
| Codex v630 `cell_stats.csv.gz` rows | 127,978 ✓ (= Lin24) |
| Codex v630 `connections.csv.gz` | 3,794,615 rows (one per pre, post, neuropil); **2,613,129 distinct pre→post pairs, all with summed syn_count ≥5** ✓ (= Lin24 exactly) |
| Codex v783 `connections.csv.gz` | 3,869,878 rows; **2,700,513 distinct pairs ≥5** ✓ (= Dork24 exactly; so Lin24's 2,701,601 is the outlier) |
| Codex v783 `cell_stats.csv.gz` rows | 139,246 (9 neurons lack morphology stats); `neurons.csv.gz` and `classification.csv.gz` both have **139,255** rows ✓ — use those for the neuron count |

M1 converter output for MaleCNS v1.0, neuron universe `superclass IS NOT NULL` (166,700), first computed 2026-09-26 and pinned as regression targets in `tests/golden/test_tier0_counts.py`:

| Quantity | Value |
|---|---|
| Neuron→neuron edges | 25,582,938 (= Phase 0 superclass→superclass count) |
| Weight sum | 124,177,617 (= Σ input_synapses_neurons = Σ output_synapses_neurons) |
| Edges ≥5 / ≥10 | 6,242,118 / 2,753,975 |
| Σ input_synapses_total (incl. fragments) / Σ output_synapses_total | 130,453,923 / 295,069,014 |
| Consensus NT over the universe | ACh 103,720; Glu 29,302; GABA 22,069; His 7,891; unclear 3,177; DA 392; OA 101; 5-HT 48 |
| Conversion cost (M4 Pro) | 26 s, peak RSS 5.7 GB; store 4.4 MB neurons + 74.9 MB edges Parquet |
| `--level nt-probs` (tbar file, 45.7 M presynapses) | 165,665 neurons with per-body mean probabilities; argmax(mean) = consensus label for 96.2 % (M3, 2026-09-27) |

FlyWire via the same converter: v783 139,255 neurons / 2,700,513 edges / 3,869,878 neuropil rows; v630 127,978 / 2,613,129.

Interpretation: the Codex `connections.csv` is the ≥5-synapse *pair* table
split by neuropil (per-row syn_count can be <5). Summing over neuropils recovers
the published connection counts exactly. Ingest must therefore aggregate
neuropil rows before applying any threshold, and must keep the unthresholded
Shiu parquet as a separate, sim-only edge source.

### 6c. Simulation golden runs reproduced by `flyconn.sim` (M4, 2026-09-27, M4 Pro, CPU float32, 30 trials x 1 s, seed 0)

| Stimulus (21 sugar GRNs) | MN9 contralateral | MN9 ipsilateral | Active neurons | Published |
|---|---|---|---|---|
| 100 Hz | 66.5 ± 4.1 Hz | 50.6 Hz | 416 | 65.7 (ST 1A) / 67.0 (repo) Hz; 49.7 Hz; 404–410 |
| 200 Hz | 94.5 Hz | – | 444 | 93.2 Hz; 455 |
| 10 Hz | 0.0 Hz | – | 39 | 0 Hz; 45 |

Rung 1 (Brian2 2.10.1 spike-for-spike parity, fixed input trains, 60- and 200-neuron random nets): identical event sets. Rung 3 throughput (`benchmarks/sim_throughput.json`): CPU 2.2 s per biological second at 30 batched trials (15.0 s single trial), MPS 2.7 s; the per-tick Python loop dominates, so MPS gives no gain yet. After the chunked engine with the `torch.compile` tick kernel (2026-10-01), the three golden 30 x 1 s runs above give the same numbers (MN9 66.5 ± 4.1 / 94.5 / 0.0 Hz; 416 / 444 / 39 active neurons; 290,693 spikes at 100 Hz) in 13 / 11 / 11 s instead of 57 / 56 / 47 s. Current throughput rows (eager and compiled, CPU and MPS) are in `benchmarks/sim_throughput.json`; the machine was shared with other jobs, so treat them as indicative.

MaleCNS calibration protocol (78 LB3 GRNs -> 2 MN9, 5 trials x 0.5 s, CPU float32): MN9 rate at 100 / 200 Hz drive for w_syn 0.1, 0.2, 0.275, 0.4, 0.6 mV = 0.0/0.6, 31.4/55.2, 42.6/75.6, 65.6/103.4, 60.4/121.6 Hz. No value reaches 80 % of maximum (best 0.63 at 0.4 mV): unresolved. Superseded by §6g (2026-10-06): the literal rule also fails on v630 itself.

### 6d. BANC v888 through the flyconn converter (M8, 2026-09-29)

| Quantity | flyconn | Published |
|---|---|---|
| Neurons (proofread or rough, excluding glia/trachea/non-neurons) | 155,858 | 155,916 proofread + roughly proofread (Bates 2026) |
| Descending / ascending | 1,316 / 1,849 | 1,316 / 1,849 |
| Edge-list rows / autapses dropped / neuron-neuron edges kept | 11,752,828 / 156,311 / 11,401,953 | doc: 11,510,975 rows, "autapses removed" |

W3 male (MaleCNS v1.0) vs female, output partners, min_weight 5, 500 permutations:

| Type | vs FlyWire v783 (statistic, verdict) | vs BANC v888, FlyWire-or-MANC vocabulary |
|---|---|---|
| PFL3 | 0.056, within between-brain range | 0.082, within range |
| EPG | 0.189, beyond range | 0.088, within range |
| DNp01 | p 0.52; 58 % of male output unmatched | p 1.0; 0 % unmatched |

### 6e. Hemibrain v1.2.1 and MANC v1.2.1 through the flyconn converters (2026-10-01)

`tests/golden/test_hemibrain_manc.py`; conversion takes about 2 s (hemibrain) and 1 s (MANC).

| Quantity | flyconn | Reference |
|---|---|---|
| Hemibrain neurons (v1.2 export `traced-neurons.csv`) | 21,739 | export README: all non-cropped Traced neurons; neuPrint v1.2.1 Traced and not cropped: 21,739 |
| Hemibrain edges / summed weight / edges with weight >= 5 | 3,550,403 / 14,329,229 / 662,578 | same file (research note section 2.3) |
| Hemibrain traced bodies with Supp. 5 side/hemilineage / with NT feather row | 21,328 / 21,709 | Supp. 5 lists 25,397 bodies (incl. cropped) |
| Hemibrain NT argmax among traced (ACh / Glu / DA / GABA / 5-HT / OA / neither or none) | 9,577 / 5,342 / 3,183 / 3,009 / 282 / 191 / 155 | NT feather, provenance UNVERIFIED |
| MANC traced bodies (Codex) / neurons kept (sjcabs meta) / glia excluded | 23,665 / 23,650 / 15 | neuPrint manc:v1.2.1 Traced 23,665 |
| MANC edge-list rows / summed count / neuron-neuron edges kept | 5,305,354 / 30,943,884 / 5,303,770 | neuPrint Traced->Traced 5,305,638 / 30,934,610 |
| MANC NT (ACh / Glu / GABA / unknown or unclear) | 11,518 / 6,274 / 5,733 / 125 | neuPrint Traced predictedNt ACh 11,518, Glu 6,283, GABA 5,738 (incl. glia) |
| MANC super_class: descending / ascending / motor / sensory / sensory_ascending | 1,322 / 1,862 / 721 / 5,925 / 535 | Cheong 2025 (v1.2.3): DN 1,328, AN 1,862, MN 733, SN 5,927, SA 535 |

`compare_type` with MaleCNS v1.0, output partners, min_weight 5, 200 permutations, seed 0:

| Pair | Type | n (a / b) | Cross / within a / within b | Statistic, p | Unmatched a / b |
|---|---|---|---|---|---|
| MaleCNS vs hemibrain | EPG | 46 / 46 | 0.976 / 1.000 / 0.970 | 0.009, p 0.005 (within between-brain range) | 0.001 / 0.0 |
| MaleCNS vs hemibrain | PFL3 | 24 / 24 | 0.832 / 0.990 / 0.307 | -0.184, p 0.995 | 0.055 / 0.011 |
| MaleCNS vs MANC | DNa02 | 2 / 2 | 0.915 / 0.970 / 0.972 | 0.056, p 0.54 | 0.202 / 0.0 |
| MaleCNS vs MANC | DNp01 | 2 / 2 | 0.726 / 0.715 / 0.904 | 0.083, p 0.54 | 0.15 / 0.0 |

The hemibrain PFL3 within-dataset similarity (0.307) shows the hemibrain left/right null is unreliable for types whose left-side arbors leave the volume (see caveats); EPG and DNa02 are pinned in the golden test, PFL3 and DNp01 were run once (scratch script) and are not tests.

### 6f. Pathway-ranked silencing screen (MaleCNS v1.0, w_syn 0.275 mV, uncalibrated; 2026-10-04; rerun at 0.188 mV in 6h)

`examples/specs/w2_sweep_pathway_mn9.yaml`: LB3* GRNs at 100 Hz; candidates on LB3* -> MN9 paths of <= 3 hops (>= 5 synapses and >= 1 % input per hop), grouped by cell type. Exactly six types qualify, the relays found in W1. MN9 in the stimulated condition 41.3 Hz; 10 trials x 300 ms:

| Silenced | Net route sign (wiring) | MN9 (Hz) | Difference (95 % CI) | q |
|---|---|---|---|---|
| GNG232 | + | 29.7 | -11.7 (-15.5, -7.8) | 0.008 |
| DNge080 | + | 32.2 | -9.2 (-13.0, -5.2) | 0.008 |
| GNG130 | - | 50.8 | +9.5 (1.5, 17.5) | 0.23 |
| DNge051 | - | 47.5 | +6.2 (1.0, 11.7) | 0.13 |
| GNG132 | - | 47.0 | +5.7 (0.7, 9.8) | 0.10 |
| GNG215 | - | 42.8 | +1.5 (-3.5, 6.5) | 0.62 |

Every effect has the direction its route sign predicts; only the two excitatory relays survive BH correction at 10 trials.

### 6g. w_syn re-calibration: positive control on v630 and MaleCNS (ADR-0009; 2026-10-06)

`tests/golden/test_shiu_calibration.py` (520 s) and `tests/golden/test_malecns_calibration.py` (1,289 s); 10 trials x 1 s per condition, CPU float32; bootstrap 1,000 replicates. Ratio = MN9 at the reference rate / MN9 at 200 Hz drive.

Shiu v630 (21 sugar GRNs -> contralateral MN9), seed 1:

| w_syn (mV) | 0.2 | 0.225 | 0.25 | 0.275 | 0.3 | 0.325 | 0.35 | 0.375 | 0.4 |
|---|---|---|---|---|---|---|---|---|---|
| MN9 at 50 / 100 / 200 Hz (Hz) | 0 / 8 / 69 | 1 / 42 / 82 | 4 / 63 / 88 | 20 / 67 / 95 | 38 / 77 / 98 | 44 / 76 / 99 | 50 / 81 / 103 | 57 / 81 / 106 | 52 / 77 / 99 |
| 100 / 200 ratio | 0.12 | 0.51 | 0.71 | 0.70 | 0.78 | 0.77 | 0.79 | 0.76 | 0.78 |

| Rule (target from v630 at 0.275 mV, seed 0) | v630 outcome |
|---|---|
| onset ratio 50 / 200, target 0.205 | calibrated, 0.273 mV (95 % CI 0.268-0.279), support 1.00 |
| 100 / 200 transfer, target 0.689 | ambiguous, crossing 0.247 (0.244-0.267), support 0.81 |
| literal 100 / 200 = 0.8 | unresolved (support 0.23) |

LB3 sugar/water assignment (output partners with >= 5 synapses, MaleCNS partners named by `fafb_783_cell_type`, cosine to FlyWire v783 profiles of Shiu's sugar and water GRN ids, 20/21 and 18/18 found in v783): LB3a 0.44 / 0.87, LB3b 0.65 / 0.49, LB3c 0.92 / 0.62, LB3d 0.86 / 0.55 (sugar / water). Per neuron, closer to sugar: LB3a 0/17, LB3b 8/11, LB3c 17/23, LB3d 24/26.

MaleCNS v1.0 -> MN9 (2 neurons), seed 1, onset-ratio transfer:

| Stimulus | Outcome | MN9 at 100 / 200 Hz near the outcome |
|---|---|---|
| LB3c + LB3d (49, sugar-like) | calibrated, **0.188 mV (95 % CI 0.185-0.192)**, support 0.99 | 29 / 43 Hz at 0.175 mV, 41 / 49 Hz at 0.2 mV |
| all LB3 (78) | ambiguous (ratio 0.18-0.22 from 0.225 to 0.4 mV) | - |

The literal and 100 / 200 transfer rules are ambiguous or unresolved for both stimuli. The LB3c+d onset ratio is not monotonic above the crossing (0.47 at 0.225 mV, 0.26 at 0.25 mV) but stays above the target there.

Sensitivity (`tests/golden/test_malecns_calibration_inputs.py`, grid 0.125-0.25 mV, 10 x 1 s, seed 1): per-neuron sugar call as the stimulus (49 neurons) 0.204 mV (95 % CI 0.199-0.209), calibrated; LB3c+d with probabilistic signs (tbar NT probabilities) 0.189 mV (0.187-0.192), calibrated. Default since 2026-10-10: 0.188 mV, range 0.185-0.209 mV.

Independent cross-check (same test file): synapses per matched type pair (sum over >= 5-synapse edges / (n_pre x n_post), types matched via MaleCNS `fafb_783_cell_type`, 227,872 shared pairs) are 1.67x (median; geometric mean 1.74) higher in MaleCNS than in FlyWire v783; 1.47x for pairs with >= 5 synapses per neuron pair. Scaling 0.275 mV by these gives 0.164-0.187 mV (0.172 at a floor of 1 synapse per neuron pair).

### 6h. MaleCNS experiments at the protocol w_syn (0.188 mV; 2026-10-10)

The three MaleCNS golden experiments rerun after the default switch (ADR-0009; same specs, seeds and trials; CPU float32): W2 190 s, GNG sweep 77 s, pathway screen 61 s. Stimulus is still all LB3* GRNs (sugar + water) at 100 Hz, as in the specs.

W2 (`w2_malecns_lb3_silence_gng232.yaml`, 20 x 0.5 s):

| Readout | Stimulated | GNG232 silenced | Difference (d, q) | At 0.275 mV (M5) |
|---|---|---|---|---|
| MN9 | 30.1 Hz | 14.1 Hz | -16.0 (-3.5, 9e-8) | 44.9 -> 23.1 (-21.8) |
| descending | 1.58 Hz | 0.84 Hz | -0.74 (-5.2) | 3.36 -> 2.65 |
| motor | 1.26 Hz | 0.45 Hz | -0.81 (-3.5) | 7.56 -> 4.91 |

Controls: degree-preserving rewiring gives MN9 0.0 Hz (as before); sign shuffling still runs away (24.8 M and 13.4 M spikes, 56,613 / 39,441 active neurons vs 203,825 spikes / 1,100 active stimulated).

GNG screen (`w2_sweep_silence_gng.yaml`, 10 x 0.3 s; stimulated MN9 28.3 Hz): MN9 effects with q < 0.05 are GNG232 -14.5 (CI -18.8 to -9.5), GNG054 +14.3, GNG115 +10.8, GNG038 +7.7, GNG452 +7.0. The 10 most active GNG types changed with w_syn (GNG019 and GNG479 dropped out; GNG054, GNG115, GNG228, GNG041, GNG042, GNG229 and GNG232 are in the top 10 now).

Pathway screen (`w2_sweep_pathway_mn9.yaml`, same six relays; stimulated MN9 28.3 Hz):

| Silenced | Route sign | MN9 difference (95 % CI) | q | At 0.275 mV (6f) |
|---|---|---|---|---|
| GNG232 | + | -14.5 (-18.8, -9.5) | 0.003 | -11.7, q 0.008 |
| DNge080 | + | -4.3 (-8.7, 0.3) | 0.26 | -9.2, q 0.008 |
| DNge051 | - | +2.5 (-2.3, 7.7) | 0.51 | +6.2 |
| GNG215 | - | -1.7 (-6.7, 3.7) | 0.59 | +1.5 |
| GNG132 | - | +1.7 (-3.8, 7.3) | 0.51 | +5.7 |
| GNG130 | - | +0.2 (-5.5, 6.0) | 0.82 | +9.5 |

Robust to the w_syn change: GNG232 silencing lowers MN9 (and the descending and motor groups); rewiring abolishes the response. Not robust: the size and significance of DNge080 and of the inhibitory-route relays, the direction-matches-route-sign pattern (GNG215 flips, n.s.), and which GNG types a top-10 activity screen selects. Screen rankings on MaleCNS should be reported with the w_syn they used.

## 7. Known inconsistencies between sources (do not "fix" tests to the wrong one)

| Item | Values | Decision |
|---|---|---|
| FlyWire v783 ≥5-synapse connections | 2,700,513 (Dork24) vs 2,701,601 (Lin24) | test against our own count of the Codex file; assert within 0.05 % of Dork24 |
| FlyWire v630 neurons | 127,978 (Lin24, Codex cell_stats) vs 127,400 (Shiu model file) | two different snapshots/filters; the Shiu parquet is the sim golden input, Codex v630 is the data-layer input |
| Lin24 clustering coefficient | 0.0463 (Table 2) vs 0.0477 (text) | accept either |
| MaleCNS types | 11,710 (Cell abstract) vs 11,751 (v1.0 file) | test the file; document the abstract |
| MaleCNS dimorphic/male-specific types | 138/289 (Cell abstract), 114/262 (preprint), file `dimorphism` column gives 102+65 "potentially" / 266+47 | not a test; documented caveat |
| MaleCNS neurons | 166,700 (Cell, v1.0 file) vs 166,691 (preprint) | 166,700 |
| Shiu JON count | 147 (paper) vs 146 IDs (notebook) | use the 146 IDs; note it |
| JO-CE / JO-F → aBN1 synapses | 103 / 78 (Shiu24 Fig. 5g text) vs **77 / 69** computed from the repo notebook ID lists on the repo's own v630 parquet (all 146 JONs → aBN1: 148) | not reproduced; the computed values are pinned in `tests/golden/test_w1_paths.py`; the paper likely used a different JON list or export (UNVERIFIED) |
| Shiu sugar/water overlap | 250 (paper) vs 280 (naive recount) | documented only |
| Hemibrain traced neurons | 21,663 (v1.1 release blog, §1.4) vs 21,739 (v1.2 export `traced-neurons.csv`, = neuPrint v1.2.1 Traced and not cropped) | test 21,739 against the v1.2 file |
| MANC edges | sjcabs `manc_121_simple_edgelist` 5,305,354 rows / 30,943,884 synapses vs neuPrint manc:v1.2.1 Traced->Traced 5,305,638 / 30,934,610 | test the pinned sjcabs file; the build's confidence threshold is undocumented |
| MANC class counts | sjcabs v1.2.1 vocabulary (DN 1,322, MN 721, SN 5,925, plus 111 visceral_circulatory) vs Cheong 2025 v1.2.3 (DN 1,328, MN 733, SN 5,927) | not a test; different annotation release and vocabulary |
| Codex "connections" header counts (e.g. 3,732,460 for v783) | unstated definition | never used as a test |

## 8. Not found (so not tests)

- Hemibrain v1.2.1 exact neuron and synapse totals in a quotable document.
- Eck24 full 6×6 confusion matrix (figure only); Lin24 Extended Data Table 2 (unthresholded stats).
- Schl24 absolute cosine-similarity values (figure only).
- Numeric giant-fiber input counts.
- Sapkal et al. 2024 firing rates (heatmaps only).
- Shiu24 erratum: none exists (PubMed and Crossref checked).
