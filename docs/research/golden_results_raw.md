# Golden results (raw) for flyconn regression tests

Compiled 2026-09-26. Rule applied: every number below is quoted from a fetched source (paper full text via Europe PMC XML / bioRxiv HTML, the authors' GitHub repo files, the Nature Supplementary Tables xlsx, or Codex pages). Nothing is from memory. Where a value was *computed by me from an authors' data file* (e.g. re-deriving a firing rate from a parquet in the Shiu repo) this is stated explicitly in the Source column as "computed from <file>" and should be treated as a derived, not published, value. Items I could not source are marked **NOT FOUND**.

Local copies of everything fetched (full texts, xlsx, parquet, notebooks) are in
`<scratchpad>/` (files `shiu_pmc.txt`, `PMC11446842.txt` (Dorkenwald), `PMC11446831.txt` (Schlegel), `PMC11446825.txt` (Lin), `PMC7546738.txt` (Scheffer), `PMC11106717.txt` (Eckstein), `PMC13518251.txt` (BANC/Bates), `PMC11446846.txt` (Sapkal), `PMC9170244.txt` (Engert), `biorxiv_*.html.txt` (MaleCNS preprint, Connectome Interpreter, MANC/Takemura), `shiu_supp/supp_tables.xlsx`, `shiu_repo/`). Note this is a session scratch dir and may be cleaned up; re-fetch instructions are in each Source cell.

Source key (short names used below):
- **Shiu24** = Shiu PK et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 634:210–219 (2024), doi:10.1038/s41586-024-07763-9, PMC11446845 (full text fetched from `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11446845/fullTextXML`). Preprint: bioRxiv 10.1101/2023.05.02.539144 (PMC10187186).
- **Shiu-repo** = github.com/philshiu/Drosophila_brain_model (files fetched from `raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/…`; file tree from GitHub API). Data mirror: Edmond doi:10.17617/3.CZODIW (version 3, 2023-09-20; contains `results.zip` 4,499,610,373 bytes with all raw model output — not downloaded).
- **Shiu-ST** = Shiu24 Supplementary Tables 1–12, single xlsx `41586_2024_7763_MOESM2_ESM.xlsx` (2,539,037 bytes) from `https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07763-9/MediaObjects/41586_2024_7763_MOESM2_ESM.xlsx`. Sheet names: 'Supplemental Table 1A Sugar Fir', 'ST 1B Sugar MN9 Activation', 'ST 1C MN9 sugar silencing A', 'ST 1D Shuffled Connectivity', 'Supplemental Table 2 Sugar Pred', 'Sup Table 3 Predicted MN9 vs. o', 'ST 4 Interaction betw Sugar, Wa', 'Supplemental Table 5 Water Pred', 'ST 6A Water Firing Rates', 'ST 6B Water MN9 Activation', 'ST 6C Water silencing data', 'Supplemental Table 7A Grooming ', 'Supp Table 7B aDN1 Activation r', 'Supp Table 7C aDN2 Activation R', 'Supp Table 7D aDN1 firing upon ', 'Supp Table 7E aDN2 firing upon ', 'Supp Table 8 JO-CE and JO-F fir', 'Supp Table 9A Behavioral Data, ', 'Supp Table 9B Water silencing', 'Supp Table 10 Overall predictio', 'Supplementary Table 11 A Robustn', 'Supp Table 11B'–'11F', 'Supplementary Table 12 Key reso'.
- **Dork24** = Dorkenwald S et al., "Neuronal wiring diagram of an adult brain", Nature 634:124–138 (2024), doi:10.1038/s41586-024-07558-y, PMC11446842.
- **Schl24** = Schlegel P et al., "Whole-brain annotation and multi-connectome cell typing of Drosophila", Nature 634:139–152 (2024), doi:10.1038/s41586-024-07686-5, PMC11446831.
- **Lin24** = Lin A et al., "Network statistics of the whole-brain connectome of Drosophila", Nature 634:153–165 (2024), doi:10.1038/s41586-024-07968-y, PMC11446825.
- **Sche20** = Scheffer LK et al., "A connectome and analysis of the adult Drosophila central brain", eLife 9:e57443 (2020), doi:10.7554/eLife.57443, PMC7546738.
- **Eck24** = Eckstein N et al., "Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster", Cell 187:2574–2594 (2024), doi:10.1016/j.cell.2024.03.016, PMC11106717.
- **Take23** = Takemura S et al., "A Connectome of the Male Drosophila Ventral Nerve Cord", bioRxiv 10.1101/2023.06.05.543757 (v1, posted 2023-06-06); eLife 13:RP97769 (2024) doi:10.7554/eLife.97769.
- **Cheong25** = Cheong HSJ et al., "Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome", eLife 2025, doi:10.7554/eLife.96084, PMC13384506.
- **MCNS-pre** = Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M et al., "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system", bioRxiv 10.1101/2025.10.09.680999 (v1, posted 2025-10-09). Published as Cell, doi:10.1016/j.cell.2026.08.015, 2026-09-03 (PubMed 42691995; Cell full text returned HTTP 403, so numbers below are from the preprint + Google/Janelia/Codex pages). Companion Cell papers (from Google Research blog): visual systems doi:10.1016/j.cell.2026.08.014 (PubMed 42691997, "The organization of visual pathways in the Drosophila brain"), taste doi:10.1016/j.cell.2026.08.016 (PubMed 42691996, "The complete gustatory connectome of adult Drosophila reveals how taste guides feeding, foraging, and social b[ehaviour]"), social behaviour doi:10.1016/j.cub.2026.08.013.
- **BANC26** = Bates AS, Phelps JS, Kim M, Yang HH et al., "Distributed control circuits across a brain-and-cord connectome", Nature (2026) doi:10.1038/s41586-026-10735-w, PMC13518251 (preprint bioRxiv 10.1101/2025.07.31.667571).
- **CI25** = Yin Y, Hoeller J, Mathiasen A, Tsang J, Charrier ME, Cardona A, "The Connectome Interpreter Toolkit", bioRxiv 10.1101/2025.09.29.679410 (v1). Repo: github.com/YijieYin/connectome_interpreter.
- **Sapkal24** = Sapkal N et al., "Neural circuit mechanisms underlying context-specific halting in Drosophila", Nature 634:191–200 (2024), doi:10.1038/s41586-024-07854-7, PMC11446846.
- **Engert22** = Engert S, Sterne GR, Bock DD, Scott K, "Drosophila gustatory projections are segregated by taste modality and connectivity", eLife 11:e78110 (2022), PMC9170244.
- **Codex** = https://codex.flywire.ai/ (dataset header on `codex.flywire.ai/stats?data_version=783`, fetched 2026-09-26).

---

## A. Shiu et al. 2024 — model definition

### A1. Shiu model parameter table

| Parameter | Value | Unit | Source location (paper) | Source location (repo `model.py`, `default_params`) | Notes |
|---|---|---|---|---|---|
| V_resting (`v_0`) | −52 | mV | Shiu24 Methods "Computational model": "V resting = −52 mV (resting potential from ref. 18)" | `model.py` line 22: `'v_0' : -52 * mV` | ref. 18 = Kakaria & de Bivort 2017 (repo comment line 21) |
| V_reset (`v_rst`) | −52 | mV | Methods: "V reset = −52 mV (reset potential after spike)" | line 23: `'v_rst' : -52 * mV` | |
| V_threshold (`v_th`) | −45 | mV | Methods: "V threshold = −45 mV (threshold for spiking)" | line 24: `'v_th' : -45 * mV`; spike condition line 50 `'eq_th' : 'v > v_th'` | strict `>` in code |
| R_mbr | 10 | kΩ·cm² | Methods: "R mbr = 10 Kohm cm2 (membrane resistance)" | (folded into `t_mbr`) | |
| C_mbr | 2 | µF·cm⁻² | Methods: "C mbr = 2 µF cm−2 (membrane capacitance)" | (folded into `t_mbr`) | |
| T_mbr = C·R (`t_mbr`) | 20 | ms | Methods: "T mbr = C mb × R mbr (definition of membrane timescale…)" | line 25: `'t_mbr' : 20 * ms, # membrane time scale (capacitance * resistance = .002 * uF * 10. * Mohm)` | 2 µF/cm² × 10 kΩ·cm² = 20 ms |
| T_refractory (`t_rfc`) | 2.2 | ms | Methods: "T refractory = 2.2 ms (refractory period 18, 59)" | line 31: `'t_rfc' : 2.2 * ms` (comment: Lazar et al. eLife.62362) | **Poisson-driven neurons get refractory 0 ms**: `model.py` line 92/103 `neu[i].rfc = 0 * ms # no refractory period for Poisson targets` |
| τ (synaptic decay, `tau`) | 5 | ms | Methods: "τ = 5 ms (synapse decay timescale 61)" | line 28: `'tau' : 5 * ms` (comment: Jürgensen et al. 10.1088/2634-4386/ac3ba6) | |
| T_dly (`t_dly`) | 1.8 | ms | Methods: "T dly = 1.8 ms (time delay from spike to change in membrane potential from ref. 62)" | line 34: `'t_dly' : 1.8*ms` (comment: Paul et al. 2015 10.3389/fncel.2015.00029); used as `Synapses(..., delay=params['t_dly'])` line 175 | |
| W_syn (`w_syn`) | 0.275 | mV | Methods: "W syn = 0.275 mV (free parameter; synaptic weight…)"; "We chose W syn such that activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 firing" | line 37: `'w_syn' : .275 * mV` | the single free parameter |
| Connection weight w_j,i | `Excitatory × Connectivity × w_syn` | mV | Methods: "the synaptic connectivity weight from the Flywire connectivity multiplied by either 1, if neuron j is excitatory or −1, if neuron j is inhibitory, multiplied by W syn" | line 183: `syn.w = df_con.loc[:,'Excitatory x Connectivity'].values * params['w_syn']` | column in `2023_03_23_connectivity_630_final.parquet` |
| Neuron equations | dv/dt = (v_0 − v + g)/t_mbr ; dg/dt = −g/tau | — | Methods eqs: "dv_i/dt = (g_i − (v_i − V_resting))/T_mbr", "dg_i/dt = −g_i/τ", "g_i ← g_i + w_j,i upon spike from neuron j" | lines 44–48 `eqs`; on_pre `'g += w'` line 175; both eqs `(unless refractory)` | α-synapse per Brian2 "converting_from_integrated_form" |
| Reset rule | v = v_rst; w = 0; g = 0 mV | — | Methods: "Upon firing, a neuron's membrane potential is reset to the resting potential, and cannot change for the duration of the refractory time period"; "g_i … after firing of the neuron, starts at 0 mV" | line 52: `'eq_rst' : 'v = v_rst; w = 0; g = 0 * mV'` | note code also resets `g` to 0 on spike |
| Integration method | linear | — | — | line 163: `method='linear'` | Brian2 exact linear integrator |
| Poisson input rate (`r_poi`) default | 150 | Hz | — | line 39: `'r_poi' : 150*Hz` | paper sweeps 10–200 Hz (sugar), 20–260 Hz (water), 20–220 Hz (JON) |
| 2nd Poisson class (`r_poi2`) default | 0 | Hz | — | line 40 | used for co-activation (e.g. sugar+bitter) |
| Poisson weight factor (`f_poi`) | 250 | × w_syn | — | line 41: `'f_poi' : 250, # scaling factor for Poisson synapse; 250 is sufficient to cause spiking`; PoissonInput weight = `w_syn*f_poi` = 68.75 mV, `N=1`, target_var `'v'` (lines 85–91) | one Poisson event ⇒ one spike (jump 68.75 mV ≫ 7 mV threshold gap) |
| Trial duration (`t_run`) | 1000 | ms | Methods: "30 simulations of 1,000 ms for each experiment were performed" | line 17 | |
| Number of trials (`n_run`) | 30 | — | same | line 18 | |
| Firing-rate definition | spikes per trial / t_run, mean & s.d. over the 30 trials (unstimulated trials count as 0) | Hz | Methods: "The firing times of all neurons that fired in any of 30 simulations was recorded, and then these data were converted into average firing per second" | `utils.py` `get_rate()` lines 32–84 | |
| "Activated" neuron definition | firing rate > 0 Hz | — | Shiu24 main text: "Activated neurons are defined as neurons that have greater than 0 Hz firing." | — | |
| Silencing definition | all output weights of neuron set to 0 | — | Methods: "each of the top 200 firing neurons was silenced by eliminating all output of that neuron" | `model.py` `silence()` lines 108–127: `syn.w[' {} == i'.format(i)] = 0*mV` (presynaptic index i) | README says "to and from" but code zeroes only outgoing |
| "Required" neuron criterion | MN9 firing ≤ 80% of control at any tested frequency | — | Main text: "any neuron whose silencing causes MN9 firing to be 80% or lower compared with control MN9 firing at any of the eight sugar activation frequencies tested (50, 60, 70, …120 Hz)" | — | |
| Brian2 version | 2.5.1 | — | (Sapkal24 Methods: "brian2 software v.2.5.1") | `environment.yml`: `brian2=2.5.1`, `python=3.10`, `numpy=1.24` | |
| Baseline firing | 0 Hz | — | Main text: "The baseline firing of each neuron in our model is 0 Hz." | `neu.v = params['v_0']`, `neu.g = 0` | |

### A2. Connectome inputs and sign assignment

| Quantity | Value | Dataset+version | Source | Suggested tolerance | Notes |
|---|---|---|---|---|---|
| FlyWire materialization | v630 ("public materialization v.630") | FlyWire v630 | Shiu24 Methods: "All 127,400 proofread neurons from Flywire materialization v.630 are included in the model." | exact | repo README: "The code is currently set up for the Flywire version 630, which the paper used. To use public version 783, change the config" (files `Completeness_783.csv`, `Connectivity_783.parquet` in repo) |
| Neurons in model | 127,400 | v630 | Shiu24 Methods (above); `2023_03_23_completeness_630_final.csv` has 127,400 rows, single column `Completed`, all `True` (computed from file) | exact | Lin24 quotes 127,978 for v630 (see B) — different snapshot/filter; the model file is the regression target |
| Synapse threshold for connections in model | **none (min 1 synapse)** | v630 | computed from `2023_03_23_connectivity_630_final.parquet`: columns `['Presynaptic_ID','Postsynaptic_ID','Presynaptic_Index','Postsynaptic_Index','Connectivity','Excitatory','Excitatory x Connectivity']`; `Connectivity` min = 1, max = 2358; counts of Connectivity=1: 7,305,126; =2: 2,611,152; =3: 1,342,881; =4: 813,991; =5: 542,616 | exact | The paper Methods do not state a threshold; the file confirms all connections ≥1 synapse are used. |
| Number of connections (edges) in model | 14,687,178 | v630 | computed from same parquet (`shape (14687178, 7)`) | exact | |
| Total synapses in model | 52,793,639 | v630 | computed: `sum(Connectivity)` | exact | paper abstract: "more than 125,000 neurons and 50 million synaptic connections" |
| Edges with Excitatory=+1 / −1 | 8,800,532 / 5,886,646 | v630 | computed from parquet (`Excitatory` value counts) | exact | |
| Max |weight| | 2358 (excitatory 1801 max; inhibitory −2358) | v630 | computed: `Excitatory x Connectivity` min −2358, max 1801 | exact | |
| Sign rule (inhibitory NTs) | GABA, glutamate ⇒ inhibitory; ACh, dopamine, octopamine, serotonin ⇒ excitatory | — | Shiu24 Methods "Neurotransmitter predictions": "We assume GABAergic and glutamatergic neurons are inhibitory 66, and that each neuron is either exclusively inhibitory or excitatory. … we used a cleft score cutoff of 50, and identified the highest neurotransmitter prediction for each presynaptic site and, if greater than half of all the presynaptic sites across the entire neuron are predicted to be inhibitory (GABA or Glut), we assigned this neuron as inhibitory. Neurons predicted to be dopaminergic, octopaminergic or serotonergic are assigned to the excitatory category." | exact rule | Sapkal24 Methods (re-using the model): "inhibitory (GABA, Glu) or excitatory (all other)" |
| Whole-brain NT fractions (as quoted by Shiu) | ≈55% cholinergic, 24% glutamatergic, 14% GABAergic, 7% DA/OA/5-HT | FlyWire (Eck24 predictions) | Shiu24 Methods: "In the entire Flywire volume, approximately 55% of neurons are predicted to be cholinergic, 24% glutamatergic, 14% GABAergic and the remaining 7% are predicted to be dopaminergic, octopaminergic or serotonergic 3." | ±2 percentage points (stated as "approximately") | good sanity check for an NT-sampling module on v630 |
| NT breakdown among 613 taste-responsive neurons | 52% ACh, 25.9% GABA, 17% Glu, 2.9% 5-HT, 2.0% DA, 0.2% OA | v630 | Shiu24 Methods; Shiu-ST 'ST 4' has 613 data rows with per-neuron NT probabilities | exact if recomputed from ST 4 | |
| Laterality convention | paper uses true biological side; FAFB image is L/R inverted | — | Shiu24 Methods: "'right hemisphere GRNs' in this paper correspond to 'left hemisphere GRNs' described previously 4,23"; "we performed unilateral left hemisphere activation for all simulations, except for Extended Data Fig. 1d" | — | Repo notebooks call the same 21-ID list "sugar-sensing neurons in the right hemisphere" and name experiments `sugarR_*`; Supp Table 1A names them `sugar_l_*`. Treat the 21-ID list as canonical, ignore side labels. |

### A3. Stimulated neuron ID sets (repo `figures.ipynb`) — regression fixtures

| Set | Count | IDs (source) | Notes |
|---|---|---|---|
| Labellar sugar GRNs (main set) | 21 | `figures.ipynb` cell 4 / `example.ipynb` cell 3 `neu_sugar`: 720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663, 720575940611875570 | Fig 1d–f, 3, 4e |
| Sugar GRNs other side (Extended Data Fig 1d) | 10 | `figures.ipynb` cell 76 `neu_sugar_left` | |
| Bitter GRNs | 21 | cell 23 `neu_bitter` | |
| Ir94e GRNs | 18 | cell 23 `neu_ir94e` | |
| Water GRNs | 18 | cell 32 `neu_water` | |
| JONs (JO-CE / JO-F / JO-D,m) | 70 / 60 / 16 = 146 listed (paper says 147) | cell 50 `neu_JON_CE`, `neu_JON_F`, `neu_JON_D_m` | Shiu24: "We activated a set of 147 previously identified JONs of the JO-C, JO-E, JO-F and JO-m subclasses"; count of IDs in notebook = 146 (cell 50 also has a bug: `neu_JON_all = neu_JON_CE + neu_JON_F + neu_JON` references undefined `neu_JON`). Supp Table 7A legend/Methods say 147. |
| MN9 | 720575940660219265 (contralateral / "MN9_r" in ST 1A / `id_mn9` in repo), 720575940645521262 (ipsilateral / "MN9_l") | `figures.ipynb` cells 8, 17 (`ids_mn9 = [720575940660219265, 720575940645521262] # left and right`) | |
| aDN1, aDN2, aBN1 | 720575940616185531 (`id_DN1_1`), 720575940629806974 (`id_DN2_l`), 720575940630907434 (`id_aBN1`) | cells 54, 62 | |
| Three inhibitory neurons silenced for JO-F→aBN1 | 720575940636066222, 720575940609957315, 720575940624986407 | cell 70 `three_inhibitory` | Extended Data Fig 4c |
| SEZ split-GAL4 cell types | 106 types, 372 neurons total | `sez_neurons.pickle` (dict name→list of IDs; computed counts) | Fig 2 |

### A4. Key quantitative results — from the paper text

| Quantity | Value | Dataset | Source | Suggested tolerance | Notes |
|---|---|---|---|---|---|
| Neurons activated by sugar GRNs at 10 Hz / 200 Hz | 45 / 455 | v630 model | Shiu24 main text: "Of the 127,400 neurons modelled, we found that 45 are predicted to respond to 10 Hz sugar GRN activation, and 455 to 200 Hz (Supplementary Table 1)". Confirmed by counting `>0` in Shiu-ST 'Supplemental Table 1A' rate columns: {10 Hz: 45, 20: 79, 30: 121, 40: 296, 50: 356, 60: 382, 70: 396, 80: 401, 90: 418, 100: 410, 110: 412, 120: 414, 130: 416, 140: 436, 150: 435, 160: 443, 170: 442, 180: 447, 190: 440, 200: 455} | ±5% of count (Poisson-seed variation across 30 trials; the count of "ever fired" neurons is noisy at the margin) | includes the 21 stimulated GRNs |
| Correct connectome activates MN9 at 100 Hz sugar | 100% of simulations; shuffled: 1 of 100 | v630 | Shiu24: "robust activation of MN9 in 100% of simulations when sugar-sensing neurons are activated at 100 Hz, only 1 of 100 shuffled simulations did (Supplementary Table 1d)". Shiu-ST 'ST 1D': MN9_r (720575940660219265) Correct = 68.0 Hz, mean over 100 shuffles 0.004333, s.d. 0.043333 (⇒ 1 shuffle >0); MN9_l 49.033333 Hz correct, 0 in all shuffles; Zorro_l 102.2 vs 2.754; roundup_l 46.333 vs 0.0 | exact for correct-connectome; shuffled ≤ 2/100 | shuffle "while maintaining the global connectivity weight distribution" |
| Sugar-responsive & sufficient for MN9 | 47 neurons; 14 also required | v630 | "our analyses identified 47 neurons predicted to be sugar-responsive, and sufficient for feeding initiation. Of these 47 neurons, 14 are also predicted to be required for MN9 activity (Fig. 1h)" | exact | requires re-running Fig 1e/1f protocol (top-200 activation at 25–200 Hz; silencing at 50–120 Hz) |
| Ten known sugar cell types predicted to respond | 10/10; 8/10 sufficient; 3/5 required (>20%) | v630 | Shiu24 main text; Shiu-ST 'Supplemental Table 2' (e.g. MN9 frequency due to 200 Hz activation: Bract 63.966667, Clavicle 77.566667, Fdg 74.766667, FMIn 32.266667, G2N-1 80.2, MN9 191.066667, Phantom 0.0, Rattle 39.6, Roundup 154.7; silencing ratio at 50 Hz sugar: Bract 1.08748, Clavicle 0.663808, Fdg 0.854203, FMIn 1.313894, G2N-1 0.619211, Phantom 1.082333, Rattle 0.864494, Roundup 0.667239) | rates ±15% rel.; categorical outcomes exact | Phantom is the known false negative (predicted inhibitory) |
| SEZ split-GAL4 screen | 106 of 138 types found; at 50 Hz 11 predicted to activate MN9, 10/11 true; of 95 predicted negative, 4 non-zero; 200 Hz adds 5 false positives; 10 Hz: 6 predicted | v630 | Shiu24 main text & Fig 2; Shiu-ST 'Sup Table 3' (106 rows, columns `10 Hz MN9_Left/Right` … `200 Hz`; at 50 Hz, 11 types have MN9_Left>0 [roundup 79.5, diatom 29.13, sink_sync 22.17, G2N_1 12.37, clavicle 10.5, Fdg 22.53, bract 33.67, rattle 1.87, FMIn 0.37, TH_VUM 0.03, kitty 7.3 Hz], 13 with either side>0) | exact set membership ±1 type | ST 10 row: "Neurons sufficient for proboscis extension … 2A: 101/106 = 0.952830" |
| Overall prediction accuracy | 150/164 = 91.46% ; excluding Fig 2: 49/58 = 84.48% | v630 | Shiu24: "Across 164 predictions … 91% were consistent … Excluding … 84%"; Shiu-ST 'Supp Table 10': Total 150 / 164, "Total Accuracy 0.914634"; "Total, excluding Figure 2" 49 / 58 = 0.844828 | exact | ST 10 rows: ipsi<contra MN9 2/2; sugar responders 12/14; required for sugar 6/10; sufficient 101/106; bitter/Ir94e aversive 4/4; water responsive 8/10; required for water 10/11; JON responders 3/3; required for aDN1 2/2; aBN1 JO-CE vs JO-F 2/2 |
| Robustness to W_syn ±30% | −30%: consistent with 90.2% of default predictions, accuracy 85%; +30%: consistent 95%, accuracy 88% | v630 | Shiu24 Methods; Shiu-ST 'Supplementary Table 11 A' (−30%: MN9 contra 33.46 Hz s.d. 6.3; +30%: MN9 contra 96.1 Hz s.d. 5.5; per-row fractions e.g. sufficient 100/106 and 99/106) | exact | Inhib:Exc ratio ±50%: "Decreasing the strength of inhibition results in predictions consistent with 95% … accuracy drops from 91% to 88%"; increased: 96%, 89%. Glutamate excitatory: false-positive rate of Fig 2 screen rises "from 1% to 16%" (ST 11A: sufficient 84/106 = 0.792; bitter/Ir94e 0/4) |
| Sugar vs water pathway overlap at 40 Hz MN9 | sugar activates 377, water 391, shared 250; sugar∩bitter 2; sugar∩Ir94e 30 | v630 | Shiu24 main text (Fig 3f, Supplementary Table 4). Recount from Shiu-ST 'ST 4' (613 rows, columns Sugar_only/Water_only/Bitter_only/Ir94e_only …): Sugar_only>0: 377, Water_only>0: 391, Bitter_only>0: 61, Ir94e_only>0: 93, sugar&water>0: 280, sugar&bitter: 2, sugar&Ir94e: 36 | 377/391 exact; overlap — use the paper's 250 but note my naive recount gives 280 (paper's Venn likely excludes GRNs or uses another rule) | ST 4 MN9_r row: Sugar_only 40.83 Hz, Water_only 40.4, Bitter_only 0, Ir94e_only 0, Sugar_Bitter 1.33, Sugar_Ir94e 1.12, Water_Bitter 10.07, Water_Ir94e 2.26 |
| Water pathway | 39 water-responsive & sufficient; 30 of 39 also sugar-activated; 9 necessary & sufficient | v630 | Shiu24 main text (Extended Data Fig 2d,e) | exact | water GRNs activated 20–260 Hz; silencing tested at 160–220 Hz (`figures.ipynb` cell 40) |
| Antennal grooming circuit | aBN1, aBN2, aDN1, aDN2 respond to 147 JONs; only 4 neurons besides aDN1 activate aDN1 (aBN1, aDN2 + two at <2 Hz); 3 neurons besides aDN1 reduce aDN1 >20% at 140 Hz JON (aBN1, a descending BN2, aDN2) | v630 | Shiu24 main text Fig 5b–f | exact set | top-300 JON-responders activated at 50/100/150/200 Hz (Methods) or 25–200 Hz (Fig 5c legend/notebook) |
| JO-CE vs JO-F synapses onto aBN1 | 103 and 78 synapses | v630 | Shiu24: "Both JO-CE and JO-F neurons synapse onto aBN1 (103 and 78 synapses, respectively; Fig. 5g)" | exact (pure connectivity check) | Directly checkable from the connectivity parquet with the ID lists in A3 |
| aBN1 firing, JO-CE vs JO-F at 150 Hz | 50.766667 Hz (s.d. 1.358512) vs 1.233333 Hz (s.d. 0.919541) | v630 | Shiu-ST 'Supp Table 8' row Flywire ID 720575940630907434 | ±15% rel. for 50.8; JO-F value < 3 Hz | aDN2 (…806974): 19.033 vs 0.0; aDN1 (…185531): 13.4 vs 0.0 |
| Ir94e result | Ir94e activation inhibits MN9 but does not eliminate MN9 response to strong sugar; bitter does | v630 + behaviour | Shiu24 Fig 3b–e; Shiu-ST 'Supp Table 9A' (50 mM sucrose: Gr66a>Chrimson light off 26/30 extended, light on 3/30; Ir94e>Chrimson off 24/30, on 3/30) | categorical | validated prediction |

### A5. Key numeric series from Supplementary Table 1A / 7A (authors' own model output; best regression targets)

Sheet 'Supplemental Table 1A Sugar Fir' = mean rate (Hz) over 30 × 1 s trials for every neuron that fired, for 21 sugar GRNs driven at 10…200 Hz; columns `sugarR_10Hz … sugarR_200Hz`, then `Standard Deviation` block.

| Neuron (Flywire ID, name in ST) | 10 Hz | 50 Hz | 100 Hz | 150 Hz | 200 Hz | s.d. at 100 / 200 Hz | Suggested tolerance |
|---|---|---|---|---|---|---|---|
| MN9 contralateral (720575940660219265, "MN9_r") | 0.0 | 19.433333 | 65.7 | 83.666667 | 93.233333 | 3.308071 / 5.18127 | mean within ±2 s.d. (≈ ±7 Hz at 100 Hz, ±10 Hz at 200 Hz) of a 30-trial run; full series: 30 Hz 0.333, 40 Hz 4.833, 60 Hz 36.4, 70 Hz 50.033, 80 Hz 58.133, 90 Hz 62.067, 110 Hz 70.9, 120 Hz 74.967, 130 Hz 78.9, 140 Hz 80.9, 160 Hz 84.6, 170 Hz 88.5, 180 Hz 89.167, 190 Hz 91.867 |
| MN9 ipsilateral (720575940645521262, "MN9_l") | 0.0 | 15.966667 | 49.666667 | 60.1 | 62.9 | 4.399495 / 3.571648 | ±2 s.d.; contralateral > ipsilateral must hold (Fig 1c) |
| MN11 (720575940618165019) | 0.0 | 12.9 | 88.966667 | 112.4 | 122.633333 | — | ±15% rel. |
| MN11 (720575940630868793) | 0.0 | 12.433333 | 85.866667 | 108.3 | 119.1 | — | ±15% rel. |
| MN8 (720575940623352063) | 0.0 | 28.5 | 68.733333 | 82.866667 | 91.666667 | — | ±15% rel. |
| MN6_r (720575940628826128) | 0.0 | 9.033333 | 31.333333 | 37.366667 | 39.666667 | — | ±20% rel. |
| MN6_l (720575940627410451) | 0.0 | 6.733333 | 24.8 | 30.333333 | 32.133333 | — | ±20% rel. |
| Top non-GRN responder (720575940622695448, unnamed) | — | — | 114.7 | — | 156.4 | — | ±15% rel. |
| Zorro_l (720575940629888530) | — | — | 102.233333 | — | 146.133333 | — | ±15% rel. |

Sheet 'Supplemental Table 7A Grooming ' (all 147 JONs at 20…220 Hz; 855 data rows):

| Neuron | 20 Hz | 60 Hz | 100 Hz | 140 Hz | 180 Hz | 220 Hz | s.d. at 140 / 220 |
|---|---|---|---|---|---|---|---|
| aBN1 (720575940630907434) | 0.133333 | 12.1 | 24.3 | 45.266667 | 63.066667 | 78.166667 | 2.475659 / 1.752776 |
| aDN1_l (720575940616185531) | 0.0 | 0.1 | 3.166667 | 16.666667 | 29.566667 | 38.433333 | 2.342838 / 0.955103 |
| aDN2_l (720575940629806974) | 0.0 | 0.3 | 3.566667 | 17.133333 | 29.833333 | 38.566667 | 2.156128 / 0.843933 |
| "Descending BN2_3" (720575940611163610) | 16.233333 | 51.6 | 74.0 | 90.633333 | 106.6 | 118.933333 | — |
| # neurons >0 Hz | 227 | 367 | 503 | 628 | 720 | 823 | (computed) |

Suggested tolerance for all Table 7A rates: ±2 s.d. listed (≈ ±5 Hz) when re-run with 30 trials; note these means already include trials in which the neuron did not fire.

### A6. Repo example outputs (`results/example/*.parquet`) — values computed by me from the authors' files with the `utils.get_rate` logic (t_run = 1 s, n_run = 30)

| File | Stimulus | # neurons that spiked in ≥1 trial | total spikes | MN9 720575940660219265 mean ± s.d. (Hz) | MN9 720575940645521262 (Hz) |
|---|---|---|---|---|---|
| `results/example/sugarR.parquet` | 21 sugar GRNs @ 200 Hz (default `r_poi` in notebook text: "By default, the neurons are excited at 200 Hz" — note `model.py` default is 150 Hz; the stored file is consistent with 200 Hz: GRN rates ≈197–202 Hz) | 448 | 511,566 | 93.267 ± 3.151 | 61.8 ± 4.175 |
| `results/example/sugarR_100Hz.parquet` | same @ 100 Hz | 404 | 289,073 | 67.033 ± 6.595 | 48.633 ± 4.672 |
| `results/example/sugarR-720575940617937543.parquet` | 100 Hz + silence GRN 720575940617937543 | 414 | 277,853 | 63.267 ± 4.732 | 45.667 ± 4.377 |

Notes: `example.ipynb` markdown states "more than 400 000 spikes were generated by activating the sugar neurons (30 trials, 1 s each)" and "only about 400 neurons show activity" — consistent. Cross-check: ST 1A gives 93.233 Hz (200 Hz) and 65.7 Hz (100 Hz) for the same neuron from an independent run — agreement within 1.3 Hz. A third-party MLX re-implementation (github.com/Kisame76/drosophila-brain-mlx README, found by web search) reports "On the FlyWire wiring MN9 fires at 67.30 Hz" for the 100 Hz example — again within noise. **Recommended golden test:** 21 sugar GRNs @100 Hz, 30 trials ⇒ MN9(…219265) mean in [58, 76] Hz (≈ ±1.5 s.d. band around 65.7–67.3) and MN9(…219265) > MN9(…521262).

Other repo regression fixtures: the notebooks write per-figure CSVs (`fig_1d_rate.csv`, `fig_1d_rate_std.csv`, `fig_1d_id_top200.pickle`, `fig_1e_{freq}_hz_rate.csv`, `fig_1f_{freq}_hz_rate.csv`, `fig_2_{freq}_hz_rate.csv`, `fig_3a_rate.csv`, `fig_3b_rate.csv`, `fig_4a_rate.csv`, `id_top200_water.pickle`, `fig_4b/4c/4e_*.csv`, `fig_5b_rate.csv`, `id_top300_JON.pickle`, `fig_5c/5d_*.csv`, `fig_5g_JON_CE_rate.csv`, `fig_5g_JON_F_rate.csv`, `fig_s5c_JON_F_silenced_rate.csv`, `fig_s1d_rate.csv`, `fig_s4a/b_*.csv`) — these are NOT in the GitHub repo; they are inside `results.zip` (4.5 GB) on Edmond doi:10.17617/3.CZODIW. The Supplementary Tables xlsx contains the same numbers for Figs 1d, 1e, 1f, 2, 3f/4, 4a–c, 5b–d, 5g, ED 4.

### A7. Experimentally validated predictions (for documentation, not numeric tests)

- Ir94e GRN activation predicted to inhibit MN9; optogenetic Ir94e activation inhibited PER to 50 mM sucrose but not to 1 M sucrose, bitter (Gr66a) eliminated both (Shiu24 Fig 3d,e; ST 9A).
- Fudog and Zorro predicted water-responsive; both responded to water in calcium imaging (ED Fig 1c, 3a).
- Of 6 neurons predicted to have water-silencing phenotypes, 5 did (G2N-1 did not); of 5 predicted not to, 4 did not (Usnea did) (Fig 4d, ED Fig 3b,c).
- JO-CE predicted to drive aBN1 robustly, JO-F not, despite 78 direct synapses; calcium imaging of aBN1 confirmed (Fig 5g,h).
- Silencing sugar GRNs reduced PER to water (Fig 4f), as predicted by sugar+water synergy (Fig 4e).

### A8. Corrections / follow-ups

- **Erratum:** none found. PubMed record 39358519 (efetch XML, 2026-09-26) lists only `CommentsCorrections RefType="UpdateOf"` → the bioRxiv preprint (PMID 37205514); no "Erratum"/"Author Correction" entries. Crossref record has no `update-to`. Web search for "author correction"/"erratum" returned nothing.
- **Follow-up reuse:** Sapkal24 (Nature 2024, same issue) "Connectome-constrained modelling: The neuronal activity was simulated as a spiking neural network in the brian2 software v.2.5.1 … Model details, including the original code, are described elsewhere [Shiu24]. The original code was modified to allow for stimulating and silencing neurons at arbitrary time points … Neurotransmitter predictions … determine whether the interaction between two neurons is inhibitory (GABA, Glu) or excitatory (all other)." Shiu24 Discussion: "Sapkal et al. use our computational model to correctly identify neurons that regulate walking". No specific Sapkal firing-rate numbers appear in the text (values are in their ED Fig heatmaps of FG/BB/DNg12 rates) — **numbers NOT FOUND in text**.
- Third-party reimplementations found by search (not peer-reviewed): github.com/Kisame76/drosophila-brain-mlx (MLX/Metal; "all 127,400 FlyWire v630 neurons and 14,687,178 directed connections … MN9 fires at 67.30 Hz" for the 100 Hz example; also claims a MaleCNS v1.0 port), github.com/BenliusYang/fly-brain (Brian2/Brian2CUDA/PyTorch/NEST ports), github.com/CC834/flywire-drone-brain (v783). Useful as independent cross-checks of the 14,687,178-edge count and the ~67 Hz MN9 value.

---

## B. Dataset-level counts

| Quantity | Value | Dataset+version | Source | Suggested tolerance | Notes |
|---|---|---|---|---|---|
| Proofread neurons | 139,255 | FlyWire v783 | Dork24 abstract & main: "Our reconstruction of an entire adult brain contains 139,255 neurons"; Schl24: "139,255 nodes"; Lin24 Methods: "v783 snapshot, containing 139,255 neurons"; Codex header "FAFB v783 (CB) … 139,255 neurons" | exact | |
| Synapses between proofread neurons | 54.5 million | v783 | Dork24: "139,255 neurons … and 54.5 million synapses between these neurons" | ±0.1 M | abstract says "5 × 10^7 chemical synapses" |
| Total detected synapses in volume (before filtering) | ~130 million (after filtering); ~244 million imported | v783 | Dork24 Methods: "contained ~244 million synapses … After filtering, we were left with ~130 million synapses"; main text "around 130 million synapses", "7.4 synapses per µm3", neuropil volume 0.0175 mm3 | approximate | filter: unassigned pre/post or score ≤ 50 removed |
| Presynapse / postsynapse attachment | ≈122 M presynapses (93.7%) attached; ≈58.1 M postsynapses (44.7%) attached | v783 | Dork24 main text | — | |
| Connections ≥5 synapses | 2,700,513 connections between 134,181 neurons | v783 | Dork24: "Setting a threshold of at least five synapses … We observed 2,700,513 such connections between 134,181 identified neurons" | exact | Lin24 Methods states "2,701,601 thresholded connections" for v783 — 1,088 discrepancy between the two papers; pick Dork24 for v783 but allow ±0.05% |
| Unthresholded weighted edges | ≈15.1 million | v783 | Schl24: "a graph with 139,255 nodes and around 15.1 million weighted edges" | ±0.1 M | |
| Connections >100 synapses / >1,000 synapses | 15,837 / 27 | v783 | Dork24: "a single connection can comprise more than 100 synapses (n = 15,837) or even more than 1,000 synapses (n = 27)" | exact | strongest: LT39 → mALC2 "more than 2,400 synapses" |
| Median in / out degree (intrinsic, ≥5 syn) | 11 / 13 | v783 | Dork24: "The median in degree and out degree of intrinsic neurons are 11 and 13 (Fig. 3g)" | exact | by NT: GABA 14/16, Glu 11/13, ACh 10/13 (Dork24) |
| Intrinsic neurons | 118,501 | v783 | Dork24: "Of the 139,255 proofread neurons in FlyWire, 118,501 are intrinsic to the brain" | exact | |
| Central-brain intrinsic / optic-lobe intrinsic | 32,388 / 77,536 | v783 | Dork24 & Schl24: "32,388 (23%) neurons are intrinsic to the central brain and 77,536 (54%) neurons are intrinsic to the optic lobes" | exact | |
| VPNs / VCNs | 8,053 / 524 | v783 | Dork24 & Schl24 | exact | |
| Sensory (afferent) neurons into central brain | 5,512 sensory; 5,375 non-visual sensory | v783 | Schl24: "5,512 sensory and 2,362 ascending neurons"; Dork24: "5,375 non-visual sensory neurons" | exact | photoreceptors: compound eye 11,118, ocelli 273, eyelets 8 (Dork24) |
| Ascending / descending neurons | 2,362 ascending; 1,303 descending | v783 | Dork24: "1,303 efferent (descending) and 2,362 afferent (ascending) neurons"; Schl24 same | exact | |
| Motor / endocrine neurons | 106 / 80 | v783 | Dork24, Schl24 | exact | |
| L/R central-brain intrinsic count difference | 27 (0.1%) | v783 | Schl24: "they differ by only 27 (0.1%) neurons" | exact | |
| Cell types annotated | 8,453 (3,643 hemibrain-derived, 4,581 new, 229 other literature); covers 96.4% of neurons | v783 | Schl24 | exact | |
| Neurons (v630 snapshot) | 127,978 neurons; 2,613,129 connections (≥5 syn); 14.7 M unthresholded connections | FlyWire v630 | Lin24 Methods: "The v630 snapshot contains 127,978 neurons and 2,613,129 thresholded connections. The central brain of the fly was fully proofread, with the optic lobes around 80% complete"; Discussion: "2.6 million connections instead of 14.7 million unthresholded connections" | exact | Shiu model file has 127,400 neurons and 14,687,178 unthresholded edges (A2) — 578 fewer neurons than Lin's v630 |
| Codex dataset headers (2026-09-26) | FAFB v783: 139,255 neurons, 3,732,460 connections; BANC v888: 158,262 neurons, 3,037,361 connections; MANC v1.1: 23,665 neurons, 5,305,638 connections; MAOL v1.1 (male right optic lobe): 52,445 neurons, 6,484,936 connections; MCNS v1.0: 166,700 neurons, 6,242,118 connections | multiple | `https://codex.flywire.ai/stats?data_version=783` page header text | exact for neurons; connection counts depend on Codex's (unspecified on that page) definition — do not use as ≥5-synapse edge counts | Codex "connections" ≠ Dork24's 2,700,513, so Codex likely counts per-neuropil or a different threshold |
| Hemibrain neurons / synapses | "≈25,000 neurons", "about 20 million chemical synapses"; "64 million PSDs and 9.5 million T-bars in the hemibrain volume"; "22,594 neurons with 5229 morphological types and 5609 connectivity types" (typed & named); "15,912 neurons" traced to cell bodies + "6682 neurons that were not traced up to their cell bodies" | hemibrain v1.1 (paper) | Sche20 main text & Table 3 | approximate (paper rounds) | Sche20: "The analyses in this article … are based on version v1.1" |
| Hemibrain traced neurons (v1.1) | 21,663 | v1.1 | dvid.io/blog/release-v1.1/: "the skeletons of the 21,663 traced neurons are available as a tar file" | exact | |
| Hemibrain v1.2.1 | "No updates to the connectome, but synapse-to-mito distances have been recomputed" (Janelia FlyEM tweet 2021-06-04, via search snippet) ; Eck24 used "all 24,666 well-reconstructed neurons in the HemiBrain dataset (hemibrain:v1.2.1)"; Schl24: hemibrain "5,235 morphology types … 640 connectivity types … final total of 5,620 types" | v1.2.1 | as stated | — | **Exact v1.2.1 neuron/synapse totals NOT FOUND in a quotable document** (neuPrint release-notes page is a JS app; dvid blog v1.2 post has no counts). Best quotable: 21,663 traced (v1.1, connectome unchanged in 1.2.1 per tweet) and 24,666 "well-reconstructed" (Eck24). |
| Hemibrain synapse-count reliability rule | connection of 10 synapses found with >99.9% probability if ≥half of synapses traced; "medium strength connection (3–9 synapses) then the connection is real" | hemibrain | Sche20 | — | |
| MANC neurons / synapses | "roughly 23 thousand traced neurons, 10 million TBars, 74 million PSDs, and 44m of neuronal cable" | MANC v1.0 | Take23 (bioRxiv v1) Results | approximate | eLife "Connectomes: Mapping the fly nerve cord" summary (via search): "over 23,000 neurons connected by more than 10 million pre-synapses, 74 million post-synapses" |
| MANC class breakdown | ~23,500 neurons: 13,066 IN, 1328 DN, 1862 AN, 733 MN, 92 EN, 9 EA, 5927 SN, 535 SA | MANC (v1.2.3 queried) | Cheong25: "In total, ~23,500 neurons were reconstructed and annotated in this dataset, which comprises 13,066 INs, 1328 DNs, 1862 ANs, 733 MNs, 92 ENs, 9 EAs, 5927 SNs, and 535 SAs"; "Connectome data was queried from MANC v1.2.3" | exact per class (v1.2.3) | Take23: "descending neurons and motor neurons (total n=2065)"; "Sensory neurons and intrinsic neurons … other than motor neurons (total n=21683)"; "approximately 1300 descending neurons and 700 motor neurons". Codex: MANC v1.1 23,665 neurons |
| MANC DN neurotransmitter prediction split | 68.4% cholinergic, 16.2% GABAergic, 7.5% glutamatergic, 7.9% other/below threshold (cutoff 0.7) | MANC | Cheong25 | exact | vs experimental 38%/37%/6% (Hsu & Bhandawat 2016) |
| MaleCNS neurons | 166,691 neurons; 11,691 cell types | MaleCNS v1.0 | MCNS-pre abstract: "This contains 166,691 neurons spanning the brain and ventral nerve cord, fully proofread and comprehensively annotated including fruitless and doublesex expression and 11,691 cell types"; Google Research blog: "over 166,000 neurons", "125 million synaptic connections"; Codex: MCNS v1.0 166,700 neurons | exact (166,691) | male-cns.janelia.org: "MaleCNS v1.0 released" 2026-06-08 (v0.9 2025-10-03); Cell paper 2026-09-03 |
| MaleCNS synapse detection | "46 million presynapses connected to 312 million PSDs across the entire volume with an average precision/recall for connections of 0.82/0.…"; "94% pre- and 42% postsynaptic completion rates in neuropils"; nuclei: "98.9% of the 141,780 detected neuron-associated nuclei are part of a proofread neuron" | MaleCNS | MCNS-pre Results | approximate | |
| MaleCNS cross-matching | 97.5% of neurons matched to FAFB/FlyWire, hemibrain and/or MANC; 96.4% central brain, 98.8% optic lobe (to FAFB/hemibrain), 93.1% VNC (to MANC) | MaleCNS vs FlyWire/MANC | MCNS-pre | exact | "we revised about 4% of FlyWire neurons in this work … whereas in earlier work matching the hemibrain to FlyWire we had to revise over 44% of cell types" |
| MaleCNS sexual dimorphism | Of 7,319 cross-matched central-brain cell types: 114 dimorphic, 262 male-specific, 69 female-specific (4.8% of male neurons, 2.4% of female); "331 sex-specific and 114 sexually dimorphic cell types"; "1,427 male-specific neurons across both hemispheres of the male CNS (vs 363 female-specific) and 924 dimorphic neurons (matched with 811 neurons in the female)"; central brain: 3.4% male-specific, 1% female-specific, 1.2% (1% female) dimorphic; optic lobe: 0.1% sex-specific, 0.3% dimorphic; 249 OL-intrinsic types: 3 sex-specific (Cm26, Tm26, Mi20), 1 dimorphic (TmY21); VPNs 99.9% isomorphic, 1 male-specific type LoVP92 (13 cells); fru+ 4,505 (2,695 high conf.), dsx+ 407 (332) neurons | MaleCNS vs FlyWire | MCNS-pre abstract & Results | exact | visual columns "male ∼900; female ∼800" |
| BANC neurons / synapses | "Total proofread neurons: 114,518"; "proofread and roughly proofread neurons (totalling 155,916)"; "171,512 accounted-for objects" (incl. 13,108 glia/trachea); synaptic links file "218,460,852 synaptic links (pre-post connections), of which 74% of presynaptic ends and 23% of postsynaptic ends are connected to a proofread neuron"; alternative synapses_v3 "259,409,001 synaptic links"; "147,846 neurons linked to a cell-type label (93.1% in the central brain and VNC)"; DNs 1,316, ANs 1,849, effector neurons 1,031 | BANC (v626 in preprint; Codex v888 = 158,262 neurons) | BANC26 (Nature) main text/figure legends (114,518 from preprint v2 via WebFetch; 155,916, 171,512, 218,460,852 from PMC text) | exact as quoted per version | brain ≈140,000 + VNC ≈20,000 expected; lamina missing (~9,390 R1–R6/Lai cells) |
| BANC synapse thresholds / prediction quality | postsynapse size threshold ≥5 for links; graph "thresholded at a synaptic count of ≥5"; synapse CNN F-score 0.83 (Nature) / 0.79, precision 0.68, recall 0.95 (preprint); NT ground truth 60,394 neurons from 3,379 cell types (16,448 train / 4,124 test) | BANC | BANC26 Methods | — | |
| Eckstein NT classifier accuracy | 87% per synapse (FAFB), 78% (hemibrain); 94% per neuron (FAFB-Catmaid), 91% (hemibrain); 91% of 624 FAFB-FlyWire cell types and 91% of 524 hemibrain cell types correct | FAFB/FlyWire v630 | Eck24 Summary, Results, Discussion; test set "40,104 presynapses from 185 neurons" | exact | per-cell-type: cholinergic 91%/91%, glutamatergic 91%/95%, GABAergic 96%/97%, dopaminergic 90%/85%, octopaminergic 85%/100%, serotonergic 33%/38% (FAFB/hemibrain) |
| Eckstein confusion-matrix entries quoted in text | ACh→ACh 0.95; ACh→GABA 0.02 (FAFB test set) | FAFB | Eck24 STAR Methods (confidence score example): "each presynapse predicted as acetylcholine would contribute a value of 0.95 … mispredicted as GABA would contribute a value of 0.02" | exact | Full 6×6 matrix is Figure 2A (image) — **remaining entries NOT FOUND in text**. "Often, glutamatergic and GABAergic neurons are mixed—our network's most common confusion" |
| Eckstein NT prediction consistency | L–R matched singletons 1,586 pairs, FAFB–hemibrain 1,318 pairs; "Matches:mismatches … FAFB-FlyWire right: 2,650:170 FAFB-FlyWire left, 1,562:40, and HemiBrain neurons, 1,088:130"; 95% of 2,626 matched cell types agree between FlyWire and hemibrain; 14% of multi-member hemibrain types inconsistent | v630 / hemibrain v1.2.1 | Eck24 Fig 4, S2 | exact | Neuron-level predictions computed for "49,985 central brain and 86,942 optic lobe FAFB-FlyWire neurons" (136,927 total) |
| Eckstein whole-brain NT fractions | central brain "largest fraction … cholinergic" (Fig 6A, not numeric in text); scRNA-seq comparison "44%–45% cholinergic, 14%–15% glutamatergic, and 10%–15% GABAergic"; predicted monoaminergic counts excluding KCs & sensory: "6052 dopaminergic, 2000 [serotonergic] neurons and 289 octopaminergic neurons in FlyWire" vs literature ∼130 DA, ∼80 5-HT, ∼44 OA | v630 | Eck24 | — | use Shiu24's "55/24/14/7%" quote (A2) for whole-volume fractions; Eck24 optic-lobe checks: 96% of ∼29,000 known cholinergic OL neurons, 87% of ∼3,600 GABAergic, 91% of ∼1,600 glutamatergic correct; Kenyon cells mispredicted dopamine (99.9%) |
| Eckstein synapse budget by compartment | axo-dendritic 55% (FlyWire) / 48% (hemibrain); axo-axonic 22% / 20%; dendro-dendritic 19% / 21%; presynapses on axon median 76% / 70% | v630 / hemibrain | Eck24 | ±2 pp | |

### B2. Lin et al. 2024 network statistics (v630, ≥5-synapse graph: 127,978 neurons, 2,613,129 connections)

| Statistic | Value | Source (Lin24) | Suggested tolerance |
|---|---|---|---|
| Connection probability | 0.000160 (Table 2) / 0.000161 (main text) | Table 2; main text "the connection probability is 0.000161" | exact to 3 s.f. |
| Connection reciprocity | 0.138 ("x858 than ER", "x43.33 than CFG") | Table 2 & main text | ±0.001 |
| Clustering coefficient | 0.0463 (Table 2) / 0.0477 (main text) — paper inconsistent | Table 2: "Clustering coefficient 0.0463 x144 than ER x7.06 than CFG"; main text: "The clustering coefficient … is 0.0477" | accept [0.046, 0.048] |
| Avg connection strength | 12.61 synapses (range 5 ~ 2358); "average connection consists of 12.6 synapses" | Table 2 | ±0.05 |
| Avg in/out degree (intrinsic) | 20.5; in vs out Pearson R = 0.76 | main text | ±0.1 |
| Giant SCC / WCC | 93.3% / 98.8% of neurons | Fig 1d,e | ±0.1 pp |
| Mean shortest path | directed within SCC 4.42 hops, max 13; undirected within WCC 3.91 hops, max 11 | main text | ±0.02 |
| Small-worldness | S = 141; ER ℓ_rand 3.57, C_rand 0.0003 | Methods eq. 1 | ±5% |
| Rich club | onset total degree > 37 (Φ_norm > 1.01); 40,218 neurons (≈30%); within-club connection prob 0.000870 (5.4× overall); peak at degree 75 (38.9% of neurons ≥75, 2.76% denser than CFG); club ends at degree 93; in-degree-only rich club 10–54; no out-degree rich club | main text & Methods | exact for thresholds; ±1% for counts |
| Reciprocal participants | 77,607 neurons in ≥1 reciprocal connection; avg 23% of incoming / 18% of outgoing connections reciprocal; 12.1% of reciprocal pairs span two neuropils | main text, Table 1 | ±0.5 pp |
| Motif participants (Table 1 data products) | feedforward-loop participants 113,978; 3-unicycle participants 66,835; highly reciprocal neurons 2,183; NSRNs 704 (text: 1,863 candidates, 54% GABA, 10% Glu); broadcasters 676; integrators 638; attractors/repellers 3,469 each | Table 1 | exact |
| 3-node motifs | feedforward motifs 1–3 under-represented vs ER & CFG; motifs 7–13 over-represented | Fig 3 | qualitative |
| Distance dependence | 3% of neuron pairs within 50 µm make 71% of connections; NND model p_close = 0.00418 | ED Fig 1d, Methods | ±0.5 pp |
| Hemisphere crossing | 11% of neurons have inputs in both hemispheres, 11% outputs; rich-club 18%/17% | main text | ±1 pp |
| Unthresholded statistics | Extended Data Table 2 (no-threshold vs 5-synapse) — values are in a table image; **NOT FOUND in text** | ED Table 2 | — |

---

## C. Cross-dataset / variability results (Schlegel 2024 unless noted; edges = cell-type→cell-type, unthresholded ≥1 synapse)

| Quantity | Value | Datasets | Source | Suggested tolerance | Notes |
|---|---|---|---|---|---|
| Pre/post-synapse counts per cross-matched cell type | within brain Pearson R = 0.99; across brains R = 0.92 (pre) and 0.76 (post) | FlyWire L vs R; FlyWire vs hemibrain | Schl24 Fig 4a,b: "pre- and post-synapse counts per cell type were highly correlated, both within brains (Pearson R = 0.99; P < 0.001) and across brains (Pearson R = 0.92 and 0.76 for pre- and post-synapses, respectively" | ±0.01 | ED Fig 4k,l slope/ratios: "1.176 (presynapses…) 0.983 (post) between FlyWire and the hemibrain" |
| Edge-weight correlation | within brain R = 0.97; across brains R = 0.8 | same | Schl24 Fig 4c: "Weights of individual edges are highly correlated within (Pearson R = 0.97, P < 0.001) and across (Pearson R = 0.8, P < 0.001) brains" | ±0.01 | |
| Cosine connectivity similarity across vs within brains | difference (effect size) 0.045 ± 0.096 (across lower, P < 0.001) | same | Schl24 Fig 4d: "the effect size is small (0.045 ± 0.096)" | exact | absolute cosine values are in Fig 4d boxplots (image) — NOT in text |
| Edge persistence | 572,980 edges in ≥1 hemisphere; 53% of hemibrain edges found in FlyWire; FlyWire L→R 61%, R→L 59% | same | Schl24 Fig 4e | ±1 pp | |
| Single-synapse edge persistence | hemibrain 1-synapse edge: 42% present in one FlyWire hemisphere, 16% in both | same | Schl24 Fig 4f | ±1 pp | |
| Strong-edge rule | edges >10 synapses or ≥0.9% of target input: >90% persistence; 16% of edges hold ~79% of synapses (>10 syn); ~7% of edges / 54% of synapses (≥0.9%); 99% persistence: >2.6% edge weight or 31 synapses | same | Schl24 Fig 4g | exact | |
| Edge weight regression | 30-synapse hemibrain edge → mean 29 in FlyWire; 25% of them <13, 5% only 1–2; 30-synapse FlyWire-left edge → mean 31 on right; 25% ≤21, 5% 1–8 | same | Schl24 Fig 4h,i | ±1 synapse | |
| Technical-noise model | 65% of L/R edge-weight variability within technical-noise 5–95% range (→100% for weak edges); "differences in edge weights of 30% or less may be entirely due to technical noise"; MB calyx postsynaptic completion L 52.5%, 6% L/R difference; synapse detection precision 0.72 | FlyWire L vs R | Schl24 Fig 4j,k & Methods | — | key null-model parameters |
| Cell-type count variability | KCs: 2,597 (FlyWire R), 2,580 (FlyWire L), 1,917 (hemibrain) — 30% larger per hemisphere; "average variation in cell counts (5 ± 12%)"; hemilineage counts differ 3% (±4%) L vs R; FlyWire vs hemibrain hemilineage R² = 0.98, FlyWire ≈5% more neurons | same | Schl24 | exact | ALPN→KCg-m input types 5.74 / 5.89 / 8.76 (FlyWire L, R, hemibrain) |
| Cell-type matching | 56% (2,920/5,235) hemibrain types unambiguously found; 664 (13%) merged/split; 1,651 (32%) not reidentified; 3,584 hemibrain types → 3,643 consensus types; 43,737 neurons validated; ~0.4% of central-brain neurons "biological oddities" | same | Schl24 Fig 3 | exact | |
| NT prediction agreement (Eck24) | see B: 1,586 L/R pairs, 1,318 FlyWire/hemibrain pairs; 95% of 2,626 cell types agree | v630 / hemibrain | Eck24 | exact | |
| Male vs female (MaleCNS) | see B MaleCNS rows: 7,319 cross-matched central-brain types; 114 dimorphic / 262 male-specific / 69 female-specific; 98.8% OL neurons matched; "comparable neuron counts per type"; only ~4% of FlyWire cell types revised | MaleCNS v1.0 vs FlyWire v783 | MCNS-pre | exact | Male DNb07→DNp63 connection "91% axo-axonic"; DN–DN dendro-dendritic 27%, AN–AN 34% |
| BANC vs FAFB / MANC matched connections | "FAFB-BANC: 483,957 matched cell type connections, MANC-BANC: 434,357 matched cell type connections"; KC outgoing links 36.6 in BANC vs ~200 in FAFB (cleft score >50) | BANC vs FAFB v783 / MANC | BANC26 ED figure legend & Methods | exact | BANC influence metric: modal adjusted influence 20 direct / 5 indirect (Nature text; preprint said 14 / 8) |

---

## D. connectome_interpreter (Yin et al. 2025) and effective connectivity

| Quantity | Value | Dataset | Source | Suggested tolerance | Notes |
|---|---|---|---|---|---|
| Central-brain in-degree (neurons / cell types) | mean ~130 partners (median ~90); ~80 cell types (median ~55) | FAFB/FlyWire | CI25 Introduction | ±10% | "on average any one neuron receives synapses from 1/1,000 of all neurons, i.e., about 140, when considering all mapped synapses" |
| Polysynaptic path density (100×100 random cell-type pairs) | threshold 0: ~70% of pairs connected within 2 hops, 100% within 5 hops; threshold 1% normalized input: ~2% at 2 hops, ~84% at 5 hops (sign ignored) | FAFB/FlyWire | CI25 Results 2.4 & Fig 4A; Discussion 3.1: "for 84% of all possible pairs of cell types there was at least 1 polysynaptic path" | ±5 pp (random sampling of 100 types) | normalization: weight / total postsynapses of target ("a number between 0 and 1") |
| Effective E/I convergence | "effective excitation and inhibition between two cell types become similar as the path length between them increases (up to 5 hops)" | FAFB/FlyWire | CI25 Fig 1L | qualitative | |
| Functional-annotation table | 2.4% (174/7175) central-brain cell types, 15% (71/481) descending, 17% (60/352) VPN, 4% (15/423) sensory types with known function; "at least 30% of the cells, comprising 5% of all cell types" | FAFB/FlyWire | CI25 Results 2.1 | exact | |
| Worked monosynaptic examples | HP5 hygrosensory → ipsilateral DNb05; JO-D → contralateral CB0916; JO-A/JO-B → Giant Fiber, DNp02, DNp11; oviLN indirectly inhibits DNc01/DNc02; LPLC1 → DNa05 bilateral indirect excitation (2-hop) | FAFB/FlyWire | CI25 Fig 1H–K, Fig 1M–N | existence of edge at 1% threshold | wiring diagrams "thresholded at 1% normalized input" |
| DNa10 visual receptive field | DNa10 receives >1% direct input from VPN types LLPC3, LPLC4, LC10d, LC10c, LTe64, LC22; 3-hop paths from L1–L3, R7, R8 | FAFB/FlyWire | CI25 Results 2.3.2, Fig 3D,E | set membership | |
| Complexity | naive matrix-power O(n³·h); toolkit "minutes instead of hours or days" for ~140,000 neurons | — | CI25 Fig 1F | — | **No specific effective-connectivity numeric values (e.g. "X% at 2 hops for pathway Y") appear in the text — NOT FOUND; they are in figure heatmaps/supplements.** |
| BANC "adjusted influence" | ~24 billion pairwise scores; modal 20 (direct) / 5 (indirect); "high influence" cutoff 17.28; AN/DN clusters 1,000 AN/DN cell types | BANC | BANC26 | — | alternative effective-connectivity metric (signal-cascade based, graph ≥5 synapses) |

---

## E. Circuit facts checkable from connectivity alone

| Quantity | Value | Dataset | Source | Suggested tolerance | Notes |
|---|---|---|---|---|---|
| Descending neurons | 1,303 | FlyWire v783 | Dork24, Schl24 | exact | BANC: 1,316 DNs; MANC: 1328 DNs (Cheong25); "about 1,300" (BANC26 intro) |
| Ascending neurons | 2,362 (brain side); 1,849 (BANC); 1862 (MANC) | v783 / BANC / MANC | Dork24, Schl24; BANC26; Cheong25 | exact per dataset | |
| Head motor neurons | 106 | v783 | Dork24: "head motor neurons (n = 106)" | exact | |
| Endocrine neurons | 80 | v783 | Dork24 | exact | |
| Proboscis motor neurons | 16 MNs control the proboscis (literature); MN9 = 2 neurons in Shiu model (IDs in A3) | — | Shiu24: "controlled by the activity of 16 MNs 32" | — | count of FlyWire MN9 neurons per se: 2 IDs used (720575940660219265, 720575940645521262) |
| Labellar sugar / bitter / water / Ir94e GRNs used in model | 21 / 21 / 18 / 18 (one hemisphere); 10 sugar on the other side | v630 | Shiu-repo `figures.ipynb` (A3) | exact | Engert22 (FAFB, CATMAID): "87 gustatory projections from the proboscis labellum in the right hemisphere and 57 from the left"; six groups of "7–23 GRNs" (right), seven groups of "4–15 neurons" (left); "90–104 GRNs per labellum" (literature); GRN synapses: "175 (±6 SE) presynaptic sites and 168 (±6 SE) postsynaptic sites"; 79% of GRN–GRN synapses within-group |
| JONs in grooming model | 147 (JO-C, JO-E, JO-F, JO-m) | v630 | Shiu24 | exact | 146 IDs in notebook (see A3) |
| Kenyon cells per hemisphere | 2,597 (R) / 2,580 (L) FlyWire; 1,917 hemibrain | v783 / hemibrain | Schl24 | exact | MBp4 hemilineage 1,335 neurons (both hemispheres counted); FLAa1 30 |
| ALPNs / canonical types | ~130 ALPNs, 58 canonical types | v783 | Schl24: "around 130 antennal lobe projection neurons (ALPNs) comprising 58 canonical types" | ±5 | |
| Central complex FC1–3 / FB1–9 | 357 / 897 neurons; 114 cross-brain types vs 146 hemibrain types | v783 | Schl24 | exact | |
| Ocellar ganglion | 63 neurons: 16 local, OCG01 12, OCG02 8, DNp28 2, 25 centrifugal; 15 DNs each receive >200 synapses from OCG01 | v783 | Dork24 | exact | DNp20/DNOVS1 gets 57% (L) / 44% (R) of brain input from ocellar PNs; DNp22/DNOVS2 36%/33% |
| Photoreceptors | compound eye 11,118; ocelli 273; eyelets 8 | v783 | Dork24 | exact | |
| JO-CE / JO-F → aBN1 | 103 / 78 synapses | v630 | Shiu24 Fig 5g | exact | pure connectivity test with the A3 ID lists |
| SEZ share of central-brain neuropil volume | 17.8% (0.0018 of 0.0103 mm³); DNs receive on average 52% of inputs in SEZ neuropils | v783 | Dork24 | ±0.1 pp | |
| Optic lobe cell types (right OL) | 156 types for 35,567 of 38,461 neurons (Schl24) vs 229 types for 37,345 neurons (Matsliah/ref. 11) | v783 | Schl24 Methods | exact | |
| Giant fiber inputs | JO-A/JO-B synapse directly onto the Giant Fiber (CI25, qualitative). **Numeric GF input counts NOT FOUND** in fetched sources. | — | CI25 | — | |
| Hemilineages | 120 neuroblast lineages, 183 hemilineages, 88% (30,233) of central-brain neurons; primary neurons 3,779 (11%); 797 (2%) unassigned secondary | v783 | Schl24 | exact | Eck24: Lacin's law holds in 88% of hemilineages |

---

## F. Not found / caveats

1. Shiu24: no erratum exists (checked PubMed, Crossref, web). The Methods contain no synapse-count threshold; the connectivity file proves none was used (min = 1).
2. Shiu24 per-neuron rates for Fig 1e/1f/4b/4c/5c/5d exist in the xlsx sheets ST 1B, 1C, 6B, 6C, 7B–7E but were not transcribed here (large matrices); they are directly loadable.
3. Sapkal24: no firing-rate numbers in text.
4. Hemibrain v1.2.1: exact neuron/synapse totals not found in a quotable document (only 21,663 traced neurons for v1.1 and "no updates to the connectome" for 1.2.1; 24,666 "well-reconstructed" neurons per Eck24).
5. Lin24 Extended Data Table 2 (unthresholded stats) and Eck24 Fig 2A full confusion matrix are images — not extracted.
6. Schl24 absolute cosine-similarity distributions (Fig 4d) are only in the figure; only the 0.045 ± 0.096 effect size is in text.
7. MaleCNS Cell version (10.1016/j.cell.2026.08.015) full text returned 403; all numbers are from the bioRxiv v1 preprint, the Google Research blog (166,000 neurons / 125 million synaptic connections) and Codex (166,700 neurons). The Cell taste companion paper (10.1016/j.cell.2026.08.016) likely contains updated GRN counts — not fetched.
8. Codex "connections" counts (e.g. 3,732,460 for v783) use an unstated definition and do not match Dork24's 2,700,513 ≥5-synapse connections.
9. Discrepancies to be aware of when writing tests: v783 ≥5-syn connections 2,700,513 (Dork24) vs 2,701,601 (Lin24); v630 neurons 127,978 (Lin24) vs 127,400 (Shiu model file); clustering coefficient 0.0463 (Lin24 Table 2) vs 0.0477 (Lin24 text); JON count 147 (paper) vs 146 IDs (notebook); paper's 250 shared sugar/water neurons vs 280 by naive recount of ST 4.
