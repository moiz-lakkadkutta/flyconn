# Tutorials

One runnable script per anchor workflow (jupytext percent format; open as a notebook with
`jupytext --to ipynb examples/<name>.py` or run with `python examples/<name>.py` after
`flyconn data pull`):

- W1 circuit tracing with signs and confidence: `examples/w1_pathways.py`
- W4 version drift: `examples/w4_version_drift.py`
- W2 in-silico experiment from one YAML file: `flyconn run examples/specs/w2_malecns_lb3_silence_gng232.yaml --out runs/w2` (controls on by default; the report states that MaleCNS runs are calibrated by protocol, not validated)
- W2 silencing screen (sweep) from one YAML file: `flyconn run examples/specs/w2_sweep_silence_gng.yaml --out runs/w2_sweep` silences each of the 10 most active GNG cell types one at a time and writes `sweep.parquet` plus one ranked report. A spec holds exactly one `sweep:` kind:

    ```yaml
    sweep:
      silence_each: {select: {cell_type: "GNG*"}, group_by: cell_type, max_items: 10}
    # or: group_by: neuron (one variant per neuron)
    # or, for readout-vs-rate curves:
    # sweep: {rate_hz: [10, 50, 100, 150, 200]}
    ```

    Shared conditions run once; BH correction spans the whole sweep; see the sweep caveats.
- W3 male vs female per cell type: `examples/w3_male_vs_female.py`
- W5 genetic access (exploratory): `examples/w5_driver_lines.py`

- Pathway-ranked silencing screen (relays on routes from the stimulated neurons to a readout): `flyconn run examples/specs/w2_sweep_pathway_mn9.yaml --out runs/screen`
