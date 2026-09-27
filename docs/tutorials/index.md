# Tutorials

One runnable script per anchor workflow (jupytext percent format; open as a notebook with
`jupytext --to ipynb examples/<name>.py` or run with `python examples/<name>.py` after
`flyconn data pull`):

- W1 circuit tracing with signs and confidence: `examples/w1_pathways.py`
- W4 version drift: `examples/w4_version_drift.py`
- W2 in-silico experiment from one YAML file: `flyconn run examples/specs/w2_malecns_lb3_silence_gng232.yaml --out runs/w2` (controls on by default; the report states that MaleCNS runs are uncalibrated)
- W3, W5: added with M6, M7.
