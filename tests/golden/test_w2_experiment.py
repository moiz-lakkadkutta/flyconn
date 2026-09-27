"""W2 acceptance: a YAML experiment on MaleCNS runs end to end and produces a standalone report."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from flyconn.experiments.runner import run_experiment
from flyconn.experiments.spec import load_spec

pytestmark = pytest.mark.golden
SPEC = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "specs"
    / "w2_malecns_lb3_silence_gng232.yaml"
)


def test_w2_yaml_experiment_runs_with_controls_and_report(tmp_path: Path):
    spec = load_spec(SPEC)
    t0 = time.time()
    res = run_experiment(spec, out_dir=tmp_path / "w2")
    dt = time.time() - t0
    print(f"\nW2 MaleCNS experiment in {dt:.0f}s")
    print(res.conditions.to_string(index=False))
    cols = [
        "readout",
        "comparison",
        "mean_test",
        "mean_control",
        "difference",
        "cohens_d",
        "q_value",
    ]
    print(res.readouts[cols].to_string(index=False))
    assert dt < 1500
    assert set(res.conditions["condition"]) == {"baseline", "stimulated", "perturbed"}
    assert (
        res.conditions.query("control == 'none' and condition == 'stimulated'")["total_spikes"] > 0
    ).all()
    assert "UNCALIBRATED" in res.provenance["network"]["calibration"]
    html = (tmp_path / "w2" / "report.html").read_text()
    assert "UNCALIBRATED" in html and "model prediction" in html.lower()
