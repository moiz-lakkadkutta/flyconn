"""W2 sweep acceptance: a silencing screen over GNG types on MaleCNS runs from one YAML."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from flyconn.experiments.spec import load_spec
from flyconn.experiments.sweep import run_sweep

pytestmark = pytest.mark.golden
SPEC = Path(__file__).resolve().parents[2] / "examples" / "specs" / "w2_sweep_silence_gng.yaml"


def test_w2_silencing_screen_ranks_gng_types(tmp_path: Path):
    spec = load_spec(SPEC)
    t0 = time.time()
    res = run_sweep(spec, out_dir=tmp_path / "w2_sweep")
    dt = time.time() - t0
    print(f"\nW2 sweep on MaleCNS in {dt:.0f}s")
    print(res.conditions.to_string(index=False))
    print(res.variants.head(15).to_string(index=False))
    cols = [
        "readout",
        "rank",
        "variant",
        "n_silenced",
        "mean_reference_hz",
        "mean_hz",
        "difference",
        "ci_low",
        "ci_high",
        "cohens_d",
        "q_value",
    ]
    print(res.sweep[cols].to_string(index=False))
    assert dt < 900
    simulated = res.variants[res.variants["simulated"]]
    assert len(simulated) == 10
    assert simulated["variant"].str.startswith("GNG").all()
    assert len(res.sweep) == 10 * 3
    assert (res.sweep["n_trials"] == 10).all()
    # shared conditions ran once: baseline, stimulated, one rewire, one shuffle, then 10 variants
    assert len(res.conditions) == 4 + 10
    stim_mn9 = res.sweep.loc[res.sweep["readout"] == "MN9", "mean_reference_hz"]
    assert (stim_mn9 > 0).all()
    assert res.provenance["network"]["calibration"].startswith("CALIBRATED BY PROTOCOL")
    html = (tmp_path / "w2_sweep" / "report.html").read_text()
    assert "CALIBRATED BY PROTOCOL" in html and "not validated" in html
    assert "model prediction" in html.lower()
    assert "n = 10 trials" in html
