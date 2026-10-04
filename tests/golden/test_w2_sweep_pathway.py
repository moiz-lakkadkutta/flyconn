"""Pathway-ranked silencing screen on MaleCNS: relays on LB3 -> MN9 routes, ranked by path strength."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from flyconn.experiments.spec import load_spec
from flyconn.experiments.sweep import run_sweep

pytestmark = pytest.mark.golden
SPEC = Path(__file__).resolve().parents[2] / "examples" / "specs" / "w2_sweep_pathway_mn9.yaml"


def test_pathway_screen_ranks_w1_relays_and_runs(tmp_path: Path):
    t0 = time.time()
    res = run_sweep(load_spec(SPEC), out_dir=tmp_path / "sweep")
    dt = time.time() - t0
    v = res.variants
    print(
        f"\npathway screen in {dt:.0f}s; {len(v)} candidate types, {int(v['simulated'].sum())} simulated"
    )
    print(
        v.head(12)[
            [
                "variant",
                "n_neurons",
                "n_paths",
                "path_strength",
                "signed_path_strength",
                "stimulated_rate_hz",
                "simulated",
            ]
        ].to_string(index=False)
    )
    mn9 = res.sweep[res.sweep["readout"] == "MN9"]
    print(
        mn9[
            [
                "variant",
                "mean_reference_hz",
                "mean_hz",
                "difference",
                "ci_low",
                "ci_high",
                "q_value",
            ]
        ].to_string(index=False)
    )
    sim = set(v.loc[v["simulated"], "variant"])
    # W1 found LB3c -> GNG232 -> DNge080 -> MN9 and LB3c -> GNG132 -> GNG130 -> MN9 among the strongest paths
    assert {"GNG232", "DNge080"} <= sim or {"GNG132", "GNG130"} <= sim
    assert "MN9" not in set(v["variant"])
    assert (v.loc[v["simulated"], "path_strength"] > 0).all()
    assert "pathway" in res.provenance["sweep"]["ranked_by"]
