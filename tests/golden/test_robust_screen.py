"""Golden: robustness-audited pathway screen on MaleCNS (ADR-0011, ADR-0012; GOLDEN_RESULTS 6k)."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import load_spec

pytestmark = pytest.mark.golden
ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "examples" / "specs" / "w2_robust_pathway_mn9.yaml"
# GOLDEN_RESULTS 6h pathway screen at 0.188 mV: relay -> (difference, ci_low, ci_high)
SIX_H = {
    "GNG232": (-14.5, -18.8, -9.5),
    "DNge080": (-4.3, -8.7, 0.3),
    "DNge051": (2.5, -2.3, 7.7),
    "GNG215": (-1.7, -6.7, 3.7),
    "GNG132": (1.7, -3.8, 7.3),
    "GNG130": (0.2, -5.5, 6.0),
}


def test_robust_screen_malecns(tmp_path: Path):
    spec = load_spec(SPEC)
    t0 = time.time()
    res = run_robust_screen(spec, out_dir=tmp_path / "robust")
    wall = time.time() - t0
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = maxrss / 1e9 if sys.platform == "darwin" else maxrss * 1024 / 1e9  # bytes vs KiB
    out = res.out_dir
    print(f"\nM14 robust screen: {wall:.0f} s, peak RSS {rss_gb:.1f} GB")
    print(res.hits.to_string(index=False))
    print(pd.read_csv(out / "handoff.csv").to_string(index=False))
    print(res.selection_stability.to_string(index=False))

    controls = pd.read_parquet(out / "controls.parquet").set_index("candidate")
    print(controls.to_string())
    for f in ("report.html", "handoff.csv", "hits.parquet", "null.parquet", "provenance.json"):
        assert (out / f).exists()
    ref = res.effects[res.effects["axis"] == "reference"].set_index("candidate")
    for relay, (_d, lo, hi) in SIX_H.items():
        assert lo <= float(ref.loc[relay, "difference"]) <= hi, relay
    # M14 structure (ADR-0012): paired statistics, active null pool, store fingerprints
    rob = res.provenance["robustness"]
    assert res.provenance["statistics"].startswith("paired")
    assert rob["random_null"]["pool"] == "active"
    assert set(rob["store_fingerprints"]) == {"malecns@1.0", "flywire@783"}
    # Controls: LC4 is silent under LB3 drive (6j), so uninformative; the automatic
    # off-route control fires and is not a hit.
    assert set(controls.index) == {"LC4", "MN11D"}  # pinned from the 2026-10-24 run (6k)
    assert not bool(controls.loc["LC4", "informative"])
    assert controls.loc["MN11D", "source"] == "auto" and bool(controls.loc["MN11D", "informative"])
    assert controls.loc["MN11D", "ci_low"] <= 0 <= controls.loc["MN11D", "ci_high"]
    assert float(controls.loc["MN11D", "null_p"]) >= 0.05  # 6k: 0.073
    # Pinned from GOLDEN_RESULTS 6k: one hit, Conditional on NT signs; the activity-matched
    # null is informative (central-brain sets -8.8 to +5.5 Hz) and GNG232 beats it.
    hits = res.hits.set_index("candidate")
    assert list(hits.index) == ["GNG232"]
    assert hits.loc["GNG232", "tier"] == "Conditional"
    assert hits.loc["GNG232", "failed_axes"] == "nt"
    assert hits.loc["GNG232", "random_null"] == "robust"
    assert hits.loc["GNG232", "uninformative_axes"] == ""
    assert hits.loc["GNG232", "random_null_p"] < 0.05
    assert rss_gb < 16
    (ROOT / "benchmarks" / "robust_screen_malecns.json").write_text(
        json.dumps(
            {
                "spec": SPEC.name,
                "wall_s": round(wall),
                "peak_rss_gb": round(rss_gb, 2),
                "variants": len(res.provenance["robustness"]["variants"]),
                "null_runs": len(res.null),
                "machine": "M4 Pro, CPU float32",
            },
            indent=1,
        )
    )
