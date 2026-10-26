"""Golden: sugar-only (LB3c + LB3d) robustness screen on MaleCNS (ADR-0012; GOLDEN_RESULTS 6k).

Same screen as test_robust_screen.py, stimulating only the GRN types that match the
MaleCNS calibration stimulus (ADR-0009). Runs in its own robust-sugar CI shard to
keep every shard under its time limit.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from flyconn.data.pull import pull
from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import load_spec

pytestmark = pytest.mark.golden
ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "examples" / "specs" / "w2_robust_pathway_mn9_sugar.yaml"


def test_robust_screen_malecns_sugar(tmp_path: Path):
    # The nt axis needs per-body NT probabilities; without them it is skipped, not run.
    pull("malecns@1.0", level="nt-probs")
    spec = load_spec(SPEC)
    t0 = time.time()
    res = run_robust_screen(spec, out_dir=tmp_path / "robust_sugar")
    wall = time.time() - t0
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = maxrss / 1e9 if sys.platform == "darwin" else maxrss * 1024 / 1e9  # bytes vs KiB
    out = res.out_dir
    print(f"\nM14 sugar-only robust screen: {wall:.0f} s, peak RSS {rss_gb:.1f} GB")
    print(res.hits.to_string(index=False))
    controls = pd.read_parquet(out / "controls.parquet").set_index("candidate")
    print(controls.to_string())

    rob = res.provenance["robustness"]
    assert res.provenance["statistics"].startswith("paired")
    assert rob["random_null"]["pool"] == "active"
    # Pinned from the 2026-10-24 run (GOLDEN_RESULTS 6k): two hits, both Conditional on
    # NT signs (GNG232 holds 5/8, DNge080 2/8); both beat an informative null.
    hits = res.hits.set_index("candidate")
    assert list(hits.index) == ["GNG232", "DNge080"]
    for h in ("GNG232", "DNge080"):
        assert hits.loc[h, "tier"] == "Conditional", h
        assert hits.loc[h, "failed_axes"] == "nt", h
        assert hits.loc[h, "random_null"] == "robust", h
        assert hits.loc[h, "direction"] == "down", h
    assert -24.0 <= hits.loc["GNG232", "difference"] <= -17.0  # 6k: -20.7
    assert -18.0 <= hits.loc["DNge080", "difference"] <= -10.0  # 6k: -14.2
    # The automatic off-route control fires (278.7 Hz summed) and does nothing.
    assert list(controls.index) == ["Tr flexor MN"]
    ctl = controls.loc["Tr flexor MN"]
    assert ctl["source"] == "auto" and bool(ctl["informative"])
    assert ctl["ci_low"] <= 0 <= ctl["ci_high"] and float(ctl["null_p"]) >= 0.05  # 6k: 0.63
    # Known limit (6k): match_types maps LB3c/d to the same 131 FlyWire GRNs as LB3*, so
    # the replicate is not sugar-only. Pinned so a better mapping shows up here.
    assert rob["replicate"]["stimulus"][0]["n_replicate_ids"] == 131
    assert rss_gb < 16
