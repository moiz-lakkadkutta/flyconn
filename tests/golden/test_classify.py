"""Zero-shot classification on real data (ADR-0010).

1. Hold-out benchmark: MaleCNS v1.0 and BANC v888 against FlyWire v783 (truth: the query's
   published fafb_783_cell_type, single names), closed and open set, type-level folds;
   fits and checks the shipped pooled calibration.
2. Secondary: BANC v888 against MANC v1.2.1 (nerve cord; reported only).
3. LB3a-d against FlyWire's sugar and water GRN sets (Shiu ids), the case that motivated
   this milestone (GOLDEN_RESULTS 6g).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from flyconn.compare import build_atlas, classify
from flyconn.compare.calibrate import Calibration, calibration_path
from flyconn.compare.classify_bench import fit_calibration, run_benchmark, write_benchmark
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix

pytestmark = pytest.mark.golden
OUT = Path("benchmarks") / "classify_flywire_783.json"
SETS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json").read_text()
)["sets"]
MAX_TYPES = 1500
# Measured top-1 (run 2026-10-10, 1,500 types per query; benchmarks/classify_flywire_783.json)
MEASURED_TOP1 = {
    "malecns@1.0": {"group": 0.909, "neuron": 0.813},
    "banc@888": {"group": 0.657, "neuron": 0.423},
}
# Atlas cosines (mean of per-neuron normalised profiles; the selection's own name "LB3" is
# blanked as a partner by default, as in the benchmark; run 2026-10-10). GOLDEN_RESULTS 6g
# pooled synapses without masking (sugar 0.44 / 0.65 / 0.92 / 0.86 for LB3a-d);
# test_lb3_pooled_synapses_reproduce_6g shows pooling on the same vocabulary reproduces 6g,
# so the gap is the averaging and the masking, not the vocabulary or the code.
LB3_ATLAS_SUGAR = {"LB3a": 0.367, "LB3b": 0.429, "LB3c": 0.776, "LB3d": 0.759}


def _update(key: str, value: object) -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")


def test_holdout_benchmark_against_flywire(tmp_path: Path):
    t0 = time.time()
    atlas = build_atlas(Store.open("flywire@783"))
    results = []
    for q in ("malecns@1.0", "banc@888"):
        store = Store.open(q)
        m = ConnectivityMatrix.from_store(store, min_weight=5)
        results.append(run_benchmark(m, atlas, query_ref=store.ref, max_types=MAX_TYPES, seed=0))
    cal = fit_calibration(results, atlas, n_folds=5, seed=0)
    write_benchmark(tmp_path, results, cal)
    elapsed = time.time() - t0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    metrics["elapsed_s"] = round(elapsed, 1)
    _update("flywire@783", metrics)

    target = calibration_path("flywire@783")
    if os.environ.get("FLYCONN_WRITE_CALIBRATION") == "1":
        cal.to_json(target)
    shipped = Calibration.from_json(target)
    assert set(shipped.models) == set(cal.models) == {"group"}  # per-neuron not calibrated
    assert np.allclose(shipped.models["group"].coef, cal.models["group"].coef, atol=1e-6)
    assert shipped.atlas_params_hash == atlas.params_hash

    m = cal.metrics
    group = m["variants"][cal.variant]["group"]
    assert group["ece_oof"] <= 0.05  # spec success criterion (pooled, out of fold)
    assert group["false_accept_open"] <= 0.10
    for q, levels in MEASURED_TOP1.items():
        for level, measured in levels.items():
            assert m["per_query"][q][level]["top1"][0] >= measured - 0.03
    assert elapsed < 20 * 60


def test_banc_against_manc_nerve_cord():
    atlas = build_atlas(Store.open("manc@1.2.1"))
    store = Store.open("banc@888")
    m = ConnectivityMatrix.from_store(store, min_weight=5)
    res = run_benchmark(
        m,
        atlas,
        query_ref=store.ref,
        truth_column="manc_121_cell_type",
        max_types=MAX_TYPES,
        seed=0,
    )
    cal = fit_calibration([res], atlas, n_folds=5, seed=0)
    _update(
        "manc@1.2.1",
        {
            "per_query": cal.metrics["per_query"],
            "variant": cal.variant,
            "variants": cal.metrics["variants"],
        },
    )
    assert cal.metrics["per_query"]["banc@888"]["group"]["n_types"] > 100


def test_lb3_sugar_water_against_shiu_sets():
    groups = {"sugar_grn": SETS["sugar_grn"], "water_grn": SETS["water_grn"]}
    atlas = build_atlas(
        Store.open("flywire@783"), groups=groups, direction="out", on_missing="drop"
    )
    assert atlas.provenance["group_ids_dropped"] == 1  # 20/21 sugar, 18/18 water in v783
    malecns = Store.open("malecns@1.0")
    found: dict[str, dict[str, object]] = {}
    for sub in ("LB3a", "LB3b", "LB3c", "LB3d"):
        res = classify(malecns, atlas, cell_type=sub, calibration=None, top_k=2)
        assert res.group is not None
        s = dict(zip(res.group.top_labels, res.group.top_s, strict=True))
        found[sub] = {
            "sugar": round(s["sugar_grn"], 3),
            "water": round(s["water_grn"], 3),
            "agreement": round(res.group.agreement, 3),
            "votes": res.group.votes,
        }
    _update("lb3_sugar_water", found)
    assert found["LB3c"]["sugar"] > found["LB3c"]["water"]
    assert found["LB3d"]["sugar"] > found["LB3d"]["water"]
    assert found["LB3a"]["water"] > found["LB3a"]["sugar"]
    margins = {k: abs(v["sugar"] - v["water"]) for k, v in found.items()}  # type: ignore[operator]
    assert min(margins, key=margins.__getitem__) == "LB3b"
    for sub, ref in LB3_ATLAS_SUGAR.items():
        assert abs(found[sub]["sugar"] - ref) <= 0.01, (sub, found[sub], ref)  # type: ignore[operator]
    # per-neuron votes as in 6g: LB3a 0/17, LB3b 8/11, LB3c 17/23 sugar (LB3d 23/26 vs 24/26)
    assert found["LB3b"]["votes"] == {"sugar_grn": 8, "water_grn": 3}


def test_lb3_pooled_synapses_reproduce_6g():
    """6g pooled synapses per set; on the atlas vocabulary that reproduces its cosines.

    This isolates why the atlas cosines above differ from 6g: averaging per-neuron
    normalised profiles (atlas) instead of pooling synapses (6g), not the vocabulary.
    """
    from flyconn.compare.profiles import partner_counts
    from flyconn.compare.types import partner_labels

    six_g = {"LB3a": (0.44, 0.87), "LB3b": (0.65, 0.49), "LB3c": (0.92, 0.62), "LB3d": (0.86, 0.55)}
    fw = ConnectivityMatrix.from_store(Store.open("flywire@783"), min_weight=5)
    mc = ConnectivityMatrix.from_store(Store.open("malecns@1.0"), min_weight=5)
    fw_names = partner_labels(fw.meta, "fafb_or_manc").tolist()
    vocab = sorted({v for v in fw_names if isinstance(v, str)})
    cf = partner_counts(fw, fw_names, vocab=vocab)
    cm = partner_counts(mc, partner_labels(mc.meta, "fafb_or_manc").tolist(), vocab=vocab)
    present = set(fw.neuron_ids.tolist())

    def pooled(counts, idx):  # type: ignore[no-untyped-def]
        return np.asarray(counts.out[idx].sum(axis=0), dtype=float).ravel()

    ref = {
        k: pooled(cf, fw.index_of(np.array([i for i in SETS[k] if i in present])))
        for k in ("sugar_grn", "water_grn")
    }
    measured = {}
    for sub, (sugar, water) in six_g.items():
        q = pooled(cm, np.flatnonzero(mc.meta["cell_type"].to_numpy() == sub))
        cos = {k: float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v))) for k, v in ref.items()}
        measured[sub] = (round(cos["sugar_grn"], 3), round(cos["water_grn"], 3))
        assert abs(cos["sugar_grn"] - sugar) <= 0.006, (sub, measured[sub])
        assert abs(cos["water_grn"] - water) <= 0.006, (sub, measured[sub])
    _update("lb3_pooled_6g_method", measured)
