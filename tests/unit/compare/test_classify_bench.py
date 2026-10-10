"""Hold-out benchmark on the synthetic pair."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flyconn.compare.atlas import build_atlas
from flyconn.compare.classify_bench import (
    fit_calibration,
    run_benchmark,
    truth_labels,
    write_benchmark,
)
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix


def _bench(male_female: tuple[Store, Store]):  # type: ignore[no-untyped-def]
    male, female = male_female
    atlas = build_atlas(female)
    mq = ConnectivityMatrix.from_store(male, min_weight=5)
    return atlas, run_benchmark(mq, atlas, query_ref=male.ref, max_types=50, seed=0)


def test_truth_labels_excludes_comma_lists_and_unknown_types():
    meta = pd.DataFrame({"x": ["A", "A,B", None, "Z", "B"]})
    truth, n_comma = truth_labels(meta, "x", {"A", "B"})
    assert truth.tolist() == ["A", None, None, None, "B"]
    assert n_comma == 1


def test_closed_and_open_records(male_female: tuple[Store, Store]):
    _, res = _bench(male_female)
    rec = res.records
    assert set(rec["protocol"]) == {"closed", "open"}
    assert set(rec["level"]) == {"neuron", "group"}
    assert set(rec["type"]) == {f"T{i}" for i in range(8)}
    closed_g = rec[(rec["protocol"] == "closed") & (rec["level"] == "group")]
    assert len(closed_g) == 8
    # the male T3 perturbation also removes T3 -> T4, so T3 and T4 are rewired; the rest match
    ok = closed_g.set_index("type")["correct"]
    assert ok[["T0", "T1", "T2", "T5", "T6", "T7"]].all()
    open_ = rec[rec["protocol"] == "open"]
    assert not open_["correct"].any()  # the true label was dropped
    assert (open_["top1"] != open_["type"]).all()
    assert (rec.loc[rec["level"] == "group", "neuron_id"] == -1).all()


def test_masking_blanks_the_held_out_type(male_female: tuple[Store, Store]):
    _, res = _bench(male_female)
    rec = res.records
    # every T_i neuron still has partners after masking T_i (ring wiring), so none is empty
    assert not rec.loc[rec["protocol"] == "closed", "empty"].any()


def test_fit_calibration_metrics_and_thresholds(male_female: tuple[Store, Store], tmp_path: Path):
    atlas, res = _bench(male_female)
    # add a second "pair" so transfer ECE is exercised
    res2 = type(res)(
        query="banc@888",
        reference=res.reference,
        records=res.records.assign(query="banc@888"),
        n_types_available=res.n_types_available,
        n_comma_excluded=0,
    )
    cal = fit_calibration([res, res2], atlas, n_folds=2, seed=0, n_boot=50)
    assert cal.fitted_on == ["malecns@1.0", "banc@888"]
    assert cal.variant in {"adjusted", "raw"}
    assert set(cal.models) == {"neuron", "group"}
    assert cal.thresholds["neuron"].s_floor >= 0.0
    m = cal.metrics
    assert set(m["per_query"]) == {"malecns@1.0", "banc@888"}
    top1 = m["per_query"]["malecns@1.0"]["group"]["top1"]
    assert len(top1) == 3 and top1[1] <= top1[0] <= top1[2]
    assert set(m["transfer_ece"]) == {"malecns@1.0", "banc@888"}
    assert set(m["variants"]) == {"adjusted", "raw"}
    assert cal.atlas_params_hash == atlas.params_hash
    out = write_benchmark(tmp_path, [res, res2], cal)
    data = json.loads(out.read_text())
    assert data["calibration_id"] == cal.id
    assert (tmp_path / "records_malecns@1.0.parquet").exists()


def test_benchmark_is_deterministic(male_female: tuple[Store, Store]):
    _, a = _bench(male_female)
    _, b = _bench(male_female)
    pd.testing.assert_frame_equal(a.records, b.records)
    assert np.isfinite(a.records["s1"]).all()
