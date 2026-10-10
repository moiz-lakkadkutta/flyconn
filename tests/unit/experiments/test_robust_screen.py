"""End-to-end robustness screen on the synthetic store (structure, resume, determinism)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store
from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import spec_from_dict
from flyconn.testing import synthetic_connectome


def _d(store: Store, **rob: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "rs",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 2}},
        "duration_ms": 60,
        "trials": 4,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        # strong enough that the readout fires in 60 ms, so reference effects are nonzero
        "params": {"w_syn_mv": 1.0},
        "controls": {"degree_preserving_rewire": 1, "sign_shuffle": 0},
        "report": {
            "alpha": 2.0
        },  # q <= 1 < 2: every nonzero candidate is a hit, to exercise scoring
        "robustness": {
            "primary_readout": "targets",
            "w_syn": [0.3],
            "nt_samples": 2,
            "min_weights": [2],
            "replicate": None,
            "random_null": {"n_sets": 3},
            **rob,
        },
    }


@pytest.fixture
def rep_store(tmp_path: Path) -> Store:
    neurons, edges = synthetic_connectome(n_neurons=40, n_edges=200, seed=22)
    neurons["dataset"] = edges["dataset"] = "flywire"
    neurons["version"] = edges["version"] = "783"
    neurons["input_synapses_total"] = (
        edges.groupby("post")["weight"].sum().reindex(neurons["neuron_id"]).fillna(0).to_numpy() + 5
    )
    d = tmp_path / "rep"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(
        d,
        dataset="flywire",
        version="783",
        converter="test",
        inputs=[],
        counts={"neurons": 40, "edges": 200},
    )
    return Store(d)


def test_outputs_and_axes(small_store: Store, tmp_path: Path):
    res = run_robust_screen(
        spec_from_dict(_d(small_store)), out_dir=tmp_path / "o", store=small_store
    )
    out = tmp_path / "o"
    for f in (
        "robustness_plan.json",
        "hits.parquet",
        "null.parquet",
        "selection_stability.parquet",
        "provenance.json",
    ):
        assert (out / f).exists(), f
    assert set(res.effects["axis"]) == {"reference", "w_syn", "nt", "min_weight"}
    assert {"label"} <= set(res.effects.columns)
    ref = res.effects[(res.effects["axis"] == "reference") & (res.effects["role"] == "candidate")]
    n_nonzero = int((ref["difference"] != 0).sum())
    assert n_nonzero >= 1
    assert len(res.hits) == n_nonzero  # alpha 2.0: every nonzero candidate is a hit
    assert {"tier", "path_stability"} <= set(res.hits.columns)
    assert set(res.selection_stability["variant"]) == {"wsyn_0.3", "nt_s0", "nt_s1", "minw_2"}
    assert len(res.null) > 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["robustness"]["control_null_p"] == {}  # no extra candidates in this spec
    assert prov["robustness"]["w_syn_values"] == [0.3]
    for v in ("wsyn_0.3", "nt_s0", "minw_2", "reference", "null"):
        assert (out / "variants" / v / "done.json").exists()


def test_deterministic(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    a = run_robust_screen(spec, out_dir=tmp_path / "a", store=small_store)
    b = run_robust_screen(spec, out_dir=tmp_path / "b", store=small_store)
    pd.testing.assert_frame_equal(a.hits, b.hits)
    pd.testing.assert_frame_equal(a.effects, b.effects)


def test_resume_moves_incomplete_and_skips_done(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    out = tmp_path / "o"
    first = run_robust_screen(spec, out_dir=out, store=small_store)
    (out / "variants" / "nt_s1" / "done.json").unlink()  # simulate a crash inside nt_s1
    shutil.rmtree(out / "variants" / "minw_2")  # and a variant that never started
    done_mtime = (out / "variants" / "wsyn_0.3" / "done.json").stat().st_mtime_ns
    again = run_robust_screen(spec, out_dir=out, store=small_store)
    assert (out / "variants" / ".incomplete" / "nt_s1-1").is_dir()
    assert (out / "variants" / "wsyn_0.3" / "done.json").stat().st_mtime_ns == done_mtime
    pd.testing.assert_frame_equal(first.hits, again.hits)


def test_changed_plan_refuses(small_store: Store, tmp_path: Path):
    out = tmp_path / "o"
    run_robust_screen(spec_from_dict(_d(small_store)), out_dir=out, store=small_store)
    with pytest.raises(FileExistsError, match="plan"):
        run_robust_screen(
            spec_from_dict(_d(small_store, nt_samples=3)), out_dir=out, store=small_store
        )


@pytest.fixture
def xref_store(small_store: Store, tmp_path: Path) -> Store:
    """small_store whose fafb_783_cell_type cross-reference equals its own cell_type."""
    neurons, edges = small_store.neurons(), small_store.edges()
    neurons["fafb_783_cell_type"] = neurons["cell_type"]
    d = tmp_path / "xref"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(
        d,
        dataset="malecns",
        version="1.0",
        converter="test",
        inputs=[],
        counts={"neurons": len(neurons), "edges": len(edges)},
    )
    return Store(d)


def test_replicate_axis(xref_store: Store, rep_store: Store, tmp_path: Path):
    ids = xref_store.neurons()["neuron_id"].tolist()
    d = _d(xref_store, replicate="flywire@783")
    # readouts and stimulus by cell type so they can be mapped
    types = xref_store.neurons().set_index("neuron_id")["cell_type"]
    d["stimulate"] = [{"select": {"cell_type": str(types[ids[0]])}, "rate_hz": 200}]
    d["readouts"] = [{"name": "targets", "select": {"cell_type": str(types[ids[-1]])}}]
    out = tmp_path / "o"
    res = run_robust_screen(
        spec_from_dict(d), out_dir=out, store=xref_store, replicate_store=rep_store
    )
    rep_rows = res.effects[res.effects["axis"] == "replicate"]
    assert set(rep_rows["variant"]) == {"rep_flywire_783"}
    assert (out / "variants" / "rep_flywire_783" / "done.json").exists()
    rep_prov = res.provenance["robustness"]["replicate"]
    assert {"dataset", "type_map_rows", "stimulus", "readouts", "candidates", "reason"} <= set(
        rep_prov
    )
    assert rep_prov["type_map_rows"] > 0
    assert rep_prov["reason"] == ""
    assert rep_prov["stimulus"][0]["n_replicate_ids"] > 0
    assert rep_prov["readouts"]["targets"]["n_replicate_ids"] > 0
    mapped = set(rep_rows["candidate"])
    assert len(res.hits) >= 1
    for cand, label in zip(res.hits["candidate"], res.hits["replicate"], strict=True):
        entry = rep_prov["candidates"][cand]
        if cand in mapped:
            assert label in {"holds", "weakens", "flips"}
            assert entry["replicate_types"] and entry["n_replicate_ids"] > 0
        else:
            assert label.startswith("not testable"), label
            assert entry["reason"].startswith("not testable")


def test_replicate_unmapped_is_not_testable(small_store: Store, rep_store: Store, tmp_path: Path):
    # small_store's fafb_783_cell_type is all null, so no type maps to the replicate
    res = run_robust_screen(
        spec_from_dict(_d(small_store, replicate="flywire@783")),
        out_dir=tmp_path / "o",
        store=small_store,
        replicate_store=rep_store,
    )
    assert "replicate" not in set(res.effects["axis"])
    assert all(str(x).startswith("not testable") for x in res.hits["replicate"])
    assert res.provenance["robustness"]["replicate"]["reason"].startswith("not testable")


def test_control_candidate_gets_null_p(small_store: Store, tmp_path: Path):
    res = run_robust_screen(
        spec_from_dict(_d(small_store, extra_candidates=[{"cell_type": "T007"}])),
        out_dir=tmp_path / "o",
        store=small_store,
    )
    ref = res.effects[res.effects["axis"] == "reference"].set_index("candidate")
    assert ref.loc["T007", "role"] == "control"
    assert "T007" not in set(res.hits["candidate"])
    rob = res.provenance["robustness"]
    assert "T007" in rob["control_null_p"]
    p = rob["control_null_p"]["T007"]
    if p is None:
        assert rob["random_null"]["skipped"], "a None null p must come with a recorded skip"
    else:
        assert isinstance(p, float) and 0 < p <= 1
