"""Tests for the experiment runner: conditions, controls by default, results and report."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from flyconn.data.store import Store
from flyconn.experiments.runner import ExperimentResult, run_experiment
from flyconn.experiments.spec import spec_from_dict


def _spec(store: Store, **over: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    d: dict[str, object] = {
        "name": "tiny",
        "dataset": store.ref,
        "network": {"min_weight": 1, "sign_policy": "argmax"},
        "stimulate": [{"select": {"ids": ids[:4]}, "rate_hz": 200}],
        "perturb": {"silence": [{"select": {"ids": ids[4:6]}}]},
        "readouts": [
            {"name": "targets", "select": {"ids": ids[-6:]}},
            {"name": "motor", "select": {"super_class": "motor"}},
        ],
        "duration_ms": 50,
        "trials": 3,
        "seed": 1,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 2, "sign_shuffle": 1},
        "report": {"top_neurons": 5},
    }
    d.update(over)
    return d


def test_runner_produces_conditions_controls_results_and_report(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_spec(small_store))
    res = run_experiment(spec, out_dir=tmp_path / "out", store=small_store)
    assert isinstance(res, ExperimentResult)
    conditions = set(res.conditions["condition"])
    assert conditions >= {"baseline", "stimulated", "perturbed"}
    assert set(res.conditions["control"]) >= {"none", "degree_preserving_rewire", "sign_shuffle"}
    counts = pq.read_table(tmp_path / "out" / "counts.parquet").to_pandas()
    assert set(counts.columns) >= {
        "condition",
        "control",
        "replicate",
        "trial",
        "neuron_id",
        "count",
    }
    assert counts["trial"].nunique() == 3
    assert (tmp_path / "out" / "provenance.json").exists()
    assert (tmp_path / "out" / "report.html").exists()
    assert (tmp_path / "out" / "readouts.parquet").exists()


def test_readout_table_has_effects_against_each_control(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_spec(small_store))
    res = run_experiment(spec, out_dir=tmp_path / "out", store=small_store)
    r = res.readouts
    assert isinstance(r, pd.DataFrame)
    assert set(r.columns) >= {
        "readout",
        "comparison",
        "mean_test",
        "mean_control",
        "difference",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "cohens_d",
    }
    assert set(r["readout"]) == {"targets", "motor"}
    comps = set(r["comparison"])
    assert {
        "perturbed_vs_stimulated",
        "stimulated_vs_baseline",
        "stimulated_vs_degree_preserving_rewire",
        "stimulated_vs_sign_shuffle",
    } <= comps
    assert ((r["q_value"] >= r["p_value"] - 1e-12) & (r["q_value"] <= 1)).all()


def test_report_html_states_model_prediction_controls_and_citations(
    small_store: Store, tmp_path: Path
):
    spec = spec_from_dict(_spec(small_store))
    run_experiment(spec, out_dir=tmp_path / "out", store=small_store)
    html = (tmp_path / "out" / "report.html").read_text()
    assert "model prediction" in html.lower()
    assert "degree_preserving_rewire" in html and "sign_shuffle" in html
    assert "10.1016/j.cell.2026.08.015" in html  # citation from the registry
    assert "sha256" in html.lower() or "provenance" in html.lower()
    assert "<img" in html or "<svg" in html


def test_runner_without_perturbation_still_runs(small_store: Store, tmp_path: Path):
    d = _spec(small_store)
    d.pop("perturb")
    res = run_experiment(spec_from_dict(d), out_dir=tmp_path / "out", store=small_store)
    assert "perturbed" not in set(res.conditions["condition"])
    assert not res.readouts.empty


def test_runner_provenance_carries_spec_hash_seeds_and_versions(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_spec(small_store))
    res = run_experiment(spec, out_dir=tmp_path / "out", store=small_store)
    prov = res.provenance
    assert prov["spec"]["name"] == "tiny"
    assert prov["seed"] == 1 and prov["trials"] == 3
    assert "flyconn_version" in prov and "torch_version" in prov
    assert prov["dataset_provenance"]["dataset"] == "malecns"
    assert prov["label"] == "model prediction"
    with pytest.raises(FileExistsError):
        run_experiment(spec, out_dir=tmp_path / "out", store=small_store)
