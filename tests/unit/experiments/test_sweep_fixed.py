"""run_sweep with a fixed candidate table, extra (control) candidates and natural selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyconn.data.store import Store
from flyconn.experiments.spec import spec_from_dict
from flyconn.experiments.sweep import Candidate, run_sweep


def _d(store: Store, **over: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    d: dict[str, object] = {
        "name": "fx",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 2}},
        "duration_ms": 60,
        "trials": 3,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
    }
    d.update(over)
    return d


def test_natural_selection_recorded(small_store: Store, tmp_path: Path):
    res = run_sweep(spec_from_dict(_d(small_store)), out_dir=tmp_path / "a", store=small_store)
    assert len(res.natural_selection) == 2
    assert [c.label for c in res.candidates] == res.natural_selection
    assert set(res.sweep["role"]) == {"candidate"}


def test_fixed_candidates_are_simulated_as_given(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    natural = run_sweep(spec, out_dir=tmp_path / "a", store=small_store).natural_selection
    ids = small_store.neurons()["neuron_id"].tolist()
    fixed = [Candidate("mine", (ids[20], ids[21]))]
    res = run_sweep(spec, out_dir=tmp_path / "b", store=small_store, fixed=fixed)
    assert set(res.sweep["variant"]) == {"mine"}
    assert res.natural_selection == natural
    res2 = run_sweep(
        spec, out_dir=tmp_path / "c", store=small_store, fixed=fixed, compute_natural=False
    )
    assert res2.natural_selection == []


def test_extra_candidates_are_controls(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    natural = run_sweep(spec, out_dir=tmp_path / "a", store=small_store).natural_selection
    types = sorted(set(small_store.neurons()["cell_type"]) - set(natural))
    spec2 = spec_from_dict(
        _d(
            small_store,
            robustness={
                "primary_readout": "targets",
                "extra_candidates": [{"cell_type": types[0]}],
            },
        )
    )
    res = run_sweep(spec2, out_dir=tmp_path / "b", store=small_store)
    roles = dict(zip(res.sweep["variant"], res.sweep["role"], strict=True))
    assert roles[types[0]] == "control"
    assert all(roles[n] == "candidate" for n in natural)


def test_extra_candidate_colliding_with_ranked_one_raises(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store))
    natural = run_sweep(spec, out_dir=tmp_path / "a", store=small_store).natural_selection
    spec2 = spec_from_dict(
        _d(
            small_store,
            robustness={
                "primary_readout": "targets",
                "extra_candidates": [{"cell_type": natural[0]}],
            },
        )
    )
    with pytest.raises(ValueError, match="already a ranked candidate"):
        run_sweep(spec2, out_dir=tmp_path / "b", store=small_store)


def test_fixed_not_allowed_for_rate_sweeps(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store, sweep={"rate_hz": [10, 50]}))
    with pytest.raises(ValueError, match="silence_each"):
        run_sweep(spec, out_dir=tmp_path / "a", store=small_store, fixed=[Candidate("x", (1,))])
