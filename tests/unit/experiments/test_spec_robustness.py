"""Parsing and validation of the robustness: block (M13, ADR-0011)."""

from __future__ import annotations

import pytest

from flyconn.data.store import Store
from flyconn.experiments.spec import RobustnessOptions, spec_from_dict


def _d(store: Store, robustness: object, sweep: object | None = None) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "r",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": sweep
        if sweep is not None
        else {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 3}},
        "robustness": robustness,
    }


def test_defaults(small_store: Store):
    spec = spec_from_dict(_d(small_store, {"primary_readout": "targets"}))
    r = spec.robustness
    assert isinstance(r, RobustnessOptions)
    assert r.w_syn == "auto" and r.nt_samples == 8 and r.min_weights == (3, 5)
    assert r.replicate == "flywire@783" and r.random_null_sets == 40
    assert r.extra_candidates == () and r.readout_proxy is None


def test_full_block(small_store: Store):
    spec = spec_from_dict(
        _d(
            small_store,
            {
                "primary_readout": "targets",
                "readout_proxy": "X drives Y (Doe 2020)",
                "w_syn": [0.2, 0.3],
                "nt_samples": 0,
                "min_weights": [],
                "replicate": None,
                "random_null": {"n_sets": 0},
                "extra_candidates": [{"cell_type": "T001"}],
            },
        )
    )
    r = spec.robustness
    assert r is not None
    assert r.w_syn == (0.2, 0.3) and r.nt_samples == 0 and r.min_weights == ()
    assert r.replicate is None and r.random_null_sets == 0
    assert r.extra_candidates[0].cell_type == "T001"


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({}, "primary_readout"),
        ({"primary_readout": "nope"}, "primary_readout"),
        ({"primary_readout": "targets", "bogus": 1}, "unknown robustness"),
        ({"primary_readout": "targets", "w_syn": "maybe"}, "w_syn"),
        ({"primary_readout": "targets", "w_syn": [0]}, "w_syn"),
        ({"primary_readout": "targets", "nt_samples": -1}, "nt_samples"),
        ({"primary_readout": "targets", "min_weights": [0]}, "min_weights"),
        ({"primary_readout": "targets", "min_weights": [3, 3]}, "min_weights"),
        ({"primary_readout": "targets", "random_null": {"n_sets": -1}}, "n_sets"),
        ({"primary_readout": "targets", "replicate": "malecns@1.0"}, "replicate"),
        ({"primary_readout": "targets", "extra_candidates": [3]}, "extra_candidates"),
    ],
)
def test_validation(small_store: Store, block: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        spec_from_dict(_d(small_store, block))


def test_needs_silence_each(small_store: Store):
    with pytest.raises(ValueError, match="silence_each"):
        spec_from_dict(_d(small_store, {"primary_readout": "targets"}, sweep={"rate_hz": [10]}))
