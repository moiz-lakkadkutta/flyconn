"""plan_variants: one axis at a time around the calibrated reference (ADR-0011)."""

from __future__ import annotations

import pytest

from flyconn.data.store import Store
from flyconn.experiments.robustness import plan_variants, w_syn_values
from flyconn.experiments.spec import derive, spec_from_dict


def _d(
    store: Store, robustness: dict[str, object], dataset: str | None = None
) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "plan",
        "dataset": dataset or store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 3}},
        "robustness": {"primary_readout": "targets", **robustness},
    }


def test_full_plan_labels(small_store: Store):
    spec = spec_from_dict(_d(small_store, {}))
    labels = [v.label for v in plan_variants(spec)]
    assert labels == [
        "reference",
        "wsyn_0.185",
        "wsyn_0.192",
        "wsyn_0.209",
        *[f"nt_s{k}" for k in range(8)],
        "minw_3",
        "minw_5",
        "rep_flywire_783",
    ]


def test_auto_w_syn_from_calibration(small_store: Store):
    assert w_syn_values(spec_from_dict(_d(small_store, {}))) == (0.185, 0.192, 0.209)


def test_axes_can_be_switched_off(small_store: Store):
    spec = spec_from_dict(
        _d(
            small_store,
            {"w_syn": "off", "nt_samples": 0, "min_weights": [], "replicate": None},
        )
    )
    assert [v.axis for v in plan_variants(spec)] == ["reference"]


def test_auto_without_calibration_fails(small_store: Store):
    spec = spec_from_dict(_d(small_store, {"replicate": None}, dataset="flywire@783"))
    with pytest.raises(ValueError, match="w_syn: auto"):
        plan_variants(spec)


def test_min_weight_equal_to_reference_is_skipped(small_store: Store):
    spec = spec_from_dict(_d(small_store, {"min_weights": [1, 3]}))
    assert [v.label for v in plan_variants(spec) if v.axis == "min_weight"] == ["minw_3"]


def test_variant_specs_are_rehashed_and_stripped(small_store: Store):
    spec = spec_from_dict(_d(small_store, {}))
    by = {v.label: v.spec for v in plan_variants(spec)}
    w = by["wsyn_0.209"]
    assert w.params.w_syn_mv == 0.209
    assert w.raw["params"]["w_syn_mv"] == 0.209
    assert w.robustness is None
    assert w.controls.degree_preserving_rewire == 0 and not w.controls.unstimulated_baseline
    assert w.source_sha256 and w.source_sha256 != spec.source_sha256
    assert w.parent_sha256 == spec.source_sha256
    nt = by["nt_s3"]
    assert nt.network.sign_policy == "sampled" and nt.network.sign_seed == spec.seed + 2003
    assert by["minw_5"].network.min_weight == 5
    rep = by["rep_flywire_783"]
    assert rep.dataset == "flywire@783" and rep.params.w_syn_mv is None


def test_derive_is_deterministic(small_store: Store):
    spec = spec_from_dict(_d(small_store, {}))
    a = derive(spec, parent_sha256="p", name="x")
    b = derive(spec, parent_sha256="p", name="x")
    assert a.source_sha256 == b.source_sha256
