"""classify(): per-neuron and group calls, calibration modes, caveats, errors."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconn.compare import ClassifyResult, build_atlas, classify
from flyconn.compare.calibrate import Calibration, LogisticModel, Thresholds
from flyconn.data.store import Store


def _cal_for(atlas, accept: float = 0.5) -> Calibration:  # type: ignore[no-untyped-def]
    m = LogisticModel(features=("s1", "margin"), coef=np.array([-4.0, 4.0, 20.0]))
    thr = Thresholds(accept=accept, s_floor=0.2, delta=0.02)
    return Calibration(
        reference=atlas.reference,
        atlas_params_hash=atlas.params_hash,
        atlas_params=atlas.params,
        variant="raw",
        models={"neuron": m, "group": m},
        thresholds={"neuron": thr, "group": thr},
        fitted_on=["banc@888"],
        metrics={"transfer_ece": {"banc@888": {"neuron": 0.04, "group": 0.06}}},
    )


def test_unperturbed_type_is_classified_correctly(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=_cal_for(atlas))
    assert isinstance(res, ClassifyResult)
    assert res.calibrated
    assert len(res.per_neuron) == 6
    assert (res.per_neuron["label"] == "T1").all()
    assert (res.per_neuron["call"] == "type").all()
    assert res.group is not None
    assert res.group.label == "T1" and res.group.call == "type"
    assert res.group.agreement == 1.0
    assert res.group.top_labels[0] == "T1"
    assert set(res.per_neuron.columns) >= {
        "neuron_id",
        "call",
        "label",
        "p",
        "s1",
        "a1",
        "margin",
        "top_labels",
        "top_s",
        "unlabelled_partner_fraction",
        "reason",
    }
    assert res.provenance["atlas_params_hash"] == atlas.params_hash
    assert res.provenance["calibration_id"] is not None
    assert "git_sha" in res.provenance["environment"]


def test_uncalibrated_mode_without_calibration(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=None)
    assert not res.calibrated
    assert (res.per_neuron["call"] == "uncalibrated").all()
    assert res.per_neuron["p"].isna().all()
    assert any(c.startswith("UNCALIBRATED") for c in res.caveats)


def test_default_calibration_missing_falls_back(
    male_female: tuple[Store, Store], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import flyconn.compare.calibrate as cal_mod

    monkeypatch.setattr(cal_mod, "CALIBRATION_DIR", tmp_path)
    male, female = male_female
    res = classify(male, build_atlas(female), cell_type="T1")
    assert not res.calibrated
    assert any("no calibration file" in c for c in res.caveats)


def test_mismatched_calibration_falls_back_with_reason(male_female: tuple[Store, Store]):
    male, female = male_female
    other = build_atlas(female, direction="out")
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=_cal_for(other))
    assert not res.calibrated
    assert any("direction" in c for c in res.caveats)


def test_calibration_caveat_quotes_transfer_ece(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=_cal_for(atlas))
    assert any("leave-one-dataset-out" in c and "0.04" in c for c in res.caveats)


def test_neuron_ids_mode_deduplicates_and_single_neuron_has_no_group(
    male_female: tuple[Store, Store],
):
    male, female = male_female
    ids = male.neurons(columns=["neuron_id", "cell_type"])
    one = int(ids.loc[ids["cell_type"] == "T2", "neuron_id"].iloc[0])
    res = classify(male, build_atlas(female), neuron_ids=[one, one], calibration=None)
    assert len(res.per_neuron) == 1
    assert res.group is None


def test_neuron_without_labelled_partners_is_unknown(male_female: tuple[Store, Store]):
    male, female = male_female
    # 901/902 (T8) have no FlyWire name and only receive input from T0; with T0 masked
    # their profiles are empty
    res = classify(male, build_atlas(female), neuron_ids=[901, 902], mask=("T0",), calibration=None)
    assert (res.per_neuron["call"] == "unknown").all()
    assert (res.per_neuron["reason"] == "no_labelled_partners").all()
    assert (res.per_neuron["label"] == "").all()
    assert res.group is not None and res.group.call == "unknown"


def test_errors(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    with pytest.raises(ValueError, match="exactly one"):
        classify(male, atlas)
    with pytest.raises(KeyError, match="no neurons of type"):
        classify(male, atlas, cell_type="NOPE")
    with pytest.raises(KeyError):
        classify(male, atlas, neuron_ids=[123456789])


def test_vocabulary_mismatch_raises(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    names = pd.Series(["zz"] * 50, index=male.neurons(columns=["neuron_id"])["neuron_id"])
    with pytest.raises(ValueError, match="vocabulary"):
        classify(male, atlas, cell_type="T1", partner_names=names)


def test_partner_names_override_is_used(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    names = pd.Series(meta["cell_type"].to_numpy(), index=meta["neuron_id"].to_numpy())
    res = classify(male, atlas, cell_type="T1", partner_names=names, calibration=None)
    assert res.provenance["partner_names"] == "override"
    assert (res.per_neuron["label"] == "T1").all()


def test_heterogeneous_group_is_flagged(male_female: tuple[Store, Store]):
    male, female = male_female
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    t1 = meta.loc[meta["cell_type"] == "T1", "neuron_id"].tolist()[:3]
    t5 = meta.loc[meta["cell_type"] == "T5", "neuron_id"].tolist()[:3]
    res = classify(male, build_atlas(female), neuron_ids=t1 + t5, calibration=None)
    assert res.group is not None
    assert res.group.agreement == pytest.approx(0.5)
    assert any("heterogeneous" in c for c in res.caveats)


def test_result_independent_of_neuron_order(male_female: tuple[Store, Store]):
    male, female = male_female
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    ids = meta.loc[meta["cell_type"] == "T4", "neuron_id"].tolist()
    atlas = build_atlas(female)
    a = classify(male, atlas, neuron_ids=ids, calibration=None)
    b = classify(male, atlas, neuron_ids=ids[::-1], calibration=None)
    pd.testing.assert_frame_equal(
        a.per_neuron.sort_values("neuron_id").reset_index(drop=True),
        b.per_neuron.sort_values("neuron_id").reset_index(drop=True),
    )
    assert a.group is not None and b.group is not None
    assert a.group.s1 == pytest.approx(b.group.s1)


def test_group_only_calibration_leaves_neurons_uncalibrated(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    cal = _cal_for(atlas)
    cal.models.pop("neuron")
    cal.thresholds.pop("neuron")
    res = classify(male, atlas, cell_type="T1", calibration=cal)
    assert res.calibrated
    assert (res.per_neuron["call"] == "uncalibrated").all()
    assert res.per_neuron["p"].isna().all()
    assert (res.per_neuron["label"] == "T1").all()
    assert res.group is not None and res.group.call == "type" and res.group.p > 0.5
    assert any("Per-neuron calls are uncalibrated" in c for c in res.caveats)


def test_top_k_one_keeps_the_true_margin(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    one = classify(male, atlas, cell_type="T1", calibration=_cal_for(atlas), top_k=1)
    five = classify(male, atlas, cell_type="T1", calibration=_cal_for(atlas), top_k=5)
    assert one.group is not None and five.group is not None
    assert one.group.margin == pytest.approx(five.group.margin)
    assert one.group.p == pytest.approx(five.group.p)
    assert np.allclose(one.per_neuron["margin"], five.per_neuron["margin"])
    assert len(one.per_neuron["top_labels"].iloc[0]) == 1


def test_query_without_cross_reference_is_uncalibrated(male_female: tuple[Store, Store]):
    _, female = male_female  # FlyWire-like store: no cross-reference values
    atlas = build_atlas(female)
    res = classify(female, atlas, cell_type="T1", calibration=_cal_for(atlas))
    assert not res.calibrated
    assert "no cross-reference" in res.provenance["partner_names"]
    assert any("no cross-reference" in c for c in res.caveats)


def test_partner_names_override_is_uncalibrated(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    names = pd.Series(meta["cell_type"].to_numpy(), index=meta["neuron_id"].to_numpy())
    res = classify(male, atlas, cell_type="T1", partner_names=names, calibration=_cal_for(atlas))
    assert not res.calibrated
    assert any("partner_names" in c for c in res.caveats)


def test_cell_type_selection_masks_its_own_partner_names(male_female: tuple[Store, Store]):
    male, female = male_female
    atlas = build_atlas(female)
    res = classify(male, atlas, cell_type="T1", calibration=None)
    assert res.provenance["mask"] == ["T1"]
    explicit = classify(
        male, atlas, cell_type="T1", calibration=None, mask_self=False, mask=("T1",)
    )
    assert np.allclose(res.per_neuron["s1"], explicit.per_neuron["s1"])
    unmasked = classify(male, atlas, cell_type="T1", calibration=None, mask_self=False)
    assert unmasked.provenance["mask"] == []


def test_query_provenance_is_recorded(male_female: tuple[Store, Store]):
    male, female = male_female
    res = classify(male, build_atlas(female), cell_type="T1", calibration=None)
    q = res.provenance["query_store"]
    assert q["dataset"] == "malecns" and "counts" in q and "inputs" in q
    assert res.provenance["query_matrix"]["dataset"] == "malecns@1.0"


def test_neuron_ids_accept_a_numpy_array(male_female: tuple[Store, Store]):
    male, female = male_female
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    ids = meta.loc[meta["cell_type"] == "T2", "neuron_id"].to_numpy()
    res = classify(male, build_atlas(female), neuron_ids=ids, calibration=None)
    assert len(res.per_neuron) == len(ids)


def test_group_counts_neurons_with_partners_separately(male_female: tuple[Store, Store]):
    male, female = male_female
    meta = male.neurons(columns=["neuron_id", "cell_type"])
    t1 = meta.loc[meta["cell_type"] == "T1", "neuron_id"].tolist()[:3]
    res = classify(
        male, build_atlas(female), neuron_ids=[*t1, 901, 902], mask=("T0",), calibration=None
    )
    assert res.group is not None
    assert res.group.n == 5
    assert res.group.n_with_partners == 3
    assert res.group.agreement == 1.0
