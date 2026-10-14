"""Reference atlas: centroids, yardsticks, custom groups, masking."""

import numpy as np
import pytest
import scipy.sparse as sp

from flyconn.compare.atlas import Atlas, atlas_from_matrix, build_atlas
from flyconn.data.store import Store
from flyconn.graph.matrix import ConnectivityMatrix


def _female_atlas(male_female: tuple[Store, Store], **kw: object) -> Atlas:
    return build_atlas(male_female[1], **kw)  # type: ignore[arg-type]


def test_cell_type_atlas_has_tight_yardsticks(male_female: tuple[Store, Store]):
    atlas = _female_atlas(male_female)
    assert atlas.labels.tolist() == [f"T{i}" for i in range(8)]
    assert atlas.n_members.tolist() == [6] * 8
    assert atlas.centroids.shape == (8, 2 * len(atlas.vocab))
    assert np.all(atlas.neuron_yardstick > 0.9)
    assert np.all(atlas.group_yardstick > 0.9)
    assert not atlas.yardstick_fallback.any()
    assert atlas.params["direction"] == "both"
    assert len(atlas.params_hash) == 16


def test_params_hash_changes_with_parameters(male_female: tuple[Store, Store]):
    a = _female_atlas(male_female)
    b = _female_atlas(male_female, direction="out")
    assert a.params_hash != b.params_hash
    assert b.centroids.shape[1] == len(b.vocab)
    m = ConnectivityMatrix.from_store(male_female[1], min_weight=5)
    other_store = atlas_from_matrix(
        m, reference="flywire@783", store_provenance={"counts": {"neurons": 1}}
    )
    assert other_store.params_hash != a.params_hash  # same name, different store


def test_custom_groups_and_missing_ids(male_female: tuple[Store, Store]):
    female = male_female[1]
    meta = female.neurons(columns=["neuron_id", "cell_type"])
    t0 = meta.loc[meta["cell_type"] == "T0", "neuron_id"].tolist()
    t1 = meta.loc[meta["cell_type"] == "T1", "neuron_id"].tolist()
    atlas = build_atlas(female, groups={"x": t0, "y": t1})
    assert atlas.labels.tolist() == ["x", "y"]
    assert atlas.params["labels_from"] == "groups"
    with pytest.raises(KeyError, match="not in"):
        build_atlas(female, groups={"x": [*t0, 999_999]})
    dropped = build_atlas(female, groups={"x": [*t0, 999_999]}, on_missing="drop")
    assert dropped.provenance["group_ids_dropped"] == 1
    with pytest.raises(ValueError, match="more than one group"):
        build_atlas(female, groups={"x": t0, "y": t0[:1]})


def test_single_member_label_uses_fallback_yardstick(male_female: tuple[Store, Store]):
    female = male_female[1]
    meta = female.neurons(columns=["neuron_id", "cell_type", "side"])
    t0 = meta.loc[meta["cell_type"] == "T0", "neuron_id"].tolist()
    one = meta.loc[(meta["cell_type"] == "T1"), "neuron_id"].tolist()[:1]
    atlas = build_atlas(female, groups={"x": t0, "solo": one})
    solo = atlas.labels.tolist().index("solo")
    assert atlas.yardstick_fallback[solo]
    assert np.isfinite(atlas.neuron_yardstick).all()
    assert np.isfinite(atlas.group_yardstick).all()
    assert atlas.neuron_yardstick[solo] == pytest.approx(atlas.neuron_yardstick[1 - solo])


def test_masked_equals_exact_recompute(male_female: tuple[Store, Store]):
    atlas = _female_atlas(male_female)
    masked = atlas.masked(("T1",))
    assert masked.mask == ("T1",)
    mc = atlas.member_counts
    prof = mc.profiles(direction="both", mask=("T1",))
    agg = sp.csr_matrix(
        (
            np.ones(len(atlas.member_codes)),
            (atlas.member_codes, np.arange(len(atlas.member_codes))),
        ),
        shape=(len(atlas.labels), len(atlas.member_codes)),
    )
    expected = (agg @ prof).toarray()
    assert np.allclose(masked.sums.toarray(), expected, atol=1e-12)
    t1_cols = np.flatnonzero(mc.column_keep(("T1",), "both") == 0)
    assert np.all(masked.centroids.toarray()[:, t1_cols] == 0)
    assert atlas.mask == ()  # original untouched


def test_drop_removes_label_and_its_members(male_female: tuple[Store, Store]):
    atlas = _female_atlas(male_female).masked(("T3",), drop=("T3",))
    assert "T3" not in atlas.labels.tolist()
    assert len(atlas.labels) == 7
    assert len(atlas.member_codes) == 42
    assert atlas.member_codes.max() == 6
    again = atlas.masked(("T2",))  # masking still works after a drop
    assert again.sums.shape[0] == 7


def test_atlas_from_matrix_rejects_empty_labels(male_female: tuple[Store, Store]):
    m = ConnectivityMatrix.from_store(male_female[1], min_weight=5)
    m.meta["cell_type"] = None
    with pytest.raises(ValueError, match="no labelled"):
        atlas_from_matrix(m, reference="flywire@783")
