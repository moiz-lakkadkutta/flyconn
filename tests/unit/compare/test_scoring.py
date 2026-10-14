"""Cosine scoring of query profiles against atlas centroids."""

import numpy as np
import pytest
import scipy.sparse as sp

from flyconn.compare.atlas import build_atlas
from flyconn.compare.scoring import Scores, score
from flyconn.data.store import Store


def test_reference_neurons_score_their_own_label_first(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    prof = atlas.member_counts.profiles(direction="both")
    sc = score(atlas, prof, top_k=3)
    assert isinstance(sc, Scores)
    assert sc.labels.shape == (48, 3)
    truth = atlas.labels[atlas.member_codes]
    assert (sc.top1 == truth).all()
    assert np.all(sc.s[:, 0] >= sc.s[:, 1])
    assert np.allclose(sc.margin, sc.s[:, 0] - sc.s[:, 1])
    assert np.allclose(sc.a1, sc.s1 / np.maximum(atlas.neuron_yardstick[atlas.member_codes], 0.05))


def test_chunking_gives_identical_scores(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    prof = atlas.member_counts.profiles(direction="both")
    a = score(atlas, prof, top_k=4, chunk=5000)
    b = score(atlas, prof, top_k=4, chunk=7)
    assert (a.labels == b.labels).all()
    assert np.array_equal(a.s, b.s)


def test_group_level_uses_group_yardstick(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    prof = sp.csr_matrix(atlas.member_counts.profiles(direction="both")[:6].mean(axis=0))
    sc = score(atlas, prof, level="group", top_k=2)
    j = atlas.labels.tolist().index(sc.top1[0])
    assert sc.a1[0] == pytest.approx(sc.s1[0] / atlas.group_yardstick[j])


def test_empty_profile_and_empty_query(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    width = atlas.sums.shape[1]
    zero = score(atlas, sp.csr_matrix((1, width)))
    assert zero.s1[0] == 0.0
    none = score(atlas, sp.csr_matrix((0, width)))
    assert none.labels.shape == (0, 5)
    with pytest.raises(ValueError, match="width"):
        score(atlas, sp.csr_matrix((1, width + 1)))


def test_top_k_capped_by_label_count(male_female: tuple[Store, Store]):
    atlas = build_atlas(male_female[1])
    sc = score(atlas, atlas.member_counts.profiles(direction="both")[:2], top_k=50)
    assert sc.labels.shape == (2, 8)
