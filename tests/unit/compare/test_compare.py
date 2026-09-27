"""Tests for cross-dataset type matching and male-vs-female comparison with a left/right null."""

import numpy as np
import pandas as pd

from flyconn.compare import TypeComparison, compare_type, match_types, type_profile
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix


def test_match_types_uses_cross_reference_columns(male_female: tuple[Store, Store]):
    male, female = male_female
    mapping = match_types(male, female)
    assert isinstance(mapping, pd.DataFrame)
    assert set(mapping.columns) >= {"type_a", "type_b", "n_a", "n_b", "source"}
    assert set(mapping["type_a"]) == {f"T{i}" for i in range(8)}
    assert (mapping["type_a"] == mapping["type_b"]).all()
    assert (mapping["n_a"] == 6).all() and (mapping["n_b"] == 6).all()
    assert (mapping["source"] == "fafb_783_cell_type").all()


def test_type_profile_is_normalised_partner_type_vector(male_female: tuple[Store, Store]):
    male, _ = male_female
    m = ConnectivityMatrix.from_store(male)
    prof = type_profile(m, "T0", direction="out", by="cell_type", side="left")
    assert isinstance(prof, pd.Series)
    assert abs(prof.sum() - 1.0) < 1e-9
    assert prof.idxmax() == "T1"
    prof_in = type_profile(m, "T1", direction="in", by="cell_type", side="left")
    assert prof_in.idxmax() == "T0"


def test_compare_type_finds_no_difference_for_conserved_type(male_female: tuple[Store, Store]):
    male, female = male_female
    res = compare_type(male, female, "T0", n_permutations=200, seed=0)
    assert isinstance(res, TypeComparison)
    assert res.cross_similarity > 0.9
    assert res.within_similarity_a > 0.9 and res.within_similarity_b > 0.9
    assert res.p_value > 0.05
    assert res.verdict == "no evidence of difference beyond left/right variability"
    assert res.n_a == 6 and res.n_b == 6


def test_compare_type_flags_rewired_type(male_female: tuple[Store, Store]):
    male, female = male_female
    res = compare_type(male, female, "T3", n_permutations=200, seed=0)
    assert res.cross_similarity < res.within_similarity_a
    assert res.p_value < 0.05
    assert res.verdict.startswith("different")
    assert set(res.partner_differences.columns) >= {
        "partner_type",
        "fraction_a",
        "fraction_b",
        "difference",
    }
    assert res.partner_differences.iloc[0]["partner_type"] in {"T4", "T0"}
    assert "predicted" in " ".join(res.caveats).lower() or "wiring" in " ".join(res.caveats).lower()


def test_compare_type_reports_provenance_and_direction(male_female: tuple[Store, Store]):
    male, female = male_female
    res = compare_type(male, female, "T0", direction="in", n_permutations=50, seed=1)
    assert res.provenance["direction"] == "in"
    assert res.provenance["datasets"] == ["malecns@1.0", "flywire@783"]
    assert np.isfinite(res.z)
