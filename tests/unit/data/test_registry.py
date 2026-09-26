"""Tests for the dataset registry (YAML → typed objects)."""

import pytest

from flyconn.data.registry import DatasetSpec, FileSpec, get_dataset, list_datasets, parse_ref


def test_parse_ref_splits_name_and_version():
    assert parse_ref("malecns@1.0") == ("malecns", "1.0")


def test_parse_ref_rejects_missing_version():
    with pytest.raises(ValueError, match="name@version"):
        parse_ref("malecns")


def test_registry_lists_bundled_datasets_with_versions():
    refs = list_datasets()
    assert "malecns@1.0" in refs
    assert "flywire@783" in refs
    assert "flywire@630" in refs


def test_malecns_spec_has_license_citation_and_default_files():
    spec = get_dataset("malecns@1.0")
    assert isinstance(spec, DatasetSpec)
    assert spec.name == "malecns"
    assert spec.version == "1.0"
    assert spec.license.startswith("CC-BY-4.0")
    assert any("10.1016/j.cell.2026.08.015" in c.doi for c in spec.citations)
    names = {f.name for f in spec.files}
    assert "body-annotations-male-cns-v1.0-minconf-0.5.feather" in names
    assert "connectome-weights-male-cns-v1.0-minconf-0.5.feather" in names


def test_file_specs_carry_url_size_checksum_and_level():
    spec = get_dataset("malecns@1.0")
    ann = spec.file("body-annotations-male-cns-v1.0-minconf-0.5.feather")
    assert isinstance(ann, FileSpec)
    assert ann.url.startswith("https://storage.googleapis.com/flyem-male-cns/v1.0/")
    assert ann.bytes == 14_483_314
    assert ann.sha256 == "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"
    assert ann.level == "meta"
    weights = spec.file("connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    assert weights.level == "weights"


def test_files_for_level_includes_lower_levels():
    spec = get_dataset("malecns@1.0")
    meta_names = {f.name for f in spec.files_for_level("meta")}
    weight_names = {f.name for f in spec.files_for_level("weights")}
    assert meta_names < weight_names
    assert not any(f.level == "synapses" for f in spec.files_for_level("weights"))


def test_unknown_dataset_raises_key_error_listing_options():
    with pytest.raises(KeyError, match=r"malecns@1\.0"):
        get_dataset("nosuch@9")
