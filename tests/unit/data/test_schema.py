"""Tests for the harmonized schema and vocabulary normalisers."""

import pandas as pd
import pyarrow as pa
import pytest

from flyconn.data.schema import (
    EDGE_REQUIRED,
    NEURON_REQUIRED,
    NT_CLASSES,
    conform_edges,
    conform_neurons,
    normalize_nt,
    normalize_side,
    normalize_super_class,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACH", "acetylcholine"),
        ("acetylcholine", "acetylcholine"),
        ("GABA", "gaba"),
        ("GLUT", "glutamate"),
        ("DA", "dopamine"),
        ("SER", "serotonin"),
        ("OCT", "octopamine"),
        ("HIST", "histamine"),
        ("TYR", "tyramine"),
        ("unclear", None),
        ("unknown", None),
        ("neither", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_nt_maps_all_source_spellings(raw: str | None, expected: str | None):
    assert normalize_nt(raw) == expected


def test_nt_classes_is_the_eight_class_union_in_fixed_order():
    assert NT_CLASSES == (
        "acetylcholine",
        "glutamate",
        "gaba",
        "dopamine",
        "octopamine",
        "serotonin",
        "histamine",
        "tyramine",
    )


@pytest.mark.parametrize(
    ("source", "raw", "expected"),
    [
        ("codex_fafb", "central", "central_brain_intrinsic"),
        ("codex_fafb", "optic", "optic_lobe_intrinsic"),
        ("codex_fafb", "descending", "descending"),
        ("malecns", "cb_intrinsic", "central_brain_intrinsic"),
        ("malecns", "ol_intrinsic", "optic_lobe_intrinsic"),
        ("malecns", "vnc_intrinsic", "ventral_nerve_cord_intrinsic"),
        ("malecns", "vnc_sensory", "sensory"),
        ("malecns", "descending_neuron", "descending"),
        ("malecns", "vnc_motor", "motor"),
        ("malecns", "cb_endocrine", "endocrine"),
        ("manc", "intrinsic neuron", "ventral_nerve_cord_intrinsic"),
        ("manc", "descending neuron", "descending"),
        ("banc", "central_brain_intrinsic", "central_brain_intrinsic"),
    ],
)
def test_normalize_super_class_maps_to_banc_vocabulary(source: str, raw: str, expected: str):
    assert normalize_super_class(source, raw) == expected


def test_normalize_super_class_keeps_unknown_values_verbatim():
    assert normalize_super_class("malecns", "vnc_tbc") == "vnc_tbc"
    assert normalize_super_class("malecns", None) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("L", "left"),
        ("R", "right"),
        ("M", "center"),
        ("LHS", "left"),
        ("RHS", "right"),
        ("Midline", "center"),
        ("left", "left"),
        ("center", "center"),
        ("na", None),
        (None, None),
    ],
)
def test_normalize_side(raw: str | None, expected: str | None):
    assert normalize_side(raw) == expected


def test_conform_neurons_fills_optional_columns_and_casts_types():
    df = pd.DataFrame(
        {"dataset": ["x"], "version": ["1"], "neuron_id": [5], "nt_pred": ["acetylcholine"]}
    )
    table = conform_neurons(df)
    assert isinstance(table, pa.Table)
    assert table.schema.field("neuron_id").type == pa.int64()
    assert table.schema.field("nt_conf").type == pa.float64()
    assert "super_class" in table.column_names
    assert table.column("super_class").null_count == 1
    assert set(NEURON_REQUIRED) <= set(table.column_names)


def test_conform_neurons_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="neuron_id"):
        conform_neurons(pd.DataFrame({"dataset": ["x"], "version": ["1"]}))


def test_conform_edges_casts_weight_to_int32_and_adds_roi():
    df = pd.DataFrame({"dataset": ["x"], "version": ["1"], "pre": [1], "post": [2], "weight": [7]})
    table = conform_edges(df)
    assert table.schema.field("weight").type == pa.int32()
    assert table.schema.field("roi").type == pa.string()
    assert set(EDGE_REQUIRED) <= set(table.column_names)
