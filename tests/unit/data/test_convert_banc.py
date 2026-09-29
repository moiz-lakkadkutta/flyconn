"""Tests for the BANC v888 compiled-data converter."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from flyconn.data.convert.banc import convert_banc


def test_neurons_are_proofread_or_rough_and_exclude_non_neurons(banc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_banc(banc_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert sorted(n.index) == [
        720575941000000001,
        720575941000000002,
        720575941000000003,
        720575941000000004,
    ]
    assert n.loc[720575941000000003, "status"] == "roughly_proofread"
    assert n.loc[720575941000000001, "status"] == "proofread"


def test_harmonized_columns_and_cross_references(banc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_banc(banc_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    dn = n.loc[720575941000000001]
    assert dn["super_class"] == "descending" and dn["cell_type"] == "DNp01"
    assert dn["fafb_783_cell_type"] == "DNp01" and dn["manc_121_cell_type"] == "DNp01"
    assert dn["malecns_09_cell_type"] == "DNp01"
    assert dn["nt_pred"] == "acetylcholine" and abs(dn["nt_conf"] - 0.9) < 1e-9
    assert n.loc[720575941000000004, "nt_pred"] == "histamine"
    assert (
        n.loc[720575941000000004, "hemibrain_121_cell_type"] == "AOTU019"
    )  # "auto:" prefix stripped
    assert n.loc[720575941000000004, "hemibrain_121_match_auto"]
    assert (n["nt_source"] == "banc_888_meta").all()


def test_edges_keep_neuron_pairs_drop_autapses_and_record_totals(banc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    prov = convert_banc(banc_raw, out)
    e = pq.read_table(out / "edges.parquet").to_pandas()
    got = {(a, b): w for a, b, w in zip(e["pre"], e["post"], e["weight"], strict=True)}
    assert got == {
        (720575941000000001, 720575941000000002): 12,
        (720575941000000001, 720575941000000003): 7,
        (720575941000000002, 720575941000000003): 30,
        (720575941000000004, 720575941000000001): 4,
    }
    assert e["weight"].dtype.name == "int32"
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert n.loc[720575941000000001, "input_synapses_total"] == 50  # post_count from the release
    assert n.loc[720575941000000003, "input_synapses_total"] == 80
    assert prov["counts"]["raw_edges"] == 7
    assert prov["counts"]["autapses_dropped"] == 1
    assert prov["counts"]["edges"] == 4
    p = json.loads((out / "provenance.json").read_text())
    assert p["neuron_definition"].startswith("proofread or roughly_proofread")
