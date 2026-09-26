"""Tests for the MaleCNS v1.0 converter (fake raw files with real column names)."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from flyconn.data.convert.malecns import convert_malecns


def test_neuron_universe_is_superclass_not_null(malecns_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    neurons = pq.read_table(out / "neurons.parquet").to_pandas()
    assert sorted(neurons["neuron_id"]) == [10001, 10002, 10003, 10005]
    assert (neurons["dataset"] == "malecns").all()
    assert (neurons["version"] == "1.0").all()


def test_neuron_columns_are_harmonized(malecns_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert n.loc[10003, "super_class"] == "central_brain_intrinsic"
    assert n.loc[10003, "super_class_raw"] == "cb_intrinsic"
    assert n.loc[10005, "super_class"] == "motor"
    assert n.loc[10001, "side"] == "right"
    assert n.loc[10003, "cell_class"] == "CX"
    assert n.loc[10003, "hemilineage"] == "ALad1"
    assert n.loc[10005, "hemilineage_truman"] == "20A.22A"
    assert n.loc[10005, "soma_neuromere"] == "T1"
    assert n.loc[10001, "fafb_783_cell_type"] == "DNp01"
    assert n.loc[10005, "manc_121_cell_type"] == "MNad07"
    assert n.loc[10003, "dimorphism"] == "sexually dimorphic"
    assert n.loc[10005, "status"] == "Out of scope"


def test_neurotransmitter_fields_use_consensus_and_confidence(malecns_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert n.loc[10003, "nt_pred"] == "gaba"
    assert n.loc[10003, "nt_conf"] == 0.88
    assert n.loc[10003, "nt_known"] == "gaba"
    # body-level unclear but type-level glutamate -> consensus glutamate, confidence from type level
    assert n.loc[10005, "nt_pred"] == "glutamate"
    assert n.loc[10005, "nt_conf"] == 0.6
    assert (n["nt_source"] == "malecns_v1.0_body_consensus").all()
    assert n["nt_p_gaba"].isna().all()  # no probabilities without the tbar file


def test_edges_restricted_to_neuron_universe_with_int32_weights(malecns_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    e = pq.read_table(out / "edges.parquet").to_pandas()
    pairs = set(zip(e["pre"], e["post"], strict=True))
    assert pairs == {(10001, 10002), (10001, 10003), (10002, 10003), (10003, 10001), (10005, 10001)}
    assert e["weight"].dtype.name == "int32"
    assert e["roi"].isna().all()


def test_per_neuron_synapse_totals_include_fragments(malecns_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    # 10001 receives 7 (from 10003) + 5 (glia 20001) + 1 (10005) = 13 total, 8 from neurons
    assert n.loc[10001, "input_synapses_total"] == 13
    assert n.loc[10001, "input_synapses_neurons"] == 8
    assert n.loc[10001, "output_synapses_total"] == 15
    assert n.loc[10002, "input_synapses_total"] == 21  # 12 + 9 (orphan)


def test_provenance_records_inputs_and_counts(malecns_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["dataset"] == "malecns"
    assert prov["version"] == "1.0"
    assert {i["name"] for i in prov["inputs"]} >= {
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    }
    assert all(len(i["sha256"]) == 64 for i in prov["inputs"])
    assert prov["counts"]["neurons"] == 4
    assert prov["counts"]["edges"] == 5
    assert prov["counts"]["raw_edges"] == 7
    assert prov["counts"]["raw_weight_sum"] == 77
    assert prov["neuron_definition"] == "superclass IS NOT NULL"


def test_meta_only_conversion_skips_edges(malecns_raw: Path, tmp_path: Path):
    (malecns_raw / "connectome-weights-male-cns-v1.0-minconf-0.5.feather").unlink()
    out = tmp_path / "store"
    convert_malecns(malecns_raw, out)
    assert (out / "neurons.parquet").exists()
    assert not (out / "edges.parquet").exists()
