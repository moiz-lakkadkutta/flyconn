"""Tests for the FlyWire Codex converter."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from flyconn.data.convert.flywire import convert_flywire


def test_neurons_join_classification_and_types(flywire_783_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_flywire(flywire_783_raw, out, version="783")
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert len(n) == 4
    assert n.loc[720575940600000001, "super_class"] == "optic_lobe_intrinsic"
    assert n.loc[720575940600000001, "super_class_raw"] == "optic"
    assert n.loc[720575940600000002, "super_class"] == "central_brain_intrinsic"
    assert n.loc[720575940600000001, "cell_type"] == "L4"
    assert n.loc[720575940600000002, "cell_type"] == "AOTU032"  # falls back to hemibrain type
    assert n.loc[720575940600000002, "hemibrain_121_cell_type"] == "AOTU032"
    assert n.loc[720575940600000002, "hemilineage"] == "ALad1"
    assert n.loc[720575940600000003, "flow"] == "afferent"
    assert n.loc[720575940600000004, "side"] == "right"


def test_neurotransmitter_probabilities_are_carried(flywire_783_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_flywire(flywire_783_raw, out, version="783")
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert n.loc[720575940600000002, "nt_pred"] == "gaba"
    assert n.loc[720575940600000002, "nt_conf"] == 0.7
    assert abs(n.loc[720575940600000002, "nt_p_gaba"] - 0.70) < 1e-6
    assert n["nt_p_histamine"].isna().all()  # not a FlyWire class
    assert (
        n.loc[720575940600000004, "nt_pred"] is None
        or n.loc[720575940600000004, "nt_pred"] != n.loc[720575940600000004, "nt_pred"]
    )
    assert (n["nt_source"] == "codex_783_neurons_csv").all()


def test_edges_are_pair_aggregated_and_roi_table_kept(flywire_783_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_flywire(flywire_783_raw, out, version="783")
    e = pq.read_table(out / "edges.parquet").to_pandas()
    got = {(a, b): w for a, b, w in zip(e["pre"], e["post"], e["weight"], strict=True)}
    assert got == {
        (720575940600000001, 720575940600000002): 7,
        (720575940600000002, 720575940600000003): 9,
        (720575940600000003, 720575940600000004): 5,
    }
    assert e["weight"].dtype.name == "int32"
    roi = pq.read_table(out / "edges_roi.parquet").to_pandas()
    assert len(roi) == 5
    assert set(roi["roi"]) == {"ME_R", "LO_R", "AL_L", "GNG"}


def test_provenance_records_threshold_and_synapse_source(flywire_783_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_flywire(flywire_783_raw, out, version="783")
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["dataset"] == "flywire"
    assert prov["version"] == "783"
    assert prov["edge_threshold"] == {"pair_min_synapses": 5, "synapse_source": "buhmann"}
    assert prov["counts"] == {"neurons": 4, "edges": 3, "edge_roi_rows": 5}
