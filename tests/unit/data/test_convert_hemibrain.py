"""Tests for the hemibrain v1.2.1 converter (v1.2 traced-adjacency export + annotations)."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from flyconn.data.convert.hemibrain import convert_hemibrain


def _neurons(out: Path):
    return pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")


def test_neuron_universe_is_the_traced_export(hemibrain_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    prov = convert_hemibrain(hemibrain_raw, out)
    n = _neurons(out)
    assert sorted(n.index) == [1001, 1002, 1003, 1004]  # Supp. 5's 9999 is not traced
    assert (n["dataset"] == "hemibrain").all() and (n["version"] == "1.2.1").all()
    assert prov["counts"]["neurons"] == 4
    assert prov["counts"]["with_supp5"] == 3


def test_types_side_hemilineage_and_super_class(hemibrain_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_hemibrain(hemibrain_raw, out)
    n = _neurons(out)
    pfl = n.loc[1001]
    assert pfl["cell_type"] == "PFL3" and pfl["hemibrain_121_cell_type"] == "PFL3"
    assert pfl["side"] == "left" and n.loc[1002, "side"] == "right"
    assert n.loc[1003, "side"] is None  # "na" in Supp. 5
    assert pfl["hemilineage"] == "DM1_CX_d2"
    assert pfl["cell_class"] == "CX"
    assert pfl["morphology_type"] == "PFL3"
    assert pfl["super_class"] == "central_brain_intrinsic" and pfl["flow"] == "intrinsic"
    assert pfl["instance"] == "PFL3_L*"
    # No published FlyWire mapping in these files: never invented.
    assert n["fafb_783_cell_type"].isna().all()


def test_nt_probabilities_from_body_mean_feather(hemibrain_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    prov = convert_hemibrain(hemibrain_raw, out)
    n = _neurons(out)
    assert n.loc[1001, "nt_pred"] == "acetylcholine"
    assert n.loc[1001, "nt_conf"] == pytest.approx(0.8)
    assert n.loc[1001, "nt_p_acetylcholine"] == pytest.approx(0.8)
    assert n.loc[1001, "nt_p_dopamine"] == pytest.approx(0.03)
    assert n.loc[1002, "nt_pred"] is None  # "neither"
    assert n.loc[1004, "nt_source"] is None  # no T-bars, no prediction
    assert "UNVERIFIED" in n.loc[1001, "nt_source"]
    assert "UNVERIFIED" in json.dumps(prov["nt_probs"])


def test_edges_from_total_connections_with_threshold_provenance(
    hemibrain_raw: Path, tmp_path: Path
):
    out = tmp_path / "store"
    prov = convert_hemibrain(hemibrain_raw, out)
    e = pq.read_table(out / "edges.parquet").to_pandas()
    got = {(a, b): w for a, b, w in zip(e["pre"], e["post"], e["weight"], strict=True)}
    assert got == {
        (1001, 1002): 12,
        (1001, 1003): 3,
        (1002, 1003): 40,
        (1003, 1001): 7,
        (1004, 1001): 1,
    }
    assert e["weight"].dtype.name == "int32"
    assert prov["counts"]["raw_edges"] == 5 and prov["counts"]["raw_weight_sum"] == 63
    assert prov["edge_threshold"]["postHighAccuracyThreshold"] == 0
    assert prov["edge_threshold"]["pair_min_synapses"] == 1
    names = {i["name"] for i in prov["inputs"]}
    assert "exported-traced-adjacencies-v1.2.tar.gz" in names
    assert "Supplemental_file5_hemibrain_meta.csv" in names


def test_optional_annotation_files_may_be_absent(hemibrain_raw: Path, tmp_path: Path):
    for name in (
        "Supplemental_file5_hemibrain_meta.csv",
        "hemibrain_121_meta.feather",
        "hemibrain-v1.2-body-mean-neurotransmitters.feather",
    ):
        (hemibrain_raw / name).unlink()
    out = tmp_path / "store"
    convert_hemibrain(hemibrain_raw, out)
    n = _neurons(out)
    assert len(n) == 4 and n["side"].isna().all() and n["nt_pred"].isna().all()
