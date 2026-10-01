"""Tests for the MANC v1.2.1 converter (sjcabs compiled meta + edgelist, Codex neurons)."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from flyconn.compare.types import partner_labels
from flyconn.data.convert.manc import convert_manc


def _neurons(out: Path):
    return pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")


def test_neurons_are_sjcabs_meta_and_glia_are_counted(manc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    prov = convert_manc(manc_raw, out)
    n = _neurons(out)
    assert sorted(n.index) == [10000, 10001, 10002, 10003]
    assert n.index.dtype.name == "int64"
    assert prov["counts"]["neurons"] == 4
    assert prov["counts"]["traced_codex"] == 5
    assert prov["counts"]["glia_excluded"] == 1


def test_harmonized_columns(manc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_manc(manc_raw, out)
    n = _neurons(out)
    dn = n.loc[10000]
    assert dn["super_class"] == "descending" and dn["cell_type"] == "DNp01"
    assert dn["side"] == "right" and dn["flow"] == "efferent"
    assert dn["manc_121_cell_type"] == "DNp01"  # MANC names are the shared vocabulary
    assert n.loc[10002, "manc_121_cell_type"] == "Ti flexor MN"  # neuPrint type, not Codex
    assert n.loc[10001, "hemilineage_truman"] == "20A.22A"
    assert n["hemilineage"].isna().all()  # ItoLee column stays empty for the VNC
    assert n.loc[10003, "nerve"] == "left_prothoracic_leg_nerve"
    assert dn["vfb_id"] == "VFB_jrchk00s"
    assert n.loc[10001, "vfb_id"] is None


def test_nt_is_three_class_with_codex_confidence_when_labels_agree(manc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_manc(manc_raw, out)
    n = _neurons(out)
    assert n.loc[10000, "nt_pred"] == "acetylcholine"
    assert n.loc[10000, "nt_conf"] == pytest.approx(0.51)
    assert n.loc[10001, "nt_pred"] == "gaba" and n.loc[10001, "nt_conf"] == pytest.approx(0.9)
    assert n.loc[10002, "nt_pred"] is None  # "unclear"
    assert n.loc[10003, "nt_pred"] == "glutamate"
    assert n.loc[10003, "nt_conf"] != n.loc[10003, "nt_conf"]  # Codex says ACH: no confidence
    assert set(n["nt_pred"].dropna()) <= {"acetylcholine", "gaba", "glutamate"}


def test_edges_drop_glia_partners_and_keep_released_totals(manc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    prov = convert_manc(manc_raw, out)
    e = pq.read_table(out / "edges.parquet").to_pandas()
    got = {(a, b): w for a, b, w in zip(e["pre"], e["post"], e["weight"], strict=True)}
    assert got == {
        (10000, 10001): 12,
        (10001, 10002): 7,
        (10002, 10001): 30,
        (10003, 10001): 2,
    }
    assert prov["counts"]["raw_edges"] == 6
    assert prov["counts"]["raw_weight_sum"] == 60
    assert prov["counts"]["autapses_dropped"] == 1
    assert prov["counts"]["edges"] == 4
    n = _neurons(out)
    assert n.loc[10001, "input_synapses_total"] == 53  # total_input incl. glia and autapse
    assert n.loc[10000, "input_synapses_total"] == 0
    assert n.loc[10001, "output_synapses_total"] == 12  # 7 + 5 (autapse) as released
    assert prov["edge_threshold"]["pair_min_synapses"] == 1


def test_partner_vocabulary_uses_manc_names(manc_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_manc(manc_raw, out)
    n = _neurons(out)
    labels = partner_labels(n, "fafb_or_manc")
    assert labels.loc[10000] == "DNp01"


def test_codex_file_is_optional(manc_raw: Path, tmp_path: Path):
    (manc_raw / "neurons.csv.gz").unlink()
    out = tmp_path / "store"
    prov = convert_manc(manc_raw, out)
    n = _neurons(out)
    assert len(n) == 4 and n["nt_conf"].isna().all()
    assert "traced_codex" not in prov["counts"]
