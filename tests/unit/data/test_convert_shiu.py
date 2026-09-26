"""Tests for the Shiu et al. 2024 model-input converter (unthresholded FlyWire v630 export)."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from flyconn.data.convert.shiu import convert_shiu


@pytest.fixture
def shiu_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "shiu" / "630"
    raw.mkdir(parents=True)
    ids = [720575940600000001, 720575940600000002, 720575940600000003]
    pd.DataFrame({"Unnamed: 0": ids, "Completed": [True, True, True]}).to_csv(
        raw / "2023_03_23_completeness_630_final.csv", index=False
    )
    conn = pd.DataFrame(
        {
            "Presynaptic_ID": [ids[0], ids[1], ids[0]],
            "Postsynaptic_ID": [ids[1], ids[2], ids[2]],
            "Presynaptic_Index": [0, 1, 0],
            "Postsynaptic_Index": [1, 2, 2],
            "Connectivity": [10, 3, 1],
            "Excitatory": [1, -1, 1],
            "Excitatory x Connectivity": [10, -3, 1],
        }
    )
    conn.to_parquet(raw / "2023_03_23_connectivity_630_final.parquet", index=False)
    return raw


def test_neurons_come_from_completeness_with_sign_derived_nt(shiu_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_shiu(shiu_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert len(n) == 3
    assert (n["dataset"] == "shiu").all() and (n["version"] == "630").all()
    # sign is pre-baked per presynaptic neuron; we record it as a coarse NT class
    assert n.loc[720575940600000001, "nt_pred"] == "acetylcholine"  # excitatory placeholder
    assert n.loc[720575940600000002, "nt_pred"] == "gaba"  # inhibitory placeholder
    assert pd.isna(n.loc[720575940600000003, "nt_pred"])  # no outgoing edges -> unknown
    assert (n["nt_source"] == "shiu2024_excitatory_flag").all()


def test_edges_keep_all_weights_including_singletons_and_sign_column(
    shiu_raw: Path, tmp_path: Path
):
    out = tmp_path / "store"
    convert_shiu(shiu_raw, out)
    e = pq.read_table(out / "edges.parquet").to_pandas()
    assert len(e) == 3
    assert e["weight"].min() == 1
    assert set(e.columns) >= {"pre", "post", "weight", "sign"}
    assert set(zip(e["pre"], e["sign"], strict=True)) == {
        (720575940600000001, 1),
        (720575940600000002, -1),
    }


def test_provenance_marks_no_threshold_and_signed_source(shiu_raw: Path, tmp_path: Path):
    out = tmp_path / "store"
    convert_shiu(shiu_raw, out)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["edge_threshold"] == {"pair_min_synapses": 1, "synapse_source": "buhmann_cleft50"}
    assert prov["counts"] == {"neurons": 3, "edges": 3, "synapses": 14}
