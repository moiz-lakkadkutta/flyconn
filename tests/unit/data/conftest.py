"""Fake raw source files with the real column layouts of each dataset."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pytest


@pytest.fixture
def malecns_raw(tmp_path: Path) -> Path:
    """Six bodies: 4 neurons (superclass set), 1 glia, 1 orphan fragment; 7 edges."""
    raw = tmp_path / "raw" / "malecns" / "1.0"
    raw.mkdir(parents=True)
    ann = pd.DataFrame(
        {
            "bodyId": [10001, 10002, 10003, 10005, 20001, 30001],
            "status": ["Traced", "Traced", "Traced", None, "Glia", "Orphan"],
            "statusLabel": [
                "Roughly traced",
                "Reviewed",
                "Prelim Roughly traced",
                "Out of scope",
                "Glia",
                "Orphan",
            ],
            "superclass": [
                "descending_neuron",
                "visual_projection",
                "cb_intrinsic",
                "vnc_motor",
                None,
                None,
            ],
            "class": [None, None, "CX", None, None, None],
            "subclass": [None, None, None, "leg", None, None],
            "type": ["DNp01", "OCG01d", "AOTU019", "MNad07", None, None],
            "instance": ["DNp01(GF)_R", "OCG01d_L", "AOTU019_R", "MNad07_L", None, None],
            "somaSide": ["R", "L", "R", "L", None, None],
            "somaNeuromere": [None, None, None, "T1", None, None],
            "flywireType": ["DNp01", "OCG01d", "AOTU019", None, None, None],
            "hemibrainType": ["DNp01", None, "AOTU019", None, None, None],
            "mancType": [None, None, None, "MNad07", None, None],
            "itoleeHl": [None, None, "ALad1", None, None, None],
            "trumanHl": [None, None, None, "20A.22A", None, None],
            "dimorphism": [None, None, "sexually dimorphic", None, None, None],
            "vfbId": ["VFB_a", "VFB_b", "VFB_c", "VFB_d", None, None],
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(ann, preserve_index=False),
        raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    )
    nt = pd.DataFrame(
        {
            "body": [10001, 10002, 10003, 10005, 30001],
            "cell_type": ["DNp01", "OCG01d", "AOTU019", "MNad07", None],
            "total_nt_predictions": [1015, 1443, 4425, 300, 2],
            "predicted_nt": ["acetylcholine", "acetylcholine", "gaba", "unclear", "unclear"],
            "predicted_nt_confidence": [0.50, 0.96, 0.88, 0.31, 0.2],
            "ground_truth": [None, None, "gaba", None, None],
            "celltype_total_nt_predictions": [1991, 2818, 8721, 300, 2],
            "celltype_predicted_nt": [
                "acetylcholine",
                "acetylcholine",
                "gaba",
                "glutamate",
                "unclear",
            ],
            "celltype_predicted_nt_confidence": [0.53, 0.96, 0.87, 0.6, None],
            "consensus_nt": ["acetylcholine", "acetylcholine", "gaba", "glutamate", "unclear"],
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(nt, preserve_index=False),
        raw / "body-neurotransmitters-male-cns-v1.0.feather",
    )
    w = pd.DataFrame(
        {
            "body_pre": [10001, 10001, 10002, 10003, 20001, 10005, 30001],
            "body_post": [10002, 10003, 10003, 10001, 10001, 10001, 10002],
            "weight": [12, 3, 40, 7, 5, 1, 9],
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(w, preserve_index=False),
        raw / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    )
    return raw


def _gz_csv(path: Path, df: pd.DataFrame) -> None:
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)


@pytest.fixture
def flywire_783_raw(tmp_path: Path) -> Path:
    """Four neurons; connections split across neuropils with per-row counts < 5."""
    raw = tmp_path / "raw" / "flywire" / "783"
    raw.mkdir(parents=True)
    ids = [720575940600000001, 720575940600000002, 720575940600000003, 720575940600000004]
    neurons = pd.DataFrame(
        {
            "root_id": ids,
            "group": ["ME.LO", "", "", ""],
            "nt_type": ["ACH", "GABA", "GLUT", ""],
            "nt_type_score": [0.9, 0.7, 0.6, np.nan],
            "da_avg": [0.02, 0.05, 0.1, 0.0],
            "ser_avg": [0.01, 0.05, 0.1, 0.0],
            "gaba_avg": [0.03, 0.70, 0.1, 0.0],
            "glut_avg": [0.02, 0.10, 0.6, 0.0],
            "ach_avg": [0.90, 0.05, 0.05, 0.0],
            "oct_avg": [0.02, 0.05, 0.05, 0.0],
        }
    )
    _gz_csv(raw / "neurons.csv.gz", neurons)
    classification = pd.DataFrame(
        {
            "root_id": ids,
            "flow": ["intrinsic", "intrinsic", "afferent", "efferent"],
            "super_class": ["optic", "central", "sensory", "descending"],
            "class": ["L1-5", "CX", "olfactory", "DN"],
            "sub_class": [None, None, None, None],
            "hemilineage": [None, "ALad1", None, None],
            "side": ["right", "left", "left", "right"],
            "nerve": [None, None, "AN", "CV"],
        }
    )
    _gz_csv(raw / "classification.csv.gz", classification)
    types = pd.DataFrame(
        {
            "root_id": ids,
            "cell_type": ["L4", None, "ORN_VA1v", "DNp01"],
            "hemibrain_type": [None, "AOTU032", "ORN_VA1v", "DNp01"],
        }
    )
    _gz_csv(raw / "fw_and_hemibrain_types.csv.gz", types)
    consolidated = pd.DataFrame(
        {
            "root_id": ids,
            "primary_type": ["L4", "AOTU032", "ORN_VA1v", "DNp01"],
            "additional_type(s)": ["", "", "", ""],
        }
    )
    _gz_csv(raw / "consolidated_cell_types.csv.gz", consolidated)
    conns = pd.DataFrame(
        {
            "pre_root_id": [ids[0], ids[0], ids[1], ids[2], ids[2]],
            "post_root_id": [ids[1], ids[1], ids[2], ids[3], ids[3]],
            "neuropil": ["ME_R", "LO_R", "AL_L", "GNG", "AL_L"],
            "syn_count": [3, 4, 9, 2, 3],
            "nt_type": ["ACH", "ACH", "GABA", "GLUT", "GLUT"],
        }
    )
    _gz_csv(raw / "connections.csv.gz", conns)
    return raw
