"""Fake raw source files with the real column layouts of each dataset."""

from __future__ import annotations

import gzip
import io
import tarfile
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


@pytest.fixture
def banc_raw(tmp_path: Path) -> Path:
    """Six segments: 4 neurons (proofread/rough), 1 glia, 1 unproofread; string ids as released."""
    raw = tmp_path / "raw" / "banc" / "888"
    raw.mkdir(parents=True)
    ids = [
        "720575941000000001",
        "720575941000000002",
        "720575941000000003",
        "720575941000000004",
        "720575941000000005",
        "720575941000000006",
    ]
    meta = pd.DataFrame(
        {
            "banc_888_id": ids,
            "root_id": ids,
            "proofread": ["TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "FALSE"],
            "roughly_proofread": ["FALSE", "FALSE", "TRUE", "FALSE", "FALSE", "FALSE"],
            "super_class": [
                "descending",
                "ventral_nerve_cord_intrinsic",
                "motor",
                "central_brain_intrinsic",
                "glia",
                None,
            ],
            "cell_class": ["descending_neuron", None, None, None, None, None],
            "cell_sub_class": [None] * 6,
            "cell_type": ["DNp01", "IN06B001", "MNad07", "AOTU019", None, None],
            "side": ["left", "right", "left", "right", "left", None],
            "hemilineage": [None, "06B", None, "ALad1", None, None],
            "region": [
                "central_brain",
                "ventral_nerve_cord",
                "ventral_nerve_cord",
                "central_brain",
                None,
                None,
            ],
            "flow": ["efferent", "intrinsic", "efferent", "intrinsic", None, None],
            "neurotransmitter_predicted": [
                "acetylcholine",
                "gaba",
                "glutamate",
                "histamine",
                None,
                None,
            ],
            "neurotransmitter_score": [0.9, 0.8, 0.7, 0.6, np.nan, np.nan],
            "fafb_cell_type": ["DNp01", None, None, "AOTU019", None, None],
            "manc_cell_type": ["DNp01", "IN06B001", "MNad07", None, None, None],
            "malecns_cell_type": ["DNp01", "IN06B001", None, "AOTU019", None, None],
            "hemibrain_cell_type": [None, None, None, "auto:AOTU019", None, None],
            "sexually_dimorphic": ["isomorphic", None, None, None, None, None],
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(meta, preserve_index=False), raw / "banc_888_meta.feather"
    )
    edges = pd.DataFrame(
        {
            "pre": [ids[0], ids[0], ids[1], ids[3], ids[4], ids[0], ids[5]],
            "post": [ids[1], ids[2], ids[2], ids[0], ids[0], ids[0], ids[1]],
            "count": np.array([12, 7, 30, 4, 9, 2, 3], dtype=np.int32),
            "norm": [0.1] * 7,
            "post_count": np.array([100, 80, 80, 50, 50, 50, 100], dtype=np.int32),
            "pre_count": np.array([20, 20, 30, 4, 9, 20, 3], dtype=np.int32),
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(edges, preserve_index=False),
        raw / "banc_888_edgelist_simple_v2.feather",
    )
    return raw


@pytest.fixture
def hemibrain_raw(tmp_path: Path) -> Path:
    """Four traced bodies in the v1.2 export tarball; Supp. 5, sjcabs meta and NT feather."""
    raw = tmp_path / "raw" / "hemibrain" / "1.2.1"
    raw.mkdir(parents=True)
    traced = pd.DataFrame(
        {
            "bodyId": [1001, 1002, 1003, 1004],
            "type": ["PFL3", "PFL3", "EPG", None],
            "instance": ["PFL3_L*", "PFL3_R", "EPG(PB08)_L7", None],
        }
    )
    total = pd.DataFrame(
        {
            "bodyId_pre": [1001, 1001, 1002, 1003, 1004],
            "bodyId_post": [1002, 1003, 1003, 1001, 1001],
            "weight": [12, 3, 40, 7, 1],
        }
    )
    roi = pd.DataFrame(
        {
            "bodyId_pre": [1001, 1001],
            "bodyId_post": [1002, 1002],
            "roi": ["LAL(R)", "NotPrimary"],
            "weight": [10, 2],
        }
    )
    with tarfile.open(raw / "exported-traced-adjacencies-v1.2.tar.gz", "w:gz") as tar:
        for name, df in [
            ("traced-neurons.csv", traced),
            ("traced-total-connections.csv", total),
            ("traced-roi-connections.csv", roi),
        ]:
            data = df.to_csv(index=False).encode()
            info = tarfile.TarInfo(f"exported-traced-adjacencies-v1.2/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    supp5 = pd.DataFrame(
        {
            "bodyId": [1001, 1002, 1003, 9999],
            "instance": ["PFL3_L*", "PFL3_R", "EPG(PB08)_L7", "x"],
            "type": ["PFL3", "PFL3", "EPG", "x"],
            "morphology_type": ["PFL3", "PFL3", "EPG", "x"],
            "cell_class": ["CX", "CX", "CX", None],
            "ito_lee_hemilineage": ["DM1_CX_d2", "DM1_CX_d2", "LALv1_dorsal", None],
            "morphology_group": [None] * 4,
            "notes": [None] * 4,
            "cellBodyFiber": ["PDM14", "PDM14", None, None],
            "side": ["left", "right", "na", "right"],
            "somaLocation": [None] * 4,
            "pre": [100.0, 120.0, 80.0, 1.0],
            "post": [400.0, 410.0, 300.0, 1.0],
            "pre_con2": [500.0, 600.0, 400.0, 1.0],
            "cropped": [False, False, False, True],
            "fbbt_id": ["FBbt_1", "FBbt_1", None, None],
        }
    )
    supp5.to_csv(raw / "Supplemental_file5_hemibrain_meta.csv", index=False)
    sj = pd.DataFrame(
        {
            "hemibrain_121_id": ["1001", "1002", "1003", "1004"],
            "instance": ["PFL3_L*", "PFL3_R", "EPG(PB08)_L7", None],
            "cell_type": ["PFL3", "PFL3", "EPG", None],
            "region": ["central_brain", "central_brain", "central_brain", None],
            "hemilineage": ["DM1_CX_d2", "DM1_CX_d2", "LALv1_dorsal", None],
            "nerve": [None] * 4,
            "flow": ["intrinsic", "intrinsic", "intrinsic", None],
            "super_class": ["central_brain_intrinsic"] * 3 + [None],
            "cell_class": ["central_complex_output_neuron"] * 2 + [None, None],
            "cell_sub_class": [None] * 4,
            "neurotransmitter_predicted": ["acetylcholine"] * 3 + [None],
            "neurotransmitter_score": ["0.9", "0.8", "0.7", None],
            "status": ["Traced"] * 4,
            "cropped": [False] * 4,
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(sj, preserve_index=False), raw / "hemibrain_121_meta.feather"
    )
    nt = pd.DataFrame(
        {
            "body": [1001, 1002, 1003, 5555],
            "type": ["PFL3", "PFL3", "EPG", None],
            "instance": ["PFL3_L*", "PFL3_R", "EPG(PB08)_L7", None],
            "statusLabel": ["Traced", "Traced", "Traced", None],
            "gaba": [0.05, 0.1, 0.1, 0.0],
            "acetylcholine": [0.8, 0.1, 0.2, 0.0],
            "glutamate": [0.05, 0.1, 0.1, 0.0],
            "serotonin": [0.02, 0.1, 0.1, 0.0],
            "octopamine": [0.02, 0.1, 0.1, 0.0],
            "dopamine": [0.03, 0.1, 0.1, 0.0],
            "neither": [0.03, 0.4, 0.3, 1.0],
            "predicted_nt": ["acetylcholine", "neither", "neither", "neither"],
        }
    ).set_index("body")  # the release stores 'body' as the pandas index
    pf.write_feather(
        pa.Table.from_pandas(nt), raw / "hemibrain-v1.2-body-mean-neurotransmitters.feather"
    )
    return raw


@pytest.fixture
def manc_raw(tmp_path: Path) -> Path:
    """Four neurons + one glia (Codex only); sjcabs meta and edgelist with string ids."""
    raw = tmp_path / "raw" / "manc" / "1.2.1"
    raw.mkdir(parents=True)
    vnc = "ventral_nerve_cord"
    meta = pd.DataFrame(
        {
            "manc_121_id": ["10000", "10001", "10002", "10003"],
            "region": ["neck_connective", vnc, vnc, vnc],
            "side": ["right", "left", "right", None],
            "hemilineage": [None, "20A.22A", "TBD", None],
            "nerve": [None, None, None, "left_prothoracic_leg_nerve"],
            "flow": ["efferent", "intrinsic", "intrinsic", "afferent"],
            "super_class": [
                "descending",
                "ventral_nerve_cord_intrinsic",
                "ventral_nerve_cord_intrinsic",
                "sensory",
            ],
            "cell_class": [None, None, None, "chordotonal_organ_neuron"],
            "cell_sub_class": [None, "ventral_nerve_cord_ipsilateral_restricted", None, None],
            "cell_type": ["DNp01", "IN20A.22A067", "Ti flexor MN", "SNpp50"],
            "neurotransmitter_predicted": ["acetylcholine", "gaba", "unclear", "glutamate"],
            "cell_function": [None] * 4,
            "cell_function_detailed": [None] * 4,
            "body_part_sensory": [None] * 4,
            "body_part_effector": [None] * 4,
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(meta, preserve_index=False), raw / "manc_121_meta.feather"
    )
    # 14223 is a glia body: traced in neuPrint/Codex, absent from the sjcabs neuron meta.
    pre = ["10000", "10001", "10002", "14223", "10003", "10001"]
    post = ["10001", "10002", "10001", "10001", "10001", "10001"]
    count = np.array([12, 7, 30, 4, 2, 5], dtype=np.int32)
    total_input = {"10001": 53, "10002": 7}
    edges = pd.DataFrame(
        {
            "pre": pre,
            "post": post,
            "count": count,
            "norm": [int(c) / total_input[p] for c, p in zip(count, post, strict=True)],
            "total_input": np.array([total_input[p] for p in post], dtype=np.int32),
        }
    )
    pf.write_feather(
        pa.Table.from_pandas(edges, preserve_index=False),
        raw / "manc_121_simple_edgelist.feather",
    )
    codex = pd.DataFrame(
        {
            "Root ID": [10000, 10001, 10002, 10003, 14223],
            "Top in/out region": ["LTCT"] * 5,
            "Community labels": [
                "description::Giant fiber,group::10000,instance::DNp01_R,vfbId::VFB_jrchk00s",
                "instance::IN20A.22A067_L",
                "",
                "vfbId::VFB_x",
                "",
            ],
            "Predicted NT type": ["ACH", "GABA", "GLUT", "ACH", ""],
            "Predicted NT confidence": [0.51, 0.9, 0.4, 0.7, np.nan],
            "Verified NT type": [None] * 5,
            "Verified Neuropeptide": [None] * 5,
            "Body Part": [None] * 5,
            "Function": [None] * 5,
            "Flow": ["efferent", "intrinsic", "intrinsic", "afferent", None],
            "Super Class": [
                "descending",
                "intrinsic_neuron",
                "intrinsic_neuron",
                "sensory",
                "glia",
            ],
            "Class": [None] * 5,
            "Sub Class": [None] * 5,
            "Hemilineage": [None] * 5,
            "Nerve": [None] * 5,
            "Soma side": ["right", "left", "right", None, None],
            "Primary Cell Type": ["DNp01", "IN20A.22A067", "tibia_flexor", "SNpp50", None],
            "Alternative Cell Type(s)": [None] * 5,
            "Cable length (nm)": [None] * 5,
            "Surface area (nm^2)": [None] * 5,
            "Volume (nm^3)": [None] * 5,
        }
    )
    _gz_csv(raw / "neurons.csv.gz", codex)
    return raw
