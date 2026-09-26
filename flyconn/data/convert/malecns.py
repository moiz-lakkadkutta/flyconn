"""MaleCNS v1.0 flat-file converter.

Inputs (official Feather exports, see ``docs/DATA_SOURCES.md`` §1):

* ``body-annotations-*.feather`` — one row per annotated body; the neuron
  universe is ``superclass IS NOT NULL`` (166,700 rows in v1.0).
* ``body-neurotransmitters-*.feather`` — body-level argmax + confidence and a
  consensus label (ground truth > type-level > body-level).
* ``connectome-weights-*.feather`` — all 151.9 M segment→segment edges. It is
  streamed in record batches: edges between neurons are kept; per-neuron
  synapse totals including unannotated fragments are accumulated so that
  "fraction of inputs accounted for" can be computed later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import pyarrow.ipc as ipc

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import (
    conform_edges,
    conform_neurons,
    normalize_nt,
    normalize_side,
    normalize_super_class,
)

DATASET = "malecns"
VERSION = "1.0"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT_SOURCE = "malecns_v1.0_body_consensus"
NEURON_DEFINITION = "superclass IS NOT NULL"

_ANN_COLUMNS = [
    "bodyId",
    "status",
    "statusLabel",
    "superclass",
    "class",
    "subclass",
    "type",
    "somaSide",
    "somaNeuromere",
    "flywireType",
    "hemibrainType",
    "mancType",
    "itoleeHl",
    "trumanHl",
    "dimorphism",
    "vfbId",
]


def _read_feather(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    table = pf.read_table(path, columns=columns)
    return table.to_pandas(types_mapper=None)


def _neurons(raw: Path) -> pd.DataFrame:
    ann = _read_feather(raw / ANNOTATIONS)
    ann = ann[[c for c in _ANN_COLUMNS if c in ann.columns]]
    ann = ann[ann["superclass"].notna()].reset_index(drop=True)
    # Categorical statusLabel -> plain strings.
    status_label = ann["statusLabel"].astype("string") if "statusLabel" in ann else None

    neurons = pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "neuron_id": ann["bodyId"].astype("int64"),
            "status": status_label,
            "super_class_raw": ann["superclass"],
            "super_class": ann["superclass"].map(lambda v: normalize_super_class("malecns", v)),
            "cell_class": ann.get("class"),
            "cell_sub_class": ann.get("subclass"),
            "cell_type": ann["type"],
            "hemilineage": ann.get("itoleeHl"),
            "hemilineage_truman": ann.get("trumanHl"),
            "side": ann["somaSide"].map(normalize_side),
            "soma_neuromere": ann.get("somaNeuromere"),
            "fafb_783_cell_type": ann.get("flywireType"),
            "hemibrain_121_cell_type": ann.get("hemibrainType"),
            "manc_121_cell_type": ann.get("mancType"),
            "dimorphism": ann.get("dimorphism"),
            "vfb_id": ann.get("vfbId"),
        }
    )

    nt_path = raw / NEUROTRANSMITTERS
    if nt_path.exists():
        nt = _read_feather(
            nt_path,
            [
                "body",
                "predicted_nt",
                "predicted_nt_confidence",
                "celltype_predicted_nt",
                "celltype_predicted_nt_confidence",
                "ground_truth",
                "consensus_nt",
            ],
        ).set_index("body")
        nt = nt.reindex(neurons["neuron_id"].to_numpy())
        pred = nt["consensus_nt"].map(normalize_nt).to_numpy(dtype=object)
        body_pred = nt["predicted_nt"].map(normalize_nt).to_numpy(dtype=object)
        type_pred = nt["celltype_predicted_nt"].map(normalize_nt).to_numpy(dtype=object)
        body_conf = nt["predicted_nt_confidence"].to_numpy(dtype=float)
        type_conf = nt["celltype_predicted_nt_confidence"].to_numpy(dtype=float)
        # Confidence of the label that won the consensus: body-level if it agrees,
        # else type-level if that agrees, else NaN (ground-truth override).
        conf = np.where(
            pred == body_pred, body_conf, np.where(pred == type_pred, type_conf, np.nan)
        )
        conf = np.where(pd.isna(pred), np.nan, conf)
        neurons["nt_pred"] = pred
        neurons["nt_conf"] = conf
        neurons["nt_known"] = nt["ground_truth"].map(normalize_nt).to_numpy(dtype=object)
        neurons["nt_source"] = NT_SOURCE
    return neurons


def _stream_edges(
    weights_path: Path, neuron_ids: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray], int, int]:
    """Stream the full weights file; keep neuron→neuron edges and per-neuron totals."""
    index = pd.Index(neuron_ids)
    n = len(index)
    totals = {
        "input_synapses_total": np.zeros(n, dtype=np.int64),
        "input_synapses_neurons": np.zeros(n, dtype=np.int64),
        "output_synapses_total": np.zeros(n, dtype=np.int64),
        "output_synapses_neurons": np.zeros(n, dtype=np.int64),
    }
    kept_pre: list[np.ndarray] = []
    kept_post: list[np.ndarray] = []
    kept_w: list[np.ndarray] = []
    raw_edges = 0
    raw_weight = 0
    with pa.memory_map(str(weights_path)) as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre = batch.column("body_pre").to_numpy()
            post = batch.column("body_post").to_numpy()
            w = pc.cast(batch.column("weight"), pa.int64()).to_numpy()
            raw_edges += len(w)
            raw_weight += int(w.sum())
            pi = index.get_indexer(pre)
            qi = index.get_indexer(post)
            pre_in = pi >= 0
            post_in = qi >= 0
            both = pre_in & post_in
            np.add.at(totals["output_synapses_total"], pi[pre_in], w[pre_in])
            np.add.at(totals["input_synapses_total"], qi[post_in], w[post_in])
            np.add.at(totals["output_synapses_neurons"], pi[both], w[both])
            np.add.at(totals["input_synapses_neurons"], qi[both], w[both])
            if both.any():
                kept_pre.append(pre[both])
                kept_post.append(post[both])
                kept_w.append(w[both])
    edges = pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "pre": np.concatenate(kept_pre) if kept_pre else np.array([], dtype=np.int64),
            "post": np.concatenate(kept_post) if kept_post else np.array([], dtype=np.int64),
            "weight": np.concatenate(kept_w) if kept_w else np.array([], dtype=np.int64),
        }
    )
    return edges, totals, raw_edges, raw_weight


def convert_malecns(raw_dir: Path, out_dir: Path) -> dict[str, object]:
    """Convert MaleCNS v1.0 flat files in ``raw_dir`` to the harmonized store in ``out_dir``.

    Writes ``neurons.parquet`` always, ``edges.parquet`` when the weights file is
    present, and ``provenance.json``. Returns the provenance record.
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons = _neurons(raw_dir)
    inputs = [raw_dir / ANNOTATIONS]
    if (raw_dir / NEUROTRANSMITTERS).exists():
        inputs.append(raw_dir / NEUROTRANSMITTERS)
    counts: dict[str, int] = {"neurons": len(neurons)}
    extra: dict[str, object] = {"neuron_definition": NEURON_DEFINITION, "nt_source": NT_SOURCE}

    weights_path = raw_dir / WEIGHTS
    if weights_path.exists():
        edges, totals, raw_edges, raw_weight = _stream_edges(
            weights_path, neurons["neuron_id"].to_numpy()
        )
        for name, arr in totals.items():
            neurons[name] = arr
        edges = edges.sort_values(["pre", "post"], kind="stable").reset_index(drop=True)
        write_parquet(conform_edges(edges), out_dir / "edges.parquet")
        counts.update(edges=len(edges), raw_edges=raw_edges, raw_weight_sum=raw_weight)
        inputs.append(weights_path)
        extra["edge_threshold"] = {"minconf": 0.5, "pair_min_synapses": 1}

    write_parquet(conform_neurons(neurons), out_dir / "neurons.parquet")
    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=VERSION,
        converter="flyconn.data.convert.malecns",
        inputs=inputs,
        counts=counts,
        **extra,
    )
