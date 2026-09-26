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
from typing import cast

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
TBAR = "tbar-neurotransmitters-male-cns-v1.0.feather"
NT_SOURCE_TBAR = "malecns_v1.0_tbar_mean"
_TBAR_CLASSES = (
    "acetylcholine",
    "dopamine",
    "gaba",
    "glutamate",
    "histamine",
    "octopamine",
    "serotonin",
)
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
    df: pd.DataFrame = table.to_pandas()
    return df


def _super_class(raw: object) -> str | None:
    return normalize_super_class("malecns", raw)


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    return cast("pd.Series", df[name]) if name in df.columns else None


def _neurons(raw: Path) -> pd.DataFrame:
    full = _read_feather(raw / ANNOTATIONS)
    keep = [c for c in _ANN_COLUMNS if c in full.columns]
    mask = cast("pd.Series", full["superclass"]).notna()
    ann: pd.DataFrame = full.loc[mask, keep].reset_index(drop=True)
    # Categorical statusLabel -> plain strings.
    status_label = ann["statusLabel"].astype("string") if "statusLabel" in ann.columns else None

    neurons = pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "neuron_id": ann["bodyId"].astype("int64"),
            "status": status_label,
            "super_class_raw": ann["superclass"],
            "super_class": ann["superclass"].map(_super_class),
            "cell_class": _col(ann, "class"),
            "cell_sub_class": _col(ann, "subclass"),
            "cell_type": ann["type"],
            "hemilineage": _col(ann, "itoleeHl"),
            "hemilineage_truman": _col(ann, "trumanHl"),
            "side": ann["somaSide"].map(normalize_side),
            "soma_neuromere": _col(ann, "somaNeuromere"),
            "fafb_783_cell_type": _col(ann, "flywireType"),
            "hemibrain_121_cell_type": _col(ann, "hemibrainType"),
            "manc_121_cell_type": _col(ann, "mancType"),
            "dimorphism": _col(ann, "dimorphism"),
            "vfb_id": _col(ann, "vfbId"),
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
            totals["output_synapses_total"] += np.bincount(pi[pre_in], w[pre_in], n).astype(
                np.int64
            )
            totals["input_synapses_total"] += np.bincount(qi[post_in], w[post_in], n).astype(
                np.int64
            )
            totals["output_synapses_neurons"] += np.bincount(pi[both], w[both], n).astype(np.int64)
            totals["input_synapses_neurons"] += np.bincount(qi[both], w[both], n).astype(np.int64)
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


def aggregate_tbar_nt(tbar_path: Path, out_path: Path, batch_rows: int = 1 << 20) -> pa.Table:
    """Stream the per-presynapse NT file and write per-body mean probabilities.

    Memory stays bounded by ``batch_rows`` rows at a time; sums are accumulated
    per body id in a dictionary-of-arrays keyed by a compact index.
    """
    cols = [f"nt_{c}_prob" for c in _TBAR_CLASSES]
    body_index: dict[int, int] = {}
    sums: list[np.ndarray] = []
    counts: list[int] = []
    sums_arr = np.zeros((0, len(cols)), dtype=np.float64)
    counts_arr = np.zeros(0, dtype=np.int64)
    with pa.memory_map(str(tbar_path)) as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            bodies = batch.column("body").to_numpy()
            probs = np.column_stack([batch.column(c).to_numpy().astype(np.float64) for c in cols])
            uniq, inv = np.unique(bodies, return_inverse=True)
            new = [b for b in uniq.tolist() if b not in body_index]
            if new:
                start = len(body_index)
                for k, b in enumerate(new):
                    body_index[b] = start + k
                sums_arr = np.vstack([sums_arr, np.zeros((len(new), len(cols)))])
                counts_arr = np.concatenate([counts_arr, np.zeros(len(new), dtype=np.int64)])
            rows = np.array([body_index[b] for b in uniq.tolist()])[inv]
            for j in range(len(cols)):
                sums_arr[:, j] += np.bincount(rows, probs[:, j], minlength=len(body_index))
            counts_arr += np.bincount(rows, minlength=len(body_index)).astype(np.int64)
    del sums, counts
    ids = np.fromiter(body_index.keys(), dtype=np.int64, count=len(body_index))
    means = sums_arr / counts_arr[:, None]
    data: dict[str, object] = {"neuron_id": ids, "n_presynapses": counts_arr}
    for j, c in enumerate(_TBAR_CLASSES):
        data[f"nt_p_{c}"] = means[:, j].astype(np.float32)
    table = pa.Table.from_pydict(data)
    write_parquet(table, out_path)
    return table


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

    tbar_path = raw_dir / TBAR
    if tbar_path.exists():
        probs = aggregate_tbar_nt(tbar_path, out_dir / "nt_probs.parquet").to_pandas()
        probs = probs.set_index("neuron_id").reindex(neurons["neuron_id"].to_numpy())
        has = probs["n_presynapses"].notna().to_numpy()
        for c in _TBAR_CLASSES:
            neurons[f"nt_p_{c}"] = probs[f"nt_p_{c}"].to_numpy(dtype=np.float32)
        neurons["nt_presynapses"] = probs["n_presynapses"].to_numpy()
        src = neurons["nt_source"].to_numpy(dtype=object) if "nt_source" in neurons else None
        if src is not None:
            neurons["nt_source"] = np.where(has, NT_SOURCE_TBAR, src)
        inputs.append(tbar_path)
        extra["nt_probs"] = {
            "source_file": TBAR,
            "aggregation": "mean of per-presynapse probabilities",
            "bodies_with_probabilities": int(has.sum()),
        }

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
