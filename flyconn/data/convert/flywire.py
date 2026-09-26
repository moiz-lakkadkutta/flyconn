"""FlyWire (FAFB) Codex export converter for snapshots 630 and 783.

Inputs (``docs/DATA_SOURCES.md`` §2): ``neurons.csv.gz`` (per-neuron NT means),
``classification.csv.gz`` (hierarchy; v630 also carries ``cell_type`` and
``hemibrain_type``), ``fw_and_hemibrain_types.csv.gz`` (v783 types),
``connections.csv.gz`` (one row per pre, post, neuropil; the ≥5 threshold is on
the pair total). Edges are aggregated per pair; the per-neuropil rows are kept
in ``edges_roi.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import (
    conform_edges,
    conform_neurons,
    normalize_nt,
    normalize_side,
    normalize_super_class,
)

DATASET = "flywire"
_NT_AVG = {
    "ach_avg": "acetylcholine",
    "glut_avg": "glutamate",
    "gaba_avg": "gaba",
    "da_avg": "dopamine",
    "oct_avg": "octopamine",
    "ser_avg": "serotonin",
}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"root_id": "int64"}, low_memory=False)


def _neurons(raw: Path, version: str) -> tuple[pd.DataFrame, list[Path]]:
    inputs = [raw / "neurons.csv.gz", raw / "classification.csv.gz"]
    nt = _read(raw / "neurons.csv.gz").set_index("root_id")
    cls = _read(raw / "classification.csv.gz").drop_duplicates("root_id").set_index("root_id")
    ids = nt.index.to_numpy()
    cls = cls.reindex(ids)

    types = pd.DataFrame(index=pd.Index(ids, name="root_id"))
    if {"cell_type", "hemibrain_type"} <= set(cls.columns):
        types["cell_type"] = cls["cell_type"]
        types["hemibrain_type"] = cls["hemibrain_type"]
    else:
        t_path = raw / "fw_and_hemibrain_types.csv.gz"
        t = _read(t_path).drop_duplicates("root_id").set_index("root_id").reindex(ids)
        types["cell_type"] = t["cell_type"]
        types["hemibrain_type"] = t["hemibrain_type"]
        inputs.append(t_path)

    cell_type = types["cell_type"].where(types["cell_type"].notna(), types["hemibrain_type"])
    neurons = pd.DataFrame(
        {
            "dataset": DATASET,
            "version": version,
            "neuron_id": ids,
            "flow": cls["flow"].to_numpy(),
            "super_class_raw": cls["super_class"].to_numpy(),
            "super_class": cls["super_class"]
            .map(lambda v: normalize_super_class("codex_fafb", v))
            .to_numpy(),
            "cell_class": cls["class"].to_numpy(),
            "cell_sub_class": cls["sub_class"].to_numpy(),
            "cell_type": cell_type.to_numpy(),
            "hemibrain_121_cell_type": types["hemibrain_type"].to_numpy(),
            "hemilineage": cls["hemilineage"].to_numpy(),
            "side": cls["side"].map(normalize_side).to_numpy(),
            "nt_pred": nt["nt_type"].map(normalize_nt).to_numpy(),
            "nt_conf": nt["nt_type_score"].to_numpy(dtype=float),
            "nt_source": f"codex_{version}_neurons_csv",
        }
    )
    for col, name in _NT_AVG.items():
        if col in nt.columns:
            neurons[f"nt_p_{name}"] = nt[col].to_numpy(dtype=np.float32)
    return neurons, inputs


def _edges(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conns = pd.read_csv(
        raw / "connections.csv.gz",
        dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"},
    )
    roi = pd.DataFrame(
        {
            "pre": conns["pre_root_id"],
            "post": conns["post_root_id"],
            "weight": conns["syn_count"],
            "roi": conns["neuropil"],
        }
    )
    pairs = (
        roi.groupby(["pre", "post"], sort=True, as_index=False)["weight"]
        .sum()
        .astype({"weight": "int32"})
    )
    return pairs, roi


def convert_flywire(raw_dir: Path, out_dir: Path, *, version: str) -> dict[str, object]:
    """Convert Codex exports for FlyWire snapshot ``version`` into the harmonized store."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons, inputs = _neurons(raw_dir, version)
    write_parquet(conform_neurons(neurons), out_dir / "neurons.parquet")
    counts: dict[str, int] = {"neurons": len(neurons)}
    extra: dict[str, object] = {"nt_source": f"codex_{version}_neurons_csv"}

    conn_path = raw_dir / "connections.csv.gz"
    if conn_path.exists():
        pairs, roi = _edges(raw_dir)
        for df in (pairs, roi):
            df.insert(0, "version", version)
            df.insert(0, "dataset", DATASET)
        write_parquet(conform_edges(pairs), out_dir / "edges.parquet")
        write_parquet(conform_edges(roi), out_dir / "edges_roi.parquet")
        counts.update(edges=len(pairs), edge_roi_rows=len(roi))
        inputs.append(conn_path)
        extra["edge_threshold"] = {"pair_min_synapses": 5, "synapse_source": "buhmann"}

    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=version,
        converter="flyconn.data.convert.flywire",
        inputs=inputs,
        counts=counts,
        **extra,
    )
