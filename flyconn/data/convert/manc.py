"""MANC v1.2.1 converter (male adult nerve cord; Takemura et al. 2024, Marin et al. 2024, CC BY).

No flat edge export exists for MANC v1.2.x, so the converter reads the sjcabs compiled data
(``gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/manc_121/``, built by
flyconnectome/bancpipeline; data CC BY by the dataset creators):

* ``manc_121_meta.feather`` (23,650 rows, string ids): the neuron universe (MANC v1.2.1 Traced
  bodies minus 15 glia), BANC-vocabulary ``super_class``, neuPrint ``type`` as ``cell_type``,
  Truman hemilineages, 3-class transmitter predictions (ACh/GABA/Glu + unknown/unclear).
* ``manc_121_simple_edgelist.feather`` (``pre, post, count, norm, total_input``): traced to
  traced connection counts with no pair threshold; ``total_input`` is the sum of ``count``
  over all traced presynaptic partners (glia included).
* Codex ``neurons.csv.gz`` (optional): the 23,665 traced bodies; supplies the transmitter
  confidence (only where Codex's label agrees) and ``vfbId``.

MANC names are themselves the ``manc_121_cell_type`` vocabulary, so that column equals
``cell_type`` and cross-dataset partner vocabularies (``fafb_or_manc``) work.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import (
    conform_edges,
    conform_neurons,
    normalize_nt,
    normalize_side,
    normalize_super_class,
)

DATASET = "manc"
VERSION = "1.2.1"
META = "manc_121_meta.feather"
EDGES = "manc_121_simple_edgelist.feather"
CODEX = "neurons.csv.gz"
NT_SOURCE = "manc_121_meta (MANC v1.2.1 3-class classifier, via sjcabs)"
NEURON_DEFINITION = "sjcabs manc_121_meta rows (MANC v1.2.1 Traced bodies excluding glia)"
_VFB = re.compile(r"vfbId::([^,]+)")


def _sjcabs_super_class(v: object) -> str | None:
    return normalize_super_class("sjcabs", v)


def _obj(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), None)


def _neurons(raw: Path) -> pd.DataFrame:
    m = pf.read_table(raw / META).to_pandas()
    sc = _obj(cast("pd.Series", m["super_class"]))
    ctype = _obj(cast("pd.Series", m["cell_type"]))
    return pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "neuron_id": m["manc_121_id"].astype("int64"),
            "status": "Traced",
            "flow": _obj(cast("pd.Series", m["flow"])),
            "super_class_raw": sc,
            "super_class": sc.map(_sjcabs_super_class),
            "cell_class": _obj(cast("pd.Series", m["cell_class"])),
            "cell_sub_class": _obj(cast("pd.Series", m["cell_sub_class"])),
            "cell_type": ctype,
            "hemilineage_truman": _obj(cast("pd.Series", m["hemilineage"])),
            "side": m["side"].map(normalize_side),
            "region": _obj(cast("pd.Series", m["region"])),
            "nerve": _obj(cast("pd.Series", m["nerve"])),
            "nt_pred": m["neurotransmitter_predicted"].map(normalize_nt),
            "nt_source": NT_SOURCE,
            "manc_121_cell_type": ctype,
        }
    )


def _vfb(labels: object) -> str | None:
    if not isinstance(labels, str):
        return None
    hit = _VFB.search(labels)
    return hit.group(1) if hit else None


def _add_codex(neurons: pd.DataFrame, path: Path, counts: dict[str, int]) -> None:
    c = pd.read_csv(
        path,
        dtype={"Root ID": "int64"},
        low_memory=False,
    ).set_index("Root ID")
    ids = neurons["neuron_id"].to_numpy()
    sub = c.reindex(ids)
    codex_nt = sub["Predicted NT type"].map(normalize_nt).to_numpy(dtype=object)
    conf = sub["Predicted NT confidence"].to_numpy(dtype=float)
    agree = neurons["nt_pred"].to_numpy(dtype=object) == codex_nt
    neurons["nt_conf"] = np.where(agree & neurons["nt_pred"].notna().to_numpy(), conf, np.nan)
    neurons["vfb_id"] = sub["Community labels"].map(_vfb).to_numpy(dtype=object)
    glia = cast("pd.Series", c["Super Class"]) == "glia"
    counts["traced_codex"] = len(c)
    counts["glia_excluded"] = int((glia & ~c.index.isin(ids)).sum())
    counts["codex_bodies_missing_from_meta"] = int((~c.index.isin(ids)).sum())


def convert_manc(raw_dir: Path, out_dir: Path) -> dict[str, object]:
    """Convert MANC v1.2.1 compiled files in ``raw_dir`` into the harmonized store ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons = _neurons(raw_dir)
    inputs = [raw_dir / META]
    counts: dict[str, int] = {"neurons": len(neurons)}
    extra: dict[str, object] = {
        "neuron_definition": NEURON_DEFINITION,
        "nt_source": NT_SOURCE,
        "nt_classes": ["acetylcholine", "gaba", "glutamate"],
    }
    codex_path = raw_dir / CODEX
    if codex_path.exists():
        _add_codex(neurons, codex_path, counts)
        inputs.append(codex_path)

    edge_path = raw_dir / EDGES
    if edge_path.exists():
        t = pf.read_table(edge_path, columns=["pre", "post", "count", "total_input"])
        pre = pc.cast(t.column("pre"), pa.int64()).to_numpy()
        post = pc.cast(t.column("post"), pa.int64()).to_numpy()
        w = t.column("count").to_numpy().astype(np.int64)
        total_input = t.column("total_input").to_numpy().astype(np.int64)
        index = pd.Index(neurons["neuron_id"].to_numpy())
        pi, qi = index.get_indexer(pre), index.get_indexer(post)
        n = len(index)
        in_tot = np.zeros(n, dtype=np.int64)
        in_tot[qi[qi >= 0]] = total_input[qi >= 0]
        neurons["input_synapses_total"] = in_tot
        neurons["output_synapses_total"] = np.bincount(pi[pi >= 0], w[pi >= 0], n).astype(np.int64)
        autapse = pre == post
        keep = (pi >= 0) & (qi >= 0) & ~autapse
        edges = pd.DataFrame(
            {
                "dataset": DATASET,
                "version": VERSION,
                "pre": pre[keep],
                "post": post[keep],
                "weight": w[keep],
            }
        ).sort_values(["pre", "post"], ignore_index=True)
        write_parquet(conform_edges(edges), out_dir / "edges.parquet")
        counts.update(
            raw_edges=len(w),
            raw_weight_sum=int(w.sum()),
            autapses_dropped=int(autapse.sum()),
            edges=len(edges),
        )
        inputs.append(edge_path)
        extra["edge_threshold"] = {
            "pair_min_synapses": 1,
            "synapse_source": "sjcabs manc_121_simple_edgelist (traced to traced)",
            "note": "confidence threshold not documented by sjcabs (neuPrint manc:v1.2.1 uses "
            "postHighAccuracyThreshold 0.4); rows and synapse sum differ slightly from neuPrint "
            "Traced->Traced (5,305,638 edges / 30,934,610); totals count traced partners only "
            "(glia included)",
        }

    write_parquet(conform_neurons(neurons), out_dir / "neurons.parquet")
    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=VERSION,
        converter="flyconn.data.convert.manc",
        inputs=inputs,
        counts=counts,
        **extra,
    )
