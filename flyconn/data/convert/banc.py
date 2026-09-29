"""BANC v888 converter (brain and nerve cord, female; Bates et al. 2026, CC BY 4.0).

Inputs: ``banc_888_meta.feather`` (188,508 annotated segments x 81 columns, string ids) and
``banc_888_edgelist_simple_v2.feather`` (pre, post, count, norm, post_count, pre_count; synapse
detector v2 as in the paper, no pair threshold). Neuron universe: segments marked proofread or
roughly proofread, excluding glia, trachea and ``not_a_neuron`` (155,858 in the pinned file; the
paper reports 155,916 proofread plus roughly proofread). Autapses present in the file are dropped
and counted.
"""

from __future__ import annotations

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

DATASET = "banc"
VERSION = "888"
META = "banc_888_meta.feather"
EDGES = "banc_888_edgelist_simple_v2.feather"
NT_SOURCE = "banc_888_meta"
NON_NEURONS = ("glia", "trachea", "not_a_neuron")
NEURON_DEFINITION = "proofread or roughly_proofread, excluding glia/trachea/not_a_neuron"

_META_COLUMNS = [
    "banc_888_id",
    "proofread",
    "roughly_proofread",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "side",
    "hemilineage",
    "region",
    "flow",
    "neurotransmitter_predicted",
    "neurotransmitter_score",
    "fafb_cell_type",
    "manc_cell_type",
    "malecns_cell_type",
    "hemibrain_cell_type",
    "sexually_dimorphic",
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return cast("pd.Series", df[name])
    return pd.Series([None] * len(df), dtype=object)


def _super_class(v: object) -> str | None:
    return normalize_super_class("banc", v)


def _is_auto(v: object) -> bool:
    return isinstance(v, str) and v.startswith("auto:")


def _strip_auto(v: object) -> str | None:
    """BANC marks automatic (NBLAST) hemibrain matches with an ``auto:`` prefix."""
    return v.removeprefix("auto:") if isinstance(v, str) else None


def _neurons(raw: Path) -> pd.DataFrame:
    schema_names = set(pa.ipc.open_file(raw / META).schema.names)
    cols = [c for c in _META_COLUMNS if c in schema_names]
    m = pf.read_table(raw / META, columns=cols).to_pandas()
    marked = (_col(m, "proofread") == "TRUE") | (_col(m, "roughly_proofread") == "TRUE")
    neuron = ~_col(m, "super_class").isin(NON_NEURONS)
    m = cast("pd.DataFrame", m[(marked & neuron).to_numpy()]).reset_index(drop=True)
    hb = _col(m, "hemibrain_cell_type").astype(object)
    hb_auto = hb.map(_is_auto)
    hb_clean = hb.map(_strip_auto)
    status = np.where(_col(m, "proofread") == "TRUE", "proofread", "roughly_proofread")
    return pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "neuron_id": _col(m, "banc_888_id").astype("int64"),
            "status": status,
            "flow": _col(m, "flow"),
            "super_class_raw": _col(m, "super_class"),
            "super_class": _col(m, "super_class").map(_super_class),
            "cell_class": _col(m, "cell_class"),
            "cell_sub_class": _col(m, "cell_sub_class"),
            "cell_type": _col(m, "cell_type"),
            "hemilineage": _col(m, "hemilineage"),
            "side": _col(m, "side").map(normalize_side),
            "region": _col(m, "region"),
            "nt_pred": _col(m, "neurotransmitter_predicted").map(normalize_nt),
            "nt_conf": pd.to_numeric(_col(m, "neurotransmitter_score"), errors="coerce"),
            "nt_source": NT_SOURCE,
            "fafb_783_cell_type": _col(m, "fafb_cell_type"),
            "manc_121_cell_type": _col(m, "manc_cell_type"),
            "hemibrain_121_cell_type": hb_clean,
            "hemibrain_121_match_auto": hb_auto,
            "malecns_09_cell_type": _col(m, "malecns_cell_type"),
            "dimorphism": _col(m, "sexually_dimorphic"),
        }
    )


def convert_banc(raw_dir: Path, out_dir: Path) -> dict[str, object]:
    """Convert BANC v888 compiled data in ``raw_dir`` into the harmonized store ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons = _neurons(raw_dir)
    inputs = [raw_dir / META]
    counts: dict[str, int] = {"neurons": len(neurons)}
    extra: dict[str, object] = {"neuron_definition": NEURON_DEFINITION, "nt_source": NT_SOURCE}

    edge_path = raw_dir / EDGES
    if edge_path.exists():
        t = pf.read_table(edge_path, columns=["pre", "post", "count", "post_count", "pre_count"])
        pre = pc.cast(t.column("pre"), pa.int64()).to_numpy()
        post = pc.cast(t.column("post"), pa.int64()).to_numpy()
        w = t.column("count").to_numpy().astype(np.int64)
        post_count = t.column("post_count").to_numpy().astype(np.int64)
        pre_count = t.column("pre_count").to_numpy().astype(np.int64)
        index = pd.Index(neurons["neuron_id"].to_numpy())
        pi, qi = index.get_indexer(pre), index.get_indexer(post)
        n = len(index)
        # Per-neuron totals as released (include partners outside the neuron universe).
        in_tot = np.zeros(n, dtype=np.int64)
        out_tot = np.zeros(n, dtype=np.int64)
        in_tot[qi[qi >= 0]] = post_count[qi >= 0]
        out_tot[pi[pi >= 0]] = pre_count[pi >= 0]
        neurons["input_synapses_total"] = in_tot
        neurons["output_synapses_total"] = out_tot
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
        extra["edge_threshold"] = {"pair_min_synapses": 1, "synapse_source": "banc_v2_size5"}

    write_parquet(conform_neurons(neurons), out_dir / "neurons.parquet")
    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=VERSION,
        converter="flyconn.data.convert.banc",
        inputs=inputs,
        counts=counts,
        **extra,
    )
