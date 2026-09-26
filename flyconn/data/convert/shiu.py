"""Converter for the Shiu et al. 2024 model inputs (unthresholded FlyWire v630 export).

Files: ``2023_03_23_completeness_630_final.csv`` (one row per neuron; first column
is the root id) and ``2023_03_23_connectivity_630_final.parquet`` with columns
``Presynaptic_ID, Postsynaptic_ID, Presynaptic_Index, Postsynaptic_Index,
Connectivity, Excitatory, Excitatory x Connectivity``. Signs are pre-baked per
presynaptic neuron (+1 excitatory, -1 inhibitory = GABA/Glu majority). We keep
the sign as an edge column and record a coarse ``nt_pred`` placeholder
(``acetylcholine`` for +1, ``gaba`` for -1) so that :class:`SignPolicy.ARGMAX`
reproduces the model's signs exactly; the true transmitter is not in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons

DATASET = "shiu"
VERSION = "630"
COMPLETENESS = "2023_03_23_completeness_630_final.csv"
CONNECTIVITY = "2023_03_23_connectivity_630_final.parquet"
NT_SOURCE = "shiu2024_excitatory_flag"


def convert_shiu(raw_dir: Path, out_dir: Path) -> dict[str, object]:
    """Convert the Shiu model inputs in ``raw_dir`` into the harmonized store ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(raw_dir / COMPLETENESS)
    ids = cast("pd.Series", comp.iloc[:, 0]).to_numpy(dtype=np.int64)
    conn = pd.read_parquet(
        raw_dir / CONNECTIVITY,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"],
    )
    pre = cast("pd.Series", conn["Presynaptic_ID"]).to_numpy(dtype=np.int64)
    post = cast("pd.Series", conn["Postsynaptic_ID"]).to_numpy(dtype=np.int64)
    weight = cast("pd.Series", conn["Connectivity"]).to_numpy(dtype=np.int64)
    sign = cast("pd.Series", conn["Excitatory"]).to_numpy(dtype=np.int8)

    # Per-neuron sign from its outgoing edges (constant per presynaptic neuron by construction).
    sign_by_pre = pd.Series(sign).groupby(pre).first()
    neuron_sign = sign_by_pre.reindex(ids)
    nt_pred = neuron_sign.map({1: "acetylcholine", -1: "gaba"}).to_numpy(dtype=object)
    neurons = pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "neuron_id": ids,
            "nt_pred": nt_pred,
            "nt_conf": np.where(pd.isna(nt_pred), np.nan, 1.0),
            "nt_source": NT_SOURCE,
        }
    )
    write_parquet(conform_neurons(neurons), out_dir / "neurons.parquet")

    edges = pd.DataFrame(
        {"dataset": DATASET, "version": VERSION, "pre": pre, "post": post, "weight": weight}
    )
    order = np.lexsort((post, pre))
    edges = edges.iloc[order].reset_index(drop=True)
    table = conform_edges(edges).append_column("sign", pa.array(sign[order], type=pa.int8()))
    write_parquet(table, out_dir / "edges.parquet")

    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=VERSION,
        converter="flyconn.data.convert.shiu",
        inputs=[raw_dir / COMPLETENESS, raw_dir / CONNECTIVITY],
        counts={"neurons": len(ids), "edges": len(edges), "synapses": int(weight.sum())},
        edge_threshold={"pair_min_synapses": 1, "synapse_source": "buhmann_cleft50"},
        nt_source=NT_SOURCE,
        note="signs pre-baked by Shiu et al. (GABA/Glu inhibitory); nt_pred is a placeholder",
    )
