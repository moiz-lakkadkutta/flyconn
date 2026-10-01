"""Hemibrain v1.2.1 converter (Janelia FlyEM; Scheffer et al. 2020, CC BY).

Inputs:

* ``exported-traced-adjacencies-v1.2.tar.gz`` (required): ``traced-neurons.csv`` (21,739
  non-cropped Traced bodies; ``bodyId, type, instance``) defines the neuron universe and
  ``traced-total-connections.csv`` (``bodyId_pre, bodyId_post, weight``) the edges. The tarball
  is the v1.2 export; neuPrint serves v1.2.1, which differs by 342 edges / 4,077 synapses.
  ``weight`` counts all PSDs (neuPrint ``postHighAccuracyThreshold`` is 0 for this dataset).
* ``Supplemental_file5_hemibrain_meta.csv`` (Schlegel et al. 2024; optional): side,
  ItoLee hemilineage, short ``cell_class``, ``morphology_type``, ``fbbt_id``.
* ``hemibrain_121_meta.feather`` (sjcabs compiled data; optional): ``super_class``, ``flow``,
  ``region`` in the BANC vocabulary.
* ``hemibrain-v1.2-body-mean-neurotransmitters.feather`` (optional): per-body mean of six
  transmitter probabilities plus ``neither``. Its provenance is UNVERIFIED (no README).

No FlyWire cross-reference is filled: none of these files publishes one.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow.feather as pf

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import (
    conform_edges,
    conform_neurons,
    normalize_nt,
    normalize_side,
    normalize_super_class,
)

DATASET = "hemibrain"
VERSION = "1.2.1"
TARBALL = "exported-traced-adjacencies-v1.2.tar.gz"
_TAR_DIR = "exported-traced-adjacencies-v1.2"
SUPP5 = "Supplemental_file5_hemibrain_meta.csv"
SJCABS_META = "hemibrain_121_meta.feather"
NT_FEATHER = "hemibrain-v1.2-body-mean-neurotransmitters.feather"
NT_SOURCE = "hemibrain_v1.2_body_mean (provenance UNVERIFIED)"
NEURON_DEFINITION = "traced-neurons.csv of the v1.2 export (Traced, not cropped)"
_NT_CLASSES = ("gaba", "acetylcholine", "glutamate", "serotonin", "octopamine", "dopamine")
_NT_ALL = (*_NT_CLASSES, "neither")


def _sjcabs_super_class(v: object) -> str | None:
    return normalize_super_class("sjcabs", v)


def _read_member(tar: tarfile.TarFile, name: str) -> pd.DataFrame:
    fh = tar.extractfile(f"{_TAR_DIR}/{name}")
    if fh is None:
        msg = f"{name} missing from {TARBALL}"
        raise FileNotFoundError(msg)
    with fh:
        return pd.read_csv(fh)


def _lookup(table: pd.DataFrame, ids: np.ndarray, column: str) -> pd.Series:
    """``table[column]`` (indexed by neuron id) aligned to ``ids``; missing rows are null."""
    s = cast("pd.Series", table[column]).reindex(ids)
    return s.astype(object).where(s.notna(), None).reset_index(drop=True)


def _annotate(neurons: pd.DataFrame, raw: Path, counts: dict[str, int], inputs: list[Path]) -> None:
    ids = neurons["neuron_id"].to_numpy()
    supp5_path = raw / SUPP5
    if supp5_path.exists():
        s5 = pd.read_csv(
            supp5_path,
            dtype={"bodyId": "int64"},
        ).set_index("bodyId")
        neurons["side"] = _lookup(s5, ids, "side").map(normalize_side)
        neurons["hemilineage"] = _lookup(s5, ids, "ito_lee_hemilineage")
        neurons["cell_class"] = _lookup(s5, ids, "cell_class")
        neurons["morphology_type"] = _lookup(s5, ids, "morphology_type")
        neurons["fbbt_id"] = _lookup(s5, ids, "fbbt_id")
        counts["with_supp5"] = int(np.isin(ids, s5.index.to_numpy()).sum())
        inputs.append(supp5_path)

    sj_path = raw / SJCABS_META
    if sj_path.exists():
        sj = pf.read_table(
            sj_path, columns=["hemibrain_121_id", "super_class", "flow", "region"]
        ).to_pandas()
        sj.index = pd.Index(sj["hemibrain_121_id"].astype("int64"))
        raw_sc = _lookup(sj, ids, "super_class")
        neurons["super_class_raw"] = raw_sc
        neurons["super_class"] = raw_sc.map(_sjcabs_super_class)
        neurons["flow"] = _lookup(sj, ids, "flow")
        neurons["region"] = _lookup(sj, ids, "region")
        counts["with_sjcabs_meta"] = int(np.isin(ids, sj.index.to_numpy()).sum())
        inputs.append(sj_path)

    nt_path = raw / NT_FEATHER
    if nt_path.exists():
        nt = pf.read_table(nt_path, columns=["body", *_NT_ALL, "predicted_nt"]).to_pandas(
            ignore_metadata=True  # the release stores "body" as the pandas index
        )
        nt = nt.set_index("body").reindex(ids)
        has = nt["predicted_nt"].notna().to_numpy()
        for c in _NT_CLASSES:
            neurons[f"nt_p_{c}"] = nt[c].to_numpy(dtype=np.float32)
        neurons["nt_p_neither"] = nt["neither"].to_numpy(dtype=np.float32)
        neurons["nt_pred"] = nt["predicted_nt"].map(normalize_nt).to_numpy(dtype=object)
        neurons["nt_conf"] = nt[list(_NT_ALL)].max(axis=1).to_numpy(dtype=float)
        neurons["nt_source"] = [NT_SOURCE if h else None for h in np.asarray(has, dtype=bool)]
        counts["with_nt"] = int(has.sum())
        inputs.append(nt_path)


def convert_hemibrain(raw_dir: Path, out_dir: Path) -> dict[str, object]:
    """Convert hemibrain v1.2.1 release files in ``raw_dir`` into the store ``out_dir``."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / TARBALL
    with tarfile.open(tar_path, "r:gz") as tar:
        traced = _read_member(tar, "traced-neurons.csv")
        total = _read_member(tar, "traced-total-connections.csv")
    ctype = traced["type"].astype(object).where(traced["type"].notna(), None)
    neurons = pd.DataFrame(
        {
            "dataset": DATASET,
            "version": VERSION,
            "neuron_id": traced["bodyId"].astype("int64"),
            "status": "Traced",
            "cell_type": ctype,
            "instance": traced["instance"].astype(object).where(traced["instance"].notna(), None),
            "hemibrain_121_cell_type": ctype,
        }
    )
    inputs = [tar_path]
    counts: dict[str, int] = {"neurons": len(neurons)}
    _annotate(neurons, raw_dir, counts, inputs)

    pre = total["bodyId_pre"].to_numpy()
    post = total["bodyId_post"].to_numpy()
    w = total["weight"].to_numpy(dtype=np.int64)
    index = pd.Index(neurons["neuron_id"].to_numpy())
    autapse = pre == post
    keep = (index.get_indexer(pre) >= 0) & (index.get_indexer(post) >= 0) & ~autapse
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

    write_parquet(conform_neurons(neurons), out_dir / "neurons.parquet")
    return write_provenance(
        out_dir,
        dataset=DATASET,
        version=VERSION,
        converter="flyconn.data.convert.hemibrain",
        inputs=inputs,
        counts=counts,
        neuron_definition=NEURON_DEFINITION,
        export_version="v1.2 (neuPrint hemibrain:v1.2.1 differs by 342 edges / 4,077 synapses)",
        edge_threshold={
            "pair_min_synapses": 1,
            "synapse_source": "neuprint_weight",
            "postHighAccuracyThreshold": 0,
            "note": "weights count all PSDs incl. low-confidence ones; neuPrint weightHP "
            "(post confidence >= 0.7) is not in the export",
        },
        side_source=SUPP5 + " (Schlegel et al. 2024)",
        nt_probs={
            "source_file": NT_FEATHER,
            "classes": [*_NT_CLASSES, "neither"],
            "provenance": "UNVERIFIED: uploaded to gs://hemibrain/v1.2/ on 2026-05-15 without "
            "a README; class set matches the Eckstein et al. 2024 classifier",
        },
    )
