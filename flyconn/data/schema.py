"""Harmonized schema and vocabulary normalisers.

Column vocabulary follows the SJCABS/BANC compiled-data convention (ADR-0001)
so that flyconn tables round-trip with bancr, coconatfly and the
fly_connectome_data_tutorial bundles. Source-specific spellings are kept in
``*_raw`` columns where they carry information.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pyarrow as pa

NT_CLASSES: tuple[str, ...] = (
    "acetylcholine",
    "glutamate",
    "gaba",
    "dopamine",
    "octopamine",
    "serotonin",
    "histamine",
    "tyramine",
)
"""Union of transmitter classes across sources (FlyWire 6, MaleCNS 7, BANC 8), fixed order."""

_NT_ALIASES: Mapping[str, str] = {
    "ach": "acetylcholine",
    "acetylcholine": "acetylcholine",
    "cholinergic": "acetylcholine",
    "gaba": "gaba",
    "gabaergic": "gaba",
    "glut": "glutamate",
    "glu": "glutamate",
    "glutamate": "glutamate",
    "glutamatergic": "glutamate",
    "da": "dopamine",
    "dopamine": "dopamine",
    "dopaminergic": "dopamine",
    "ser": "serotonin",
    "5ht": "serotonin",
    "5-ht": "serotonin",
    "serotonin": "serotonin",
    "serotonergic": "serotonin",
    "oct": "octopamine",
    "oa": "octopamine",
    "octopamine": "octopamine",
    "octopaminergic": "octopamine",
    "hist": "histamine",
    "his": "histamine",
    "histamine": "histamine",
    "histaminergic": "histamine",
    "tyr": "tyramine",
    "ta": "tyramine",
    "tyramine": "tyramine",
    "tyraminergic": "tyramine",
}
_NT_UNKNOWN = {"", "unclear", "unknown", "neither", "none", "nan", "na"}


def normalize_nt(raw: object) -> str | None:
    """Map a source neurotransmitter label to a canonical lower-case name (or ``None``)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    key = str(raw).strip().lower()
    if key in _NT_UNKNOWN:
        return None
    try:
        return _NT_ALIASES[key]
    except KeyError:
        msg = f"unrecognised neurotransmitter label {raw!r}"
        raise ValueError(msg) from None


# BANC / SJCABS super_class vocabulary is the target.
_SUPER_CLASS_MAPS: Mapping[str, Mapping[str, str]] = {
    "codex_fafb": {
        "central": "central_brain_intrinsic",
        "optic": "optic_lobe_intrinsic",
    },
    "malecns": {
        "cb_intrinsic": "central_brain_intrinsic",
        "ol_intrinsic": "optic_lobe_intrinsic",
        "vnc_intrinsic": "ventral_nerve_cord_intrinsic",
        "cb_sensory": "sensory",
        "ol_sensory": "sensory",
        "vnc_sensory": "sensory",
        "ascending_neuron": "ascending",
        "descending_neuron": "descending",
        "vnc_motor": "motor",
        "cb_motor": "motor",
        "cb_endocrine": "endocrine",
        "vnc_endocrine": "endocrine",
        "vnc_efferent": "efferent",
        "cb_efferent": "efferent",
        "efferent_ascending": "efferent_ascending",
        "efferent_descending": "efferent_descending",
    },
    "manc": {
        "intrinsic neuron": "ventral_nerve_cord_intrinsic",
        "intrinsic_neuron": "ventral_nerve_cord_intrinsic",
        "sensory neuron": "sensory",
        "ascending neuron": "ascending",
        "descending neuron": "descending",
        "motor neuron": "motor",
        "efferent neuron": "efferent",
        "sensory ascending": "sensory_ascending",
        "efferent ascending": "efferent_ascending",
    },
    "banc": {},
    "sjcabs": {},
}


def normalize_super_class(source: str, raw: object) -> str | None:
    """Map a source ``super_class`` to the BANC vocabulary; unknown values pass through."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    key = str(raw).strip()
    if not key:
        return None
    try:
        table = _SUPER_CLASS_MAPS[source]
    except KeyError:
        msg = f"unknown super_class source {source!r}; known: {sorted(_SUPER_CLASS_MAPS)}"
        raise KeyError(msg) from None
    return table.get(key, table.get(key.lower(), key))


_SIDES: Mapping[str, str] = {
    "l": "left",
    "left": "left",
    "lhs": "left",
    "r": "right",
    "right": "right",
    "rhs": "right",
    "m": "center",
    "mid": "center",
    "midline": "center",
    "middle": "center",
    "center": "center",
    "centre": "center",
    "bil": "center",
}


def normalize_side(raw: object) -> str | None:
    """Map side spellings (L/R/M, LHS/RHS, left/right/center) to left/right/center."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return _SIDES.get(str(raw).strip().lower())


NEURON_REQUIRED: tuple[str, ...] = ("dataset", "version", "neuron_id")
EDGE_REQUIRED: tuple[str, ...] = ("dataset", "version", "pre", "post", "weight")

NEURON_FIELDS: list[pa.Field] = [
    pa.field("dataset", pa.string()),
    pa.field("version", pa.string()),
    pa.field("neuron_id", pa.int64()),
    pa.field("status", pa.string()),
    pa.field("flow", pa.string()),
    pa.field("super_class", pa.string()),
    pa.field("super_class_raw", pa.string()),
    pa.field("cell_class", pa.string()),
    pa.field("cell_sub_class", pa.string()),
    pa.field("cell_type", pa.string()),
    pa.field("hemilineage", pa.string()),
    pa.field("hemilineage_truman", pa.string()),
    pa.field("side", pa.string()),
    pa.field("soma_neuromere", pa.string()),
    pa.field("nt_pred", pa.string()),
    pa.field("nt_conf", pa.float64()),
    *[pa.field(f"nt_p_{nt}", pa.float32()) for nt in NT_CLASSES],
    pa.field("nt_source", pa.string()),
    pa.field("nt_known", pa.string()),
    pa.field("fafb_783_cell_type", pa.string()),
    pa.field("hemibrain_121_cell_type", pa.string()),
    pa.field("manc_121_cell_type", pa.string()),
    pa.field("malecns_10_cell_type", pa.string()),
    pa.field("dimorphism", pa.string()),
    pa.field("vfb_id", pa.string()),
]
NEURONS_SCHEMA = pa.schema(NEURON_FIELDS)

EDGE_FIELDS: list[pa.Field] = [
    pa.field("dataset", pa.string()),
    pa.field("version", pa.string()),
    pa.field("pre", pa.int64()),
    pa.field("post", pa.int64()),
    pa.field("weight", pa.int32()),
    pa.field("roi", pa.string()),
]
EDGES_SCHEMA = pa.schema(EDGE_FIELDS)


def _conform(df: pd.DataFrame, schema: pa.Schema, required: tuple[str, ...]) -> pa.Table:
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = f"missing required columns: {missing}"
        raise ValueError(msg)
    table = pa.Table.from_pandas(df, preserve_index=False)
    columns: list[pa.Array | pa.ChunkedArray] = []
    for f in schema:
        if f.name in table.column_names:
            columns.append(table.column(f.name).cast(f.type))
        else:
            columns.append(pa.nulls(table.num_rows, type=f.type))
    extras = [c for c in table.column_names if c not in schema.names]
    out = pa.table(columns, schema=schema)
    for name in extras:
        out = out.append_column(name, table.column(name))
    return out


def conform_neurons(df: pd.DataFrame) -> pa.Table:
    """Cast a neuron table to :data:`NEURONS_SCHEMA`, adding missing optional columns as null."""
    return _conform(df, NEURONS_SCHEMA, NEURON_REQUIRED)


def conform_edges(df: pd.DataFrame) -> pa.Table:
    """Cast an edge table to :data:`EDGES_SCHEMA`, adding ``roi`` as null when absent."""
    return _conform(df, EDGES_SCHEMA, EDGE_REQUIRED)
