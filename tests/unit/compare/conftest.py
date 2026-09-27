"""Two synthetic 'datasets' sharing cell types, with left/right copies, for compare tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store


def _make(tmp_path: Path, name: str, version: str, seed: int, perturb_type: str | None) -> Store:
    """8 types x 2 sides x 3 neurons; type-level wiring identical except for perturb_type."""
    rng = np.random.default_rng(seed)
    types = [f"T{i}" for i in range(8)]
    rows = []
    nid = 1
    for t in types:
        for side in ("left", "right"):
            for _ in range(3):
                rows.append(
                    {
                        "neuron_id": nid,
                        "cell_type": t,
                        "side": side,
                        "super_class": "central_brain_intrinsic",
                        "nt_pred": "acetylcholine",
                        "nt_conf": 0.9,
                    }
                )
                nid += 1
    neurons = pd.DataFrame(rows)
    # type-level template: T_i -> T_{i+1} strong, T_i -> T_{i+2} weak (same in both datasets)
    tmpl = {}
    for i in range(8):
        tmpl[(types[i], types[(i + 1) % 8])] = 30
        tmpl[(types[i], types[(i + 2) % 8])] = 8
    if perturb_type:
        tmpl[(perturb_type, types[(types.index(perturb_type) + 1) % 8])] = 2
        tmpl[(perturb_type, types[(types.index(perturb_type) + 5) % 8])] = 40
    e = []
    by_type = neurons.groupby(["cell_type", "side"])["neuron_id"].apply(list)
    for (a, b), w in tmpl.items():
        for side in ("left", "right"):
            for pre in by_type[(a, side)]:
                for post in by_type[(b, side)]:
                    e.append({"pre": pre, "post": post, "weight": int(max(1, rng.poisson(w)))})
    edges = pd.DataFrame(e)
    if name == "malecns":  # cross-reference column as MaleCNS publishes it
        neurons["fafb_783_cell_type"] = neurons["cell_type"]
        # a male-only partner type (e.g. a VNC neuron) with no FlyWire counterpart
        extra = pd.DataFrame(
            {
                "neuron_id": [901, 902],
                "cell_type": ["T8", "T8"],
                "side": ["left", "right"],
                "super_class": "ventral_nerve_cord_intrinsic",
                "nt_pred": "acetylcholine",
                "nt_conf": 0.9,
                "fafb_783_cell_type": [None, None],
            }
        )
        neurons = pd.concat([neurons, extra], ignore_index=True)
        t0 = neurons[neurons["cell_type"] == "T0"]
        more = [
            {"pre": int(pre), "post": 901 if side == "left" else 902, "weight": 20}
            for pre, side in zip(t0["neuron_id"], t0["side"], strict=True)
        ]
        edges = pd.concat([edges, pd.DataFrame(more)], ignore_index=True)
    neurons["dataset"] = edges["dataset"] = name
    neurons["version"] = edges["version"] = version
    d = tmp_path / name
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(d, dataset=name, version=version, converter="t", inputs=[], counts={})
    return Store(d)


@pytest.fixture
def male_female(tmp_path: Path) -> tuple[Store, Store]:
    male = _make(tmp_path, "malecns", "1.0", seed=1, perturb_type="T3")
    female = _make(tmp_path, "flywire", "783", seed=2, perturb_type=None)
    return male, female
