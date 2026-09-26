"""A small deterministic Store for graph tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store
from flyconn.testing import synthetic_connectome


@pytest.fixture
def small_store(tmp_path: Path) -> Store:
    neurons, edges = synthetic_connectome(n_neurons=40, n_edges=200, seed=21)
    neurons["dataset"] = edges["dataset"] = "malecns"
    neurons["version"] = edges["version"] = "1.0"
    # Force a known layout for sign tests: first four neurons get fixed NT calls.
    neurons.loc[0, "nt_pred"] = "acetylcholine"
    neurons.loc[1, "nt_pred"] = "gaba"
    neurons.loc[2, "nt_pred"] = "histamine"
    neurons.loc[3, "nt_pred"] = None
    neurons.loc[3, "nt_conf"] = np.nan
    # Give neuron 0 a clean probability vector: 0.6 ACh, 0.4 GABA.
    for col in [c for c in neurons.columns if c.startswith("nt_p_")]:
        neurons.loc[0, col] = 0.0
    neurons.loc[0, "nt_p_acetylcholine"] = 0.6
    neurons.loc[0, "nt_p_gaba"] = 0.4
    neurons["input_synapses_total"] = (
        edges.groupby("post")["weight"].sum().reindex(neurons["neuron_id"]).fillna(0).to_numpy() + 5
    )
    d = tmp_path / "store"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(
        d,
        dataset="malecns",
        version="1.0",
        converter="test",
        inputs=[],
        counts={"neurons": 40, "edges": 200},
    )
    return Store(d)


@pytest.fixture
def small_frames(small_store: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    return small_store.neurons(), small_store.edges()
