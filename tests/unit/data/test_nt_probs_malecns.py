"""Tests for aggregating MaleCNS per-presynapse NT probabilities into per-body means (ADR-0008)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

from flyconn.data.convert.malecns import TBAR, aggregate_tbar_nt, convert_malecns


def _write_tbar(raw: Path) -> None:
    rows = pd.DataFrame(
        {
            "point_id": np.arange(6, dtype=np.uint64),
            "x": np.zeros(6, dtype=np.int32),
            "y": np.zeros(6, dtype=np.int32),
            "z": np.zeros(6, dtype=np.int32),
            "conf": np.full(6, 0.9, dtype=np.float32),
            "sv": np.zeros(6, dtype=np.int64),
            "body": np.array([10001, 10001, 10003, 10003, 10003, 99999], dtype=np.int64),
            "nt_acetylcholine_prob": np.array([0.8, 0.6, 0.1, 0.1, 0.1, 1.0], dtype=np.float32),
            "nt_dopamine_prob": np.zeros(6, dtype=np.float32),
            "nt_gaba_prob": np.array([0.1, 0.2, 0.7, 0.8, 0.6, 0.0], dtype=np.float32),
            "nt_glutamate_prob": np.array([0.1, 0.2, 0.2, 0.1, 0.3, 0.0], dtype=np.float32),
            "nt_histamine_prob": np.zeros(6, dtype=np.float32),
            "nt_octopamine_prob": np.zeros(6, dtype=np.float32),
            "nt_serotonin_prob": np.zeros(6, dtype=np.float32),
        }
    )
    pf.write_feather(pa.Table.from_pandas(rows, preserve_index=False), raw / TBAR, chunksize=2)


def test_aggregate_tbar_writes_per_body_means_and_counts(malecns_raw: Path, tmp_path: Path):
    _write_tbar(malecns_raw)
    out = tmp_path / "store"
    table = aggregate_tbar_nt(malecns_raw / TBAR, out / "nt_probs.parquet", batch_rows=2)
    df = table.to_pandas().set_index("neuron_id")
    assert df.loc[10001, "n_presynapses"] == 2
    assert abs(df.loc[10001, "nt_p_acetylcholine"] - 0.7) < 1e-6
    assert abs(df.loc[10003, "nt_p_gaba"] - 0.7) < 1e-6
    assert 99999 in df.index  # all bodies aggregated; join filters later
    assert pq.read_table(out / "nt_probs.parquet").num_rows == 3


def test_convert_joins_probabilities_into_neurons_when_tbar_present(
    malecns_raw: Path, tmp_path: Path
):
    _write_tbar(malecns_raw)
    out = tmp_path / "store"
    prov = convert_malecns(malecns_raw, out)
    n = pq.read_table(out / "neurons.parquet").to_pandas().set_index("neuron_id")
    assert abs(n.loc[10003, "nt_p_gaba"] - 0.7) < 1e-6
    assert np.isnan(n.loc[10002, "nt_p_gaba"])  # no presynapses in the tbar file
    assert n.loc[10003, "nt_source"] == "malecns_v1.0_tbar_mean"
    assert n.loc[10002, "nt_source"] == "malecns_v1.0_body_consensus"
    assert prov["nt_probs"]["bodies_with_probabilities"] == 2
