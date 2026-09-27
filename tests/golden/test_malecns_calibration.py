"""Apply the w_syn re-calibration protocol to MaleCNS v1.0 (LB3 labellar GRNs -> MN9) and report.

This documents the procedure of ADR-0004; it does not validate MaleCNS simulations,
which remain uncalibrated until compared with independent data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import LIFNetwork, calibrate_w_syn

pytestmark = pytest.mark.golden


def test_malecns_calibration_protocol_runs_and_reports():
    store = Store(pull("malecns@1.0", level="weights").store_dir)
    m = ConnectivityMatrix.from_store(store, sign_policy=SignPolicy.ARGMAX)
    net = LIFNetwork.from_matrix(m)
    assert net.provenance["calibration"].startswith("UNCALIBRATED")
    meta = m.meta
    sources = meta.index[meta["cell_type"].astype(str).str.match(r"^LB3")].tolist()
    mn9 = meta.index[meta["cell_type"] == "MN9"].tolist()
    t0 = time.time()
    res = calibrate_w_syn(
        net,
        stimulate=dict.fromkeys(sources, 100.0),
        readout=mn9,
        w_syn_grid_mv=[0.1, 0.2, 0.275, 0.4, 0.6],
        rates_hz=[100.0, 200.0],
        n_steps=5000,
        n_trials=5,
        seed=0,
        device="cpu",
        dtype="float32",
    )
    dt = time.time() - t0
    print(f"\nMaleCNS calibration ({len(sources)} LB3 -> {len(mn9)} MN9) in {dt:.0f}s")
    print(
        res.curve.pivot(index="w_syn_mv", columns="rate_hz", values="readout_rate_hz").to_string()
    )
    print(res.summary())
    out = Path("benchmarks") / "malecns_calibration.json"
    out.write_text(json.dumps({"curve": res.curve.to_dict("records"), **res.provenance}, indent=1))
    assert len(res.curve) == 10
    assert res.provenance["status"] in {"calibrated", "unresolved"}
