"""Validation rung 3: throughput of the LIF engine on the full Shiu v630 network per device.

Run: FLYCONN_CACHE=... uv run python benchmarks/bench_sim.py
Writes benchmarks/sim_throughput.json (device, dtype, trials, seconds per biological second).
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import LIFNetwork, simulate

SETS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "shiu_2024_neuron_sets.json"
    ).read_text()
)["sets"]


def main() -> None:
    m = ConnectivityMatrix.from_store(
        Store(pull("shiu@630", level="weights").store_dir), sign_policy=SignPolicy.ARGMAX
    )
    net = LIFNetwork.from_matrix(m)
    stim = dict.fromkeys(SETS["sugar_grn"], 100.0)
    devices = (
        ["cpu"]
        + (["mps"] if torch.backends.mps.is_available() else [])
        + (["cuda"] if torch.cuda.is_available() else [])
    )
    rows = []
    for dev in devices:
        for n_trials in (1, 8, 30):
            n_steps = 2000
            t0 = time.time()
            simulate(
                net,
                stimulate=stim,
                n_steps=n_steps,
                n_trials=n_trials,
                seed=0,
                dtype="float32",
                device=dev,
                record=False,
            )
            el = time.time() - t0
            bio = n_steps * 1e-4 * n_trials
            rows.append(
                {
                    "device": dev,
                    "dtype": "float32",
                    "n_trials": n_trials,
                    "n_steps": n_steps,
                    "seconds": round(el, 2),
                    "seconds_per_bio_second": round(el / bio, 2),
                }
            )
            print(rows[-1])
    out = {
        "machine": platform.platform(),
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "network": {"neurons": net.n, "edges": int(net.weights_mv.nnz)},
        "rows": rows,
    }
    Path(__file__).with_name("sim_throughput.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
