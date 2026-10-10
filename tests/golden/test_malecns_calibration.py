"""Apply the w_syn re-calibration protocol to MaleCNS v1.0 (sugar-like LB3 GRNs -> MN9).

Shiu et al. drove the sugar GRNs only; MaleCNS types its labellar GRNs as LB3a-d without a
taste label. LB3c and LB3d are the sugar-like subtypes by output-partner profile against
the FlyWire sugar and water GRN sets (cosine 0.92 / 0.86 to sugar; LB3a is water-like,
0.87; see GOLDEN_RESULTS 6g). The primary target is the 50 / 200 Hz (onset) ratio the positive
control measured on v630 at the published w_syn (ADR-0009); the 100 / 200 Hz transfer
and the literal 0.8 rule are reported alongside. All LB3 is run as the previous protocol's
stimulus for comparison.

This documents the procedure of ADR-0004 / ADR-0009; it does not validate MaleCNS
simulations, which stay labelled uncalibrated until compared with independent data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import LIFNetwork, ShiuParams, calibrate_w_syn, select_w_syn

pytestmark = pytest.mark.golden

# v630 MN9 readout ratios at w_syn = 0.275 mV, keyed by reference rate (max 200 Hz);
# tests/golden/test_shiu_calibration.py, seed 0, 10 x 1 s trials (benchmarks/shiu_calibration.json)
V630_FRACTION = {50.0: 0.20504731861198738, 100.0: 0.6887486855941115}
GRID = [round(0.1 + 0.025 * i, 3) for i in range(13)]  # 0.1 .. 0.4 mV
SUGAR_SUBTYPES = ("LB3c", "LB3d")


def test_malecns_calibration_protocol_runs_and_reports():
    store = Store(pull("malecns@1.0", level="weights").store_dir)
    m = ConnectivityMatrix.from_store(store, sign_policy=SignPolicy.ARGMAX)
    net = LIFNetwork.from_matrix(m, ShiuParams())  # published constant as the scan base
    assert net.provenance["calibration"].startswith("UNCALIBRATED")
    meta = m.meta
    cell_type = meta["cell_type"].astype(str)
    sugar = meta.index[cell_type.isin(SUGAR_SUBTYPES)].tolist()
    all_lb3 = meta.index[cell_type.str.match(r"^LB3")].tolist()
    mn9 = meta.index[cell_type == "MN9"].tolist()
    assert len(sugar) == 49 and len(all_lb3) == 78 and len(mn9) == 2
    report: dict[str, object] = {
        "grid_mv": GRID,
        "v630_fraction": {f"{k:g}": v for k, v in V630_FRACTION.items()},
    }
    for name, sources in (("sugar_LB3cd", sugar), ("all_LB3", all_lb3)):
        t0 = time.time()
        res = calibrate_w_syn(
            net,
            stimulate=dict.fromkeys(sources, 100.0),
            readout=mn9,
            w_syn_grid_mv=GRID,
            rates_hz=[50.0, 100.0, 200.0],
            target_fraction=V630_FRACTION[50.0],
            reference_rate_hz=50.0,
            n_steps=10_000,
            n_trials=10,
            seed=1,
            device="cpu",
            dtype="float32",
        )
        others = {
            "transfer_100": select_w_syn(res.trials, target_fraction=V630_FRACTION[100.0], seed=1),
            "literal_100": select_w_syn(res.trials, target_fraction=0.8, seed=1),
        }
        dt = time.time() - t0
        print(f"\nMaleCNS {name} ({len(sources)} GRNs -> {len(mn9)} MN9), {dt:.0f}s")
        print(res.curve.pivot(index="w_syn_mv", columns="rate_hz", values="readout_rate_hz"))
        print(res.selection.fractions.to_string(index=False) if res.selection else "")
        print("transfer_50:", res.summary())
        for label, sel in others.items():
            print(f"{label}: {sel.status} {sel.crossing_mv} CI {sel.crossing_ci_mv}")
        report[name] = {
            "n_sources": len(sources),
            "curve": res.curve.to_dict("records"),
            "transfer_50": {k: v for k, v in res.provenance.items() if k != "grid_mv"},
            **{
                label: {
                    "status": sel.status,
                    "crossing_mv": sel.crossing_mv,
                    "crossing_ci_mv": sel.crossing_ci_mv,
                    "support": sel.crossing_support,
                    "fractions": sel.fractions.to_dict("records"),
                }
                for label, sel in others.items()
            },
            "seconds": dt,
        }
        assert res.provenance["status"] in {"calibrated", "ambiguous", "unresolved"}
    out = Path("benchmarks") / "malecns_calibration.json"
    out.write_text(json.dumps(report, indent=1))
