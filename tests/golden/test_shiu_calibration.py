"""Positive control for the w_syn re-calibration protocol on FlyWire v630 (ADR-0009).

The protocol must recover Shiu et al.'s published w_syn = 0.275 mV on the network it
was fitted on before it may label any other dataset. Two runs with independent seeds:
seed 0 measures v630's MN9 readout ratios at 0.275 mV; seed 1 calibrates against them.
The 50 / 200 Hz (onset) ratio must recover 0.275 mV inside the bootstrap interval of the
crossing. The 100 / 200 Hz ratio plateaus above ~0.25 mV and the literal 0.8 is never
reached; both are reported, and the literal rule is asserted not to calibrate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import LIFNetwork, calibrate_w_syn, select_w_syn

pytestmark = pytest.mark.golden
SETS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json").read_text()
)["sets"]
PUBLISHED_W_SYN = 0.275
GRID = [0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]


def test_protocol_recovers_published_w_syn_on_v630():
    m = ConnectivityMatrix.from_store(
        Store(pull("shiu@630", level="weights").store_dir), sign_policy=SignPolicy.ARGMAX
    )
    net = LIFNetwork.from_matrix(m)
    kwargs = {
        "stimulate": dict.fromkeys(SETS["sugar_grn"], 100.0),
        "readout": [SETS["mn9"][0]],
        "w_syn_grid_mv": GRID,
        "rates_hz": [50.0, 100.0, 200.0],
        "n_steps": 10_000,
        "n_trials": 10,
        "device": "cpu",
        "dtype": "float32",
    }
    t0 = time.time()
    ref = calibrate_w_syn(net, seed=0, **kwargs)  # pyright: ignore[reportArgumentType]
    run = calibrate_w_syn(net, seed=1, **kwargs)  # pyright: ignore[reportArgumentType]
    dt = time.time() - t0
    report: dict[str, object] = {"seconds": dt, "grid_mv": GRID}
    report["curve_seed0"] = ref.curve.to_dict("records")
    report["curve_seed1"] = run.curve.to_dict("records")
    print(f"\nShiu v630 calibration positive control, 2 x {len(GRID)} x 3 conditions, {dt:.0f}s")
    print(run.curve.pivot(index="w_syn_mv", columns="rate_hz", values="readout_rate_hz"))
    outcomes = {}
    for reference_rate in (50.0, 100.0):
        measured = select_w_syn(ref.trials, reference_rate_hz=reference_rate, seed=0)
        target = float(measured.fractions.set_index("w_syn_mv").loc[PUBLISHED_W_SYN, "fraction"])
        for label, frac in (
            (f"transfer_{reference_rate:g}", target),
            (f"literal_{reference_rate:g}", 0.8),
        ):
            sel = select_w_syn(
                run.trials, target_fraction=frac, reference_rate_hz=reference_rate, seed=1
            )
            outcomes[label] = sel
            print(
                f"{label} (target {frac:.3f}): {sel.status} {sel.crossing_mv} "
                f"CI {sel.crossing_ci_mv} support {sel.crossing_support:.2f}"
            )
            print(sel.fractions.to_string(index=False))
            report[label] = {
                "reference_rate_hz": reference_rate,
                "target_fraction": frac,
                "status": sel.status,
                "crossing_mv": sel.crossing_mv,
                "crossing_ci_mv": sel.crossing_ci_mv,
                "support": sel.crossing_support,
                "fractions": sel.fractions.to_dict("records"),
            }
    out = Path("benchmarks") / "shiu_calibration.json"
    out.write_text(json.dumps(report, indent=1))
    # the literal rule cannot be met on the network it was stated for
    assert outcomes["literal_100"].status != "calibrated"
    # the onset ratio (50 / 200 Hz) transfers and recovers the published value
    onset = outcomes["transfer_50"]
    assert onset.status == "calibrated"
    assert onset.crossing_ci_mv is not None
    lo, hi = onset.crossing_ci_mv
    assert lo <= PUBLISHED_W_SYN <= hi
