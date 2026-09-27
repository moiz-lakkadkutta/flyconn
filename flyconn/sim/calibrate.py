"""Re-calibration protocol for datasets other than FlyWire v630 (ADR-0004).

Shiu et al. 2024 fixed the single free parameter ``w_syn`` so that driving the
sugar GRNs at 100 Hz produced roughly 80 % of MN9's maximal firing rate (their
Methods). :func:`calibrate_w_syn` reproduces that protocol for any network and
readout: scan a grid of ``w_syn`` values, simulate a reference rate and a
saturating rate, and pick the ``w_syn`` whose readout at the reference rate is
closest to ``target_fraction`` of the readout at the saturating rate.

The result is a *protocol outcome*, not a validated calibration: until a
dataset's readout has been compared with independent data, simulations on it
stay labelled uncalibrated (``CalibrationResult.summary`` says so).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from flyconn.sim.engine import DType, LIFNetwork, simulate


@dataclass
class CalibrationResult:
    w_syn_mv: float
    achieved_fraction: float
    curve: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = self.provenance.get("status", "unresolved")
        return (
            f"w_syn = {self.w_syn_mv:g} mV gives {self.achieved_fraction:.2f} of the saturating "
            f"readout at the reference rate (target {self.provenance.get('target_fraction')}); "
            f"status: {status}. Simulations remain uncalibrated until validated against data."
        )


def calibrate_w_syn(
    net: LIFNetwork,
    *,
    stimulate: Mapping[int, float],
    readout: Sequence[int],
    w_syn_grid_mv: Sequence[float],
    rates_hz: Sequence[float] = (100.0, 200.0),
    target_fraction: float = 0.8,
    reference_rate_hz: float = 100.0,
    max_rate_hz: float = 200.0,
    n_steps: int = 10_000,
    n_trials: int = 10,
    seed: int = 0,
    device: str = "auto",
    dtype: DType = "float32",
) -> CalibrationResult:
    """Scan ``w_syn`` and pick the value matching Shiu's 80 %-of-maximum rule for ``readout``.

    ``stimulate`` gives the driven neurons; their rates are overridden by each value
    in ``rates_hz``. ``readout`` neurons' mean rate is the calibration target.
    """
    base = net.params
    rows: list[dict[str, float]] = []
    for w in w_syn_grid_mv:
        scaled = LIFNetwork(
            net.neuron_ids,
            net.weights_mv * (w / base.w_syn_mv),
            replace(base, w_syn_mv=float(w)),
            dict(net.provenance),
        )
        for r in rates_hz:
            res = simulate(
                scaled,
                stimulate=dict.fromkeys(stimulate, float(r)),
                n_steps=n_steps,
                n_trials=n_trials,
                seed=seed,
                device=device,
                dtype=dtype,
                record=False,
            )
            idx = scaled.index_of(list(readout))
            seconds = n_steps * base.dt_ms / 1000.0
            rate = float(res.counts[:, idx].mean() / seconds)
            rows.append({"w_syn_mv": float(w), "rate_hz": float(r), "readout_rate_hz": rate})
    curve = pd.DataFrame(rows)
    fractions: list[tuple[float, float]] = []
    for w in w_syn_grid_mv:
        sub = curve[curve["w_syn_mv"] == float(w)]
        lookup: dict[float, float] = dict(
            zip(sub["rate_hz"].tolist(), sub["readout_rate_hz"].tolist(), strict=True)
        )
        ref = lookup.get(float(reference_rate_hz), float("nan"))
        mx = lookup.get(float(max_rate_hz), float("nan"))
        frac = ref / mx if mx > 0 else 0.0
        fractions.append((float(w), frac))
    best_w, best_frac = min(fractions, key=lambda t: abs(t[1] - target_fraction))
    status = (
        "calibrated" if abs(best_frac - target_fraction) < 0.1 and best_frac > 0 else "unresolved"
    )
    prov = {
        "protocol": "Shiu et al. 2024: w_syn such that the reference-rate readout is ~80% of max",
        "target_fraction": target_fraction,
        "reference_rate_hz": reference_rate_hz,
        "max_rate_hz": max_rate_hz,
        "grid_mv": [float(w) for w in w_syn_grid_mv],
        "fractions": fractions,
        "n_steps": n_steps,
        "n_trials": n_trials,
        "seed": seed,
        "status": status,
        "label": "protocol outcome; dataset remains uncalibrated until validated",
    }
    return CalibrationResult(best_w, best_frac, curve, prov)
