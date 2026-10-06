"""Re-calibration protocol for datasets other than FlyWire v630 (ADR-0004, ADR-0009).

Shiu et al. 2024 fixed the single free parameter ``w_syn`` so that driving the
sugar GRNs at 100 Hz produced roughly 80 % of MN9's maximal firing rate (their
Methods). :func:`calibrate_w_syn` reproduces that protocol for any network and
readout: scan a grid of ``w_syn`` values, simulate a reference rate and a
saturating rate, and find the ``w_syn`` at which the readout ratio
(reference / saturating) crosses ``target_fraction``.

MN9 does not saturate in the v630 model (it still rises at 400 Hz drive), and at the
published 0.275 mV the 100 / 200 Hz ratio is about 0.69, not 0.8 (ADR-0009). The
target is therefore a parameter: the literal 0.8, or the ratio measured on the
reference network, which transfers v630's operating point to another dataset.

:func:`select_w_syn` is the pure decision step: bootstrap confidence intervals on
each grid point's ratio (trials resampled per condition), a linearly interpolated
crossing, and a bootstrap interval on the crossing. Status is ``calibrated`` only
for a single crossing found in at least 95 % of bootstrap replicates, ``ambiguous``
for several crossings or a weakly supported one, and ``unresolved`` when the
target is never reached.

The result is a *protocol outcome*, not a validated calibration: until a
dataset's readout has been compared with independent data, simulations on it
stay labelled uncalibrated (``CalibrationResult.summary`` says so).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from flyconn.sim.engine import DType, LIFNetwork, simulate

Status = Literal["calibrated", "ambiguous", "unresolved"]

_MIN_SUPPORT = 0.95


@dataclass
class Selection:
    """Outcome of :func:`select_w_syn`."""

    w_syn_mv: float
    achieved_fraction: float
    status: Status
    fractions: pd.DataFrame  # w_syn_mv, fraction, ci_low, ci_high
    crossing_mv: float | None
    crossing_ci_mv: tuple[float, float] | None
    crossing_support: float


@dataclass
class CalibrationResult:
    w_syn_mv: float
    achieved_fraction: float
    curve: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)
    selection: Selection | None = None

    def summary(self) -> str:
        status = self.provenance.get("status", "unresolved")
        ci = self.provenance.get("crossing_ci_mv")
        ci_text = f" (95 % CI {ci[0]:.3g}-{ci[1]:.3g} mV)" if ci else ""
        return (
            f"w_syn = {self.w_syn_mv:.3g} mV{ci_text} gives {self.achieved_fraction:.2f} of the "
            f"saturating readout at the reference rate (target "
            f"{self.provenance.get('target_fraction')}); status: {status}. "
            "Simulations remain uncalibrated until validated against data."
        )


def _crossings(ws: np.ndarray, fr: np.ndarray, target: float) -> list[float]:
    d = fr - target
    out: list[float] = []
    for i in range(len(ws) - 1):
        a, b = float(d[i]), float(d[i + 1])
        if a == 0.0:
            out.append(float(ws[i]))
        elif a * b < 0.0:
            out.append(float(ws[i] + (ws[i + 1] - ws[i]) * a / (a - b)))
    if len(ws) and float(d[-1]) == 0.0:
        out.append(float(ws[-1]))
    return out


def _ratio(ref: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return np.where(mx > 0, ref / np.where(mx > 0, mx, 1.0), 0.0)


def select_w_syn(
    trials: pd.DataFrame,
    *,
    target_fraction: float = 0.8,
    reference_rate_hz: float = 100.0,
    max_rate_hz: float = 200.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> Selection:
    """Pick ``w_syn`` from per-trial readout rates.

    ``trials`` has columns ``w_syn_mv``, ``rate_hz`` and ``readout_rate_hz`` (one row per trial).
    """
    rng = np.random.default_rng(seed)
    ws = np.array(sorted(set(trials["w_syn_mv"].astype(float))))
    point = np.empty(len(ws))
    boot = np.empty((n_boot, len(ws)))
    for j, w in enumerate(ws):
        sub = trials[trials["w_syn_mv"] == w]
        ref = sub.loc[sub["rate_hz"] == reference_rate_hz, "readout_rate_hz"].to_numpy(float)
        mx = sub.loc[sub["rate_hz"] == max_rate_hz, "readout_rate_hz"].to_numpy(float)
        if ref.size == 0 or mx.size == 0:
            msg = f"w_syn {w}: need trials at {reference_rate_hz} and {max_rate_hz} Hz"
            raise ValueError(msg)
        point[j] = _ratio(np.array(ref.mean()), np.array(mx.mean()))
        ref_b = ref[rng.integers(0, ref.size, (n_boot, ref.size))].mean(axis=1)
        mx_b = mx[rng.integers(0, mx.size, (n_boot, mx.size))].mean(axis=1)
        boot[:, j] = _ratio(ref_b, mx_b)
    fractions = pd.DataFrame(
        {
            "w_syn_mv": ws,
            "fraction": point,
            "ci_low": np.quantile(boot, 0.025, axis=0),
            "ci_high": np.quantile(boot, 0.975, axis=0),
        }
    )
    found = _crossings(ws, point, target_fraction)
    boot_cross: list[float] = []
    for b in range(n_boot):
        c = _crossings(ws, boot[b], target_fraction)
        if len(c) == 1:
            boot_cross.append(c[0])
    support = len(boot_cross) / n_boot if n_boot else 0.0
    crossing = found[0] if len(found) == 1 else None
    crossing_ci = (
        (float(np.quantile(boot_cross, 0.025)), float(np.quantile(boot_cross, 0.975)))
        if crossing is not None and boot_cross
        else None
    )
    status: Status
    if not found:
        status = "unresolved"
    elif crossing is not None and support >= _MIN_SUPPORT:
        status = "calibrated"
    else:
        status = "ambiguous"
    if status == "calibrated" and crossing is not None:
        w_best, frac_best = crossing, target_fraction
    else:
        k = int(np.argmin(np.abs(point - target_fraction)))
        w_best, frac_best = float(ws[k]), float(point[k])
    return Selection(
        w_syn_mv=w_best,
        achieved_fraction=frac_best,
        status=status,
        fractions=fractions,
        crossing_mv=crossing,
        crossing_ci_mv=crossing_ci,
        crossing_support=support,
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
    n_boot: int = 1000,
) -> CalibrationResult:
    """Scan ``w_syn`` and find where ``readout``'s reference / saturating ratio crosses the target.

    ``stimulate`` gives the driven neurons; their rates are overridden by each value
    in ``rates_hz``. The per-trial readout rate is the mean over ``readout`` neurons.
    """
    base = net.params
    seconds = n_steps * base.dt_ms / 1000.0
    rows: list[dict[str, float]] = []
    for w in w_syn_grid_mv:
        scaled = LIFNetwork(
            net.neuron_ids,
            net.weights_mv * (w / base.w_syn_mv),
            replace(base, w_syn_mv=float(w)),
            dict(net.provenance),
        )
        idx = scaled.index_of(list(readout))
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
            per_trial = res.counts[:, idx].mean(axis=1) / seconds
            rows.extend(
                {"w_syn_mv": float(w), "rate_hz": float(r), "trial": t, "readout_rate_hz": float(v)}
                for t, v in enumerate(per_trial)
            )
    trials = pd.DataFrame(rows)
    grouped = trials.groupby(["w_syn_mv", "rate_hz"], sort=False)["readout_rate_hz"]
    curve = pd.DataFrame(
        {
            "readout_rate_hz": grouped.mean(),
            "readout_sd_hz": cast("pd.Series", grouped.std(ddof=1)).fillna(0.0),
            "n_trials": grouped.size(),
        }
    ).reset_index()
    sel = select_w_syn(
        trials,
        target_fraction=target_fraction,
        reference_rate_hz=reference_rate_hz,
        max_rate_hz=max_rate_hz,
        n_boot=n_boot,
        seed=seed,
    )
    prov = {
        "protocol": "Shiu et al. 2024: w_syn such that the reference-rate readout is ~80% of max",
        "target_fraction": target_fraction,
        "reference_rate_hz": reference_rate_hz,
        "max_rate_hz": max_rate_hz,
        "grid_mv": [float(w) for w in w_syn_grid_mv],
        "fractions": sel.fractions.to_dict("records"),
        "crossing_mv": sel.crossing_mv,
        "crossing_ci_mv": list(sel.crossing_ci_mv) if sel.crossing_ci_mv else None,
        "crossing_support": sel.crossing_support,
        "n_boot": n_boot,
        "n_steps": n_steps,
        "n_trials": n_trials,
        "seed": seed,
        "status": sel.status,
        "label": "protocol outcome; dataset remains uncalibrated until validated",
    }
    return CalibrationResult(sel.w_syn_mv, sel.achieved_fraction, curve, prov, trials, sel)
