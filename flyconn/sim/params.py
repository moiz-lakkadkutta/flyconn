"""Leaky integrate-and-fire parameters of Shiu et al. 2024 (Nature 634:210) and derived constants.

Values are quoted from the paper's Methods and the reference ``model.py``
(``docs/GOLDEN_RESULTS.md`` section 3.1). They were fitted to FlyWire v630.
Other datasets are uncalibrated (ADR-0004) unless :data:`CALIBRATIONS` holds a
w_syn from the re-calibration protocol (ADR-0009); :func:`default_params` applies it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ShiuParams:
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    tau_membrane_ms: float = 20.0
    tau_synapse_ms: float = 5.0
    t_refractory_ms: float = 2.2
    delay_ms: float = 1.8
    w_syn_mv: float = 0.275
    poisson_factor: float = 250.0
    dt_ms: float = 0.1
    source: str = "Shiu et al. 2024, Nature 634:210-219, doi:10.1038/s41586-024-07763-9"

    @property
    def delay_steps(self) -> int:
        return round(self.delay_ms / self.dt_ms)

    @property
    def refractory_steps(self) -> int:
        return round(self.t_refractory_ms / self.dt_ms)

    @property
    def poisson_kick_mv(self) -> float:
        """Voltage jump per Poisson event: ``w_syn * poisson_factor`` (68.75 mV)."""
        return self.w_syn_mv * self.poisson_factor

    @property
    def decay_v(self) -> float:
        return math.exp(-self.dt_ms / self.tau_membrane_ms)

    @property
    def decay_g(self) -> float:
        return math.exp(-self.dt_ms / self.tau_synapse_ms)

    @property
    def g_to_v(self) -> float:
        """Exact one-step contribution of ``g`` to ``v - v_rest`` (Brian2 ``method='exact'``).

        For ``dv/dt = (v0 - v + g)/tm`` and ``dg/dt = -g/ts`` the closed form is
        ``v(t+dt) - v0 = (v - v0) e^{-dt/tm} + g * ts/(ts - tm) * (e^{-dt/ts} - e^{-dt/tm})``.
        """
        tm, ts = self.tau_membrane_ms, self.tau_synapse_ms
        return ts / (ts - tm) * (self.decay_g - self.decay_v)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            delay_steps=self.delay_steps,
            refractory_steps=self.refractory_steps,
            poisson_kick_mv=self.poisson_kick_mv,
        )
        return d


@dataclass(frozen=True)
class DatasetCalibration:
    """A w_syn from the re-calibration protocol: a protocol outcome, not a validation."""

    dataset: str
    w_syn_mv: float
    ci_mv: tuple[float, float]
    sensitivity_mv: tuple[float, float]
    method: str
    evidence: str

    def note(self) -> str:
        return (
            f"CALIBRATED BY PROTOCOL, not validated against data: w_syn = {self.w_syn_mv:g} mV "
            f"(95 % CI {self.ci_mv[0]:g}-{self.ci_mv[1]:g}; {self.sensitivity_mv[0]:g}-"
            f"{self.sensitivity_mv[1]:g} across stimulus-set and sign variants) by "
            f"{self.method} (ADR-0009); "
            "spike counts are model predictions"
        )


CALIBRATIONS: dict[str, DatasetCalibration] = {
    "malecns@1.0": DatasetCalibration(
        dataset="malecns@1.0",
        w_syn_mv=0.188,
        ci_mv=(0.185, 0.192),
        sensitivity_mv=(0.185, 0.209),
        method=(
            "transfer of FlyWire v630's MN9 onset ratio (50 / 200 Hz drive of sugar-like "
            "LB3c+LB3d GRNs; argmax signs)"
        ),
        evidence=(
            "docs/GOLDEN_RESULTS.md 6g; benchmarks/malecns_calibration.json, "
            "benchmarks/malecns_calibration_inputs.json"
        ),
    ),
}


def default_params(dataset: str) -> ShiuParams:
    """Shiu's parameters, with the protocol w_syn for datasets in :data:`CALIBRATIONS`."""
    cal = CALIBRATIONS.get(dataset)
    return ShiuParams(w_syn_mv=cal.w_syn_mv) if cal else ShiuParams()
