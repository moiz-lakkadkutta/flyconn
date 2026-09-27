"""Leaky integrate-and-fire parameters of Shiu et al. 2024 (Nature 634:210) and derived constants.

Values are quoted from the paper's Methods and the reference ``model.py``
(``docs/GOLDEN_RESULTS.md`` section 3.1). They were fitted to FlyWire v630;
simulations on other datasets are uncalibrated (ADR-0004).
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
