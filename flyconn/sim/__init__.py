"""flyconn.sim: cross-platform LIF engine reproducing Shiu et al. 2024 (model predictions only)."""

from flyconn.sim.calibrate import CalibrationResult, calibrate_w_syn
from flyconn.sim.engine import LIFNetwork, SimResult, simulate
from flyconn.sim.params import ShiuParams

__all__ = [
    "CalibrationResult",
    "LIFNetwork",
    "ShiuParams",
    "SimResult",
    "calibrate_w_syn",
    "simulate",
]
