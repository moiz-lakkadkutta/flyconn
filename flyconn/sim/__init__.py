"""flyconn.sim: cross-platform LIF engine reproducing Shiu et al. 2024 (model predictions only)."""

from flyconn.sim.calibrate import CalibrationResult, Selection, calibrate_w_syn, select_w_syn
from flyconn.sim.engine import LIFNetwork, SimResult, simulate
from flyconn.sim.params import CALIBRATIONS, DatasetCalibration, ShiuParams, default_params

__all__ = [
    "CALIBRATIONS",
    "CalibrationResult",
    "DatasetCalibration",
    "LIFNetwork",
    "Selection",
    "ShiuParams",
    "SimResult",
    "calibrate_w_syn",
    "default_params",
    "select_w_syn",
    "simulate",
]
