"""flyconn.compare: cross-dataset type matching, comparison and zero-shot classification."""

from flyconn.compare.atlas import Atlas, build_atlas
from flyconn.compare.calibrate import Calibration
from flyconn.compare.classify import ClassifyResult, GroupCall, classify
from flyconn.compare.types import (
    TypeComparison,
    compare_type,
    match_types,
    partner_labels,
    type_profile,
    verdict_for,
)

__all__ = [
    "Atlas",
    "Calibration",
    "ClassifyResult",
    "GroupCall",
    "TypeComparison",
    "build_atlas",
    "classify",
    "compare_type",
    "match_types",
    "partner_labels",
    "type_profile",
    "verdict_for",
]
