"""flyconn.experiments: YAML experiment specs, runner with controls by default, statistics."""

from flyconn.experiments.spec import (
    ExperimentSpec,
    Selection,
    load_spec,
    resolve_selection,
    spec_from_dict,
)
from flyconn.experiments.stats import benjamini_hochberg, cohens_d, compare_groups

__all__ = [
    "ExperimentSpec",
    "Selection",
    "benjamini_hochberg",
    "cohens_d",
    "compare_groups",
    "load_spec",
    "resolve_selection",
    "spec_from_dict",
]
