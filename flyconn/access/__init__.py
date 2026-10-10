"""flyconn.access: driver lines -> cell types and NeuronBridge off-target candidates.

Exploratory (M7). Data: Meissner et al. 2025 line table (CC BY 4.0) and NeuronBridge
precomputed colour-depth matches (CC BY 4.0). Matches are similarity candidates, not
expression calls.
"""

from flyconn.access.handoff import attach_lines, load_catalog
from flyconn.access.lines import LineCatalog, convert_meissner_lines
from flyconn.access.neuronbridge import (
    aggregate_cds_matches,
    fetch_line_off_targets,
    parse_published_name,
)

__all__ = [
    "LineCatalog",
    "aggregate_cds_matches",
    "attach_lines",
    "convert_meissner_lines",
    "fetch_line_off_targets",
    "load_catalog",
    "parse_published_name",
]
