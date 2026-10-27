"""Every example spec parses; the sugar-only spec matches the MaleCNS calibration stimulus."""

from __future__ import annotations

from pathlib import Path

import pytest

from flyconn.experiments.spec import load_spec

SPECS = Path(__file__).resolve().parents[3] / "examples" / "specs"


@pytest.mark.parametrize("path", sorted(SPECS.glob("*.yaml")), ids=lambda p: p.name)
def test_example_spec_parses(path: Path):
    load_spec(path)


def test_sugar_spec_stimulates_lb3c_and_lb3d_only():
    s = load_spec(SPECS / "w2_robust_pathway_mn9_sugar.yaml")
    assert [st.select.cell_type for st in s.stimulate] == ["LB3[cd]"]
    assert s.robustness is not None and s.robustness.negative_controls == 1
    assert s.report.statistics == "paired"
