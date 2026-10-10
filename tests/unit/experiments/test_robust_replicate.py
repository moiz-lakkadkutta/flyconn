"""Replicate mapping MaleCNS -> FlyWire by cell type (spec 2.2, cross-dataset replicate)."""

from __future__ import annotations

import pandas as pd

from flyconn.experiments.robust_replicate import map_candidates, map_selection, replicate_spec
from flyconn.experiments.spec import Selection, spec_from_dict
from flyconn.experiments.sweep import Candidate

Q = pd.DataFrame(
    {
        "neuron_id": [1, 2, 3, 4, 5, 6],
        "cell_type": ["LB3c", "LB3d", "MN9", "GNG232", "X1", None],
        "super_class": ["sensory", "sensory", "motor", "central", "central", "descending"],
        "side": ["left", "right", "left", "left", "left", "left"],
    }
)
R = pd.DataFrame(
    {
        "neuron_id": [11, 12, 13, 14, 15],
        "cell_type": ["LB3", "LB3", "MN9", "GNG232", "DNx"],
        "super_class": ["sensory", "sensory", "motor", "central", "descending"],
        "side": ["left", "right", "left", "left", "left"],
    }
)
TM = pd.DataFrame(
    {
        "type_a": ["LB3c", "LB3d", "MN9", "GNG232"],
        "type_b": ["LB3", "LB3", "MN9", "GNG232"],
        "n_a": [1, 1, 1, 1],
        "n_b": [2, 2, 1, 1],
        "source": "fafb_783_cell_type",
    }
)


def test_glob_selection_maps_through_types():
    ids, why = map_selection(Selection(cell_type="LB3*"), Q, R, TM)
    assert why == "" and ids is not None and ids.tolist() == [11, 12]


def test_side_filter_is_kept():
    ids, _ = map_selection(Selection(cell_type="LB3*", side="left"), Q, R, TM)
    assert ids is not None and ids.tolist() == [11]


def test_super_class_only_passes_through():
    ids, _ = map_selection(Selection(super_class="descending"), Q, R, TM)
    assert ids is not None and ids.tolist() == [15]


def test_unmapped_type_is_reported():
    ids, why = map_selection(Selection(cell_type="X1"), Q, R, TM)
    assert ids is None and "X1" in why


def test_candidates():
    mapped, reasons = map_candidates(
        [Candidate("GNG232", (4,)), Candidate("X1", (5,)), Candidate("K", (6,), "control")],
        R,
        TM,
        "cell_type",
    )
    assert [(c.label, c.neuron_ids, c.role) for c in mapped] == [("GNG232", (14,), "candidate")]
    assert reasons["X1"].startswith("not testable") and reasons["K"].startswith("not testable")


def test_candidates_exclude_stimulus_and_readout_neurons():
    cands = [Candidate("GNG232", (4,)), Candidate("MN9", (3,)), Candidate("LB3c", (1,))]
    mapped, reasons = map_candidates(cands, R, TM, "cell_type", exclude={11, 12, 13})
    assert [c.label for c in mapped] == ["GNG232"]
    assert reasons["MN9"] == "not testable: maps only to replicate stimulus/readout neurons"
    assert reasons["LB3c"] == "not testable: maps only to replicate stimulus/readout neurons"
    mapped, _ = map_candidates(cands, R, TM, "cell_type", exclude={13})
    assert [(c.label, c.neuron_ids) for c in mapped] == [
        ("GNG232", (14,)),
        ("LB3c", (11, 12)),
    ]


def test_candidates_need_cell_type_grouping():
    mapped, reasons = map_candidates([Candidate("4", (4,))], R, TM, "neuron")
    assert mapped == [] and "not grouped by cell type" in reasons["4"]


def _spec(stim_type: str):
    return spec_from_dict(
        {
            "name": "s",
            "dataset": "malecns@1.0",
            "stimulate": [{"select": {"cell_type": stim_type}, "rate_hz": 100}],
            "readouts": [{"name": "mn", "select": {"cell_type": "MN9"}}],
            "sweep": {
                "silence_each": {"select": {"super_class": "central"}, "group_by": "cell_type"}
            },
        }
    )


def test_replicate_spec_maps_stimulus_and_readouts():
    orig = _spec("LB3*")
    out, why = replicate_spec(orig, orig, Q, R, TM)
    assert why == "" and out is not None
    assert out.stimulate[0].select.ids == [11, 12]
    assert out.readouts[0].select.ids == [13]
    assert out.source_sha256 != orig.source_sha256


def test_replicate_spec_unmappable_stimulus():
    orig = _spec("X1")
    out, why = replicate_spec(orig, orig, Q, R, TM)
    assert out is None and why.startswith("not testable")
