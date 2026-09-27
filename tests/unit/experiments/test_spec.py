"""Tests for the declarative experiment spec (YAML -> ExperimentSpec) and neuron selection."""

from pathlib import Path

import pytest

from flyconn.data.store import Store
from flyconn.experiments.spec import ExperimentSpec, Selection, load_spec, resolve_selection

YAML = """
name: activate_motor_probe
dataset: malecns@1.0
network:
  min_weight: 5
  sign_policy: argmax
stimulate:
  - select: {cell_type: "T00*"}
    rate_hz: 100
perturb:
  silence:
    - select: {ids: [10001, 10002]}
readouts:
  - name: motor
    select: {super_class: motor}
  - name: two_cells
    select: {ids: [10005, 10006]}
duration_ms: 200
trials: 4
seed: 3
device: cpu
dtype: float64
controls:
  degree_preserving_rewire: 2
  sign_shuffle: 2
report:
  top_neurons: 20
"""


def test_load_spec_parses_all_sections(tmp_path: Path):
    p = tmp_path / "exp.yaml"
    p.write_text(YAML)
    spec = load_spec(p)
    assert isinstance(spec, ExperimentSpec)
    assert spec.name == "activate_motor_probe"
    assert spec.dataset == "malecns@1.0"
    assert spec.network.min_weight == 5 and spec.network.sign_policy == "argmax"
    assert spec.stimulate[0].rate_hz == 100 and spec.stimulate[0].select.cell_type == "T00*"
    assert spec.perturb is not None and spec.perturb.silence[0].select.ids == [10001, 10002]
    assert [r.name for r in spec.readouts] == ["motor", "two_cells"]
    assert spec.duration_ms == 200 and spec.trials == 4 and spec.seed == 3
    assert spec.controls.degree_preserving_rewire == 2 and spec.controls.sign_shuffle == 2
    assert spec.n_steps == 2000
    assert spec.source_sha256 is not None and len(spec.source_sha256) == 64


def test_controls_default_on_and_validation_errors(tmp_path: Path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "name: x\ndataset: malecns@1.0\n"
        "stimulate: [{select: {ids: [1]}, rate_hz: 50}]\n"
        "readouts: [{name: r, select: {ids: [2]}}]\n"
    )
    spec = load_spec(p)
    assert spec.controls.degree_preserving_rewire >= 1
    assert spec.controls.sign_shuffle >= 1
    assert spec.trials >= 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ndataset: malecns@1.0\nreadouts: []\n")
    with pytest.raises(ValueError, match="readout"):
        load_spec(bad)


def test_resolve_selection_by_ids_type_glob_and_class(small_store: Store):
    neurons = small_store.neurons()
    ids = resolve_selection(Selection(ids=[10001, 10002]), neurons)
    assert ids.tolist() == [10001, 10002]
    by_type = resolve_selection(Selection(cell_type="T00*"), neurons)
    assert len(by_type) > 0
    assert set(neurons.set_index("neuron_id").loc[by_type, "cell_type"].str.startswith("T00"))
    by_class = resolve_selection(Selection(super_class="motor"), neurons)
    assert (neurons.set_index("neuron_id").loc[by_class, "super_class"] == "motor").all()
    with pytest.raises(KeyError, match="not in dataset"):
        resolve_selection(Selection(ids=[999]), neurons)
    with pytest.raises(ValueError, match="empty"):
        resolve_selection(Selection(cell_type="ZZZ*"), neurons)
