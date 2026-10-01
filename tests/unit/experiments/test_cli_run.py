"""Tests for `flyconn run exp.yaml`."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from flyconn.cli.main import app
from flyconn.data.store import Store


def test_run_command_executes_spec_and_writes_report(
    small_store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ids = small_store.neurons()["neuron_id"].tolist()
    spec = tmp_path / "exp.yaml"
    spec.write_text(
        f"""
name: cli_tiny
dataset: {small_store.ref}
stimulate: [{{select: {{ids: {ids[:3]}}}, rate_hz: 200}}]
readouts: [{{name: r, select: {{ids: {ids[-4:]}}}}}]
duration_ms: 30
trials: 2
device: cpu
dtype: float64
controls: {{degree_preserving_rewire: 1, sign_shuffle: 1}}
"""
    )
    # Point the runner at the synthetic store instead of pulling real data.
    import flyconn.experiments.runner as runner

    monkeypatch.setattr(
        runner, "pull", lambda ref, level: type("R", (), {"store_dir": small_store.root})()
    )
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["run", str(spec), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report.html").exists()
    assert "report.html" in result.output
    assert "model prediction" in result.output.lower()


def test_run_command_handles_sweep_spec(
    small_store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ids = small_store.neurons()["neuron_id"].tolist()
    spec = tmp_path / "sweep.yaml"
    spec.write_text(
        f"""
name: cli_sweep
dataset: {small_store.ref}
stimulate: [{{select: {{ids: {ids[:3]}}}, rate_hz: 200}}]
readouts: [{{name: r, select: {{ids: {ids[-4:]}}}}}]
sweep: {{rate_hz: [20, 200]}}
duration_ms: 30
trials: 2
device: cpu
dtype: float64
controls: {{degree_preserving_rewire: 1, sign_shuffle: 0}}
"""
    )
    import flyconn.experiments.runner as runner

    monkeypatch.setattr(
        runner, "pull", lambda ref, level: type("R", (), {"store_dir": small_store.root})()
    )
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["run", str(spec), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "sweep.parquet").exists()
    assert "2 variants" in result.output
    assert "model prediction" in result.output.lower()
