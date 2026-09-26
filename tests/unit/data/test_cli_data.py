"""Tests for the `flyconn data` CLI group."""

from typer.testing import CliRunner

from flyconn.cli.main import app


def test_data_list_shows_registered_datasets():
    result = CliRunner().invoke(app, ["data", "list"])
    assert result.exit_code == 0, result.output
    assert "malecns@1.0" in result.output
    assert "flywire@783" in result.output


def test_data_info_shows_files_levels_and_license():
    result = CliRunner().invoke(app, ["data", "info", "malecns@1.0"])
    assert result.exit_code == 0, result.output
    assert "CC-BY-4.0" in result.output
    assert "body-annotations-male-cns-v1.0-minconf-0.5.feather" in result.output
    assert "weights" in result.output


def test_data_info_unknown_dataset_fails_cleanly():
    result = CliRunner().invoke(app, ["data", "info", "nosuch@1"])
    assert result.exit_code != 0
    assert "unknown dataset" in result.output
