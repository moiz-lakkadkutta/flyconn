"""Tests for the flyconn CLI entry point."""

from typer.testing import CliRunner

import flyconn
from flyconn.cli.main import app


def test_version_flag_prints_package_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert flyconn.__version__ in result.stdout
