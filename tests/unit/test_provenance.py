"""Run-environment provenance helper."""

import re

import flyconn
from flyconn.provenance import git_sha, run_environment


def test_run_environment_has_versions_and_sha():
    env = run_environment()
    assert env["flyconn_version"] == flyconn.__version__
    assert set(env["packages"]) >= {"numpy", "scipy", "pandas"}
    sha = git_sha()
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha)
    assert env["git_sha"] == sha
