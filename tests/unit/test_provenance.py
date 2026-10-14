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


def test_git_sha_ignores_a_foreign_repository(tmp_path):  # type: ignore[no-untyped-def]
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "user project",
        ],
        check=True,
    )  # a user's own repo with a HEAD, flyconn installed in its .venv
    pkg = tmp_path / ".venv" / "site-packages" / "flyconn"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    assert git_sha(pkg) is None


def test_run_environment_flags_dirty_checkout():
    env = run_environment()
    assert "git_dirty" in env
    assert env["git_dirty"] in (True, False, None)
