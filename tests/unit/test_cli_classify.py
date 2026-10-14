"""flyconn classify run / bench on the synthetic stores."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from flyconn.cli import classify as cli_classify
from flyconn.cli.main import app
from flyconn.data.store import Store
from tests.unit.compare.conftest import _make


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Store]:
    s = {
        "malecns@1.0": _make(tmp_path, "malecns", "1.0", seed=1, perturb_type="T3"),
        "flywire@783": _make(tmp_path, "flywire", "783", seed=2, perturb_type=None),
    }
    monkeypatch.setattr(cli_classify.Store, "open", classmethod(lambda cls, ref: s[ref]))
    return s


def test_read_ids_parses_whitespace_blanks_and_duplicates(tmp_path: Path):
    p = tmp_path / "ids.txt"
    p.write_text(" 12\n\n7\n12\n  3 \n")
    assert cli_classify.read_ids(p) == [3, 7, 12]


def test_run_by_type_writes_parquet_and_json(stores: dict[str, Store], tmp_path: Path):
    out = tmp_path / "t1.parquet"
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "run",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--type",
            "T1",
            "--out",
            str(out),
            "--uncalibrated",
        ],
    )
    assert res.exit_code == 0, res.stdout
    df = pd.read_parquet(out)
    assert (df["label"] == "T1").all()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["group"]["top_labels"][0] == "T1"
    assert meta["caveats"]
    assert "T1" in res.stdout


def test_run_with_groups_file(stores: dict[str, Store], tmp_path: Path):
    female = stores["flywire@783"].neurons(columns=["neuron_id", "cell_type"])
    groups = {
        "first": female.loc[female["cell_type"] == "T1", "neuron_id"].tolist(),
        "second": female.loc[female["cell_type"] == "T5", "neuron_id"].tolist(),
    }
    gpath = tmp_path / "groups.json"
    gpath.write_text(json.dumps(groups))
    out = tmp_path / "g.parquet"
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "run",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--type",
            "T1",
            "--groups",
            str(gpath),
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (pd.read_parquet(out)["label"] == "first").all()


def test_bench_writes_metrics(stores: dict[str, Store], tmp_path: Path):
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "bench",
            "--query",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--out",
            str(tmp_path / "b"),
            "--max-types",
            "8",
            "--folds",
            "2",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (tmp_path / "b" / "metrics.json").exists()


def test_run_group_without_labelled_partners_does_not_crash(
    stores: dict[str, Store], tmp_path: Path
):
    out = tmp_path / "t8.parquet"
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "run",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--type",
            "T8",
            "--direction",
            "out",
            "--out",
            str(out),
            "--uncalibrated",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "group: unknown" in res.stdout


def test_run_rejects_missing_selection_before_building_atlas(
    stores: dict[str, Store], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import flyconn.compare as compare

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("atlas built before validating arguments")

    monkeypatch.setattr(compare, "build_atlas", boom)
    for extra in ([], ["--type", "T1", "--direction", "bth"]):
        res = CliRunner().invoke(
            app,
            [
                "classify",
                "run",
                "malecns@1.0",
                "--reference",
                "flywire@783",
                "--out",
                str(tmp_path / "x.parquet"),
                *extra,
            ],
        )
        assert res.exit_code != 0
        assert not isinstance(res.exception, AssertionError)


def test_run_reports_dropped_group_ids(stores: dict[str, Store], tmp_path: Path):
    female = stores["flywire@783"].neurons(columns=["neuron_id", "cell_type"])
    groups = {"first": [*female.loc[female["cell_type"] == "T1", "neuron_id"].tolist(), 999999]}
    gpath = tmp_path / "groups.json"
    gpath.write_text(json.dumps(groups))
    res = CliRunner().invoke(
        app,
        [
            "classify",
            "run",
            "malecns@1.0",
            "--reference",
            "flywire@783",
            "--type",
            "T1",
            "--groups",
            str(gpath),
            "--out",
            str(tmp_path / "g.parquet"),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "1 group id(s) not in flywire@783" in res.stdout
