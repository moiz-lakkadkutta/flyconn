"""Evidence cards, handoff.csv and the CLI route for robustness specs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from flyconn.cli.main import app
from flyconn.data.store import Store
from flyconn.experiments.robustness import run_robust_screen
from flyconn.experiments.spec import spec_from_dict
from flyconn.sim.params import CALIBRATIONS

HANDOFF_COLS = [
    "label",
    "tier",
    "direction",
    "effect_hz",
    "ci_low",
    "ci_high",
    "q_value",
    "failed_axes",
    "untested_axes",
    "uninformative_axes",
    "thinly_tested",
    "random_null_note",
    "replicate",
    "path_stability",
    "best_line",
    "line_quality",
    "line_specific",
    "line_match",
    "lines",
    "off_target_bodies",
    "suggested_test",
]


def _d(store: Store, alpha: float = 2.0) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    return {
        "name": "card",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "sweep": {"silence_each": {"select": {"cell_type": "T*"}, "max_items": 2}},
        "duration_ms": 60,
        "trials": 4,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
        "report": {"alpha": alpha},
        "params": {"w_syn_mv": 1.0},
        "robustness": {
            "primary_readout": "targets",
            "readout_proxy": "targets drive nothing in particular (test 2026)",
            "w_syn": [0.3],
            "nt_samples": 0,
            "min_weights": [],
            "replicate": None,
            "random_null": {"n_sets": 2},
        },
    }


def test_report_and_handoff(small_store: Store, tmp_path: Path):
    res = run_robust_screen(
        spec_from_dict(_d(small_store)), out_dir=tmp_path / "o", store=small_store, catalog=None
    )
    n = len(res.hits)
    assert n >= 1
    h = pd.read_csv(tmp_path / "o" / "handoff.csv")
    assert list(h.columns) == HANDOFF_COLS and len(h) == n
    assert set(h["line_match"]) == {"not looked up"}
    assert all("model predicts" in s and "targets drive nothing" in s for s in h["suggested_test"])
    html = (tmp_path / "o" / "report.html").read_text()
    assert "Model prediction." in html and "Calibration:" in html
    assert html.count("class='card'") == n
    assert html.count("Caveats") >= n
    assert "lines not looked up" in html and "flyconn data pull flylight_lines@meissner2025" in html


def test_no_hits(small_store: Store, tmp_path: Path):
    run_robust_screen(
        spec_from_dict(_d(small_store, alpha=1e-12)),
        out_dir=tmp_path / "o",
        store=small_store,
        catalog=None,
    )
    h = pd.read_csv(tmp_path / "o" / "handoff.csv")
    assert list(h.columns) == HANDOFF_COLS and h.empty
    assert "No reference hits" in (tmp_path / "o" / "report.html").read_text()


def test_cli_routes_robustness_specs(small_store: Store, tmp_path: Path, monkeypatch):
    import flyconn.experiments.robustness as rob

    seen: dict[str, object] = {}

    def fake(spec, *, out_dir, **kw):
        seen["name"] = spec.name
        return rob.RobustScreenResult(
            Path(out_dir),
            spec,
            pd.DataFrame({"tier": []}),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    monkeypatch.setattr(rob, "run_robust_screen", fake)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(_d(small_store)))
    res = CliRunner().invoke(app, ["run", str(p), "--out", str(tmp_path / "o")])
    assert res.exit_code == 0, res.output
    assert seen["name"] == "card" and "robustness screen" in res.output


def _row_hit(p: float) -> tuple[pd.Series, pd.Series]:
    row = pd.Series(
        {
            "label": "T1", "tier": "strong", "direction": "down", "effect_hz": -1.0,
            "ci_low": -2.0, "ci_high": -0.5, "q_value": 0.01, "thinly_tested": False,
            "replicate": "n/a", "path_stability": float("nan"), "lines": "",
            "line_match": "none", "suggested_test": "x",
        }
    )  # fmt: skip
    hit = pd.Series(
        {
            "w_syn": "holds", "w_syn_detail": "d", "nt": "holds", "nt_holds": "1/1",
            "nt_flips": "", "min_weight": "holds", "min_weight_detail": "d",
            "random_null": "not run", "random_null_p": p,
        }
    )  # fmt: skip
    return row, hit


def test_null_p_nan_renders_na():
    from flyconn.report.robust_html import _card  # pyright: ignore[reportPrivateUsage]

    row, hit = _row_hit(float("nan"))
    html = _card(row, hit, True, ())
    assert "p n/a" in html and "p nan" not in html
    row, hit = _row_hit(0.25)
    assert "p 0.25" in _card(row, hit, True, ())


def test_conditional_caveats(small_store: Store):
    from flyconn.report.robust_html import caveats_for

    d = _d(small_store)
    cav = " ".join(caveats_for(spec_from_dict(d)))
    assert "replicate runs" not in cav
    d["dataset"] = "malecns@1.0"
    d["robustness"] = {**d["robustness"], "replicate": "flywire@783"}  # type: ignore[dict-item]
    cav = " ".join(caveats_for(spec_from_dict(d)))
    assert "malecns@1.0" in CALIBRATIONS
    assert "malecns@1.0 w_syn calibrated by protocol, not validated (ADR-0009)." in cav
    assert "replicate runs" in cav
    d["dataset"] = "other@1"
    cav = " ".join(caveats_for(spec_from_dict(d)))
    assert "other@1 runs at Shiu's published w_syn, uncalibrated for this dataset." in cav
    assert "calibrated by protocol" not in cav


def test_skipped_null_renders_reason():
    from flyconn.report.robust_html import _card  # pyright: ignore[reportPrivateUsage]

    row, hit = _row_hit(float("nan"))
    hit["random_null"] = "not_run"
    hit["random_null_skipped"] = "pool of 3 descending neurons is smaller than the set size 4"
    html = _card(row, hit, True, ())
    assert "not_run (skipped: pool of 3 descending neurons is smaller than the set size 4)" in html
    assert "p n/a" not in html


def test_degenerate_null_note():
    from flyconn.experiments.robustness import degenerate_null_note

    assert degenerate_null_note(20, 40) == (
        "degenerate null: 20/40 random sets had no effect; p cannot go below 1/41 and is weak evidence"
    )
    assert degenerate_null_note(39, 40).startswith("degenerate null: 39/40 ")
    assert degenerate_null_note(19, 40) == ""
    assert degenerate_null_note(0, 0) == ""


def test_degenerate_null_note_on_card():
    from flyconn.report.robust_html import _card  # pyright: ignore[reportPrivateUsage]

    row, hit = _row_hit(1 / 41)
    note = "degenerate null: 30/40 random sets had no effect; p cannot go below 1/41 and is weak evidence"
    row["random_null_note"] = note
    assert note in _card(row, hit, True, ())
    row["random_null_note"] = ""
    assert "degenerate null" not in _card(row, hit, True, ())


def test_uninformative_null_renders_reason():
    from flyconn.report.robust_html import _null_text  # pyright: ignore[reportPrivateUsage]

    hit = pd.Series(
        {
            "random_null": "uninformative",
            "random_null_p": 1 / 41,
            "random_null_skipped": "",
            "random_null_note": "degenerate null: 40/40 random sets had no effect; "
            "p cannot go below 1/41 and is weak evidence",
        }
    )
    text = _null_text(hit)
    assert text.startswith("uninformative (p 0.024")
    assert "not counted in the tier" in text
