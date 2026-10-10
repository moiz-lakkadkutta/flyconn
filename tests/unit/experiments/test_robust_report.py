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
    "thinly_tested",
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
