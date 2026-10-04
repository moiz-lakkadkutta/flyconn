"""Tests for sweep experiments: one YAML, many variants, one combined result and report."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

import flyconn.experiments.runner as runner_mod
from flyconn.data.store import Store
from flyconn.experiments.spec import RateSweep, SilenceEachSweep, spec_from_dict
from flyconn.experiments.sweep import SweepResult, run_sweep

BASE_YAML_KEYS = ("name", "dataset", "stimulate", "readouts")


def _spec(store: Store, sweep: dict[str, object], **over: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    d: dict[str, object] = {
        "name": "tiny_sweep",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [
            {"name": "targets", "select": {"ids": ids[-10:]}},
            {"name": "motor", "select": {"super_class": "motor"}},
        ],
        "sweep": sweep,
        "duration_ms": 60,
        "trials": 4,
        "seed": 2,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 1, "sign_shuffle": 1},
    }
    d.update(over)
    return d


# --- spec parsing -------------------------------------------------------------------------


def test_parse_silence_each_sweep(small_store: Store):
    spec = spec_from_dict(
        _spec(
            small_store,
            {
                "silence_each": {
                    "select": {"cell_type": "T*"},
                    "group_by": "cell_type",
                    "max_items": 3,
                }
            },
        )
    )
    assert isinstance(spec.sweep, SilenceEachSweep)
    assert spec.sweep.select.cell_type == "T*"
    assert spec.sweep.group_by == "cell_type" and spec.sweep.max_items == 3


def test_parse_rate_sweep(small_store: Store):
    spec = spec_from_dict(_spec(small_store, {"rate_hz": [10, 50, 100]}))
    assert isinstance(spec.sweep, RateSweep)
    assert spec.sweep.rates_hz == [10.0, 50.0, 100.0]


def test_no_sweep_is_none(small_store: Store):
    d = _spec(small_store, {})
    d.pop("sweep")
    assert spec_from_dict(d).sweep is None


@pytest.mark.parametrize(
    ("sweep", "match"),
    [
        ({"rate_hz": [10], "silence_each": {"select": {"cell_type": "T*"}}}, "exactly one"),
        ({"bogus": 1}, "unknown sweep"),
        ({"rate_hz": []}, "rate_hz"),
        ({"rate_hz": [-5]}, "rate_hz"),
        ({"silence_each": {"group_by": "cell_type"}}, "select"),
        ({"silence_each": {"select": {"cell_type": "T*"}, "max_items": 0}}, "max_items"),
    ],
)
def test_sweep_validation_errors(small_store: Store, sweep: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        spec_from_dict(_spec(small_store, sweep))


def test_sweep_with_perturb_is_rejected(small_store: Store):
    ids = small_store.neurons()["neuron_id"].tolist()
    d = _spec(
        small_store, {"rate_hz": [10, 20]}, perturb={"silence": [{"select": {"ids": ids[8:9]}}]}
    )
    with pytest.raises(ValueError, match="perturb"):
        spec_from_dict(d)


def test_rate_sweep_needs_stimulation(small_store: Store):
    d = _spec(small_store, {"rate_hz": [10, 20]}, stimulate=[])
    with pytest.raises(ValueError, match="stimulate"):
        spec_from_dict(d)


# --- silence_each ---------------------------------------------------------------------------


def _count_simulate_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    real = runner_mod.simulate

    def spy(*args: object, **kw: object):
        calls.append(kw)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_mod, "simulate", spy)
    return calls


def test_silence_each_runs_shared_conditions_once_and_one_run_per_variant(
    small_store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _count_simulate_calls(monkeypatch)
    spec = spec_from_dict(
        _spec(
            small_store,
            {
                "silence_each": {
                    "select": {"cell_type": "T*"},
                    "group_by": "cell_type",
                    "max_items": 3,
                }
            },
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    assert isinstance(res, SweepResult)
    n_variants = int(res.variants["simulated"].sum())
    assert 1 <= n_variants <= 3 < len(res.variants)
    # baseline + stimulated + 1 rewire + 1 shuffle, then one per variant
    assert len(calls) == 4 + n_variants
    silenced = [c.get("silence") for c in calls if c.get("silence")]
    assert len(silenced) == n_variants


def test_silence_each_picks_most_active_groups_and_reports_skipped(
    small_store: Store, tmp_path: Path
):
    spec = spec_from_dict(
        _spec(
            small_store,
            {
                "silence_each": {
                    "select": {"cell_type": "T*"},
                    "group_by": "cell_type",
                    "max_items": 2,
                }
            },
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    v = res.variants
    assert set(v.columns) >= {"variant", "n_neurons", "stimulated_rate_hz", "simulated"}
    sim = v[v["simulated"]]
    assert len(sim) <= 2
    # chosen variants are the most active candidate groups in the stimulated condition
    others = v[~v["simulated"]]
    if len(others) and len(sim):
        assert sim["stimulated_rate_hz"].min() >= others["stimulated_rate_hz"].max()
    assert (sim["stimulated_rate_hz"] > 0).all()


def test_silence_each_sweep_table_and_bh_across_whole_sweep(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(
        _spec(
            small_store,
            {"silence_each": {"select": {"cell_type": "T*"}, "group_by": "neuron", "max_items": 4}},
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    t = pq.read_table(tmp_path / "out" / "sweep.parquet").to_pandas()
    assert set(t.columns) >= {
        "sweep_kind",
        "variant",
        "readout",
        "mean_hz",
        "mean_ci_low",
        "mean_ci_high",
        "reference",
        "mean_reference_hz",
        "difference",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "significant",
        "rank",
        "n_trials",
    }
    assert (t["sweep_kind"] == "silence_each").all()
    assert (t["reference"] == "stimulated").all()
    assert (t["n_trials"] == 4).all()
    n_var = res.variants["simulated"].sum()
    assert len(t) == n_var * 2  # variants x readouts
    # BH over the whole sweep: q equals BH of all p values together
    from flyconn.experiments.stats import benjamini_hochberg

    pd.testing.assert_series_equal(
        t["q_value"].reset_index(drop=True),
        pd.Series(benjamini_hochberg(t["p_value"].to_numpy()), name="q_value"),
        check_exact=False,
    )
    # rank 1 within each readout is the largest absolute difference
    for _, g in t.groupby("readout"):
        top = g.sort_values("rank").iloc[0]
        assert abs(top["difference"]) == pytest.approx(g["difference"].abs().max())
    for f in (
        "sweep.parquet",
        "readouts.parquet",
        "conditions.parquet",
        "provenance.json",
        "report.html",
    ):
        assert (tmp_path / "out" / f).exists(), f


def test_silence_each_by_neuron_covers_every_active_driven_neuron(
    small_store: Store, tmp_path: Path
):
    """Externally driven neurons always fire, so each one becomes its own simulated variant."""
    ids = small_store.neurons()["neuron_id"].tolist()
    spec = spec_from_dict(
        _spec(
            small_store,
            {"silence_each": {"select": {"ids": ids[:8]}, "group_by": "neuron", "max_items": 8}},
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    assert int(res.variants["simulated"].sum()) == 8
    assert sorted(res.variants["variant"]) == sorted(str(i) for i in ids[:8])


# --- rate sweep -----------------------------------------------------------------------------


def test_rate_sweep_curves_reference_baseline_and_monotone_drive(
    small_store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _count_simulate_calls(monkeypatch)
    spec = spec_from_dict(_spec(small_store, {"rate_hz": [0.5, 50, 400]}))
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    t = res.sweep
    assert (t["sweep_kind"] == "rate_hz").all()
    assert sorted(t["rate_hz"].unique().tolist()) == [0.5, 50.0, 400.0]
    assert (t["reference"] == "baseline").all()
    assert len(calls) == 4 + 3
    rates_seen = sorted(
        {float(next(iter(c["stimulate"].values()))) for c in calls if c.get("stimulate")}  # type: ignore[union-attr]
    )
    assert rates_seen == [0.5, 50.0, 200.0, 400.0]  # 200 = spec's own rate (shared stimulated)
    stim_counts = res.variants.set_index("rate_hz")["total_spikes"]
    assert stim_counts.loc[400.0] >= stim_counts.loc[0.5]


def test_rate_sweep_without_baseline_uses_stimulated_reference(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(
        _spec(
            small_store,
            {"rate_hz": [20, 100]},
            controls={
                "degree_preserving_rewire": 0,
                "sign_shuffle": 0,
                "unstimulated_baseline": False,
            },
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    assert (res.sweep["reference"] == "stimulated").all()


# --- report & provenance --------------------------------------------------------------------


def test_sweep_report_has_banners_ranked_table_figure_and_n_trials_statement(
    small_store: Store, tmp_path: Path
):
    spec = spec_from_dict(
        _spec(
            small_store,
            {
                "silence_each": {
                    "select": {"cell_type": "T*"},
                    "group_by": "cell_type",
                    "max_items": 3,
                }
            },
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    html = (tmp_path / "out" / "report.html").read_text()
    low = html.lower()
    assert "model prediction" in low
    assert "uncalibrated" in low  # synthetic store is labelled malecns -> uncalibrated
    assert "n = 4 trials" in low
    assert "benjamini-hochberg" in low and "whole sweep" in low
    assert "<img" in html
    assert "10.1016/j.cell.2026.08.015" in html
    assert "silence_each" in html
    prov = res.provenance
    assert prov["label"] == "model prediction"
    assert prov["sweep"]["kind"] == "silence_each"
    assert prov["seed"] == 2 and prov["trials"] == 4
    assert "flyconn_version" in prov


def test_rate_sweep_report_has_curve_figure(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_spec(small_store, {"rate_hz": [10, 100]}))
    run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    html = (tmp_path / "out" / "report.html").read_text()
    assert "readout vs stimulation rate" in html.lower()
    assert "<img" in html


def test_run_experiment_refuses_sweep_spec(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_spec(small_store, {"rate_hz": [10, 100]}))
    with pytest.raises(ValueError, match="run_sweep"):
        runner_mod.run_experiment(spec, out_dir=tmp_path / "out", store=small_store)


def test_sweep_refuses_to_overwrite(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_spec(small_store, {"rate_hz": [10]}))
    run_sweep(spec, out_dir=tmp_path / "out", store=small_store)
    with pytest.raises(FileExistsError):
        run_sweep(spec, out_dir=tmp_path / "out", store=small_store)


# --- pathway-ranked silencing screens -----------------------------------------------------


@pytest.fixture
def relay_store(tmp_path: Path) -> Store:
    """Stimulated S1,S2 -> relay A (strong) -> R; S1 -> relay B (weak) -> R; S -> C (no route to R).

    Path strengths (product of input fractions): via A 0.5*0.6 per source = 0.6 total;
    via B 1.0*0.1 = 0.1; C is very active but has no path to the readout R.
    """
    import numpy as np

    from flyconn.data.convert.common import write_parquet, write_provenance
    from flyconn.data.schema import conform_edges, conform_neurons

    neurons = pd.DataFrame(
        {
            "dataset": "malecns",
            "version": "1.0",
            "neuron_id": [1, 2, 3, 4, 5, 6],
            "cell_type": ["S", "S", "RelayA", "RelayB", "SinkC", "R"],
            "super_class": "central_brain_intrinsic",
            "nt_pred": "acetylcholine",
            "nt_conf": 1.0,
            "input_synapses_total": [1.0, 1.0, 100.0, 10.0, 160.0, 100.0],
        }
    )
    edges = pd.DataFrame(
        {
            "dataset": "malecns",
            "version": "1.0",
            "pre": [1, 2, 3, 1, 4, 1, 2],
            "post": [3, 3, 6, 4, 6, 5, 5],
            "weight": np.array([50, 50, 60, 10, 10, 80, 80], dtype=np.int32),
        }
    )
    d = tmp_path / "relay_store"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(edges), d / "edges.parquet")
    write_provenance(d, dataset="malecns", version="1.0", converter="t", inputs=[], counts={})
    return Store(d)


def _relay_spec(store: Store, silence_each: dict[str, object]) -> dict[str, object]:
    return {
        "name": "relay_screen",
        "dataset": store.ref,
        "stimulate": [{"select": {"cell_type": "S"}, "rate_hz": 200}],
        "readouts": [{"name": "R", "select": {"cell_type": "R"}}],
        "sweep": {"silence_each": silence_each},
        "duration_ms": 40,
        "trials": 2,
        "seed": 0,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
    }


def test_parse_pathway_ranking_with_optional_select(relay_store: Store):
    spec = spec_from_dict(
        _relay_spec(
            relay_store,
            {"rank_by": "pathway", "pathway": {"readout": "R", "max_hops": 2}, "max_items": 5},
        )
    )
    assert isinstance(spec.sweep, SilenceEachSweep)
    assert spec.sweep.rank_by == "pathway"
    assert spec.sweep.pathway is not None
    assert spec.sweep.pathway.readout == "R" and spec.sweep.pathway.max_hops == 2
    assert spec.sweep.select.ids is None and spec.sweep.select.cell_type is None


@pytest.mark.parametrize(
    ("silence_each", "match"),
    [
        ({"rank_by": "pathway"}, "pathway"),
        ({"rank_by": "pathway", "pathway": {"readout": "nope"}}, "readout"),
        ({"rank_by": "magic", "select": {"cell_type": "*"}}, "rank_by"),
        ({"rank_by": "pathway", "pathway": {"readout": "R", "max_hops": 0}}, "max_hops"),
    ],
)
def test_pathway_ranking_validation(
    relay_store: Store, silence_each: dict[str, object], match: str
):
    with pytest.raises(ValueError, match=match):
        spec_from_dict(_relay_spec(relay_store, silence_each))


def test_pathway_ranking_orders_relays_by_path_strength_and_skips_off_path(
    relay_store: Store, tmp_path: Path
):
    spec = spec_from_dict(
        _relay_spec(
            relay_store,
            {
                "rank_by": "pathway",
                "pathway": {"readout": "R", "max_hops": 2, "min_weight": 1},
                "max_items": 5,
            },
        )
    )
    res = run_sweep(spec, out_dir=tmp_path / "out", store=relay_store)
    v = res.variants.set_index("variant")
    assert list(res.variants.loc[res.variants["simulated"], "variant"]) == ["RelayA", "RelayB"]
    assert v.loc["RelayA", "path_strength"] == pytest.approx(0.6)
    assert v.loc["RelayB", "path_strength"] == pytest.approx(0.1)
    assert v.loc["RelayA", "n_paths"] == 2 and v.loc["RelayB", "n_paths"] == 1
    # stimulated sources and readout targets are never candidates
    assert "S" not in v.index and "R" not in v.index
    # SinkC fires (it receives 80+80 synapses) but lies on no route to the readout
    assert not v.loc["SinkC", "simulated"]
    assert v.loc["SinkC", "reason"] == "not on a path to the readout"
    assert "pathway" in res.provenance["sweep"]["ranked_by"]


def test_activity_ranking_still_default_and_picks_the_busy_sink(relay_store: Store, tmp_path: Path):
    spec = spec_from_dict(_relay_spec(relay_store, {"select": {"cell_type": "*"}, "max_items": 1}))
    assert spec.sweep is not None and spec.sweep.rank_by == "activity"  # type: ignore[union-attr]
    res = run_sweep(spec, out_dir=tmp_path / "out", store=relay_store)
    chosen = res.variants.loc[res.variants["simulated"], "variant"].tolist()
    assert len(chosen) == 1 and chosen[0] in {"SinkC", "S", "R"}
