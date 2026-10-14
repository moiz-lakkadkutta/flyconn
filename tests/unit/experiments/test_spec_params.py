"""Spec-level simulator overrides: w_syn and the sampled-sign seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flyconn.data.convert.common import write_parquet, write_provenance
from flyconn.data.schema import conform_edges, conform_neurons
from flyconn.data.store import Store
from flyconn.experiments.runner import prepare, sim_params, simulate_condition
from flyconn.experiments.spec import spec_from_dict


def _d(store: Store, **over: object) -> dict[str, object]:
    ids = store.neurons()["neuron_id"].tolist()
    d: dict[str, object] = {
        "name": "p",
        "dataset": store.ref,
        "stimulate": [{"select": {"ids": ids[:8]}, "rate_hz": 200}],
        "readouts": [{"name": "targets", "select": {"ids": ids[-10:]}}],
        "duration_ms": 60,
        "trials": 3,
        "seed": 1,
        "device": "cpu",
        "dtype": "float64",
        "controls": {"degree_preserving_rewire": 0, "sign_shuffle": 0},
    }
    d.update(over)
    return d


def test_parse_params_and_sign_seed(small_store: Store):
    spec = spec_from_dict(
        _d(
            small_store,
            params={"w_syn_mv": 0.3},
            network={"sign_policy": "sampled", "sign_seed": 7},
        )
    )
    assert spec.params.w_syn_mv == 0.3
    assert spec.network.sign_seed == 7


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"params": {"w_syn_mv": 0}}, "w_syn_mv"),
        ({"params": {"tau": 1}}, "unknown params"),
        ({"network": {"sign_policy": "sampled"}}, "sign_seed"),
        ({"network": {"sign_policy": "argmax", "sign_seed": 3}}, "only valid"),
    ],
)
def test_param_validation(small_store: Store, over: dict[str, object], match: str):
    with pytest.raises(ValueError, match=match):
        spec_from_dict(_d(small_store, **over))


def test_w_syn_override_reaches_the_network(small_store: Store, tmp_path: Path):
    spec = spec_from_dict(_d(small_store, params={"w_syn_mv": 0.6}))
    assert sim_params(spec).w_syn_mv == 0.6
    ctx = prepare(spec, tmp_path / "a", small_store)
    assert ctx.net.params.w_syn_mv == 0.6
    base = spec_from_dict(_d(small_store))
    ctx0 = prepare(base, tmp_path / "b", small_store)
    assert ctx0.net.params.w_syn_mv == pytest.approx(0.188)  # MaleCNS calibration default
    a = simulate_condition(ctx.net, spec, ctx.background, [])
    b = simulate_condition(ctx0.net, base, ctx0.background, [])
    assert not np.array_equal(a, b)


def test_sampled_signs_are_reproducible(small_store: Store, tmp_path: Path):
    def signed(seed: int, out: str) -> np.ndarray:
        spec = spec_from_dict(
            _d(small_store, network={"sign_policy": "sampled", "sign_seed": seed})
        )
        return prepare(spec, tmp_path / out, small_store).matrix.signed_weights.toarray()

    assert np.array_equal(signed(5, "a"), signed(5, "b"))
    assert not np.array_equal(signed(5, "c"), signed(6, "d"))


def test_sampled_without_probabilities_raises(small_store: Store, tmp_path: Path):
    neurons = small_store.neurons()
    for c in [c for c in neurons.columns if c.startswith("nt_p_")]:
        neurons[c] = np.nan
    d = tmp_path / "noprob"
    d.mkdir()
    write_parquet(conform_neurons(neurons), d / "neurons.parquet")
    write_parquet(conform_edges(small_store.edges()), d / "edges.parquet")
    write_provenance(
        d,
        dataset="malecns",
        version="1.0",
        converter="test",
        inputs=[],
        counts={"neurons": len(neurons), "edges": 200},
    )
    store = Store(d)
    spec = spec_from_dict(_d(store, network={"sign_policy": "sampled", "sign_seed": 1}))
    with pytest.raises(ValueError, match="nt-probs"):
        prepare(spec, tmp_path / "x", store)
