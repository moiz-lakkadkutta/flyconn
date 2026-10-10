"""Inputs and sensitivity of the MaleCNS w_syn calibration (ADR-0009, GOLDEN_RESULTS 6g).

1. Which LB3 subtypes are sugar-like: output-partner type profiles of MaleCNS LB3a-d against
   FlyWire v783 profiles of Shiu's sugar and water GRN ids (partners >= 5 synapses; MaleCNS
   partners named by ``fafb_783_cell_type``).
2. Synapses per matched type pair, MaleCNS vs FlyWire v783 (independent w_syn scale check).
3. Onset-ratio calibration with the per-neuron sugar call instead of LB3c+d, and with
   probabilistic signs (tbar-aggregated NT probabilities) instead of argmax signs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import duckdb
import numpy as np
import pandas as pd
import pytest

from flyconn.data.pull import pull
from flyconn.data.store import Store
from flyconn.graph import ConnectivityMatrix, SignPolicy
from flyconn.sim import LIFNetwork, ShiuParams, calibrate_w_syn

pytestmark = pytest.mark.golden
SETS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "shiu_2024_neuron_sets.json").read_text()
)["sets"]
V630_ONSET_FRACTION = 0.20504731861198738  # test_shiu_calibration.py, seed 0
GRID = [0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
OUT = Path("benchmarks") / "malecns_calibration_inputs.json"


def _update(key: str, value: object) -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, indent=1))


def _dirs() -> tuple[Path, Path]:
    return (
        pull("flywire@783", level="weights").store_dir,
        pull("malecns@1.0", level="weights").store_dir,
    )


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else float("nan")


def _profiles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(reference profiles FW sugar/water by partner type, MaleCNS LB3 per-neuron profiles)."""
    fw, mc = _dirs()
    con = duckdb.connect()
    ref: dict[str, pd.Series] = {}
    for key in ("sugar_grn", "water_grn"):
        ids = ",".join(map(str, SETS[key]))
        df = con.sql(
            f"""select n.cell_type t, sum(e.weight) w from '{fw}/edges.parquet' e
            join '{fw}/neurons.parquet' n on n.neuron_id = e.post
            where e.pre in ({ids}) and e.weight >= 5 group by 1"""
        ).df()
        ref[key] = cast("pd.Series", df.set_index("t")["w"])
    per = con.sql(
        f"""select p.neuron_id pre, p.cell_type sub,
               coalesce(n.fafb_783_cell_type, n.cell_type) t, sum(e.weight) w
        from '{mc}/edges.parquet' e
        join '{mc}/neurons.parquet' p on p.neuron_id = e.pre
        join '{mc}/neurons.parquet' n on n.neuron_id = e.post
        where p.cell_type like 'LB3%' and e.weight >= 5 group by all"""
    ).df()
    ref_df = pd.DataFrame(ref).fillna(0.0)
    ref_df.index = ref_df.index.astype(str)
    per["t"] = per["t"].astype(str)
    return ref_df, per


def sugar_calls() -> pd.DataFrame:
    """Per LB3 neuron: subtype, cosine to the sugar and water reference profiles, call."""
    ref, per = _profiles()
    mat = per.pivot_table(index="t", columns="pre", values="w", aggfunc="sum").fillna(0.0)
    idx = mat.index.union(ref.index)
    mat, ref = mat.reindex(idx).fillna(0.0), ref.reindex(idx).fillna(0.0)
    rows = [
        {
            "neuron_id": int(pre),
            "sugar": _cos(mat[pre].to_numpy(), ref["sugar_grn"].to_numpy()),
            "water": _cos(mat[pre].to_numpy(), ref["water_grn"].to_numpy()),
        }
        for pre in mat.columns
    ]
    subs = cast("pd.DataFrame", per[["pre", "sub"]]).drop_duplicates()
    subs.columns = ["neuron_id", "sub"]
    calls = pd.DataFrame(rows).merge(subs, on="neuron_id")
    calls["call"] = np.where(calls["sugar"] > calls["water"], "sugar", "water")
    return calls


def test_lb3c_and_lb3d_are_sugar_like_lb3a_water_like():
    ref, per = _profiles()
    sub = cast("pd.DataFrame", per.groupby(["sub", "t"])["w"].sum().unstack(0)).fillna(0.0)
    idx = sub.index.union(ref.index)
    sub, ref = sub.reindex(idx).fillna(0.0), ref.reindex(idx).fillna(0.0)
    cos = {
        s: {k: round(_cos(sub[s].to_numpy(), ref[k].to_numpy()), 3) for k in ref}
        for s in sub.columns
    }
    calls = sugar_calls()
    table = calls.groupby(["sub", "call"]).size().unstack(fill_value=0)
    print("\n", pd.DataFrame(cos).T, "\n", table)
    _update(
        "lb3_assignment",
        {"subtype_cosine": cos, "per_neuron_calls": cast("pd.DataFrame", table).to_dict("index")},
    )
    assert cos["LB3c"]["sugar_grn"] > cos["LB3c"]["water_grn"]
    assert cos["LB3d"]["sugar_grn"] > cos["LB3d"]["water_grn"]
    assert cos["LB3a"]["water_grn"] > cos["LB3a"]["sugar_grn"]


def test_synapses_per_matched_type_pair_malecns_vs_flywire():
    fw, mc = _dirs()
    con = duckdb.connect()

    def density(d: Path, type_col: str) -> pd.DataFrame:
        return con.sql(
            f"""with n as (select neuron_id, {type_col} t from '{d}/neurons.parquet'
                  where {type_col} is not null and {type_col} not like '%,%'),
            cnt as (select t, count(*) c from n group by t),
            pair as (select a.t pt, b.t qt, sum(e.weight) w from '{d}/edges.parquet' e
                  join n a on a.neuron_id = e.pre join n b on b.neuron_id = e.post
                  where e.weight >= 5 group by 1, 2)
            select pt, qt, w / (c1.c * c2.c) d from pair
            join cnt c1 on c1.t = pt join cnt c2 on c2.t = qt"""
        ).df()

    joined = density(fw, "cell_type").merge(
        density(mc, "fafb_783_cell_type"), on=["pt", "qt"], suffixes=("_fw", "_mc")
    )
    out = {}
    for floor in (0.0, 1.0, 5.0):
        j = joined[(joined["d_fw"] >= floor) & (joined["d_mc"] >= floor)]
        r = np.log(np.asarray(j["d_mc"], float) / np.asarray(j["d_fw"], float))
        out[f"{floor:g}"] = {
            "n_pairs": len(j),
            "median_ratio": float(np.exp(np.median(r))),
            "geomean_ratio": float(np.exp(r.mean())),
            "implied_w_syn_mv": float(0.275 / np.exp(np.median(r))),
        }
    print("\n", json.dumps(out, indent=1))
    _update("synapse_scale", out)
    assert 1.0 < out["0"]["median_ratio"] < 3.0


@pytest.mark.parametrize(
    ("label", "policy", "source"),
    [
        ("per_neuron_sugar_call_argmax", SignPolicy.ARGMAX, "calls"),
        ("lb3cd_probabilistic_signs", SignPolicy.PROBABILISTIC, "lb3cd"),
    ],
)
def test_onset_calibration_sensitivity(label: str, policy: SignPolicy, source: str):
    store = Store(pull("malecns@1.0", level="weights").store_dir)
    m = ConnectivityMatrix.from_store(store, sign_policy=policy)
    net = LIFNetwork.from_matrix(m, ShiuParams())
    cell_type = m.meta["cell_type"].astype(str)
    if source == "calls":
        calls = sugar_calls()
        sources = calls.loc[calls["call"] == "sugar", "neuron_id"].tolist()
    else:
        sources = m.meta.index[cell_type.isin(("LB3c", "LB3d"))].tolist()
    mn9 = m.meta.index[cell_type == "MN9"].tolist()
    t0 = time.time()
    res = calibrate_w_syn(
        net,
        stimulate=dict.fromkeys(sources, 100.0),
        readout=mn9,
        w_syn_grid_mv=GRID,
        rates_hz=[50.0, 200.0],
        target_fraction=V630_ONSET_FRACTION,
        reference_rate_hz=50.0,
        n_steps=10_000,
        n_trials=10,
        seed=1,
        device="cpu",
        dtype="float32",
    )
    dt = time.time() - t0
    print(f"\n{label} ({len(sources)} GRNs), {dt:.0f}s: {res.summary()}")
    _update(
        label,
        {
            "n_sources": len(sources),
            "sign_policy": str(policy),
            "curve": res.curve.to_dict("records"),
            **{k: v for k, v in res.provenance.items() if k != "grid_mv"},
            "seconds": dt,
        },
    )
    assert res.provenance["status"] in {"calibrated", "ambiguous", "unresolved"}
