"""Print the numbers GOLDEN_RESULTS 6j cites from a robustness-screen output directory.

Usage (after ``flyconn run examples/specs/w2_robust_pathway_mn9.yaml --out DIR``):

    FLYCONN_CACHE=... uv run python benchmarks/robust_screen_details.py DIR [--paths]

Reads hits.parquet, null.parquet, selection_stability.parquet, provenance.json,
handoff.csv and variants/*/sweep.parquet / variants/reference/counts.parquet.
``--paths`` also reruns the negative-control path query on the stimulus -> readout graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--paths", action="store_true", help="rerun the LB3* -> MN9 path query")
    args = ap.parse_args()
    out: Path = args.out_dir
    prov = json.loads((out / "provenance.json").read_text())
    rob = prov["robustness"]

    print("== handoff.csv")
    print(pd.read_csv(out / "handoff.csv").drop(columns=["suggested_test"]).to_string(index=False))

    print("\n== effects on the primary readout (hits.parquet), rounded to 0.01")
    eff = pd.read_parquet(out / "hits.parquet")
    print(eff.round(2).to_string(index=False))

    print("\n== label counts per candidate on the NT-sign axis")
    nt = eff[eff["axis"] == "nt"]
    print(pd.crosstab(nt["candidate"], nt["label"]).to_string())

    print("\n== stimulated primary-readout rate per variant (Hz)")
    readout = prov["spec"]["robustness"]["primary_readout"]
    for sweep in sorted((out / "variants").glob("*/sweep.parquet")):
        s = pd.read_parquet(sweep)
        m = s[s["readout"] == readout]
        print(f"{sweep.parent.name:18s} {float(m['mean_reference_hz'].iloc[0]):.1f}")

    print("\n== selection stability")
    print(pd.read_parquet(out / "selection_stability.parquet").to_string(index=False))

    print("\n== random null per key (size bin x super_class)")
    null = pd.read_parquet(out / "null.parquet")
    summary = null.groupby("key")["difference"].agg(
        n="size", n_zero=lambda d: int((d == 0).sum()), min="min", max="max"
    )
    print(summary.round(2).to_string())
    print("skipped:", json.dumps(rob["random_null"].get("skipped", {})))
    print("control_null_p:", json.dumps(rob["control_null_p"]))

    print("\n== replicate mapping")
    print(json.dumps(rob["replicate"].get("candidates", {}), indent=None))

    print("\n== spikes of control neurons in the reference stimulated condition")
    from flyconn.data.store import Store

    store = Store.open(prov["dataset"])
    neurons = store.neurons()
    counts = pd.read_parquet(out / "variants" / "reference" / "counts.parquet")
    stim = counts[(counts["condition"] == "stimulated") & (counts["control"] == "none")]
    for sel in rob["extra_candidates"]:
        ctype = sel.split("=", 1)[1]
        ids = set(neurons.loc[neurons["cell_type"] == ctype, "neuron_id"])
        n_spk = int(stim.loc[stim["neuron_id"].isin(ids), "count"].sum())
        print(f"{ctype}: {len(ids)} neurons, {n_spk} spikes over {stim['trial'].nunique()} trials")

    if args.paths:
        from flyconn.graph.matrix import ConnectivityMatrix
        from flyconn.graph.paths import find_paths

        print("\n== LB3* -> MN9 paths (max_hops 3, min_weight 1)")
        src = neurons[neurons["cell_type"].fillna("").str.startswith("LB3")]["neuron_id"].tolist()
        dst = neurons[neurons["cell_type"] == "MN9"]["neuron_id"].tolist()
        m = ConnectivityMatrix.from_store(store, min_weight=1)
        p = find_paths(m, src, dst, max_hops=3, min_weight=1)
        on = {i for path in p["path"] for i in path}
        print(f"sources {len(src)}, targets {len(dst)}, paths {len(p)}, neurons on paths {len(on)}")
        for t in ("LC4", "T4a", "Mi1", "LPLC2"):
            ids = set(neurons.loc[neurons["cell_type"] == t, "neuron_id"])
            print(t, len(ids), "on path" if ids & on else "no route")


if __name__ == "__main__":
    main()
