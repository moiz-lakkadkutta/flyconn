"""``flyconn classify``: zero-shot cell-type classification by partner profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from flyconn.data.store import Store

app = typer.Typer(
    help="Zero-shot cell-type classification by partner profile.", no_args_is_help=True
)


def read_ids(path: Path) -> list[int]:
    """Neuron ids from a text file, one per line; blanks skipped, duplicates removed."""
    tokens = [t.strip() for t in Path(path).read_text().splitlines()]
    return sorted({int(t) for t in tokens if t})


@app.command("run")
def run_cmd(
    query: Annotated[str, typer.Argument(help="query dataset, e.g. malecns@1.0")],
    out: Annotated[
        Path, typer.Option("--out", help="output .parquet (a .json sidecar is written too)")
    ],
    reference: str = typer.Option(..., "--reference", help="reference dataset, e.g. flywire@783"),
    cell_type: str | None = typer.Option(None, "--type", help="query cell type to classify"),
    ids: Annotated[
        Path | None, typer.Option("--ids", help="text file, one neuron id per line")
    ] = None,
    groups: Annotated[
        Path | None, typer.Option("--groups", help="JSON {label: [reference ids]}")
    ] = None,
    direction: str = typer.Option("both", "--direction", help="both | out | in"),
    min_weight: int = typer.Option(5, "--min-weight"),
    uncalibrated: bool = typer.Option(False, "--uncalibrated", help="skip calibration"),
) -> None:
    """Classify query neurons against a reference atlas."""
    from flyconn.compare import build_atlas, classify

    group_map = json.loads(groups.read_text()) if groups is not None else None
    atlas = build_atlas(
        Store.open(reference),
        groups=group_map,
        min_weight=min_weight,
        direction=direction,  # type: ignore[arg-type]
        on_missing="drop" if group_map is not None else "raise",
    )
    res = classify(
        Store.open(query),
        atlas,
        cell_type=cell_type,
        neuron_ids=read_ids(ids) if ids is not None else None,
        calibration=None if uncalibrated else "default",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    res.per_neuron.to_parquet(out, index=False)
    side = {
        "group": vars(res.group) if res.group is not None else None,
        "calibrated": res.calibrated,
        "caveats": res.caveats,
        "provenance": res.provenance,
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str) + "\n")
    if res.group is not None:
        g = res.group
        typer.echo(
            f"group: {g.call} {g.label or '-'} (s1 {g.s1:.3f}, margin {g.margin:.3f}, "
            f"agreement {g.agreement:.0%} of {g.n})"
        )
    typer.echo(res.per_neuron["call"].value_counts().to_string())
    for c in res.caveats:
        typer.echo(f"caveat: {c}")
    typer.echo(f"wrote {out}")


@app.command("bench")
def bench_cmd(
    query: Annotated[
        list[str], typer.Option("--query", help="query dataset(s) with a cross-reference")
    ],
    out: Annotated[Path, typer.Option("--out", help="output directory")],
    reference: str = typer.Option(..., "--reference"),
    truth_column: str = typer.Option("fafb_783_cell_type", "--truth-column"),
    max_types: int = typer.Option(2000, "--max-types"),
    folds: int = typer.Option(5, "--folds"),
    seed: int = typer.Option(0, "--seed"),
    write_calibration: bool = typer.Option(
        False, "--write-calibration", help="write the shipped calibration file for REFERENCE"
    ),
) -> None:
    """Hold-out benchmark: closed- and open-set accuracy, calibration, transfer ECE."""
    from flyconn.compare import build_atlas
    from flyconn.compare.calibrate import calibration_path
    from flyconn.compare.classify_bench import fit_calibration, run_benchmark, write_benchmark
    from flyconn.graph.matrix import ConnectivityMatrix

    atlas = build_atlas(Store.open(reference))
    results = []
    for q in query:
        store = Store.open(q)
        m = ConnectivityMatrix.from_store(store, min_weight=int(atlas.params["min_weight"]))
        results.append(
            run_benchmark(
                m,
                atlas,
                query_ref=store.ref,
                truth_column=truth_column,
                max_types=max_types,
                seed=seed,
            )
        )
        typer.echo(f"{q}: {results[-1].records['type'].nunique()} types benchmarked")
    cal = fit_calibration(results, atlas, n_folds=folds, seed=seed)
    path = write_benchmark(out, results, cal)
    for q, levels in cal.metrics["per_query"].items():
        for level, m in levels.items():
            typer.echo(
                f"{q} {level}: top-1 {m['top1'][0]:.3f} ({m['top1'][1]:.3f}-{m['top1'][2]:.3f}), "
                f"coverage {m['coverage']:.2f}, accuracy covered {m['accuracy_covered']:.3f}, "
                f"open-set false accept {m['false_accept_open']:.3f}"
            )
    if write_calibration:
        cal.to_json(calibration_path(atlas.reference))
        typer.echo(f"calibration {cal.id} -> {calibration_path(atlas.reference)}")
    typer.echo(f"wrote {path}")
