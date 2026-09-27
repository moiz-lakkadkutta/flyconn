"""``flyconn`` command-line interface (typer)."""

from __future__ import annotations

import typer

import flyconn
from flyconn.cli.data import app as data_app

app = typer.Typer(
    name="flyconn",
    help="Research-grade toolkit over public Drosophila connectomes.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"flyconn {flyconn.__version__}")
        raise typer.Exit()


@app.command("run")
def run_cmd(
    spec_path: str = typer.Argument(..., help="YAML experiment spec"),
    out: str = typer.Option(..., "--out", help="output directory (must not exist or be empty)"),
) -> None:
    """Run a YAML experiment (controls on by default) and write Parquet results + HTML report."""
    from flyconn.experiments.runner import run_experiment
    from flyconn.experiments.spec import load_spec

    spec = load_spec(spec_path)
    typer.echo(
        f"running {spec.name} on {spec.dataset}: {spec.trials} trials x {spec.duration_ms:g} ms"
    )
    result = run_experiment(spec, out_dir=out)
    n_sig = int(result.readouts["significant"].sum()) if "significant" in result.readouts else 0
    typer.echo(f"{len(result.readouts)} comparisons, {n_sig} significant at q<{spec.report.alpha}")
    typer.echo(f"model prediction; report: {result.out_dir / 'report.html'}")


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the flyconn version and exit.",
    ),
) -> None:
    """flyconn CLI."""
