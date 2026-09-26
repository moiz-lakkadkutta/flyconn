"""``flyconn`` command-line interface (typer)."""

from __future__ import annotations

import typer

import flyconn

app = typer.Typer(
    name="flyconn",
    help="Research-grade toolkit over public Drosophila connectomes.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"flyconn {flyconn.__version__}")
        raise typer.Exit()


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
