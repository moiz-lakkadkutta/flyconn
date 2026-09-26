"""``flyconn data`` commands: list, info, pull."""

from __future__ import annotations

import typer

from flyconn.data.pull import pull as _pull
from flyconn.data.registry import LEVELS, Level, get_dataset, list_datasets

app = typer.Typer(help="Dataset registry, downloads and conversion.", no_args_is_help=True)


@app.command("list")
def list_cmd() -> None:
    """List registered datasets (name@version)."""
    for ref in list_datasets():
        spec = get_dataset(ref)
        typer.echo(f"{ref:<16} {spec.license:<12} {spec.title}")


@app.command()
def info(ref: str) -> None:
    """Show licence, citations and files of a dataset."""
    try:
        spec = get_dataset(ref)
    except (KeyError, ValueError) as exc:
        typer.echo(str(exc).strip('"'), err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"{spec.ref}: {spec.title}")
    typer.echo(f"licence: {spec.license}  {spec.license_url}")
    typer.echo(f"homepage: {spec.homepage}")
    typer.echo("citations:")
    for c in spec.citations:
        typer.echo(f"  - {c.text} doi:{c.doi}")
    typer.echo("files:")
    for f in spec.files:
        mb = f.bytes / 1e6
        typer.echo(f"  [{f.level:<9}] {f.name}  {mb:,.1f} MB  {f.role}")


@app.command("pull")
def pull_cmd(
    ref: str,
    level: str = typer.Option("meta", "--level", help=f"one of {', '.join(LEVELS)}"),
    force: bool = typer.Option(False, "--force", help="re-run conversion even if up to date"),
) -> None:
    """Download (resumably, checksummed) and convert a dataset up to LEVEL."""
    if level not in LEVELS:
        typer.echo(f"level must be one of {LEVELS}", err=True)
        raise typer.Exit(code=2)
    lvl: Level = level  # type: ignore[assignment]

    def progress(name: str, done: int, total: int | None) -> None:
        if total and done % (64 << 20) < (1 << 20):
            typer.echo(f"  {name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")

    try:
        result = _pull(ref, level=lvl, force=force, progress=progress)
    except (KeyError, ValueError, NotImplementedError) as exc:
        typer.echo(str(exc).strip('"'), err=True)
        raise typer.Exit(code=1) from None
    for d in result.downloads:
        state = "cached" if d.skipped else "downloaded"
        typer.echo(f"{state:<10} {d.path.name}  {d.bytes / 1e6:,.1f} MB  sha256={d.sha256[:12]}…")
    typer.echo(f"{'converted' if result.converted else 'up to date'}: {result.store_dir}")
