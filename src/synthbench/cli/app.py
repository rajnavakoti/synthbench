"""Synthbench CLI application."""

import typer
from rich import print as rprint

app = typer.Typer(
    name="synthbench",
    help="Quality-under-load benchmarking for AI media generation APIs.",
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Quality-under-load benchmarking for AI media generation APIs."""


@app.command()
def run(
    scenario: str = typer.Option(
        ...,
        "--scenario",
        "-s",
        help="Path to TOML scenario file",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print plan without executing",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON report to file",
    ),
) -> None:
    """Run a benchmark scenario against a provider."""
    rprint(f"[bold]Loading scenario:[/bold] {scenario}")
    if dry_run:
        rprint("[dim]Dry run mode — printing plan without calling provider APIs.[/dim]")
