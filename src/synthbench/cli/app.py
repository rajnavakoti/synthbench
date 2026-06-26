"""Synthbench CLI application."""

import typer

from synthbench.cli.run import run

app = typer.Typer(
    name="synthbench",
    help="Quality-under-load benchmarking for AI media generation APIs.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Quality-under-load benchmarking for AI media generation APIs."""


app.command()(run)
