"""The ``synthbench run`` command."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from synthbench.config.scenario import Scenario, ScenarioError, load_scenario
from synthbench.pricing import estimate_text_cost

console = Console()
err_console = Console(stderr=True)


def run(
    scenario: Path = typer.Option(
        ...,
        "--scenario",
        "-s",
        help="Path to the TOML scenario file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the execution plan and estimated cost without calling any "
        "provider.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the JSON degradation report to this path.",
    ),
) -> None:
    """Run a quality-under-load benchmark scenario against a provider."""
    try:
        scn = load_scenario(scenario)
        prompts = scn.resolve_prompts()
    except ScenarioError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_summary(scn, prompts)

    if dry_run:
        _print_dry_run(scn, prompts)
        return

    # The workload engine (Epic 5) executes the plan and the reporting engine
    # (Epic 7) renders the curve. Until then, point the user at --dry-run.
    err_console.print(
        "\n[yellow]The execution engine is not implemented yet.[/yellow] "
        "Re-run with [bold]--dry-run[/bold] to preview the plan."
    )


def _print_summary(scn: Scenario, prompts: list[str]) -> None:
    provider_cfg = scn.provider_config
    body = "\n".join(
        [
            f"[bold]Provider:[/bold]    {scn.provider}",
            f"[bold]Model:[/bold]       {provider_cfg.model or '(provider default)'}",
            f"[bold]Modality:[/bold]    {scn.modality}",
            f"[bold]Concurrency:[/bold] {', '.join(str(c) for c in scn.concurrency)}",
            f"[bold]Prompts:[/bold]     {len(prompts)} ({scn.prompts.source.value})",
            f"[bold]Metrics:[/bold]     {', '.join(scn.scoring.metrics)}",
            f"[bold]Budget:[/bold]      ${scn.budget_limit_usd:.2f}",
        ]
    )
    console.print(Panel(body, title=f"[bold]{scn.name}[/bold]", expand=False))


def _print_dry_run(scn: Scenario, prompts: list[str]) -> None:
    model = scn.provider_config.model
    # Each concurrency level runs the full prompt set, so per-level cost is the
    # same; total scales by the number of levels.
    per_level_cost = sum(estimate_text_cost(p, scn.provider, model) for p in prompts)

    table = Table(title="Dry run — execution plan")
    table.add_column("Concurrency", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Est. cost", justify="right")
    for level in scn.concurrency:
        table.add_row(str(level), str(len(prompts)), f"${per_level_cost:.4f}")
    console.print(table)

    total_requests = len(prompts) * len(scn.concurrency)
    total_cost = per_level_cost * len(scn.concurrency)
    over_budget = total_cost > scn.budget_limit_usd
    status = (
        "[red]exceeds budget[/red]" if over_budget else "[green]within budget[/green]"
    )
    console.print(
        f"\n[bold]Total:[/bold] {total_requests} requests, "
        f"est. ${total_cost:.4f} / ${scn.budget_limit_usd:.2f} budget ({status})"
    )
    console.print(
        "[dim]Cost is an estimate from per-character provider pricing; actual "
        "cost depends on the provider response.[/dim]"
    )
