"""The ``synthbench run`` command."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from synthbench.config.scenario import Scenario, ScenarioError, load_scenario
from synthbench.engine.runner import ProgressEvent, run_scenario
from synthbench.models import RunResult
from synthbench.pricing import estimate_text_cost
from synthbench.providers.base import ProviderAdapter, ProviderError
from synthbench.providers.registry import create_adapter
from synthbench.reporting.analysis import analyze
from synthbench.reporting.json_report import write_report
from synthbench.reporting.terminal import print_report
from synthbench.scoring.base import Scorer, ScoringError
from synthbench.scoring.registry import build_scorers

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
        err_console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    _print_summary(scn, prompts)

    if dry_run:
        _print_dry_run(scn, prompts)
        return

    try:
        adapter = create_adapter(scn.provider, scn.provider_config)
        scorers = build_scorers(scn.scoring)
    except (ProviderError, ScoringError) as exc:
        err_console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    result = _execute(scn, prompts, adapter, scorers)

    summary = analyze(result, scn.thresholds)
    print_report(result, summary, console=console)
    if output is not None:
        write_report(result, scn.thresholds, summary, output)
        console.print(f"\n[dim]JSON report written to {escape(str(output))}[/dim]")


def _execute(
    scn: Scenario,
    prompts: list[str],
    adapter: ProviderAdapter,
    scorers: list[Scorer],
) -> RunResult:
    total_requests = sum(level * scn.requests_multiplier for level in scn.concurrency)
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]Benchmarking[/bold]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} reqs"),
        TextColumn("[dim]{task.fields[cost]}[/dim]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("run", total=total_requests, cost="$0.0000")
        budget = scn.budget_limit_usd

        def on_progress(event: ProgressEvent) -> None:
            progress.update(
                task,
                advance=1,
                cost=f"${event.cumulative_cost:.4f} / ${budget:.2f}",
            )

        async def _amain() -> RunResult:
            try:
                return await run_scenario(
                    scn, adapter, prompts, scorers=scorers, on_progress=on_progress
                )
            finally:
                await adapter.aclose()

        return asyncio.run(_amain())


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
    override = scn.provider_config.cost_per_million_chars
    k = scn.requests_multiplier
    # Per-prompt cost; the engine cycles the prompt set to fill each level.
    prompt_costs = [
        estimate_text_cost(p, scn.provider, model, override_per_million=override)
        for p in prompts
    ]
    num_prompts = len(prompts)

    table = Table(title="Dry run — execution plan")
    table.add_column("Concurrency", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Est. cost", justify="right")

    total_requests = 0
    total_cost = 0.0
    for level in scn.concurrency:
        n_requests = level * k
        level_cost = sum(prompt_costs[i % num_prompts] for i in range(n_requests))
        total_requests += n_requests
        total_cost += level_cost
        table.add_row(str(level), str(n_requests), f"${level_cost:.4f}")
    console.print(table)

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
