"""Render the degradation curve as a Rich terminal table — the product output."""

from rich.console import Console
from rich.table import Table

from synthbench.models import RunResult, Verdict
from synthbench.reporting.analysis import DegradationSummary

_VERDICT_STYLE = {
    Verdict.PASS: "green",
    Verdict.WARN: "yellow",
    Verdict.FAIL: "red",
}


def print_report(
    run: RunResult,
    summary: DegradationSummary,
    *,
    console: Console | None = None,
) -> None:
    """Print the per-level curve table plus the degradation summary.

    Verdicts must already be assigned (call ``analyze`` first).
    """
    console = console or Console()
    title = f"Synthbench degradation curve — {run.provider} {run.model}".strip()
    table = Table(title=title)
    table.add_column("Concurrency", justify="right")
    table.add_column("Requests", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Status", justify="center")

    for level in run.concurrency_results:
        wer = f"{level.avg_wer:.3f}" if level.avg_wer is not None else "—"
        requests = str(len(level.generations))
        if level.incomplete:
            requests += f" (+{level.incomplete})"
        style = _VERDICT_STYLE[level.verdict]
        table.add_row(
            str(level.concurrency_level),
            requests,
            f"{level.avg_latency:.2f}s",
            f"{level.p50_latency:.2f}s",
            f"{level.p95_latency:.2f}s",
            wer,
            f"{level.success_rate * 100:.0f}%",
            f"${level.total_cost:.4f}",
            f"[{style}]{level.verdict.value}[/{style}]",
        )
    console.print(table)

    budget_status = (
        "[red]budget exceeded[/red]"
        if run.budget_exceeded
        else "[green]within budget[/green]"
    )
    console.print(
        f"\n[bold]Total:[/bold] ${run.total_cost:.4f} / "
        f"${run.budget_limit_usd:.2f} ({budget_status}) in "
        f"{run.total_duration_seconds:.1f}s"
    )

    if summary.onset_concurrency is not None:
        console.print(
            f"\n[bold yellow]Quality degradation detected at concurrency ≥ "
            f"{summary.onset_concurrency}[/bold yellow]"
        )
        for note in summary.notes:
            console.print(f"  - {note}")
    else:
        console.print(
            "\n[bold green]No quality degradation detected across the tested "
            "concurrency range.[/bold green]"
        )
