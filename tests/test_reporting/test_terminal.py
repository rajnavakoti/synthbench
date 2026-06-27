"""Tests for the terminal degradation-curve renderer."""

import io

from rich.console import Console

from synthbench.config.scenario import Thresholds
from synthbench.models import ConcurrencyResult, GenerationResult, RunResult
from synthbench.reporting.analysis import analyze
from synthbench.reporting.terminal import print_report

THRESHOLDS = Thresholds()


def _level(concurrency: int, *, avg_wer: float, p95: float) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency_level=concurrency,
        generations=[
            GenerationResult(
                prompt="x",
                concurrency_level=concurrency,
                latency_seconds=p95,
                cost_usd=0.1,
            )
        ],
        avg_latency=p95,
        p50_latency=p95,
        p95_latency=p95,
        avg_wer=avg_wer,
        total_cost=0.1,
        success_rate=1.0,
    )


def _render(run: RunResult) -> str:
    summary = analyze(run, THRESHOLDS)
    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    print_report(run, summary, console=console)
    return console.file.getvalue()


def test_renders_curve_with_verdicts() -> None:
    run = RunResult(
        scenario_name="s",
        provider="elevenlabs",
        model="m",
        concurrency_results=[
            _level(1, avg_wer=0.02, p95=1.0),
            _level(50, avg_wer=0.15, p95=35.0),
        ],
        total_cost=0.2,
        total_duration_seconds=4.0,
        budget_limit_usd=5.0,
    )
    output = _render(run)
    assert "degradation curve" in output
    assert "PASS" in output
    assert "FAIL" in output
    assert "Quality degradation detected at concurrency" in output


def test_renders_no_degradation_message() -> None:
    run = RunResult(
        scenario_name="s",
        provider="openai",
        model="tts-1",
        concurrency_results=[
            _level(1, avg_wer=0.01, p95=1.0),
            _level(5, avg_wer=0.02, p95=2.0),
        ],
        total_cost=0.2,
        total_duration_seconds=2.0,
        budget_limit_usd=5.0,
    )
    output = _render(run)
    assert "No quality degradation detected" in output
