"""Tests for verdict assignment and degradation analysis."""

from synthbench.config.scenario import Thresholds
from synthbench.models import ConcurrencyResult, GenerationResult, RunResult, Verdict
from synthbench.reporting.analysis import analyze, level_verdict

THRESHOLDS = Thresholds()  # warn_wer 0.05 / fail 0.10, warn p95 10 / fail 30


def _level(
    concurrency: int,
    *,
    avg_wer: float | None = None,
    p95: float = 1.0,
    success_rate: float = 1.0,
    cost: float = 0.1,
    incomplete: int = 0,
    n: int = 2,
) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency_level=concurrency,
        generations=[
            GenerationResult(
                prompt="x",
                concurrency_level=concurrency,
                latency_seconds=p95,
                cost_usd=cost,
            )
            for _ in range(n)
        ],
        avg_latency=p95,
        p50_latency=p95,
        p95_latency=p95,
        avg_wer=avg_wer,
        total_cost=cost,
        success_rate=success_rate,
        incomplete=incomplete,
    )


def _run(levels: list[ConcurrencyResult]) -> RunResult:
    return RunResult(
        scenario_name="s",
        provider="elevenlabs",
        model="m",
        concurrency_results=levels,
        total_cost=sum(level.total_cost for level in levels),
        total_duration_seconds=1.0,
        budget_limit_usd=5.0,
    )


def test_pass_under_thresholds() -> None:
    verdict, reasons = level_verdict(_level(1, avg_wer=0.02, p95=1.0), THRESHOLDS)
    assert verdict is Verdict.PASS
    assert reasons == []


def test_warn_and_fail_on_wer() -> None:
    assert level_verdict(_level(5, avg_wer=0.06), THRESHOLDS)[0] is Verdict.WARN
    assert level_verdict(_level(5, avg_wer=0.12), THRESHOLDS)[0] is Verdict.FAIL


def test_warn_and_fail_on_latency() -> None:
    assert level_verdict(_level(5, p95=12.0), THRESHOLDS)[0] is Verdict.WARN
    assert level_verdict(_level(5, p95=35.0), THRESHOLDS)[0] is Verdict.FAIL


def test_low_success_rate_escalates() -> None:
    assert level_verdict(_level(5, success_rate=0.8), THRESHOLDS)[0] is Verdict.WARN
    assert level_verdict(_level(5, success_rate=0.4), THRESHOLDS)[0] is Verdict.FAIL


def test_worst_signal_wins() -> None:
    # WER fine, latency fails -> FAIL
    verdict, _ = level_verdict(_level(5, avg_wer=0.01, p95=35.0), THRESHOLDS)
    assert verdict is Verdict.FAIL


def test_analyze_assigns_verdicts_and_finds_onset() -> None:
    run = _run(
        [
            _level(1, avg_wer=0.02, p95=1.0),
            _level(10, avg_wer=0.04, p95=2.0),
            _level(25, avg_wer=0.08, p95=12.0),  # WARN: wer + latency
            _level(50, avg_wer=0.15, p95=35.0),  # FAIL
        ]
    )
    summary = analyze(run, THRESHOLDS)
    verdicts = [level.verdict for level in run.concurrency_results]
    assert verdicts == [Verdict.PASS, Verdict.PASS, Verdict.WARN, Verdict.FAIL]
    assert summary.onset_concurrency == 25
    assert any("WER" in note for note in summary.notes)


def test_no_degradation_when_all_pass() -> None:
    run = _run([_level(1, avg_wer=0.01), _level(5, avg_wer=0.02)])
    summary = analyze(run, THRESHOLDS)
    assert summary.onset_concurrency is None
    assert all(level.verdict is Verdict.PASS for level in run.concurrency_results)
