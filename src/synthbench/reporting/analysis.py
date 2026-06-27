"""Turn raw run results into PASS/WARN/FAIL verdicts and a degradation summary.

This is where the numbers become the product: each concurrency level gets a
verdict against the scenario's thresholds, and we locate where quality starts to
break — the headline of the degradation curve.
"""

from dataclasses import dataclass, field

from synthbench.config.scenario import Thresholds
from synthbench.models import ConcurrencyResult, RunResult, Verdict

_SEVERITY = {Verdict.PASS: 0, Verdict.WARN: 1, Verdict.FAIL: 2}

# A level with most of its requests failing is itself a failure, independent of
# the latency/WER thresholds.
_FAIL_SUCCESS_RATE = 0.5


def level_verdict(
    level: ConcurrencyResult, thresholds: Thresholds
) -> tuple[Verdict, list[str]]:
    """Compute a level's verdict and the human reasons behind it."""
    verdict = Verdict.PASS
    reasons: list[str] = []

    def escalate(to: Verdict, reason: str) -> None:
        nonlocal verdict
        if _SEVERITY[to] > _SEVERITY[verdict]:
            verdict = to
        reasons.append(reason)

    if level.avg_wer is not None:
        if level.avg_wer >= thresholds.fail_wer:
            escalate(
                Verdict.FAIL, f"WER {level.avg_wer:.3f} ≥ fail {thresholds.fail_wer}"
            )
        elif level.avg_wer >= thresholds.warn_wer:
            escalate(
                Verdict.WARN, f"WER {level.avg_wer:.3f} ≥ warn {thresholds.warn_wer}"
            )

    if level.p95_latency >= thresholds.fail_latency_p95:
        escalate(
            Verdict.FAIL,
            f"P95 {level.p95_latency:.1f}s ≥ fail {thresholds.fail_latency_p95}s",
        )
    elif level.p95_latency >= thresholds.warn_latency_p95:
        escalate(
            Verdict.WARN,
            f"P95 {level.p95_latency:.1f}s ≥ warn {thresholds.warn_latency_p95}s",
        )

    if level.success_rate < 1.0:
        pct = level.success_rate * 100
        if level.success_rate < _FAIL_SUCCESS_RATE:
            escalate(Verdict.FAIL, f"success rate {pct:.0f}%")
        else:
            escalate(Verdict.WARN, f"success rate {pct:.0f}%")

    return verdict, reasons


@dataclass
class DegradationSummary:
    """Where quality first breaks, and why."""

    onset_concurrency: int | None = None
    notes: list[str] = field(default_factory=list)


def analyze(run: RunResult, thresholds: Thresholds) -> DegradationSummary:
    """Assign each level's verdict (in place) and locate degradation onset."""
    baseline = run.concurrency_results[0] if run.concurrency_results else None
    summary = DegradationSummary()

    for level in run.concurrency_results:
        verdict, reasons = level_verdict(level, thresholds)
        level.verdict = verdict
        if verdict is not Verdict.PASS and summary.onset_concurrency is None:
            summary.onset_concurrency = level.concurrency_level
            summary.notes = _onset_notes(baseline, level, reasons)

    return summary


def _onset_notes(
    baseline: ConcurrencyResult | None,
    level: ConcurrencyResult,
    reasons: list[str],
) -> list[str]:
    notes: list[str] = []
    if baseline is not None and baseline is not level:
        if (
            baseline.avg_wer is not None
            and level.avg_wer is not None
            and baseline.avg_wer > 0
        ):
            factor = level.avg_wer / baseline.avg_wer
            notes.append(
                f"WER {baseline.avg_wer:.3f} → {level.avg_wer:.3f} ({factor:.1f}x)"
            )
        notes.append(
            f"P95 latency {baseline.p95_latency:.1f}s → {level.p95_latency:.1f}s"
        )
    notes.extend(reasons)
    return notes
