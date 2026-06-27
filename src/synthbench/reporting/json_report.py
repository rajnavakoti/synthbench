"""JSON degradation report — the stable CI/CD integration surface.

The shape here is a contract: Phase 3 builds CI gates on it, so additions should
be backward-compatible and ``SCHEMA_VERSION`` bumped on any breaking change.
"""

import json
from pathlib import Path

from synthbench.config.scenario import Thresholds
from synthbench.models import RunResult
from synthbench.reporting.analysis import DegradationSummary

SCHEMA_VERSION = 1


def build_report(
    run: RunResult, thresholds: Thresholds, summary: DegradationSummary
) -> dict:
    """Build the JSON-serializable report. Verdicts must already be assigned."""
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": run.scenario_name,
        "provider": run.provider,
        "model": run.model,
        "thresholds": {
            "warn_wer": thresholds.warn_wer,
            "fail_wer": thresholds.fail_wer,
            "warn_latency_p95": thresholds.warn_latency_p95,
            "fail_latency_p95": thresholds.fail_latency_p95,
        },
        "levels": [
            {
                "concurrency": level.concurrency_level,
                "requests": len(level.generations),
                "incomplete": level.incomplete,
                "avg_latency_s": round(level.avg_latency, 4),
                "p50_latency_s": round(level.p50_latency, 4),
                "p95_latency_s": round(level.p95_latency, 4),
                "avg_wer": None if level.avg_wer is None else round(level.avg_wer, 4),
                "success_rate": round(level.success_rate, 4),
                "cost_usd": round(level.total_cost, 6),
                "verdict": level.verdict.value,
            }
            for level in run.concurrency_results
        ],
        "summary": {
            "total_cost_usd": round(run.total_cost, 6),
            "total_duration_s": round(run.total_duration_seconds, 2),
            "budget_limit_usd": run.budget_limit_usd,
            "budget_exceeded": run.budget_exceeded,
            "degradation_onset_concurrency": summary.onset_concurrency,
            "degradation_notes": summary.notes,
        },
    }


def write_report(
    run: RunResult,
    thresholds: Thresholds,
    summary: DegradationSummary,
    path: Path,
) -> None:
    """Write the JSON report to ``path``."""
    data = build_report(run, thresholds, summary)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
