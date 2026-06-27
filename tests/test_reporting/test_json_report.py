"""Tests for the JSON degradation report."""

import json
from pathlib import Path

from synthbench.config.scenario import Thresholds
from synthbench.models import ConcurrencyResult, GenerationResult, RunResult
from synthbench.reporting.analysis import analyze
from synthbench.reporting.json_report import SCHEMA_VERSION, build_report, write_report

THRESHOLDS = Thresholds()


def _run() -> RunResult:
    levels = [
        ConcurrencyResult(
            concurrency_level=1,
            generations=[
                GenerationResult(
                    prompt="x", concurrency_level=1, latency_seconds=1.0, cost_usd=0.1
                )
            ],
            avg_latency=1.0,
            p50_latency=1.0,
            p95_latency=1.0,
            avg_wer=0.02,
            total_cost=0.1,
            success_rate=1.0,
        ),
        ConcurrencyResult(
            concurrency_level=50,
            generations=[
                GenerationResult(
                    prompt="x", concurrency_level=50, latency_seconds=35.0, cost_usd=0.1
                )
            ],
            avg_latency=35.0,
            p50_latency=35.0,
            p95_latency=35.0,
            avg_wer=None,
            total_cost=0.1,
            success_rate=0.4,
            incomplete=3,
        ),
    ]
    return RunResult(
        scenario_name="s",
        provider="elevenlabs",
        model="m",
        concurrency_results=levels,
        total_cost=0.2,
        total_duration_seconds=12.5,
        budget_limit_usd=5.0,
        budget_exceeded=True,
    )


def test_build_report_shape() -> None:
    run = _run()
    summary = analyze(run, THRESHOLDS)
    report = build_report(run, THRESHOLDS, summary)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["provider"] == "elevenlabs"
    assert len(report["levels"]) == 2
    first, second = report["levels"]
    assert first["concurrency"] == 1
    assert first["verdict"] == "PASS"
    assert second["verdict"] == "FAIL"
    assert second["avg_wer"] is None  # serialized as null
    assert second["incomplete"] == 3
    assert report["summary"]["budget_exceeded"] is True
    assert report["summary"]["degradation_onset_concurrency"] == 50
    assert report["thresholds"]["fail_wer"] == THRESHOLDS.fail_wer


def test_write_report_round_trips(tmp_path: Path) -> None:
    run = _run()
    summary = analyze(run, THRESHOLDS)
    path = tmp_path / "report.json"
    write_report(run, THRESHOLDS, summary, path)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["scenario"] == "s"
    assert loaded["levels"][1]["success_rate"] == 0.4
