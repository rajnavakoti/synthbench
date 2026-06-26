"""Tests for scorer base class and models."""

from synthbench.models import GenerationResult, ScoreResult, Verdict


def test_score_result_defaults() -> None:
    result = ScoreResult(metric="wer", value=0.02, unit="ratio")
    assert result.verdict == Verdict.PASS
    assert result.detail == ""


def test_generation_result_defaults() -> None:
    gen = GenerationResult(
        prompt="hello",
        concurrency_level=1,
        latency_seconds=1.5,
        cost_usd=0.01,
    )
    assert gen.success is True
    assert gen.scores == []


def test_verdict_values() -> None:
    assert Verdict.PASS.value == "PASS"
    assert Verdict.WARN.value == "WARN"
    assert Verdict.FAIL.value == "FAIL"
