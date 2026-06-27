"""Tests for the scorer registry and optional-dependency handling."""

import importlib.util

import pytest

from synthbench.config.scenario import ScoringConfig
from synthbench.scoring.audio.file_integrity import FileIntegrityScorer
from synthbench.scoring.audio.wer import WERScorer
from synthbench.scoring.base import ScoringError, load_optional
from synthbench.scoring.registry import build_scorers


def test_engine_metrics_produce_no_scorers() -> None:
    assert build_scorers(ScoringConfig(metrics=["latency", "cost"])) == []


def test_unknown_metric_raises() -> None:
    with pytest.raises(ScoringError) as exc:
        build_scorers(ScoringConfig(metrics=["bogus"]))
    assert "unknown scoring metric" in str(exc.value)


def test_builds_available_scorers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    scorers = build_scorers(ScoringConfig(metrics=["latency", "wer", "file_integrity"]))
    assert any(isinstance(s, WERScorer) for s in scorers)
    assert any(isinstance(s, FileIntegrityScorer) for s in scorers)


def test_whisper_model_size_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    scorers = build_scorers(ScoringConfig(metrics=["wer"], whisper_model="small"))
    wer = next(s for s in scorers if isinstance(s, WERScorer))
    assert wer.model_size == "small"


def test_missing_dependency_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ScoringError) as exc:
        build_scorers(ScoringConfig(metrics=["wer"]))
    assert "synthbench[audio]" in str(exc.value)


def test_load_optional_missing_raises() -> None:
    with pytest.raises(ScoringError) as exc:
        load_optional("definitely_not_a_real_module_zzz")
    assert "synthbench[audio]" in str(exc.value)
