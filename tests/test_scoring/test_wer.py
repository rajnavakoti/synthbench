"""Tests for the WER scorer (transcription + WER injected — no whisper/jiwer)."""

import asyncio
import time

import pytest

from synthbench.scoring.audio import wer as wer_module
from synthbench.scoring.audio.wer import WERScorer, normalize_text
from synthbench.scoring.base import ScoringError


def test_normalize_text_lowercases_and_strips_punctuation() -> None:
    assert normalize_text("The Quick, Brown FOX!") == "the quick brown fox"
    assert normalize_text("  multiple   spaces  ") == "multiple spaces"
    assert normalize_text("Numbers 27 and 3.5?") == "numbers 27 and 3 5"


async def test_passes_normalized_text_to_wer_fn() -> None:
    captured: dict[str, str] = {}

    def fake_wer(reference: str, hypothesis: str) -> float:
        captured["ref"] = reference
        captured["hyp"] = hypothesis
        return 0.25

    scorer = WERScorer(transcribe_fn=lambda data: "The QUICK brown!", wer_fn=fake_wer)
    result = await scorer.score(b"audio", "the quick brown")

    assert captured["ref"] == "the quick brown"
    assert captured["hyp"] == "the quick brown"  # normalized before comparison
    assert result.value == 0.25
    assert result.metric == "wer"
    assert "heard" in result.detail


async def test_empty_reference_returns_zero_without_calling_wer() -> None:
    def boom(reference: str, hypothesis: str) -> float:
        raise AssertionError("should not be called for empty reference")

    scorer = WERScorer(transcribe_fn=lambda data: "anything", wer_fn=boom)
    result = await scorer.score(b"audio", "")
    assert result.value == 0.0


async def test_missing_whisper_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, **kwargs: object) -> object:
        raise ScoringError("no whisper")

    monkeypatch.setattr(wer_module, "load_optional", missing)
    scorer = WERScorer()  # no transcribe_fn -> needs whisper
    with pytest.raises(ScoringError):
        await scorer.score(b"audio", "hello")


async def test_concurrent_scoring_serializes_transcription() -> None:
    # Regression test: one Whisper model is shared across concurrent scorings,
    # and its inference is not thread-safe. Transcription must never overlap, or
    # outputs get corrupted (which produced fake "degradation" in a real run).
    state = {"active": 0, "max_active": 0}

    def transcribe_fn(data: bytes) -> str:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.02)  # hold the model so any overlap would be observed
        state["active"] -= 1
        return "the text"

    scorer = WERScorer(transcribe_fn=transcribe_fn, wer_fn=lambda ref, hyp: 0.0)
    await asyncio.gather(*(scorer.score(b"x", "the text") for _ in range(8)))

    assert state["max_active"] == 1  # transcription ran strictly one at a time
