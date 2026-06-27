"""Tests for the file-integrity scorer (decode injected — no soundfile needed)."""

import pytest

from synthbench.models import Verdict
from synthbench.scoring.audio import file_integrity as fi_module
from synthbench.scoring.audio.file_integrity import FileIntegrityScorer
from synthbench.scoring.base import ScoringError


async def test_empty_audio_fails() -> None:
    scorer = FileIntegrityScorer(decode_fn=lambda data: 1000)
    result = await scorer.score(b"", "hello")
    assert result.verdict is Verdict.FAIL
    assert result.value == 0.0
    assert "empty" in result.detail


async def test_valid_audio_passes() -> None:
    scorer = FileIntegrityScorer(decode_fn=lambda data: 44100)
    result = await scorer.score(b"audio-bytes", "hello")
    assert result.verdict is Verdict.PASS
    assert result.value == 1.0
    assert "frames" in result.detail


async def test_zero_frames_fails() -> None:
    scorer = FileIntegrityScorer(decode_fn=lambda data: 0)
    result = await scorer.score(b"audio-bytes", "hello")
    assert result.verdict is Verdict.FAIL


async def test_corrupt_audio_fails() -> None:
    def boom(data: bytes) -> int:
        raise ValueError("bad header")

    scorer = FileIntegrityScorer(decode_fn=boom)
    result = await scorer.score(b"audio-bytes", "hello")
    assert result.verdict is Verdict.FAIL
    assert "decode failed" in result.detail


async def test_missing_soundfile_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, **kwargs: object) -> object:
        raise ScoringError("no soundfile")

    monkeypatch.setattr(fi_module, "load_optional", missing)
    scorer = FileIntegrityScorer()
    with pytest.raises(ScoringError):
        await scorer.score(b"audio-bytes", "hello")
