"""Tests for the duration-accuracy scorer (duration injected — no soundfile)."""

from synthbench.scoring.audio.duration import DurationScorer


async def test_on_target_ratio_is_one() -> None:
    # prompt of 20 chars at 10 chars/s -> expected 2.0s; actual 2.0s -> ratio 1.0
    scorer = DurationScorer(chars_per_second=10.0, duration_fn=lambda data: 2.0)
    result = await scorer.score(b"audio", "x" * 20)
    assert abs(result.value - 1.0) < 1e-9
    assert result.metric == "duration_accuracy"
    assert result.unit == "ratio"


async def test_truncated_audio_has_low_ratio() -> None:
    scorer = DurationScorer(chars_per_second=10.0, duration_fn=lambda data: 0.5)
    result = await scorer.score(b"audio", "x" * 20)  # expected 2.0s, actual 0.5s
    assert abs(result.value - 0.25) < 1e-9


async def test_overlong_audio_has_high_ratio() -> None:
    scorer = DurationScorer(chars_per_second=10.0, duration_fn=lambda data: 6.0)
    result = await scorer.score(b"audio", "x" * 20)  # expected 2.0s, actual 6.0s
    assert result.value > 2.0
