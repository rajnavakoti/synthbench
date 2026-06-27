"""Audio-duration accuracy scorer.

Compares the generated audio's actual duration to an estimate from the text
length, surfacing truncation (too short) or runaway/looping output (too long).
Reported as a ratio: actual / expected, where 1.0 is on target.
"""

import asyncio
import io
from collections.abc import Callable

from synthbench.models import ScoreResult
from synthbench.scoring.base import Scorer, load_optional

# Rough conversational speaking rate. Configurable per scorer.
DEFAULT_CHARS_PER_SECOND = 15.0


class DurationScorer(Scorer):
    """Scores actual audio duration against a text-length estimate."""

    required_modules = ("soundfile",)

    def __init__(
        self,
        chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
        *,
        duration_fn: Callable[[bytes], float] | None = None,
    ) -> None:
        # duration_fn(bytes) -> seconds. Injectable for tests; defaults to
        # soundfile.
        self.chars_per_second = chars_per_second
        self._duration_fn = duration_fn

    @property
    def metric_name(self) -> str:
        return "duration_accuracy"

    async def score(
        self, artifact: bytes, prompt: str, **kwargs: object
    ) -> ScoreResult:
        actual = await asyncio.to_thread(self._duration, artifact)
        expected = max(len(prompt) / self.chars_per_second, 1e-6)
        ratio = actual / expected
        return ScoreResult(
            metric=self.metric_name,
            value=float(ratio),
            unit="ratio",
            detail=f"actual {actual:.2f}s vs expected {expected:.2f}s",
        )

    def _duration(self, artifact: bytes) -> float:
        if self._duration_fn is not None:
            return self._duration_fn(artifact)
        soundfile = load_optional("soundfile")
        with soundfile.SoundFile(io.BytesIO(artifact)) as audio:
            return len(audio) / audio.samplerate
