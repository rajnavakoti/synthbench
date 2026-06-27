"""File-integrity scorer — is the returned audio a valid, non-empty file?

Catches the cheapest TTS failure mode: empty, truncated, or corrupt audio that
still came back with HTTP 200. Decodes with soundfile; a decode failure or
zero-length file is a FAIL.
"""

import asyncio
import io
from collections.abc import Callable

from synthbench.models import ScoreResult, Verdict
from synthbench.scoring.base import Scorer, ScoringError, load_optional


class FileIntegrityScorer(Scorer):
    """Verifies the artifact decodes to a non-empty audio stream."""

    required_modules = ("soundfile",)

    def __init__(self, *, decode_fn: Callable[[bytes], int] | None = None) -> None:
        # decode_fn(bytes) -> frame count. Injectable for tests; defaults to
        # soundfile.
        self._decode_fn = decode_fn

    @property
    def metric_name(self) -> str:
        return "file_integrity"

    async def score(
        self, artifact: bytes, prompt: str, **kwargs: object
    ) -> ScoreResult:
        if not artifact:
            return ScoreResult(
                metric=self.metric_name,
                value=0.0,
                unit="bool",
                verdict=Verdict.FAIL,
                detail="empty audio (zero bytes)",
            )
        try:
            frames = await asyncio.to_thread(self._decode, artifact)
        except ScoringError:
            raise  # missing dependency — surface the install hint
        except Exception as exc:  # noqa: BLE001 - any decode failure means corrupt
            return ScoreResult(
                metric=self.metric_name,
                value=0.0,
                unit="bool",
                verdict=Verdict.FAIL,
                detail=f"decode failed: {exc}",
            )

        valid = frames > 0
        return ScoreResult(
            metric=self.metric_name,
            value=1.0 if valid else 0.0,
            unit="bool",
            verdict=Verdict.PASS if valid else Verdict.FAIL,
            detail=f"{frames} frames" if valid else "decoded to 0 frames",
        )

    def _decode(self, artifact: bytes) -> int:
        if self._decode_fn is not None:
            return self._decode_fn(artifact)
        soundfile = load_optional("soundfile")
        with soundfile.SoundFile(io.BytesIO(artifact)) as audio:
            return len(audio)
