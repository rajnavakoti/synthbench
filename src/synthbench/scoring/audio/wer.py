"""Word Error Rate scorer — the core TTS quality signal.

Transcribes the generated audio with Whisper and compares it to the input text
with jiwer. A rising WER under load is the silent degradation synthbench exists
to catch: the model says fewer of the right words before it ever fails outright.
"""

import asyncio
import re
import tempfile
from collections.abc import Callable

from synthbench.models import ScoreResult
from synthbench.scoring.base import Scorer, load_optional

DEFAULT_MODEL_SIZE = "base"

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for fair WER.

    Pure and dependency-free, so the comparison is identical regardless of which
    transcriber produced the text.
    """
    text = _PUNCTUATION.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()


class WERScorer(Scorer):
    """Whisper transcription + jiwer WER against the input text."""

    required_modules = ("whisper", "jiwer")

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        transcribe_fn: Callable[[bytes], str] | None = None,
        wer_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        # transcribe_fn / wer_fn are injectable so the scorer's logic is testable
        # without Whisper or jiwer installed (and without a model download).
        self.model_size = model_size
        self._transcribe_fn = transcribe_fn
        self._wer_fn = wer_fn
        self._model = None

    @property
    def metric_name(self) -> str:
        return "wer"

    async def score(
        self, artifact: bytes, prompt: str, **kwargs: object
    ) -> ScoreResult:
        transcription = await asyncio.to_thread(self._transcribe, artifact)
        reference = normalize_text(prompt)
        hypothesis = normalize_text(transcription)
        wer = 0.0 if not reference else self._compute_wer(reference, hypothesis)
        return ScoreResult(
            metric=self.metric_name,
            value=float(wer),
            unit="ratio",
            detail=f"heard: {transcription!r}",
        )

    def _transcribe(self, artifact: bytes) -> str:
        if self._transcribe_fn is not None:
            return self._transcribe_fn(artifact)
        whisper = load_optional("whisper")
        if self._model is None:
            self._model = whisper.load_model(self.model_size)
        with tempfile.NamedTemporaryFile(suffix=".audio") as tmp:
            tmp.write(artifact)
            tmp.flush()
            result = self._model.transcribe(tmp.name)
        return str(result.get("text", "")).strip()

    def _compute_wer(self, reference: str, hypothesis: str) -> float:
        if self._wer_fn is not None:
            return self._wer_fn(reference, hypothesis)
        jiwer = load_optional("jiwer")
        return jiwer.wer(reference, hypothesis)
