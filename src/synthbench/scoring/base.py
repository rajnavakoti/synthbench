"""Abstract scorer interface."""

from abc import ABC, abstractmethod

from synthbench.models import ScoreResult


class Scorer(ABC):
    """Base class for all quality scorers."""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the metric this scorer produces."""

    @abstractmethod
    async def score(self, artifact: bytes, prompt: str, **kwargs) -> ScoreResult:
        """Score a single generated artifact. Returns a ScoreResult."""
