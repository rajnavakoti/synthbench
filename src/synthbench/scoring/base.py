"""Abstract scorer interface and optional-dependency helpers.

Quality scorers depend on heavy, opt-in libraries (Whisper, jiwer, soundfile)
shipped via the ``[audio]`` extra. Scorers lazy-import them and raise a clear
``ScoringError`` when missing, so the core package stays light and a user
without the extra gets an actionable message instead of an ImportError.
"""

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Iterable
from importlib import import_module
from typing import Any

from synthbench.models import ScoreResult


class ScoringError(Exception):
    """Raised when a scorer is misconfigured or its dependencies are missing.

    Carries a user-facing message; the CLI surfaces it without a traceback.
    """


def load_optional(module_name: str, *, extra: str = "audio") -> Any:
    """Import an optional module, or raise ``ScoringError`` with an install hint."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ScoringError(
            f"'{module_name}' is required for this scorer. "
            f"Install audio support with: pip install 'synthbench[{extra}]'"
        ) from exc


def require_available(
    metric: str, modules: Iterable[str], *, extra: str = "audio"
) -> None:
    """Fail fast if any module for ``metric`` is not importable.

    Uses ``find_spec`` so it does not import heavy packages (e.g. torch) — just
    checks availability before a run starts, so a missing dependency fails
    before any paid API call.
    """
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        raise ScoringError(
            f"scoring metric '{metric}' needs {', '.join(missing)}. "
            f"Install audio support with: pip install 'synthbench[{extra}]'"
        )


class Scorer(ABC):
    """Base class for all quality scorers."""

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Name of the metric this scorer produces (matches ScoreResult.metric)."""

    #: Modules that must be importable for this scorer to run for real.
    required_modules: tuple[str, ...] = ()

    @abstractmethod
    async def score(self, artifact: bytes, prompt: str, **kwargs: Any) -> ScoreResult:
        """Score a single generated artifact. Returns a ScoreResult."""
