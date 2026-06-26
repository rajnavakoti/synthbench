"""Abstract provider adapter interface.

Each adapter normalizes a provider-specific API into one interface the workload
engine drives: ``submit`` a request, ``poll`` until it is done, ``retrieve`` the
artifact, plus ``estimate_cost`` and ``parse_rate_limit``.

The lifecycle is modeled around async job queues (the general case for image and
video providers in later phases). Synchronous providers — most TTS APIs return
audio in a single response — complete the work in ``submit`` and hand back a
``GenerationJob`` that already carries the artifact, so ``poll`` and ``retrieve``
are trivial. Carrying state on the job (not the adapter) keeps a single adapter
instance safe to share across concurrent requests.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ProviderError(Exception):
    """Raised when a provider request fails or an adapter is misconfigured.

    Carries a user-facing message; the CLI surfaces it without a traceback.
    """


@dataclass
class RateLimitInfo:
    """Rate limit state parsed from a provider response."""

    remaining: int | None = None
    reset_seconds: float | None = None
    limit: int | None = None


@dataclass
class GenerationJob:
    """Handle to an in-flight or completed generation request.

    Synchronous providers return a job with ``done=True`` and ``artifact`` set.
    Async providers return a pending job whose ``id`` the engine polls until the
    artifact is ready, then downloads via ``retrieve``.
    """

    id: str
    done: bool = False
    artifact: bytes | None = None
    response_headers: dict[str, str] = field(default_factory=dict)


class ProviderAdapter(ABC):
    """Base class for all provider adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, used for display and config matching."""

    @abstractmethod
    async def submit(self, prompt: str, params: dict) -> GenerationJob:
        """Submit a generation request.

        For synchronous providers this performs the full round-trip and returns
        a completed job. For async providers it enqueues the work and returns a
        pending job carrying the remote id.
        """

    @abstractmethod
    async def poll(self, job: GenerationJob) -> bool:
        """Return ``True`` when ``job`` has completed, polling the remote if needed."""

    @abstractmethod
    async def retrieve(self, job: GenerationJob) -> bytes:
        """Return the generated binary artifact for a completed ``job``."""

    @abstractmethod
    def estimate_cost(self, prompt: str, params: dict) -> float:
        """Estimate the USD cost of a single generation before submitting it."""

    def parse_rate_limit(self, response_headers: dict) -> RateLimitInfo:
        """Parse rate limit info from response headers. Override per provider."""
        return RateLimitInfo()
