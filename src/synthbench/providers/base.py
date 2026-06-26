"""Abstract provider adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RateLimitInfo:
    """Rate limit information parsed from provider response."""

    remaining: int | None = None
    reset_seconds: float | None = None
    limit: int | None = None


class ProviderAdapter(ABC):
    """Base class for all provider adapters.

    Each adapter normalizes a provider-specific API into a common interface
    for the workload engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display and config matching."""

    @abstractmethod
    async def submit(self, prompt: str, params: dict) -> str:
        """Submit a generation request. Returns a job/request ID."""

    @abstractmethod
    async def poll(self, job_id: str) -> bool:
        """Check if a job is complete. Returns True when done."""

    @abstractmethod
    async def retrieve(self, job_id: str) -> bytes:
        """Retrieve the generated binary artifact."""

    @abstractmethod
    def estimate_cost(self, prompt: str, params: dict) -> float:
        """Estimate cost in USD for a single generation."""

    def parse_rate_limit(self, response_headers: dict) -> RateLimitInfo:
        """Parse rate limit info from response headers. Override per provider."""
        return RateLimitInfo()
