"""Abstract provider adapter interface and its generation vocabulary.

The adapter layer normalizes provider-specific APIs into one lifecycle the
workload engine drives:

    GenerationRequest  ->  GenerationJob  ->  GenerationArtifact
       (what we ask)        (lifecycle)        (raw bytes + type)

From a terminal job the *engine* (not the adapter) builds the benchmarkable
``GenerationResult`` in ``synthbench.models`` — adapters return raw outputs;
the engine owns timing, cost roll-up, and scoring. Keeping that boundary sharp
is what lets one adapter instance stay stateless and reusable across concurrent
requests: all per-request state lives on the ``GenerationJob``, never on the
adapter.

The lifecycle is modeled on async job queues (the general case for image and
video providers in later phases). Synchronous providers — most TTS APIs return
audio in a single response — complete the work in ``submit`` and return a job
already in a terminal ``SUCCEEDED`` state carrying the artifact, so ``poll`` and
``retrieve`` are trivial.
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# A plain USD amount. Aliased for readability at call sites and to leave room to
# promote it to a richer CostEstimate later without churning every signature.
CostUsd = float


class ProviderError(Exception):
    """Raised when a provider request fails or an adapter is misconfigured.

    Carries a user-facing message; the CLI surfaces it without a traceback.
    Transport/HTTP failures raise this (the engine catches it and records a
    failed result). A remote job that *reports* failure is instead surfaced as
    ``GenerationStatus.FAILED`` on the job — see the module docstring.
    """


def resolve_api_key(config_value: str | None, env_var: str, provider_label: str) -> str:
    """Resolve an API key from a scenario value or the environment.

    The scenario loader expands ``${VAR}`` references, so a resolved value is
    used directly. An unresolved token (the env var was unset at load time) or
    an omitted key falls back to reading ``env_var`` from the environment.
    Shared by all adapters so auth resolution lives in one place.
    """
    if config_value and "${" not in config_value:
        return config_value
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    raise ProviderError(
        f"missing {provider_label} API key: set the {env_var} environment "
        "variable or provide 'api_key' in the scenario"
    )


class GenerationStatus(StrEnum):
    """Lifecycle state of a single generation request."""

    PENDING = "pending"  # submitted, not yet started (async queue)
    RUNNING = "running"  # in progress (async)
    SUCCEEDED = "succeeded"  # artifact available
    FAILED = "failed"  # provider/remote-reported failure (error set)
    CANCELLED = "cancelled"  # engine stopped tracking (budget guard, timeout)


@dataclass
class RateLimitInfo:
    """Rate limit state parsed from a provider response."""

    remaining: int | None = None
    reset_seconds: float | None = None
    limit: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable description of one generation to perform.

    ``provider_params`` carries provider-specific knobs (voice_id, output
    format, guidance scale…) that the matching adapter understands; it is named
    explicitly to keep the boundary with future engine-level request fields
    clear.
    """

    prompt: str
    provider: str
    model: str | None = None
    provider_params: Mapping[str, Any] = field(default_factory=dict)
    index: int | None = None  # engine-assigned position within a run

    @property
    def input_hash(self) -> str:
        """Stable hash of the generation inputs — for dedup and baselines.

        Excludes ``index`` (a run-position detail, not a content input) so the
        same prompt/model/params hashes identically across runs.
        """
        payload = json.dumps(
            {
                "provider": self.provider,
                "model": self.model,
                "prompt": self.prompt,
                "params": {
                    k: self.provider_params[k] for k in sorted(self.provider_params)
                },
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GenerationArtifact:
    """The binary output of a successful generation."""

    data: bytes
    content_type: str = "application/octet-stream"

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass
class GenerationJob:
    """Mutable lifecycle handle for one in-flight or completed request.

    Synchronous providers return a job already ``SUCCEEDED`` with ``artifact``
    set. Async providers return a ``PENDING`` job whose ``provider_job_id`` is
    polled until terminal, then downloaded via ``retrieve``.
    """

    request: GenerationRequest
    status: GenerationStatus = GenerationStatus.PENDING
    provider_job_id: str | None = None
    artifact: GenerationArtifact | None = None
    error: str | None = None
    estimated_cost_usd: CostUsd = 0.0
    actual_cost_usd: CostUsd | None = None
    response_headers: dict[str, str] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            GenerationStatus.SUCCEEDED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        )


class ProviderAdapter(ABC):
    """Base class for all provider adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, used for display and config matching."""

    @abstractmethod
    def estimate_cost_usd(self, request: GenerationRequest) -> CostUsd:
        """Estimate the USD cost of a single generation before submitting it."""

    @abstractmethod
    async def submit(self, request: GenerationRequest) -> GenerationJob:
        """Submit a generation request.

        Synchronous providers perform the full round-trip and return a terminal
        job. Async providers enqueue the work and return a ``PENDING`` job
        carrying the provider-native id. Raises ``ProviderError`` on a
        transport/HTTP failure.
        """

    @abstractmethod
    async def poll(self, job: GenerationJob) -> GenerationJob:
        """Refresh and return ``job`` — querying the remote when needed.

        For synchronous providers the job is already terminal and is returned
        unchanged.
        """

    @abstractmethod
    async def retrieve(self, job: GenerationJob) -> GenerationArtifact:
        """Return the artifact for a succeeded ``job``."""

    def parse_rate_limit(self, response_headers: Mapping[str, str]) -> RateLimitInfo:
        """Parse rate limit info from response headers. Override per provider."""
        return RateLimitInfo()
