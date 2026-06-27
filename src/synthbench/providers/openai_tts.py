"""OpenAI text-to-speech provider adapter.

OpenAI TTS is synchronous: a single POST to ``/audio/speech`` returns the audio
bytes directly. ``submit`` performs that round-trip and returns a terminal
:class:`GenerationJob`; ``poll`` returns it unchanged and ``retrieve`` reads the
artifact off the job — the same shape as the ElevenLabs adapter, which is the
point: it proves the interface generalizes across providers.
"""

import aiohttp

from synthbench.config.scenario import ProviderConfig
from synthbench.pricing import estimate_text_cost
from synthbench.providers.base import (
    CostUsd,
    GenerationArtifact,
    GenerationJob,
    GenerationRequest,
    GenerationStatus,
    ProviderAdapter,
    ProviderError,
    RateLimitInfo,
    resolve_api_key,
)

DEFAULT_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"
DEFAULT_RESPONSE_FORMAT = "mp3"
ENV_API_KEY = "OPENAI_API_KEY"


def _content_type(response_format: str) -> str:
    """Map an OpenAI response_format to a MIME content type."""
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/L16",
    }.get(response_format, "application/octet-stream")


class OpenAITTSAdapter(ProviderAdapter):
    """Adapter for the OpenAI TTS API (``/v1/audio/speech``)."""

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        *,
        cost_per_million_chars: float | None = None,
        base_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.response_format = response_format
        self.cost_per_million_chars = cost_per_million_chars
        self.base_url = base_url or self.BASE_URL
        self._session = session
        self._owns_session = session is None

    @classmethod
    def from_config(cls, config: ProviderConfig) -> "OpenAITTSAdapter":
        """Build an adapter from a validated ``[provider.openai]`` section.

        ``voice`` defaults to ``alloy`` — OpenAI voices are a fixed, well-known
        set, so a minimal scenario works out of the box.
        """
        return cls(
            api_key=resolve_api_key(config.api_key, ENV_API_KEY, "OpenAI"),
            voice=getattr(config, "voice", None) or DEFAULT_VOICE,
            model=config.model or DEFAULT_MODEL,
            response_format=getattr(config, "response_format", None)
            or DEFAULT_RESPONSE_FORMAT,
            cost_per_million_chars=config.cost_per_million_chars,
        )

    @property
    def name(self) -> str:
        return "openai"

    def estimate_cost_usd(self, request: GenerationRequest) -> CostUsd:
        model = request.model or self.model
        return estimate_text_cost(
            request.prompt,
            "openai",
            model,
            override_per_million=self.cost_per_million_chars,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def submit(self, request: GenerationRequest) -> GenerationJob:
        session = await self._get_session()
        model = request.model or self.model
        voice = request.provider_params.get("voice", self.voice)
        response_format = request.provider_params.get(
            "response_format", self.response_format
        )
        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "input": request.prompt,
            "voice": voice,
            "response_format": response_format,
        }

        async with session.post(url, json=body, headers=headers) as response:
            data = await response.read()
            resp_headers = dict(response.headers)
            if response.status != 200:
                snippet = data[:200].decode("utf-8", errors="replace")
                raise ProviderError(
                    f"OpenAI returned HTTP {response.status}: {snippet}"
                )

        return GenerationJob(
            request=request,
            status=GenerationStatus.SUCCEEDED,
            provider_job_id=resp_headers.get("x-request-id"),
            artifact=GenerationArtifact(
                data=data, content_type=_content_type(response_format)
            ),
            estimated_cost_usd=self.estimate_cost_usd(request),
            response_headers=resp_headers,
        )

    async def poll(self, job: GenerationJob) -> GenerationJob:
        # Synchronous provider — the job is already terminal after submit.
        return job

    async def retrieve(self, job: GenerationJob) -> GenerationArtifact:
        if job.artifact is None:
            raise ProviderError("OpenAI job has no artifact to retrieve")
        return job.artifact

    def parse_rate_limit(self, response_headers: dict) -> RateLimitInfo:
        headers = {k.lower(): v for k, v in response_headers.items()}
        info = RateLimitInfo()
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                info.reset_seconds = float(retry_after)
            except ValueError:
                pass
        remaining = headers.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            try:
                info.remaining = int(remaining)
            except ValueError:
                pass
        limit = headers.get("x-ratelimit-limit-requests")
        if limit is not None:
            try:
                info.limit = int(limit)
            except ValueError:
                pass
        return info

    async def aclose(self) -> None:
        """Close the HTTP session if this adapter created it."""
        if self._session is not None and self._owns_session:
            await self._session.close()
            self._session = None
