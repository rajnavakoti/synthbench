"""ElevenLabs text-to-speech provider adapter.

ElevenLabs TTS is synchronous: a single POST to the text-to-speech endpoint
returns the audio bytes directly. ``submit`` performs that round-trip and
returns a completed :class:`GenerationJob`; ``poll`` and ``retrieve`` then read
from the job.
"""

import os

import aiohttp

from synthbench.config.scenario import ProviderConfig
from synthbench.pricing import estimate_text_cost
from synthbench.providers.base import (
    GenerationJob,
    ProviderAdapter,
    ProviderError,
    RateLimitInfo,
)

DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
ENV_API_KEY = "ELEVENLABS_API_KEY"


def _resolve_api_key(config_value: str | None, env_var: str) -> str:
    """Resolve an API key from the scenario value or the environment.

    The scenario loader already expands ``${VAR}`` references, so a resolved
    value is used directly. An unresolved token (env var was not set) or an
    omitted key falls back to reading ``env_var`` directly.
    """
    if config_value and "${" not in config_value:
        return config_value
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    raise ProviderError(
        f"missing ElevenLabs API key: set ${{{env_var}}} or provide "
        "'api_key' in the [provider.elevenlabs] section"
    )


class ElevenLabsAdapter(ProviderAdapter):
    """Adapter for the ElevenLabs TTS API."""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = DEFAULT_MODEL,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        *,
        base_url: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.output_format = output_format
        self.base_url = base_url or self.BASE_URL
        self._session = session
        self._owns_session = session is None

    @classmethod
    def from_config(cls, config: ProviderConfig) -> "ElevenLabsAdapter":
        """Build an adapter from a validated ``[provider.elevenlabs]`` section."""
        voice_id = getattr(config, "voice_id", None)
        if not voice_id:
            raise ProviderError(
                "ElevenLabs requires 'voice_id' in the [provider.elevenlabs] section"
            )
        return cls(
            api_key=_resolve_api_key(config.api_key, ENV_API_KEY),
            voice_id=voice_id,
            model=config.model or DEFAULT_MODEL,
            output_format=getattr(config, "output_format", None)
            or DEFAULT_OUTPUT_FORMAT,
        )

    @property
    def name(self) -> str:
        return "elevenlabs"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def submit(self, prompt: str, params: dict) -> GenerationJob:
        session = await self._get_session()
        model = params.get("model", self.model)
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        body = {"text": prompt, "model_id": model}
        query = {"output_format": params.get("output_format", self.output_format)}

        async with session.post(
            url, json=body, headers=headers, params=query
        ) as response:
            data = await response.read()
            resp_headers = dict(response.headers)
            if response.status != 200:
                snippet = data[:200].decode("utf-8", errors="replace")
                raise ProviderError(
                    f"ElevenLabs returned HTTP {response.status}: {snippet}"
                )

        return GenerationJob(
            id=resp_headers.get("request-id", "elevenlabs"),
            done=True,
            artifact=data,
            response_headers=resp_headers,
        )

    async def poll(self, job: GenerationJob) -> bool:
        # Synchronous provider — the job is already complete after submit.
        return job.done

    async def retrieve(self, job: GenerationJob) -> bytes:
        if job.artifact is None:
            raise ProviderError("ElevenLabs job has no artifact to retrieve")
        return job.artifact

    def estimate_cost(self, prompt: str, params: dict) -> float:
        model = params.get("model", self.model)
        return estimate_text_cost(prompt, "elevenlabs", model)

    def parse_rate_limit(self, response_headers: dict) -> RateLimitInfo:
        headers = {k.lower(): v for k, v in response_headers.items()}
        info = RateLimitInfo()
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                info.reset_seconds = float(retry_after)
            except ValueError:
                pass
        remaining = headers.get("x-ratelimit-remaining")
        if remaining is not None:
            try:
                info.remaining = int(remaining)
            except ValueError:
                pass
        limit = headers.get("x-ratelimit-limit")
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
