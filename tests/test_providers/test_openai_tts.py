"""Tests for the OpenAI TTS adapter, using a mocked HTTP session."""

import pytest

from synthbench.config.scenario import ProviderConfig
from synthbench.pricing import estimate_text_cost
from synthbench.providers.base import (
    GenerationJob,
    GenerationRequest,
    GenerationStatus,
    ProviderError,
)
from synthbench.providers.openai_tts import OpenAITTSAdapter


def _request(
    prompt: str = "hello world",
    model: str | None = None,
    **params: object,
) -> GenerationRequest:
    return GenerationRequest(
        prompt=prompt, provider="openai", model=model, provider_params=params
    )


class _FakeResponse:
    def __init__(
        self, status: int = 200, body: bytes = b"FAKE_MP3", headers: dict | None = None
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append((url, kwargs))
        return self._response


async def test_submit_returns_succeeded_job_with_artifact() -> None:
    session = _FakeSession(
        _FakeResponse(status=200, body=b"AUDIODATA", headers={"x-request-id": "rq-9"})
    )
    adapter = OpenAITTSAdapter(api_key="key", session=session)

    job = await adapter.submit(_request())

    assert job.status is GenerationStatus.SUCCEEDED
    assert job.provider_job_id == "rq-9"
    assert job.estimated_cost_usd > 0
    assert (await adapter.poll(job)).status is GenerationStatus.SUCCEEDED
    artifact = await adapter.retrieve(job)
    assert artifact.data == b"AUDIODATA"
    assert artifact.content_type == "audio/mpeg"


async def test_submit_sends_openai_request_shape() -> None:
    session = _FakeSession(_FakeResponse())
    adapter = OpenAITTSAdapter(
        api_key="secret", voice="nova", model="tts-1-hd", session=session
    )

    await adapter.submit(_request(prompt="speak this", model="tts-1-hd"))

    url, kwargs = session.calls[0]
    assert url.endswith("/audio/speech")
    assert kwargs["headers"]["Authorization"] == "Bearer secret"
    assert kwargs["json"] == {
        "model": "tts-1-hd",
        "input": "speak this",
        "voice": "nova",
        "response_format": "mp3",
    }


async def test_submit_non_200_raises_provider_error() -> None:
    session = _FakeSession(_FakeResponse(status=401, body=b"unauthorized"))
    adapter = OpenAITTSAdapter(api_key="bad", session=session)
    with pytest.raises(ProviderError) as exc:
        await adapter.submit(_request())
    assert "401" in str(exc.value)


async def test_retrieve_without_artifact_raises() -> None:
    adapter = OpenAITTSAdapter(api_key="key")
    job = GenerationJob(request=_request(), status=GenerationStatus.SUCCEEDED)
    with pytest.raises(ProviderError):
        await adapter.retrieve(job)


async def test_provider_params_override_voice_and_format() -> None:
    session = _FakeSession(_FakeResponse())
    adapter = OpenAITTSAdapter(api_key="key", session=session)

    await adapter.submit(_request(prompt="hi", voice="shimmer", response_format="wav"))

    _, kwargs = session.calls[0]
    assert kwargs["json"]["voice"] == "shimmer"
    assert kwargs["json"]["response_format"] == "wav"


async def test_wav_response_sets_content_type() -> None:
    session = _FakeSession(_FakeResponse(body=b"WAVDATA"))
    adapter = OpenAITTSAdapter(api_key="key", session=session)
    job = await adapter.submit(_request(response_format="wav"))
    artifact = await adapter.retrieve(job)
    assert artifact.content_type == "audio/wav"


def test_estimate_cost_usd_uses_openai_pricing() -> None:
    adapter = OpenAITTSAdapter(api_key="key", model="tts-1")
    request = _request(prompt="The quick brown fox.")
    expected = estimate_text_cost(request.prompt, "openai", "tts-1")
    assert adapter.estimate_cost_usd(request) == expected


def test_estimate_cost_usd_uses_scenario_override() -> None:
    adapter = OpenAITTSAdapter(api_key="key", cost_per_million_chars=50.0)
    request = _request(prompt="hello")
    assert adapter.estimate_cost_usd(request) == len("hello") * (50.0 / 1_000_000)


def test_from_config_defaults_voice_to_alloy() -> None:
    config = ProviderConfig(api_key="sk-1", model="tts-1")
    adapter = OpenAITTSAdapter.from_config(config)
    assert adapter.voice == "alloy"
    assert adapter.model == "tts-1"


def test_from_config_reads_voice_and_override() -> None:
    config = ProviderConfig(
        api_key="sk-1", model="tts-1-hd", voice="echo", cost_per_million_chars=30.0
    )
    adapter = OpenAITTSAdapter.from_config(config)
    assert adapter.voice == "echo"
    assert adapter.model == "tts-1-hd"
    assert adapter.cost_per_million_chars == 30.0


def test_from_config_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    config = ProviderConfig(api_key="${OPENAI_API_KEY}")
    adapter = OpenAITTSAdapter.from_config(config)
    assert adapter.api_key == "env-key"


def test_from_config_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ProviderConfig()
    with pytest.raises(ProviderError) as exc:
        OpenAITTSAdapter.from_config(config)
    assert "API key" in str(exc.value)


def test_parse_rate_limit_reads_openai_headers() -> None:
    adapter = OpenAITTSAdapter(api_key="key")
    info = adapter.parse_rate_limit(
        {
            "retry-after": "12",
            "x-ratelimit-remaining-requests": "7",
            "x-ratelimit-limit-requests": "50",
        }
    )
    assert info.reset_seconds == 12.0
    assert info.remaining == 7
    assert info.limit == 50
