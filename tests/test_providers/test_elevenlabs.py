"""Tests for the ElevenLabs adapter, using a mocked HTTP session."""

import pytest

from synthbench.config.scenario import ProviderConfig
from synthbench.pricing import estimate_text_cost
from synthbench.providers.base import GenerationJob, ProviderError
from synthbench.providers.elevenlabs import ElevenLabsAdapter


class _FakeResponse:
    """Stands in for an aiohttp response used as an async context manager."""

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


async def test_submit_returns_completed_job_with_artifact() -> None:
    session = _FakeSession(
        _FakeResponse(status=200, body=b"AUDIODATA", headers={"request-id": "req-1"})
    )
    adapter = ElevenLabsAdapter(api_key="key", voice_id="voice", session=session)

    job = await adapter.submit("hello world", {})

    assert job.done is True
    assert job.id == "req-1"
    assert await adapter.poll(job) is True
    assert await adapter.retrieve(job) == b"AUDIODATA"


async def test_submit_sends_correct_request_shape() -> None:
    session = _FakeSession(_FakeResponse())
    adapter = ElevenLabsAdapter(
        api_key="secret", voice_id="voice-x", model="eleven_turbo_v2", session=session
    )

    await adapter.submit("speak this", {})

    url, kwargs = session.calls[0]
    assert url.endswith("/text-to-speech/voice-x")
    assert kwargs["headers"]["xi-api-key"] == "secret"
    assert kwargs["json"] == {"text": "speak this", "model_id": "eleven_turbo_v2"}
    assert kwargs["params"]["output_format"] == "mp3_44100_128"


async def test_submit_non_200_raises_provider_error() -> None:
    session = _FakeSession(_FakeResponse(status=429, body=b"rate limited"))
    adapter = ElevenLabsAdapter(api_key="key", voice_id="voice", session=session)

    with pytest.raises(ProviderError) as exc:
        await adapter.submit("hi", {})
    assert "429" in str(exc.value)


async def test_retrieve_without_artifact_raises() -> None:
    adapter = ElevenLabsAdapter(api_key="key", voice_id="voice")
    job = GenerationJob(id="x", done=True, artifact=None)
    with pytest.raises(ProviderError):
        await adapter.retrieve(job)


async def test_params_override_model_and_format() -> None:
    session = _FakeSession(_FakeResponse())
    adapter = ElevenLabsAdapter(api_key="key", voice_id="voice", session=session)

    await adapter.submit(
        "hi", {"model": "eleven_flash_v2", "output_format": "pcm_16000"}
    )

    _, kwargs = session.calls[0]
    assert kwargs["json"]["model_id"] == "eleven_flash_v2"
    assert kwargs["params"]["output_format"] == "pcm_16000"


def test_estimate_cost_delegates_to_pricing() -> None:
    adapter = ElevenLabsAdapter(
        api_key="key", voice_id="voice", model="eleven_multilingual_v2"
    )
    prompt = "The quick brown fox."
    expected = estimate_text_cost(prompt, "elevenlabs", "eleven_multilingual_v2")
    assert adapter.estimate_cost(prompt, {}) == expected


def test_parse_rate_limit_reads_headers() -> None:
    adapter = ElevenLabsAdapter(api_key="key", voice_id="voice")
    info = adapter.parse_rate_limit(
        {
            "Retry-After": "30",
            "x-ratelimit-remaining": "5",
            "x-ratelimit-limit": "100",
        }
    )
    assert info.reset_seconds == 30.0
    assert info.remaining == 5
    assert info.limit == 100


def test_parse_rate_limit_handles_missing_headers() -> None:
    adapter = ElevenLabsAdapter(api_key="key", voice_id="voice")
    info = adapter.parse_rate_limit({})
    assert info.reset_seconds is None
    assert info.remaining is None


def test_from_config_reads_all_fields() -> None:
    config = ProviderConfig(api_key="sk-1", model="eleven_turbo_v2", voice_id="vv")
    adapter = ElevenLabsAdapter.from_config(config)
    assert adapter.api_key == "sk-1"
    assert adapter.voice_id == "vv"
    assert adapter.model == "eleven_turbo_v2"


def test_from_config_defaults_model() -> None:
    config = ProviderConfig(api_key="sk-1", voice_id="vv")
    adapter = ElevenLabsAdapter.from_config(config)
    assert adapter.model == "eleven_multilingual_v2"


def test_from_config_api_key_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    # Unresolved token (env was unset at load time) falls back to the environment.
    config = ProviderConfig(api_key="${ELEVENLABS_API_KEY}", voice_id="vv")
    adapter = ElevenLabsAdapter.from_config(config)
    assert adapter.api_key == "env-key"


def test_from_config_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    config = ProviderConfig(voice_id="vv")
    with pytest.raises(ProviderError) as exc:
        ElevenLabsAdapter.from_config(config)
    assert "API key" in str(exc.value)


def test_from_config_missing_voice_id_raises() -> None:
    config = ProviderConfig(api_key="sk-1")
    with pytest.raises(ProviderError) as exc:
        ElevenLabsAdapter.from_config(config)
    assert "voice_id" in str(exc.value)
