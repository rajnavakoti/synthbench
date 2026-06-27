"""Tests for the provider registry."""

import textwrap
from pathlib import Path

import pytest

from synthbench.config.scenario import ProviderConfig, load_scenario
from synthbench.providers.base import ProviderError
from synthbench.providers.elevenlabs import ElevenLabsAdapter
from synthbench.providers.openai_tts import OpenAITTSAdapter
from synthbench.providers.registry import available_providers, create_adapter


def test_create_adapter_returns_elevenlabs() -> None:
    config = ProviderConfig(api_key="key", voice_id="vv")
    adapter = create_adapter("elevenlabs", config)
    assert isinstance(adapter, ElevenLabsAdapter)


def test_create_adapter_returns_openai() -> None:
    config = ProviderConfig(api_key="key")
    adapter = create_adapter("openai", config)
    assert isinstance(adapter, OpenAITTSAdapter)


def test_create_adapter_is_case_insensitive() -> None:
    config = ProviderConfig(api_key="key", voice_id="vv")
    assert isinstance(create_adapter("ElevenLabs", config), ElevenLabsAdapter)
    assert isinstance(
        create_adapter("OpenAI", ProviderConfig(api_key="k")), OpenAITTSAdapter
    )


def test_unknown_provider_raises_with_available_list() -> None:
    with pytest.raises(ProviderError) as exc:
        create_adapter("does-not-exist", ProviderConfig())
    message = str(exc.value)
    assert "unknown provider" in message
    assert "elevenlabs" in message
    assert "openai" in message


def test_available_providers_lists_both_tts_providers() -> None:
    providers = available_providers()
    assert "elevenlabs" in providers
    assert "openai" in providers


def test_same_scenario_format_works_for_openai(tmp_path: Path) -> None:
    # The identical scenario shape used for ElevenLabs works with provider="openai".
    content = textwrap.dedent("""
        [scenario]
        name = "openai run"
        provider = "openai"
        concurrency = [1, 5]
        budget_limit_usd = 2.0

        [provider.openai]
        api_key = "sk-test"
        model = "tts-1"
        voice = "alloy"

        [prompts]
        source = "inline"
        texts = ["Hello world."]
    """)
    path = tmp_path / "openai.toml"
    path.write_text(content, encoding="utf-8")

    scn = load_scenario(path)
    adapter = create_adapter(scn.provider, scn.provider_config)

    assert isinstance(adapter, OpenAITTSAdapter)
    assert adapter.model == "tts-1"
    assert adapter.voice == "alloy"
