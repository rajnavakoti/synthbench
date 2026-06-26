"""Tests for the provider registry."""

import pytest

from synthbench.config.scenario import ProviderConfig
from synthbench.providers.base import ProviderError
from synthbench.providers.elevenlabs import ElevenLabsAdapter
from synthbench.providers.registry import available_providers, create_adapter


def test_create_adapter_returns_elevenlabs() -> None:
    config = ProviderConfig(api_key="key", voice_id="vv")
    adapter = create_adapter("elevenlabs", config)
    assert isinstance(adapter, ElevenLabsAdapter)


def test_create_adapter_is_case_insensitive() -> None:
    config = ProviderConfig(api_key="key", voice_id="vv")
    assert isinstance(create_adapter("ElevenLabs", config), ElevenLabsAdapter)


def test_unknown_provider_raises_with_available_list() -> None:
    with pytest.raises(ProviderError) as exc:
        create_adapter("does-not-exist", ProviderConfig())
    message = str(exc.value)
    assert "unknown provider" in message
    assert "elevenlabs" in message


def test_available_providers_lists_elevenlabs() -> None:
    assert "elevenlabs" in available_providers()
