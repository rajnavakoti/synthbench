"""Tests for the pricing estimator."""

from synthbench.pricing import (
    estimate_text_cost,
    per_character_rate,
    per_million_rate,
)


def test_defaults_loaded_from_bundled_data_file() -> None:
    # Values come from the bundled pricing.toml, not a hardcoded dict.
    assert per_million_rate("openai", "tts-1") == 15.0
    assert per_million_rate("openai", "tts-1-hd") == 30.0
    assert per_million_rate("elevenlabs") == 300.0


def test_known_model_rate() -> None:
    assert per_character_rate("openai", "tts-1") == 15.0 / 1_000_000
    assert per_character_rate("openai", "tts-1-hd") == 30.0 / 1_000_000


def test_provider_default_for_unknown_model() -> None:
    assert per_character_rate("openai", "tts-9-ultra") == 15.0 / 1_000_000
    assert per_character_rate("elevenlabs", "any-model") == 0.30 / 1_000


def test_unknown_provider_falls_back_to_global_default() -> None:
    assert per_character_rate("nonexistent") == 0.30 / 1_000


def test_provider_name_is_case_insensitive() -> None:
    assert per_character_rate("OpenAI", "tts-1") == per_character_rate(
        "openai", "tts-1"
    )


def test_estimate_text_cost_scales_with_length() -> None:
    rate = per_character_rate("openai", "tts-1")
    text = "hello"
    assert estimate_text_cost(text, "openai", "tts-1") == len(text) * rate


def test_override_per_million_wins_over_defaults() -> None:
    # A scenario's cost_per_million_chars overrides the bundled default.
    assert (
        per_character_rate("elevenlabs", override_per_million=180.0)
        == 180.0 / 1_000_000
    )


def test_estimate_text_cost_honors_override() -> None:
    text = "hello"
    assert estimate_text_cost(text, "elevenlabs", override_per_million=180.0) == len(
        text
    ) * (180.0 / 1_000_000)
