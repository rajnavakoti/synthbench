"""Tests for scenario file parsing and validation."""

import textwrap
from pathlib import Path

import pytest

from synthbench.config.scenario import (
    PromptSource,
    ScenarioError,
    load_scenario,
)

VALID_SCENARIO = """\
[scenario]
name = "Test stress"
provider = "elevenlabs"
modality = "tts"
concurrency = [1, 5, 10]
budget_limit_usd = 5.0

[provider.elevenlabs]
api_key = "sk-test"
model = "eleven_multilingual_v2"
voice_id = "abc123"

[prompts]
source = "inline"
texts = ["Hello world.", "The quick brown fox."]

[scoring]
metrics = ["latency", "wer"]
whisper_model = "base"

[thresholds]
warn_wer = 0.05
fail_wer = 0.1
"""


def _write(tmp_path: Path, content: str, name: str = "scenario.toml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_valid_scenario_parses(tmp_path: Path) -> None:
    scn = load_scenario(_write(tmp_path, VALID_SCENARIO))
    assert scn.name == "Test stress"
    assert scn.provider == "elevenlabs"
    assert scn.concurrency == [1, 5, 10]
    assert scn.budget_limit_usd == 5.0
    assert scn.provider_config.model == "eleven_multilingual_v2"
    # Provider-specific extras are preserved.
    assert scn.provider_config.voice_id == "abc123"
    assert scn.resolve_prompts() == ["Hello world.", "The quick brown fox."]


def test_thresholds_default_when_section_absent(tmp_path: Path) -> None:
    content = """\
    [scenario]
    name = "No thresholds"
    provider = "openai"
    concurrency = [1]
    budget_limit_usd = 1.0

    [provider.openai]
    model = "tts-1"

    [prompts]
    source = "inline"
    texts = ["hi"]
    """
    scn = load_scenario(_write(tmp_path, content))
    assert scn.thresholds.warn_wer == 0.05
    assert scn.thresholds.fail_wer == 0.10
    assert scn.scoring.metrics == ["latency", "file_integrity"]


def test_file_based_prompts(tmp_path: Path) -> None:
    (tmp_path / "prompts.txt").write_text(
        "First prompt.\n\n  Second prompt.  \nThird prompt.\n", encoding="utf-8"
    )
    content = """\
    [scenario]
    name = "File prompts"
    provider = "openai"
    concurrency = [1]
    budget_limit_usd = 1.0

    [provider.openai]
    model = "tts-1"

    [prompts]
    source = "file"
    path = "prompts.txt"
    """
    scn = load_scenario(_write(tmp_path, content))
    assert scn.prompts.source is PromptSource.file
    # Blank lines dropped, surrounding whitespace stripped.
    assert scn.resolve_prompts() == [
        "First prompt.",
        "Second prompt.",
        "Third prompt.",
    ]


def test_env_var_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TTS_KEY", "secret-value")
    content = VALID_SCENARIO.replace('api_key = "sk-test"', 'api_key = "${MY_TTS_KEY}"')
    scn = load_scenario(_write(tmp_path, content))
    assert scn.provider_config.api_key == "secret-value"


def test_unset_env_var_left_as_literal(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace(
        'api_key = "sk-test"', 'api_key = "${DEFINITELY_UNSET_VAR}"'
    )
    scn = load_scenario(_write(tmp_path, content))
    # Left as the literal token so dry-run works without secrets present.
    assert scn.provider_config.api_key == "${DEFINITELY_UNSET_VAR}"


def test_missing_required_field_raises(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace("budget_limit_usd = 5.0\n", "")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(_write(tmp_path, content))
    assert "budget_limit_usd" in str(exc.value)


def test_invalid_concurrency_value_raises(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace("concurrency = [1, 5, 10]", "concurrency = [1, 0]")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(_write(tmp_path, content))
    assert "concurrency" in str(exc.value)


def test_empty_concurrency_raises(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace("concurrency = [1, 5, 10]", "concurrency = []")
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, content))


def test_provider_section_missing_raises(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace("[provider.elevenlabs]", "[provider.openai]")
    with pytest.raises(ScenarioError) as exc:
        load_scenario(_write(tmp_path, content))
    message = str(exc.value)
    assert "elevenlabs" in message
    assert "openai" in message  # lists what is configured


def test_unsupported_modality_raises(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace('modality = "tts"', 'modality = "image"')
    with pytest.raises(ScenarioError) as exc:
        load_scenario(_write(tmp_path, content))
    assert "modality" in str(exc.value)


def test_inline_prompts_empty_raises(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace(
        'texts = ["Hello world.", "The quick brown fox."]', "texts = []"
    )
    with pytest.raises(ScenarioError) as exc:
        load_scenario(_write(tmp_path, content))
    assert "texts" in str(exc.value)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    content = VALID_SCENARIO.replace(
        'name = "Test stress"', 'name = "Test stress"\ntypo_field = 1'
    )
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, content))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError) as exc:
        load_scenario(tmp_path / "nope.toml")
    assert "not found" in str(exc.value)


def test_invalid_toml_raises(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError) as exc:
        load_scenario(_write(tmp_path, "this is = = not toml"))
    assert "invalid TOML" in str(exc.value)


def test_missing_prompt_file_raises(tmp_path: Path) -> None:
    content = """\
    [scenario]
    name = "Missing prompts"
    provider = "openai"
    concurrency = [1]
    budget_limit_usd = 1.0

    [provider.openai]
    model = "tts-1"

    [prompts]
    source = "file"
    path = "does-not-exist.txt"
    """
    scn = load_scenario(_write(tmp_path, content))
    with pytest.raises(ScenarioError) as exc:
        scn.resolve_prompts()
    assert "could not read prompt file" in str(exc.value)
