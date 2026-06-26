"""TOML scenario file parsing and validation.

A scenario file fully describes one benchmark run: the provider and model, the
concurrency levels to sweep, the prompt set, the metrics to score, and the
PASS/WARN/FAIL thresholds. It is the single entry point the CLI validates
before any provider is ever called.
"""

import os
import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)


class ScenarioError(Exception):
    """Raised when a scenario file cannot be loaded, parsed, or validated.

    Carries a human-readable message suitable for printing directly to the
    user — the CLI surfaces it without a traceback.
    """


class PromptSource(StrEnum):
    """Where the prompt set comes from."""

    inline = "inline"
    file = "file"


class PromptConfig(BaseModel):
    """The ``[prompts]`` section."""

    model_config = ConfigDict(extra="forbid")

    source: PromptSource = PromptSource.inline
    texts: list[str] = Field(default_factory=list)
    path: str | None = None

    @model_validator(mode="after")
    def _check_source(self) -> "PromptConfig":
        if self.source is PromptSource.inline and not self.texts:
            raise ValueError(
                "inline prompts require a non-empty 'texts' list "
                '(e.g. texts = ["..."])'
            )
        if self.source is PromptSource.file and not self.path:
            raise ValueError("file prompts require a 'path' to a prompt file")
        return self


class ProviderConfig(BaseModel):
    """A ``[provider.<name>]`` section.

    Common fields are declared; provider-specific extras (e.g. ``voice_id``
    for ElevenLabs) are preserved via ``extra="allow"`` and consumed by the
    matching adapter.
    """

    model_config = ConfigDict(extra="allow")

    api_key: str | None = None
    model: str | None = None


class ScoringConfig(BaseModel):
    """The ``[scoring]`` section."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=lambda: ["latency", "file_integrity"])
    whisper_model: str = "base"


class Thresholds(BaseModel):
    """PASS/WARN/FAIL boundaries — the ``[thresholds]`` section."""

    model_config = ConfigDict(extra="forbid")

    warn_wer: float = 0.05
    fail_wer: float = 0.10
    warn_latency_p95: float = 10.0
    fail_latency_p95: float = 30.0


class Scenario(BaseModel):
    """A fully validated benchmark scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    modality: Literal["tts"] = "tts"
    concurrency: list[PositiveInt] = Field(min_length=1)
    budget_limit_usd: PositiveFloat
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    prompts: PromptConfig
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # Directory of the scenario file, used to resolve relative prompt paths.
    # Excluded from serialization — it is execution context, not config.
    base_dir: Path = Field(default=Path(), exclude=True)

    @model_validator(mode="after")
    def _check_provider_present(self) -> "Scenario":
        if self.provider not in self.providers:
            available = ", ".join(sorted(self.providers)) or "none"
            raise ValueError(
                f"provider '{self.provider}' has no [provider.{self.provider}] "
                f"section (configured providers: {available})"
            )
        return self

    @property
    def provider_config(self) -> ProviderConfig:
        """The config block for the active provider."""
        return self.providers[self.provider]

    def resolve_prompts(self) -> list[str]:
        """Return the concrete prompt list, reading the file if needed.

        Raises ``ScenarioError`` if a referenced prompt file is missing or
        empty.
        """
        if self.prompts.source is PromptSource.inline:
            return list(self.prompts.texts)

        # source == file; path is guaranteed present by PromptConfig validation.
        assert self.prompts.path is not None
        path = Path(self.prompts.path)
        if not path.is_absolute():
            path = self.base_dir / path
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ScenarioError(f"could not read prompt file '{path}': {exc}") from exc
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]
        if not prompts:
            raise ScenarioError(f"prompt file '{path}' contains no prompts")
        return prompts


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references using the environment.

    Unset variables are left as the literal ``${VAR}`` token so that
    ``--dry-run`` works without secrets present; the engine validates real
    credentials before it calls a provider.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["scenario validation failed:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_scenario(path: str | Path) -> Scenario:
    """Load, env-interpolate, and validate a TOML scenario file.

    All failure modes raise ``ScenarioError`` with a user-facing message.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ScenarioError(f"scenario file not found: {file_path}")

    try:
        with file_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ScenarioError(f"invalid TOML in '{file_path}': {exc}") from exc

    raw = _interpolate_env(raw)

    scenario_table = raw.get("scenario", {})
    if not isinstance(scenario_table, dict):
        raise ScenarioError("[scenario] section must be a table")

    data = {
        **scenario_table,
        "providers": raw.get("provider", {}),
        "prompts": raw.get("prompts", {}),
        "scoring": raw.get("scoring", {}),
        "thresholds": raw.get("thresholds", {}),
        "base_dir": file_path.parent,
    }

    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(_format_validation_error(exc)) from exc
