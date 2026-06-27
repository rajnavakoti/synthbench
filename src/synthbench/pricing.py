"""Cost estimation for TTS providers.

Default rates live in the bundled ``pricing.toml`` data file — not in this
module — so they can be edited without touching code. A scenario may override a
provider's rate for a single run via ``cost_per_million_chars`` in its
``[provider.<name>]`` section; callers pass that through as
``override_per_million``.

Rates are USD per million characters and are approximate — planning estimates,
not invoices. Because TTS APIs do not return a charged cost, this estimate (and
its override) fully determines the cost axis of the degradation curve.
"""

import tomllib
from importlib import resources

# Last-resort rate (USD / 1M chars) if the data file lacks a fallback.
_FALLBACK_PER_MILLION = 300.0


def _load_defaults() -> dict[str, dict[str, float]]:
    with resources.files("synthbench").joinpath("pricing.toml").open("rb") as handle:
        return tomllib.load(handle)


_DEFAULTS = _load_defaults()


def _global_default_per_million() -> float:
    return _DEFAULTS.get("fallback", {}).get("default", _FALLBACK_PER_MILLION)


def per_million_rate(provider: str, model: str | None = None) -> float:
    """Return the default USD-per-million-character rate for a provider/model.

    Falls back to the provider default, then a global default, so callers always
    receive a usable estimate even for unconfigured models.
    """
    table = _DEFAULTS.get(provider.lower())
    if table is None:
        return _global_default_per_million()
    if model is not None and model in table:
        return table[model]
    return table.get("default", _global_default_per_million())


def per_character_rate(
    provider: str,
    model: str | None = None,
    *,
    override_per_million: float | None = None,
) -> float:
    """Return the USD-per-character rate, honoring a per-run override.

    When ``override_per_million`` is given it wins over the bundled defaults —
    this is the scenario's ``cost_per_million_chars``.
    """
    per_million = (
        override_per_million
        if override_per_million is not None
        else per_million_rate(provider, model)
    )
    return per_million / 1_000_000


def estimate_text_cost(
    text: str,
    provider: str,
    model: str | None = None,
    *,
    override_per_million: float | None = None,
) -> float:
    """Estimate the USD cost of generating audio for ``text``."""
    return len(text) * per_character_rate(
        provider, model, override_per_million=override_per_million
    )
