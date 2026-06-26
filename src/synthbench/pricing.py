"""Per-character pricing estimates for TTS providers.

Used by the ``--dry-run`` pre-flight cost estimate and, once provider adapters
land, by their ``estimate_cost()`` implementations so a single pricing source
of truth backs both the plan and the live run.

Rates are USD per character and are approximate — providers change pricing and
tiers frequently. Treat these as planning estimates, not invoices.
"""

# USD per character, keyed by provider then model. ``"_default"`` is the
# per-provider fallback used when the configured model is not listed.
_PRICING: dict[str, dict[str, float]] = {
    "elevenlabs": {
        # ~$0.30 per 1,000 characters on mid usage tiers (plan-dependent).
        "_default": 0.30 / 1_000,
    },
    "openai": {
        "tts-1": 15.0 / 1_000_000,
        "tts-1-hd": 30.0 / 1_000_000,
        "_default": 15.0 / 1_000_000,
    },
}

# Fallback when the provider itself is unknown.
_GLOBAL_DEFAULT = 0.30 / 1_000


def per_character_rate(provider: str, model: str | None = None) -> float:
    """Return the USD-per-character rate for a provider/model.

    Falls back to the provider default, then a global default, so callers
    always receive a usable estimate even for unconfigured models.
    """
    table = _PRICING.get(provider.lower())
    if table is None:
        return _GLOBAL_DEFAULT
    if model is not None and model in table:
        return table[model]
    return table.get("_default", _GLOBAL_DEFAULT)


def estimate_text_cost(text: str, provider: str, model: str | None = None) -> float:
    """Estimate the USD cost of generating audio for ``text``."""
    return len(text) * per_character_rate(provider, model)
