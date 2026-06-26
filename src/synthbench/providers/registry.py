"""Provider registry — look up and construct adapters by name."""

from collections.abc import Callable

from synthbench.config.scenario import ProviderConfig
from synthbench.providers.base import ProviderAdapter, ProviderError
from synthbench.providers.elevenlabs import ElevenLabsAdapter

# Maps a provider name (as written in a scenario's ``provider = "..."``) to a
# factory that builds the adapter from its config section.
_REGISTRY: dict[str, Callable[[ProviderConfig], ProviderAdapter]] = {
    "elevenlabs": ElevenLabsAdapter.from_config,
}


def available_providers() -> list[str]:
    """Return the names of all registered providers, sorted."""
    return sorted(_REGISTRY)


def create_adapter(name: str, config: ProviderConfig) -> ProviderAdapter:
    """Construct the adapter for ``name`` from its config section.

    Raises ``ProviderError`` for an unknown provider name.
    """
    factory = _REGISTRY.get(name.lower())
    if factory is None:
        available = ", ".join(available_providers()) or "none"
        raise ProviderError(f"unknown provider '{name}' (available: {available})")
    return factory(config)
