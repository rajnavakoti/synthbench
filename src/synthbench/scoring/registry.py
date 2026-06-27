"""Scorer registry — build the scorer list for a scenario's metrics."""

from collections.abc import Callable

from synthbench.config.scenario import ScoringConfig
from synthbench.scoring.audio.duration import DurationScorer
from synthbench.scoring.audio.file_integrity import FileIntegrityScorer
from synthbench.scoring.audio.wer import WERScorer
from synthbench.scoring.base import Scorer, ScoringError, require_available

# Metrics the workload engine measures itself (not via a Scorer).
_ENGINE_METRICS = frozenset({"latency", "cost"})

_BUILDERS: dict[str, Callable[[ScoringConfig], Scorer]] = {
    "wer": lambda cfg: WERScorer(model_size=cfg.whisper_model),
    "duration_accuracy": lambda cfg: DurationScorer(),
    "file_integrity": lambda cfg: FileIntegrityScorer(),
}


def build_scorers(config: ScoringConfig) -> list[Scorer]:
    """Construct the scorers for the configured metrics.

    Engine-measured metrics (latency, cost) are skipped. Raises ``ScoringError``
    for an unknown metric or a metric whose dependencies are not installed —
    fail fast, before any paid API call.
    """
    scorers: list[Scorer] = []
    for metric in config.metrics:
        if metric in _ENGINE_METRICS:
            continue
        builder = _BUILDERS.get(metric)
        if builder is None:
            known = ", ".join(sorted(_BUILDERS) + sorted(_ENGINE_METRICS))
            raise ScoringError(
                f"unknown scoring metric '{metric}' (available: {known})"
            )
        scorer = builder(config)
        require_available(metric, scorer.required_modules)
        scorers.append(scorer)
    return scorers
