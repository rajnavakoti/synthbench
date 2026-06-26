"""Core data models for synthbench."""

from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class ScoreResult:
    """Result from a single scorer for a single generation."""

    metric: str
    value: float
    unit: str
    verdict: Verdict = Verdict.PASS
    detail: str = ""


@dataclass
class GenerationResult:
    """Result of a single generation request at a specific concurrency level."""

    prompt: str
    concurrency_level: int
    latency_seconds: float
    cost_usd: float
    artifact_bytes: int = 0
    success: bool = True
    error: str = ""
    scores: list[ScoreResult] = field(default_factory=list)


@dataclass
class ConcurrencyResult:
    """Aggregated results for a single concurrency level."""

    concurrency_level: int
    generations: list[GenerationResult] = field(default_factory=list)
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    avg_wer: float | None = None
    total_cost: float = 0.0
    success_rate: float = 1.0
    verdict: Verdict = Verdict.PASS


@dataclass
class RunResult:
    """Complete result of a benchmark run."""

    scenario_name: str
    provider: str
    model: str
    concurrency_results: list[ConcurrencyResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration_seconds: float = 0.0
    budget_limit_usd: float = 0.0
    budget_exceeded: bool = False
