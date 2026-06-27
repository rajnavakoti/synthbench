"""Workload engine — the spine that produces the degradation curve.

For each concurrency level the engine issues ``level * requests_multiplier``
requests (cycling the prompt set) with at most ``level`` in flight, times each
request's full lifecycle, tracks cost against a budget guard, and aggregates the
results per level into a :class:`RunResult`.

The engine owns timing (a monotonic clock around submit -> retrieve), so latency
is one authoritative measurement. It is scorer-agnostic: pass scorers to attach
quality metrics (Epic 6); with none, results carry latency/cost/success only.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, floor
from statistics import mean

from synthbench.config.scenario import Scenario
from synthbench.engine.budget import BudgetGuard
from synthbench.models import ConcurrencyResult, GenerationResult, RunResult, Verdict
from synthbench.providers.base import (
    GenerationArtifact,
    GenerationRequest,
    GenerationStatus,
    ProviderAdapter,
    ProviderError,
)
from synthbench.scoring.base import Scorer

# How often to re-poll an async job. No-op for synchronous TTS providers, whose
# jobs are already terminal after submit.
POLL_INTERVAL_S = 0.5
_BACKOFF_BASE_S = 0.5


@dataclass
class ProgressEvent:
    """Emitted once per completed request for real-time reporting."""

    level: int
    cumulative_cost: float
    budget_limit: float


ProgressCallback = Callable[[ProgressEvent], None]


async def run_scenario(
    scenario: Scenario,
    adapter: ProviderAdapter,
    prompts: list[str],
    *,
    scorers: list[Scorer] | None = None,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    """Run every concurrency level and return the aggregated result."""
    scorers = scorers or []
    guard = BudgetGuard(scenario.budget_limit_usd, scenario.budget_guard_pct)
    model = scenario.provider_config.model
    started = time.perf_counter()

    concurrency_results: list[ConcurrencyResult] = []
    for level in scenario.concurrency:
        result = await _run_level(
            level, scenario, adapter, prompts, model, guard, scorers, on_progress
        )
        concurrency_results.append(result)
        # Budget guard halts the whole run, not just the current level.
        if guard.exceeded:
            break

    return RunResult(
        scenario_name=scenario.name,
        provider=scenario.provider,
        model=model or "",
        concurrency_results=concurrency_results,
        total_cost=guard.spent,
        total_duration_seconds=time.perf_counter() - started,
        budget_limit_usd=scenario.budget_limit_usd,
        budget_exceeded=guard.exceeded,
    )


async def _run_level(
    level: int,
    scenario: Scenario,
    adapter: ProviderAdapter,
    prompts: list[str],
    model: str | None,
    guard: BudgetGuard,
    scorers: list[Scorer],
    on_progress: ProgressCallback | None,
) -> ConcurrencyResult:
    num_requests = level * scenario.requests_multiplier
    semaphore = asyncio.Semaphore(level)
    results: list[GenerationResult] = []
    results_lock = asyncio.Lock()

    async def worker(index: int) -> bool:
        prompt = prompts[index % len(prompts)]
        request = GenerationRequest(
            prompt=prompt,
            provider=scenario.provider,
            model=model,
            provider_params={},
            index=index,
        )
        est_cost = adapter.estimate_cost_usd(request)
        # Reserve budget before taking a concurrency slot; refused -> cancelled.
        if not await guard.reserve(est_cost):
            return False
        async with semaphore:
            result = await _execute_one(
                request, level, adapter, scenario, scorers, est_cost
            )
        async with results_lock:
            results.append(result)
        if on_progress is not None:
            on_progress(ProgressEvent(level, guard.spent, scenario.budget_limit_usd))
        return True

    outcomes = await asyncio.gather(*(worker(i) for i in range(num_requests)))
    incomplete = sum(1 for ran in outcomes if not ran)
    return _aggregate(level, results, incomplete)


async def _execute_one(
    request: GenerationRequest,
    level: int,
    adapter: ProviderAdapter,
    scenario: Scenario,
    scorers: list[Scorer],
    est_cost: float,
) -> GenerationResult:
    start = time.perf_counter()
    artifact: GenerationArtifact | None = None
    error = ""
    attempts = scenario.max_retries + 1
    for attempt in range(attempts):
        try:
            artifact = await asyncio.wait_for(
                _submit_poll_retrieve(adapter, request),
                timeout=scenario.request_timeout_s,
            )
            error = ""
            break
        except TimeoutError:
            error = f"timeout after {scenario.request_timeout_s}s"
        except ProviderError as exc:
            error = str(exc) or "provider error"
        if attempt < attempts - 1:
            await asyncio.sleep(_BACKOFF_BASE_S * (2**attempt))

    latency = time.perf_counter() - start
    success = artifact is not None

    scores = []
    if success and scorers:
        for scorer in scorers:
            scores.append(await scorer.score(artifact.data, request.prompt))

    return GenerationResult(
        prompt=request.prompt,
        concurrency_level=level,
        latency_seconds=latency,
        cost_usd=est_cost,
        artifact_bytes=artifact.size_bytes if artifact else 0,
        success=success,
        error=error,
        scores=scores,
    )


async def _submit_poll_retrieve(
    adapter: ProviderAdapter, request: GenerationRequest
) -> GenerationArtifact:
    job = await adapter.submit(request)
    while not job.is_terminal:
        await asyncio.sleep(POLL_INTERVAL_S)
        job = await adapter.poll(job)
    if job.status is not GenerationStatus.SUCCEEDED:
        raise ProviderError(job.error or f"generation ended {job.status}")
    return await adapter.retrieve(job)


def _aggregate(
    level: int, results: list[GenerationResult], incomplete: int
) -> ConcurrencyResult:
    successful = [r for r in results if r.success]
    latencies = sorted(r.latency_seconds for r in successful)
    wers = [s.value for r in successful for s in r.scores if s.metric == "wer"]
    return ConcurrencyResult(
        concurrency_level=level,
        generations=results,
        avg_latency=mean(latencies) if latencies else 0.0,
        p50_latency=_percentile(latencies, 50),
        p95_latency=_percentile(latencies, 95),
        avg_wer=mean(wers) if wers else None,
        total_cost=sum(r.cost_usd for r in results),
        success_rate=len(successful) / len(results) if results else 0.0,
        incomplete=incomplete,
        verdict=Verdict.PASS,  # PASS/WARN/FAIL is assigned by the reporter.
    )


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (q / 100)
    lo, hi = floor(rank), ceil(rank)
    if lo == hi:
        return values[int(rank)]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)
