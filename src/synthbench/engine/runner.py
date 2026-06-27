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
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from statistics import mean

from synthbench.config.scenario import Scenario
from synthbench.engine.budget import BudgetGuard
from synthbench.models import (
    ConcurrencyResult,
    GenerationResult,
    RunResult,
    ScoreResult,
    Verdict,
)
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

# Content-type -> file extension for saved artifacts.
_AUDIO_EXT = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/L16": "pcm",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/opus": "opus",
}


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
    artifact_dir: Path | None = None,
) -> RunResult:
    """Run every concurrency level and return the aggregated result.

    When ``artifact_dir`` is set, every generated clip is written to
    ``artifact_dir/audio/`` and a ``manifest.json`` (prompt, transcription,
    scores, latency, cost per clip) is written alongside — so a paid run's
    outputs can always be replayed and ground-truthed by ear.
    """
    scorers = scorers or []
    guard = BudgetGuard(scenario.budget_limit_usd, scenario.budget_guard_pct)
    model = scenario.provider_config.model
    started = time.perf_counter()

    audio_dir: Path | None = None
    manifest: list[dict] = []
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        audio_dir = artifact_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

    concurrency_results: list[ConcurrencyResult] = []
    for level in scenario.concurrency:
        result = await _run_level(
            level,
            scenario,
            adapter,
            prompts,
            model,
            guard,
            scorers,
            on_progress,
            audio_dir,
            manifest,
        )
        concurrency_results.append(result)
        # Budget guard halts the whole run, not just the current level.
        if guard.exceeded:
            break

    if artifact_dir is not None:
        manifest.sort(key=lambda record: (record["concurrency"], record["index"]))
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

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
    audio_dir: Path | None,
    manifest: list[dict],
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
            result, artifact = await _run_generation(
                request, level, adapter, scenario, est_cost
            )
        # Score outside the concurrency slot: scoring is local post-processing
        # and must not hold a provider slot, which would throttle real load.
        if result.success and artifact is not None and scorers:
            result.scores = await _score(artifact, request.prompt, scorers)
        if audio_dir is not None:
            record = _persist_artifact(
                audio_dir, level, index, request, result, artifact
            )
            async with results_lock:
                manifest.append(record)
        async with results_lock:
            results.append(result)
        if on_progress is not None:
            on_progress(ProgressEvent(level, guard.spent, scenario.budget_limit_usd))
        return True

    outcomes = await asyncio.gather(*(worker(i) for i in range(num_requests)))
    incomplete = sum(1 for ran in outcomes if not ran)
    return _aggregate(level, results, incomplete)


def _persist_artifact(
    audio_dir: Path,
    level: int,
    index: int,
    request: GenerationRequest,
    result: GenerationResult,
    artifact: GenerationArtifact | None,
) -> dict:
    """Write a generated clip to disk and return its manifest record.

    Failed generations have no file but are still recorded (with their error) so
    the manifest is a complete account of what the paid run produced.
    """
    file_ref: str | None = None
    if artifact is not None:
        ext = _AUDIO_EXT.get(artifact.content_type, "bin")
        name = f"c{level:03d}_r{index:03d}.{ext}"
        (audio_dir / name).write_bytes(artifact.data)
        file_ref = f"audio/{name}"
    return {
        "concurrency": level,
        "index": index,
        "prompt": request.prompt,
        "file": file_ref,
        "success": result.success,
        "error": result.error or None,
        "latency_s": round(result.latency_seconds, 3),
        "cost_usd": round(result.cost_usd, 6),
        "scores": [
            {"metric": s.metric, "value": s.value, "detail": s.detail}
            for s in result.scores
        ],
    }


async def _run_generation(
    request: GenerationRequest,
    level: int,
    adapter: ProviderAdapter,
    scenario: Scenario,
    est_cost: float,
) -> tuple[GenerationResult, GenerationArtifact | None]:
    """Run one request's lifecycle (timed, with retries). Scoring is separate."""
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
    result = GenerationResult(
        prompt=request.prompt,
        concurrency_level=level,
        latency_seconds=latency,
        cost_usd=est_cost,
        artifact_bytes=artifact.size_bytes if artifact else 0,
        success=artifact is not None,
        error=error,
        scores=[],
    )
    return result, artifact


async def _score(
    artifact: GenerationArtifact, prompt: str, scorers: list[Scorer]
) -> list[ScoreResult]:
    scores: list[ScoreResult] = []
    for scorer in scorers:
        try:
            scores.append(await scorer.score(artifact.data, prompt))
        except Exception as exc:  # noqa: BLE001 - one bad scorer must not sink the run
            scores.append(
                ScoreResult(
                    metric=scorer.metric_name,
                    value=0.0,
                    unit="",
                    verdict=Verdict.FAIL,
                    detail=f"scoring error: {exc}",
                )
            )
    return scores


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
    wers = [
        s.value
        for r in successful
        for s in r.scores
        if s.metric == "wer" and s.verdict is not Verdict.FAIL
    ]
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
