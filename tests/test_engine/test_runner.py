"""Tests for the workload engine, driven by a mock adapter (no real API calls)."""

import asyncio

from synthbench.config.scenario import PromptConfig, ProviderConfig, Scenario
from synthbench.engine.runner import ProgressEvent, run_scenario
from synthbench.models import ScoreResult, Verdict
from synthbench.providers.base import (
    GenerationArtifact,
    GenerationJob,
    GenerationRequest,
    GenerationStatus,
    ProviderAdapter,
    ProviderError,
)
from synthbench.scoring.base import Scorer


class MockAdapter(ProviderAdapter):
    """Controllable synchronous adapter for engine tests."""

    def __init__(
        self,
        *,
        latency: float = 0.01,
        cost: float = 0.01,
        fail_indices: set[int] | None = None,
        timeout_indices: set[int] | None = None,
    ) -> None:
        self.latency = latency
        self.cost = cost
        self.fail_indices = fail_indices or set()
        self.timeout_indices = timeout_indices or set()
        self.in_flight = 0
        self.max_in_flight = 0
        self.submitted = 0

    @property
    def name(self) -> str:
        return "mock"

    def estimate_cost_usd(self, request: GenerationRequest) -> float:
        return self.cost

    async def submit(self, request: GenerationRequest) -> GenerationJob:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.submitted += 1
        try:
            if request.index in self.timeout_indices:
                await asyncio.sleep(3600)  # forces wait_for to time out
            await asyncio.sleep(self.latency)
            if request.index in self.fail_indices:
                raise ProviderError(f"mock failure {request.index}")
            return GenerationJob(
                request=request,
                status=GenerationStatus.SUCCEEDED,
                artifact=GenerationArtifact(
                    data=b"audio-bytes", content_type="audio/mpeg"
                ),
            )
        finally:
            self.in_flight -= 1

    async def poll(self, job: GenerationJob) -> GenerationJob:
        return job

    async def retrieve(self, job: GenerationJob) -> GenerationArtifact:
        if job.artifact is None:
            raise ProviderError("no artifact")
        return job.artifact


def make_scenario(
    concurrency: list[int],
    *,
    k: int = 1,
    budget: float = 1000.0,
    guard_pct: float = 0.9,
    timeout: float = 60.0,
    retries: int = 1,
) -> Scenario:
    return Scenario(
        name="test",
        provider="mock",
        concurrency=concurrency,
        budget_limit_usd=budget,
        requests_multiplier=k,
        budget_guard_pct=guard_pct,
        request_timeout_s=timeout,
        max_retries=retries,
        providers={"mock": ProviderConfig()},
        prompts=PromptConfig(texts=["x"]),
    )


async def test_runs_every_level_with_n_requests() -> None:
    scn = make_scenario([1, 2, 3])
    adapter = MockAdapter()
    result = await run_scenario(scn, adapter, ["a", "b"])

    assert [cr.concurrency_level for cr in result.concurrency_results] == [1, 2, 3]
    assert [len(cr.generations) for cr in result.concurrency_results] == [1, 2, 3]
    assert adapter.submitted == 6
    assert result.budget_exceeded is False


async def test_requests_multiplier_scales_requests() -> None:
    scn = make_scenario([2], k=3)
    adapter = MockAdapter()
    result = await run_scenario(scn, adapter, ["a"])
    # level 2 * k 3 = 6 requests
    assert len(result.concurrency_results[0].generations) == 6


async def test_concurrency_never_exceeds_level() -> None:
    scn = make_scenario([5], k=4)  # 20 requests at concurrency 5
    adapter = MockAdapter(latency=0.02)
    await run_scenario(scn, adapter, ["a"])
    assert adapter.max_in_flight <= 5
    assert adapter.max_in_flight > 1  # genuinely parallel


async def test_prompts_are_cycled() -> None:
    scn = make_scenario([5])  # 5 requests, 3 prompts -> a b c a b
    adapter = MockAdapter()
    result = await run_scenario(scn, adapter, ["a", "b", "c"])
    prompts_used = [g.prompt for g in result.concurrency_results[0].generations]
    assert sorted(prompts_used) == ["a", "a", "b", "b", "c"]


async def test_budget_guard_halts_run() -> None:
    # cost 1.0/req, threshold = 3.0 * 1.0 = 3.0
    scn = make_scenario([2, 5], budget=3.0, guard_pct=1.0)
    adapter = MockAdapter(cost=1.0)
    result = await run_scenario(scn, adapter, ["a"])

    assert result.budget_exceeded is True
    assert result.total_cost == 3.0
    # level 1 ran 2; level 2 could afford only 1 more, 4 cancelled
    level2 = result.concurrency_results[1]
    assert len(level2.generations) == 1
    assert level2.incomplete == 4
    # run halts — no third level attempted even if more were configured
    assert len(result.concurrency_results) == 2


async def test_failures_are_recorded() -> None:
    scn = make_scenario([3], retries=0)
    adapter = MockAdapter(fail_indices={1})
    result = await run_scenario(scn, adapter, ["a"])
    level = result.concurrency_results[0]
    assert abs(level.success_rate - (2 / 3)) < 1e-9
    failed = [g for g in level.generations if not g.success]
    assert len(failed) == 1
    assert "mock failure" in failed[0].error


async def test_timeout_is_recorded_as_failure() -> None:
    scn = make_scenario([1], timeout=0.05, retries=0)
    adapter = MockAdapter(timeout_indices={0})
    result = await run_scenario(scn, adapter, ["a"])
    gen = result.concurrency_results[0].generations[0]
    assert gen.success is False
    assert "timeout" in gen.error


async def test_latency_is_measured() -> None:
    scn = make_scenario([2])
    adapter = MockAdapter(latency=0.03)
    result = await run_scenario(scn, adapter, ["a"])
    assert result.concurrency_results[0].avg_latency >= 0.03
    assert result.total_duration_seconds > 0


async def test_progress_callback_fires_per_completed_request() -> None:
    scn = make_scenario([3])
    adapter = MockAdapter()
    events: list[ProgressEvent] = []
    await run_scenario(scn, adapter, ["a"], on_progress=events.append)
    assert len(events) == 3
    assert events[-1].cumulative_cost > 0


class _FakeWERScorer(Scorer):
    @property
    def metric_name(self) -> str:
        return "wer"

    async def score(
        self, artifact: bytes, prompt: str, **kwargs: object
    ) -> ScoreResult:
        return ScoreResult(metric="wer", value=0.1, unit="ratio")


class _BoomScorer(Scorer):
    @property
    def metric_name(self) -> str:
        return "wer"

    async def score(
        self, artifact: bytes, prompt: str, **kwargs: object
    ) -> ScoreResult:
        raise RuntimeError("boom")


async def test_scores_are_attached_and_aggregated() -> None:
    scn = make_scenario([2])
    result = await run_scenario(scn, MockAdapter(), ["a"], scorers=[_FakeWERScorer()])
    level = result.concurrency_results[0]
    assert all(any(s.metric == "wer" for s in g.scores) for g in level.generations)
    assert level.avg_wer == 0.1


async def test_scorer_failure_does_not_crash_run() -> None:
    scn = make_scenario([1])
    result = await run_scenario(scn, MockAdapter(), ["a"], scorers=[_BoomScorer()])
    gen = result.concurrency_results[0].generations[0]
    assert gen.success is True  # generation succeeded; only scoring failed
    wer_score = next(s for s in gen.scores if s.metric == "wer")
    assert wer_score.verdict is Verdict.FAIL
    # a failed WER score is excluded from the level average
    assert result.concurrency_results[0].avg_wer is None
