# Synthbench Architecture

Synthbench is quality-under-load benchmarking for AI media generation APIs. The
category-defining output is the **degradation curve**: a chart showing exactly
where quality, latency, and cost break as concurrency rises. Everything in the
codebase exists to make that curve trustworthy, clear, and easy to generate.

This document describes the technical architecture and the contracts between
layers. It is meant to stay in sync with the code — update it in the same PR as
any change to a public interface.

> **Scope boundary.** Synthbench tests AI-generated **media** — audio, image,
> video, music. Not text, not embeddings, not RAG, not LLM inference. This
> boundary keeps the tool focused and is enforced in review.

## Current phase

**Phase 1 — TTS foundation.** ElevenLabs + OpenAI TTS adapters, an async
workload engine with budget guards, WER/duration/file-integrity scoring via
Whisper, and a terminal + JSON degradation-curve report with PASS/WARN/FAIL
zones. Image and video scoring, MOS prediction, historical baselines, and the
hosted tier are later phases and intentionally do not exist yet.

## Layered architecture

```
+----------------------------------------------------------------------+
|                          synthbench CLI                              |
|                    (Typer + Rich; stateless)                        |
+----------------------------------------------------------------------+
          |                    |                     |
          v                    v                     v
+------------------+  +------------------+  +------------------+
|  Workload Engine |  | Quality Scoring  |  | Reporting Engine |
| - job lifecycle  |  | - audio scorers  |  | - terminal table |
| - concurrency    |  | - Scorer iface   |  | - JSON report    |
| - budget guards  |  | - ScoreResult    |  | - PASS/WARN/FAIL |
| - owns timing    |  |                  |  |                  |
+------------------+  +------------------+  +------------------+
          |                    |                     |
          v                    v                     v
+----------------------------------------------------------------------+
|                     Provider Adapter Layer                           |
|        (one adapter per provider, shared interface)                 |
|   ElevenLabs (TTS)  |  OpenAI (TTS)  |  ... (community-contributed)  |
+----------------------------------------------------------------------+
```

- **CLI layer** (`cli/`) — parses commands, loads and validates the TOML
  scenario, orchestrates the engine, pipes results to reporting. Stateless.
- **Config** (`config/`) — Pydantic-validated scenario schema and TOML loader
  with `${ENV_VAR}` interpolation. The single validation boundary; nothing
  downstream re-checks config.
- **Workload engine** (`engine/`) — drives the generation lifecycle at each
  configured concurrency level, **owns timing** (see below), enforces budget
  guards, and produces the per-request `GenerationResult` records.
- **Provider adapters** (`providers/`) — normalize provider APIs into one
  lifecycle (see the contract below). Pluggable; looked up by name via the
  registry.
- **Scoring** (`scoring/`) — pluggable `Scorer`s consume artifacts and return
  `ScoreResult`s. Phase 1 is audio-only (WER, duration, file integrity).
- **Reporting** (`reporting/`) — renders the degradation curve as a Rich
  terminal table and a stable JSON document.

## The provider adapter contract

The adapter layer speaks an explicit vocabulary so the boundary with the engine
stays sharp:

```
GenerationRequest  ->  GenerationJob  ->  GenerationArtifact      [adapter domain]
   (what we ask)        (lifecycle)        (raw bytes + type)
                              │
                              ▼
                       GenerationResult                            [engine domain]
            (request + latency + cost + scores) — scorers & reports consume this
```

Adapter-domain models (`providers/base.py`):

- **`GenerationRequest`** (frozen) — `prompt`, `provider`, `model`,
  `provider_params` (voice id, output format, …), `index`. Exposes a stable
  `input_hash` over the content inputs for dedup and future baselines.
- **`GenerationJob`** — lifecycle handle: `status`, `provider_job_id`,
  `artifact`, `error`, `estimated_cost_usd`, `actual_cost_usd`,
  `response_headers`. **All per-request state lives on the job, never on the
  adapter**, so one adapter instance is safe to share across concurrent
  requests.
- **`GenerationArtifact`** — `data: bytes`, `content_type`, `size_bytes`.
- **`GenerationStatus`** — `pending` / `running` / `succeeded` / `failed` /
  `cancelled`.

Interface (`ProviderAdapter`):

| Method | Returns | Notes |
|---|---|---|
| `estimate_cost_usd(request)` | `float` (USD) | Pre-flight estimate; backs dry-run and budget guards |
| `submit(request)` | `GenerationJob` | Sync providers return a terminal job; async return `pending` |
| `poll(job)` | `GenerationJob` | Refreshed job; no-op for sync providers |
| `retrieve(job)` | `GenerationArtifact` | Raw output for a succeeded job |
| `parse_rate_limit(headers)` | `RateLimitInfo` | Optional; override per provider |

**Synchronous vs. async.** Most TTS APIs return audio in a single response;
those adapters complete the work in `submit` and return a `succeeded` job
carrying the artifact. Async job-queue providers (image/video, later phases)
return a `pending` job whose `provider_job_id` the engine polls until terminal,
then downloads via `retrieve`. The same vocabulary covers both.

**Engine owns timing.** Latency is the engine's authoritative measurement: it
wraps `submit` through a successful `retrieve` with a monotonic clock, giving one
consistent latency per request for the curve. Adapters never stamp timestamps.

**Failure model.** Transport/HTTP errors raise `ProviderError` (the engine
catches it and records a failed result). A remote job that *reports* failure
surfaces as `status=failed` on the job. `cancelled` is reserved for the engine's
own stop decisions (budget guard, timeout). Adapters return raw outputs only —
the engine builds the benchmarkable `GenerationResult` from a terminal job.

## Scoring contract

`Scorer` (`scoring/base.py`) is the pluggable scoring interface; each scorer
returns a `ScoreResult` (`metric`, `value`, `unit`, `verdict`, `detail`). The
engine attaches scores to each `GenerationResult`, and the reporting layer
aggregates them per concurrency level into the curve. Phase 1 ships audio
scorers only; advanced scorers (MOS, speaker similarity) and other modalities
arrive in later phases behind the same interface.

## Pricing

`pricing.py` is the single source of truth for per-character cost estimates. It
backs both the CLI dry-run estimate and each adapter's `estimate_cost_usd`, so
the plan and the live run never disagree. Rates are approximate planning
estimates, not invoices.

## Technology stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Async | asyncio + aiohttp |
| CLI | Typer + Rich |
| Config validation | Pydantic (TOML scenarios) |
| Audio quality | openai-whisper, jiwer, soundfile (optional `[audio]` extra) |
| Packaging | Poetry |
| Lint/format | Ruff (line length 88) |
| Testing | pytest + pytest-asyncio (providers mocked in CI) |

ML-heavy dependencies are opt-in via extras (`pip install synthbench[audio]`);
the core package stays light.

## Repository structure (Phase 1)

```
src/synthbench/
  cli/         # Typer app + run command
  config/      # scenario schema + TOML loader
  engine/      # workload engine (concurrency, lifecycle, budget) — in progress
  providers/   # ProviderAdapter base + adapters + registry
  scoring/     # Scorer base + audio scorers
  reporting/   # terminal + JSON report
  models.py    # engine-domain records (GenerationResult, RunResult, …)
  pricing.py   # shared cost estimation
tests/         # mirrors src/; integration/ holds live tests (skipped in CI)
```

Directories for unbuilt phases (image/video scoring, baseline store, HTML
reports) are intentionally absent until their phase begins.

## Conventions

- Type hints on all signatures; async throughout the engine and provider layers.
- Pydantic validates at boundaries; nothing downstream re-validates config.
- Ruff for lint + format. Structured errors over bare exceptions
  (`ProviderError`, `ScenarioError`) carry user-facing messages.
- Provider responses are mocked in unit tests so CI never spends API credits;
  live tests live under `tests/integration/` and are skipped without keys.

---

Strategic context — positioning, roadmap rationale, and feature prioritization —
lives in an internal blueprint kept outside this public repository.
