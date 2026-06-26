# Synthbench

Quality-under-load benchmarking for AI media generation APIs.

## Product Context

Synthbench produces trusted quality-under-load degradation curves for AI media generation — showing exactly where quality, latency, and cost break as concurrency rises. The degradation curve IS the product.

Technical architecture (public, kept in sync with code): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Full strategic blueprint (moat, segments, pricing, roadmap) is **private** and
**not** in version control — it lives as a gitignored `SYNTHBENCH-BLUEPRINT.md`
in the repo root (mirrored from the workspace copy). Do not commit it or quote
its strategy sections into public files.

## Cutting Rule

Every feature must directly make the degradation curve more trustworthy, clearer, or easier to generate. If it doesn't, defer it.

## Current Phase

Phase 1: TTS foundation. ElevenLabs + OpenAI TTS adapters, async workload engine with budget guards, WER scoring via Whisper, terminal + JSON degradation curve report with PASS/WARN/FAIL zones.

## Architecture

- Python 3.11+, asyncio, Poetry
- CLI: Typer + Rich
- Config: TOML scenario files, validated with Pydantic
- Provider adapters: pluggable interface (`ProviderAdapter` base class)
- Scoring: pluggable interface (`Scorer` base class), audio-only for Phase 1
- Reporting: terminal (Rich tables) + JSON

## Code Conventions

- Type hints on all function signatures
- Ruff for linting and formatting (line-length 88)
- pytest + pytest-asyncio for testing
- Pydantic for config/scenario validation at boundaries
- Structured logging via structlog (not print)
- No `any` types if using type annotations
- Async/await throughout the engine and provider layers

## Repository Identity

- Owner: rajnavakoti (personal account)
- Email: rajnavakoti@gmail.com
- License: Apache 2.0

## What Does NOT Exist Yet (Phase 2+)

Do not create directories or files for: image scoring, video scoring, MOS prediction, speaker similarity, corpus loader, SQLite baseline store, HTML reports, dashboard, SaaS layer.
