# synthbench

**Know where your AI media pipeline breaks — before your users do.**

Synthbench produces trusted quality-under-load degradation curves for AI media
generation — showing exactly where quality, latency, and cost break as
concurrency rises.

---

AI media generation APIs (ElevenLabs, OpenAI, fal.ai, Runway) are shipping to
production at scale, but the worst failures are **silent**: before a provider
returns an HTTP error, quality quietly degrades — words get dropped, audio gets
truncated — and traditional load tools (k6, JMeter, Gatling) never see it because
they measure HTTP latency, not whether the generated media is still good. At a
2026 conference, a founder asked ~50 people to scan a QR code for personalized
AI voice clips; his ElevenLabs-backed demo crashed live on stage — and there was
no tool he could have used beforehand to find the concurrency level where it
would break. Synthbench is that tool.

## What synthbench measures that other tools can't

| Capability | k6 / JMeter / Gatling | Synthbench |
|---|---|---|
| Async job lifecycle (submit → poll → receive) | No | Yes |
| Binary artifact quality scoring (WER today; MOS/CLIPScore later) | No | Yes |
| **Silent** quality degradation detection | No | Yes |
| Credit-based cost tracking + budget guards | No | Yes |
| Quality-under-load degradation curves with PASS/WARN/FAIL zones | No | Yes |

> **Scope:** synthbench tests AI-generated **media** — audio now; image, video,
> and music next. Not text, not embeddings, not RAG.

## Quick start

```bash
pip install 'synthbench[audio]'          # [audio] adds WER scoring (Whisper + jiwer)
export ELEVENLABS_API_KEY=sk-...
synthbench run --scenario examples/tts-stress.toml --dry-run   # preview plan + cost
synthbench run --scenario examples/tts-stress.toml             # run, then print the curve
```

`--dry-run` shows the concurrency plan and estimated cost without calling the
provider. Add `--output report.json` to write the machine-readable report for CI.

> Pre-PyPI: until `v0.1.0` is published, install from source —
> `git clone https://github.com/rajnavakoti/synthbench && cd synthbench && poetry install --extras audio`,
> then run with `poetry run synthbench ...`.

## Example output

```
        Synthbench degradation curve — elevenlabs eleven_multilingual_v2
 Concurrency  Requests    Avg     P50     P95    WER   Success    Cost   Status
          1         1   1.12s   0.98s   1.40s  0.020    100%    $0.12   PASS
          5         5   2.32s   2.03s   2.90s  0.020    100%    $0.60   PASS
         10        10   4.16s   3.64s   5.20s  0.040    100%    $1.20   PASS
         25        25   9.68s   8.47s  12.10s  0.080    100%    $3.00   WARN
         50    38 (+12) 28.00s  24.50s  35.00s  0.200     40%    $6.00   FAIL

Total: $7.32 / $5.00 (budget exceeded) in 4m 12s

Quality degradation detected at concurrency ≥ 25
  - WER 0.020 → 0.080 (4.0x)
  - P95 latency 1.4s → 12.1s
```

The curve is the product: it tells you the **safe operating range** (1–10 here),
where to **watch** (25), and where it **fails** (50) — before your users find out.

## Scenario file

A scenario fully describes one benchmark run, in TOML:

```toml
[scenario]
name = "ElevenLabs TTS stress test"
provider = "elevenlabs"
modality = "tts"
concurrency = [1, 5, 10, 25, 50]   # the x-axis of the curve
budget_limit_usd = 5.00            # hard stop — never overspend on a test

[provider.elevenlabs]
api_key = "${ELEVENLABS_API_KEY}"  # ${VAR} reads from the environment
model = "eleven_multilingual_v2"
voice_id = "21m00Tcm4TlvDq8ikWAM"
# cost_per_million_chars = 180.0   # optional: your plan's rate, for an accurate cost axis

[prompts]
source = "inline"
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your appointment for Tuesday at three thirty PM.",
]
# or: source = "file", path = "prompts/tts-stress.txt"  (one prompt per line)

[scoring]
metrics = ["latency", "wer", "duration_accuracy", "file_integrity"]
whisper_model = "base"             # tiny | base | small

[thresholds]
warn_wer = 0.05
fail_wer = 0.10
warn_latency_p95 = 10.0
fail_latency_p95 = 30.0
```

See [`examples/`](examples/) for full ElevenLabs and OpenAI scenarios. The same
file format works for any provider — just change `provider` and its section.

## Supported providers

| Provider | Modality | Status |
|---|---|---|
| ElevenLabs | TTS | ✅ v0.1 |
| OpenAI TTS | TTS | ✅ v0.1 |
| fal.ai / Stability | Image | Planned |
| Runway / Luma | Video | Planned |

Adapters are pluggable behind one interface — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) to add one.

## Quality metrics (TTS)

| Metric | What it catches |
|---|---|
| **WER** (Whisper + jiwer) | Pronunciation/accuracy degradation — the headline silent-failure signal |
| **duration_accuracy** | Truncated or runaway audio |
| **file_integrity** | Empty or corrupt audio returned with HTTP 200 |
| **latency** | Full async lifecycle, p50/p95 |
| **cost** | Per-request and cumulative, with budget guards |

## How it works

`config → provider adapters → workload engine → quality scoring → curve.` The
engine drives each concurrency level under an asyncio semaphore, times the full
request lifecycle, enforces the budget guard, and the reporter turns the results
into the PASS/WARN/FAIL curve. Architecture details:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## License

Apache 2.0
