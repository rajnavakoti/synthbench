# synthbench

**Know where your AI media pipeline breaks — before your users do.**

Synthbench produces trusted quality-under-load degradation curves for AI media generation — showing exactly where quality, latency, and cost break as concurrency rises.

> Traditional load testing tools (k6, JMeter, Gatling) measure HTTP latency in milliseconds. AI media generation APIs return binary artifacts (audio, images, video) via async job queues that take seconds to minutes — and the worst failures are silent quality degradation, not HTTP 500s. Synthbench measures what no other tool can.

## Quick Start

```bash
pip install synthbench[audio]

synthbench run --scenario tts-stress.toml
```

## Example Output

```
Synthbench TTS Benchmark — ElevenLabs eleven_multilingual_v2
================================================================

Concurrency  Avg Latency   P95 Latency   WER      Cost     Status
-----------  -----------   -----------   ------   ------   ------
1            1.2s          1.4s          0.02     $0.12    PASS
5            1.8s          2.9s          0.02     $0.60    PASS
10           3.1s          5.2s          0.04     $1.20    PASS
25           7.4s          12.1s         0.08     $3.00    WARN
50           timeout       timeout       N/A      $2.40    FAIL

Quality degradation detected at concurrency >= 25:
  - WER increased 4x (0.02 -> 0.08)
  - P95 latency exceeded 10s threshold
```

## What Synthbench Measures That Other Tools Cannot

| Capability | k6 / JMeter / Gatling | Synthbench |
|---|---|---|
| Async job lifecycle (submit → poll → receive) | No | Yes |
| Binary artifact quality scoring (WER, MOS, CLIPScore) | No | Yes |
| Silent quality degradation detection | No | Yes |
| Credit-based cost tracking and budget guards | No | Yes |
| Provider-specific rate limit handling | No | Yes |
| Quality-under-load degradation curves | No | Yes |

## Supported Providers

| Provider | Modality | Status |
|---|---|---|
| ElevenLabs | TTS | v0.1.0 |
| OpenAI TTS | TTS | v0.1.0 |
| fal.ai | Image | Planned |
| Stability AI | Image | Planned |
| Runway | Video | Planned |

## Scenario File

```toml
[scenario]
name = "ElevenLabs TTS stress test"
provider = "elevenlabs"
modality = "tts"
concurrency = [1, 5, 10, 25, 50]
budget_limit_usd = 5.00

[provider.elevenlabs]
api_key = "${ELEVENLABS_API_KEY}"
model = "eleven_multilingual_v2"
voice_id = "21m00Tcm4TlvDq8ikWAM"

[prompts]
source = "inline"
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Please confirm your appointment for Tuesday at three thirty PM.",
]

[scoring]
metrics = ["latency", "wer", "duration_accuracy", "file_integrity"]
whisper_model = "base"

[thresholds]
warn_wer = 0.05
fail_wer = 0.10
```

## License

Apache 2.0
