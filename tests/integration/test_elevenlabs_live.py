"""Live ElevenLabs integration test.

Skipped unless ``ELEVENLABS_API_KEY`` is set, and excluded from CI via
``--ignore=tests/integration``. Run locally with a real key:

    ELEVENLABS_API_KEY=sk-... poetry run pytest tests/integration -v
"""

import os

import pytest

from synthbench.providers.elevenlabs import ElevenLabsAdapter

pytestmark = pytest.mark.skipif(
    not os.environ.get("ELEVENLABS_API_KEY"),
    reason="ELEVENLABS_API_KEY not set; live integration test skipped",
)


async def test_live_generation_returns_audio() -> None:
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    adapter = ElevenLabsAdapter(
        api_key=os.environ["ELEVENLABS_API_KEY"], voice_id=voice_id
    )
    try:
        job = await adapter.submit("This is a synthbench integration test.", {})
        assert await adapter.poll(job) is True
        audio = await adapter.retrieve(job)
        # A real generation returns a non-trivial audio payload.
        assert len(audio) > 1000
    finally:
        await adapter.aclose()
