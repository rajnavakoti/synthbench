"""Tests for the provider generation vocabulary (request/job/artifact)."""

from synthbench.providers.base import (
    GenerationArtifact,
    GenerationJob,
    GenerationRequest,
    GenerationStatus,
)


def _request(**params: object) -> GenerationRequest:
    return GenerationRequest(
        prompt="hello", provider="elevenlabs", model="m", provider_params=params
    )


def test_input_hash_is_stable_and_order_independent() -> None:
    a = GenerationRequest(
        prompt="hi", provider="p", model="m", provider_params={"x": 1, "y": 2}
    )
    b = GenerationRequest(
        prompt="hi", provider="p", model="m", provider_params={"y": 2, "x": 1}
    )
    assert a.input_hash == b.input_hash


def test_input_hash_ignores_index() -> None:
    a = GenerationRequest(prompt="hi", provider="p", index=0)
    b = GenerationRequest(prompt="hi", provider="p", index=99)
    assert a.input_hash == b.input_hash


def test_input_hash_changes_with_prompt() -> None:
    a = GenerationRequest(prompt="one", provider="p")
    b = GenerationRequest(prompt="two", provider="p")
    assert a.input_hash != b.input_hash


def test_job_terminal_states() -> None:
    req = _request()
    assert not GenerationJob(request=req, status=GenerationStatus.PENDING).is_terminal
    assert not GenerationJob(request=req, status=GenerationStatus.RUNNING).is_terminal
    assert GenerationJob(request=req, status=GenerationStatus.SUCCEEDED).is_terminal
    assert GenerationJob(request=req, status=GenerationStatus.FAILED).is_terminal
    assert GenerationJob(request=req, status=GenerationStatus.CANCELLED).is_terminal


def test_artifact_size() -> None:
    artifact = GenerationArtifact(data=b"abcd", content_type="audio/mpeg")
    assert artifact.size_bytes == 4
