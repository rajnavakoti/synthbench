"""Tests for the CLI app entry point and the run command."""

import re
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synthbench.cli.app import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI color codes so assertions are terminal-independent.

    Rich highlights option names by wrapping each dash in its own color span
    (``\\x1b[36m-\\x1b[0m\\x1b[36m-scenario``), which splits the literal
    ``--scenario`` token. CI emits color; a local non-tty run does not — so we
    normalize before matching.
    """
    return _ANSI.sub("", text)


VALID_SCENARIO = """\
[scenario]
name = "CLI test"
provider = "openai"
modality = "tts"
concurrency = [1, 5]
budget_limit_usd = 5.0

[provider.openai]
model = "tts-1"

[prompts]
source = "inline"
texts = ["Hello world.", "Second prompt."]
"""


def _scenario(tmp_path: Path, content: str = VALID_SCENARIO) -> Path:
    path = tmp_path / "scenario.toml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_no_args_shows_usage() -> None:
    result = runner.invoke(app, [])
    output = _plain(result.output)
    assert "Usage" in output
    assert "COMMAND" in output


def test_run_requires_scenario() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "scenario" in _plain(result.output).lower()


def test_run_help_lists_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "--scenario" in output
    assert "--dry-run" in output
    assert "--output" in output


def test_run_prints_summary(tmp_path: Path) -> None:
    # The summary panel renders in dry-run too; assert on it without executing.
    result = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path)), "--dry-run"]
    )
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "CLI test" in output
    assert "openai" in output
    assert "tts-1" in output


def test_dry_run_prints_plan_and_cost(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path)), "--dry-run"]
    )
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "Dry run" in output
    # concurrency [1, 5], k=1 -> 1 + 5 = 6 requests.
    assert "6 requests" in output
    assert "budget" in output.lower()


def test_dry_run_cost_override_changes_total(tmp_path: Path) -> None:
    # Same scenario, with a high per-million override, should cost more.
    base = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path)), "--dry-run"]
    )
    with_override = VALID_SCENARIO.replace(
        'model = "tts-1"', 'model = "tts-1"\ncost_per_million_chars = 5000.0'
    )
    overridden = runner.invoke(
        app,
        ["run", "--scenario", str(_scenario(tmp_path, with_override)), "--dry-run"],
    )
    assert base.exit_code == 0
    assert overridden.exit_code == 0
    # The override is ~333x the OpenAI tts-1 default, so the total must differ.
    assert _plain(base.output) != _plain(overridden.output)


def test_invalid_scenario_exits_nonzero(tmp_path: Path) -> None:
    bad = VALID_SCENARIO.replace("budget_limit_usd = 5.0\n", "")
    result = runner.invoke(app, ["run", "--scenario", str(_scenario(tmp_path, bad))])
    assert result.exit_code == 1
    assert "budget_limit_usd" in _plain(result.output)


def test_missing_scenario_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--scenario", str(tmp_path / "missing.toml")])
    assert result.exit_code == 1
    assert "not found" in _plain(result.output).lower()


def test_run_executes_with_mock_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch the registry so the run executes against an in-memory adapter
    # instead of calling a real provider.
    from synthbench.cli import run as run_module
    from synthbench.providers.base import (
        GenerationArtifact,
        GenerationJob,
        GenerationRequest,
        GenerationStatus,
        ProviderAdapter,
    )

    class _CliMockAdapter(ProviderAdapter):
        @property
        def name(self) -> str:
            return "openai"

        def estimate_cost_usd(self, request: GenerationRequest) -> float:
            return 0.001

        async def submit(self, request: GenerationRequest) -> GenerationJob:
            return GenerationJob(
                request=request,
                status=GenerationStatus.SUCCEEDED,
                artifact=GenerationArtifact(data=b"x", content_type="audio/mpeg"),
            )

        async def poll(self, job: GenerationJob) -> GenerationJob:
            return job

        async def retrieve(self, job: GenerationJob) -> GenerationArtifact:
            assert job.artifact is not None
            return job.artifact

    monkeypatch.setattr(
        run_module, "create_adapter", lambda name, config: _CliMockAdapter()
    )
    # metrics = ["latency"] needs no scoring libs, so this exercises pure
    # execution wiring without the [audio] extra. chdir to tmp so the
    # save-by-default run bundle lands under tmp, not the repo.
    monkeypatch.chdir(tmp_path)
    content = VALID_SCENARIO + '\n[scoring]\nmetrics = ["latency"]\n'
    out_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--scenario",
            str(_scenario(tmp_path, content)),
            "--output",
            str(out_path),
        ],
    )
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "degradation curve" in output
    assert "within budget" in output

    import json

    # Explicit --output report.
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["provider"] == "openai"
    assert len(report["levels"]) == 2  # concurrency [1, 5]
    assert all("verdict" in level for level in report["levels"])

    # Save-by-default: a run bundle exists with audio, manifest, snapshot, report.
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    bundle = run_dirs[0]
    assert (bundle / "manifest.json").exists()
    assert (bundle / "report.json").exists()
    assert list((bundle / "audio").glob("*.mp3"))  # clips saved
    # Snapshot must NOT contain the real API key.
    snapshot = (bundle / "scenario.snapshot.json").read_text()
    assert "sk-" not in snapshot or "redacted" in snapshot


def test_run_no_save_skips_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from synthbench.cli import run as run_module
    from synthbench.providers.base import (
        GenerationArtifact,
        GenerationJob,
        GenerationRequest,
        GenerationStatus,
        ProviderAdapter,
    )

    class _Mock(ProviderAdapter):
        @property
        def name(self) -> str:
            return "openai"

        def estimate_cost_usd(self, request: GenerationRequest) -> float:
            return 0.001

        async def submit(self, request: GenerationRequest) -> GenerationJob:
            return GenerationJob(
                request=request,
                status=GenerationStatus.SUCCEEDED,
                artifact=GenerationArtifact(data=b"x", content_type="audio/mpeg"),
            )

        async def poll(self, job: GenerationJob) -> GenerationJob:
            return job

        async def retrieve(self, job: GenerationJob) -> GenerationArtifact:
            assert job.artifact is not None
            return job.artifact

    monkeypatch.setattr(run_module, "create_adapter", lambda name, config: _Mock())
    monkeypatch.chdir(tmp_path)
    content = VALID_SCENARIO + '\n[scoring]\nmetrics = ["latency"]\n'
    result = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path, content)), "--no-save"]
    )
    assert result.exit_code == 0
    assert not (tmp_path / "runs").exists()  # nothing saved


def test_run_missing_api_key_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    # No OPENAI_API_KEY and none in the scenario -> adapter construction fails.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    content = VALID_SCENARIO + '\n[scoring]\nmetrics = ["latency"]\n'
    result = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path, content))]
    )
    assert result.exit_code == 1
    assert "api key" in _plain(result.output).lower()


def test_run_missing_audio_extra_exits_nonzero(tmp_path: Path, monkeypatch) -> None:
    # A scenario asking for WER fails fast (before any API call) when the audio
    # extra is not installed.
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    # Provide an API key so adapter creation succeeds and we reach scorer build.
    content = VALID_SCENARIO.replace(
        'model = "tts-1"', 'model = "tts-1"\napi_key = "sk-test"'
    )
    content += '\n[scoring]\nmetrics = ["wer"]\n'
    result = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path, content))]
    )
    assert result.exit_code == 1
    assert "synthbench[audio]" in _plain(result.output)
