"""Tests for the CLI app entry point and the run command."""

import textwrap
from pathlib import Path

from typer.testing import CliRunner

from synthbench.cli.app import app

runner = CliRunner()

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
    assert "Usage" in result.output
    assert "COMMAND" in result.output


def test_run_requires_scenario() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "scenario" in result.output.lower()


def test_run_help_lists_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--scenario" in result.output
    assert "--dry-run" in result.output
    assert "--output" in result.output


def test_run_prints_summary(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--scenario", str(_scenario(tmp_path))])
    assert result.exit_code == 0
    assert "CLI test" in result.output
    assert "openai" in result.output
    assert "tts-1" in result.output


def test_dry_run_prints_plan_and_cost(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "--scenario", str(_scenario(tmp_path)), "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Dry run" in result.output
    # 2 prompts x 2 concurrency levels = 4 requests.
    assert "4 requests" in result.output
    assert "budget" in result.output.lower()


def test_invalid_scenario_exits_nonzero(tmp_path: Path) -> None:
    bad = VALID_SCENARIO.replace("budget_limit_usd = 5.0\n", "")
    result = runner.invoke(app, ["run", "--scenario", str(_scenario(tmp_path, bad))])
    assert result.exit_code == 1
    assert "budget_limit_usd" in result.output


def test_missing_scenario_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--scenario", str(tmp_path / "missing.toml")])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()
