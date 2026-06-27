"""Tests for the CLI app entry point and the run command."""

import re
import textwrap
from pathlib import Path

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
    result = runner.invoke(app, ["run", "--scenario", str(_scenario(tmp_path))])
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
    # 2 prompts x 2 concurrency levels = 4 requests.
    assert "4 requests" in output
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
