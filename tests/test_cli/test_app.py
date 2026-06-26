"""Tests for CLI app entry point."""

from typer.testing import CliRunner

from synthbench.cli.app import app

runner = CliRunner()


def test_no_args_shows_usage() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    assert "COMMAND" in result.output


def test_run_requires_scenario() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0
    assert "scenario" in result.output.lower()


def test_run_dry_run_with_scenario(tmp_path) -> None:
    scenario = tmp_path / "test.toml"
    scenario.write_text('[scenario]\nname = "test"')
    result = runner.invoke(app, ["run", "--scenario", str(scenario), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output
