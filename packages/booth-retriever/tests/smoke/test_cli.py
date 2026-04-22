"""Tier-1 smoke tests for the `booth` CLI.

We only verify that the Typer app is wired up correctly and that --help and
--version work. Business-logic subcommands currently raise NotImplementedError
and are covered by tests/unit/test_cli.py once they're implemented.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from booth_retriever import __version__
from booth_retriever.cli import app

pytestmark = pytest.mark.smoke

runner = CliRunner()


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stdout
    assert "booth" in result.stdout.lower()


def test_version_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.stdout
    assert __version__ in result.stdout


def test_curate_subcommand_is_registered() -> None:
    """`booth curate --help` lists the curation subcommands."""
    result = runner.invoke(app, ["curate", "--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("list", "show", "approve", "reject", "edit"):
        assert cmd in result.stdout, f"missing curate subcommand: {cmd}"


def test_top_level_commands_are_registered() -> None:
    """`booth --help` lists init, stats, feedback, curate."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.stdout
    for cmd in ("init", "stats", "feedback", "curate"):
        assert cmd in result.stdout, f"missing top-level command: {cmd}"
