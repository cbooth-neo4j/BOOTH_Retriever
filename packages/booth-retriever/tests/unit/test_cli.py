"""Tier-2 CLI tests using typer's CliRunner.

These tests monkey-patch ``booth_retriever.cli._driver_factory`` so commands
don't try to open a real Neo4j connection. They then assert on the CLI's
stdout/stderr/exit code for each command path.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from booth_retriever.cli import _load_env_file, app, set_driver_factory
from booth_retriever.curator import (
    ApprovalResult,
    CuratorStats,
    PendingQuery,
    QueryDetail,
)

pytestmark = pytest.mark.unit
runner = CliRunner()


@pytest.fixture
def fake_driver_factory():
    """Install a fake driver factory for the duration of a test.

    Returns the factory so tests can inspect calls to it.
    """
    factory = MagicMock(return_value=MagicMock())
    set_driver_factory(factory)
    try:
        yield factory
    finally:
        # Restore the default on teardown
        from booth_retriever.cli import _default_driver_factory

        set_driver_factory(_default_driver_factory)


@pytest.fixture
def patched_curator(monkeypatch, fake_driver_factory):
    """Patch BOOTHCurator in cli.py to return a pre-built MagicMock.

    Tests receive the MagicMock so they can program return values and
    assert on method calls.
    """
    mock_curator = MagicMock()
    monkeypatch.setattr(
        "booth_retriever.cli.BOOTHCurator",
        lambda *args, **kwargs: mock_curator,
    )
    return mock_curator


# ---------- Basic surface ----------------------------------------------------


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "BOOTH" in result.stdout


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "booth-retriever" in result.stdout


def test_subcommand_help() -> None:
    for args in [
        ["curate", "--help"],
        ["curate", "list", "--help"],
        ["curate", "approve", "--help"],
        ["stats", "--help"],
        ["feedback", "--help"],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"{args} failed: {result.stdout}\n{result.stderr}"


# ---------- init requires password ------------------------------------------


def test_init_fails_without_password(monkeypatch, fake_driver_factory) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 2
    assert "password" in result.stderr.lower()


# ---------- .env auto-loading ------------------------------------------------


def test_load_env_file_reads_dotenv_from_cwd(tmp_path, monkeypatch) -> None:
    """``_load_env_file`` should populate os.environ from a .env in CWD."""
    (tmp_path / ".env").write_text(
        "NEO4J_PASSWORD=from-dotenv\nNEO4J_URI=bolt://example:7687\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("NEO4J_URI", raising=False)

    _load_env_file()

    assert os.environ["NEO4J_PASSWORD"] == "from-dotenv"
    assert os.environ["NEO4J_URI"] == "bolt://example:7687"


def test_load_env_file_does_not_override_real_env(tmp_path, monkeypatch) -> None:
    """Real environment variables must win over .env values."""
    (tmp_path / ".env").write_text("NEO4J_PASSWORD=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEO4J_PASSWORD", "from-real-env")

    _load_env_file()

    assert os.environ["NEO4J_PASSWORD"] == "from-real-env"


# ---------- stats -----------------------------------------------------------


def test_stats_text_output(patched_curator) -> None:
    patched_curator.stats.return_value = CuratorStats(
        counts={"approved": 3, "pending_approval": 2, "rejected": 0}
    )
    result = runner.invoke(
        app,
        ["stats", "--uri", "bolt://x", "--password", "pw"],
    )
    assert result.exit_code == 0
    assert "Total queries: 5" in result.stdout
    assert "approved" in result.stdout
    assert "pending_approval" in result.stdout


def test_stats_json_output(patched_curator) -> None:
    patched_curator.stats.return_value = CuratorStats(counts={"approved": 7})
    result = runner.invoke(
        app,
        ["stats", "--json", "--uri", "bolt://x", "--password", "pw"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total"] == 7
    assert payload["by_status"] == {"approved": 7}


# ---------- curate list -----------------------------------------------------


def test_curate_list_empty(patched_curator) -> None:
    patched_curator.list_pending.return_value = []
    result = runner.invoke(
        app, ["curate", "list", "--uri", "bolt://x", "--password", "pw"]
    )
    assert result.exit_code == 0
    assert "No queries found." in result.stdout


def test_curate_list_renders_rows(patched_curator) -> None:
    patched_curator.list_pending.return_value = [
        PendingQuery(
            query_id="11111111-2222-3333-4444-555555555555",
            query_text="count all the users please",
            status="pending_approval",
            risk_level="low",
            timestamp="2026-04-01T10:00Z",
            has_fewshot=False,
        )
    ]
    result = runner.invoke(
        app, ["curate", "list", "--uri", "bolt://x", "--password", "pw"]
    )
    assert result.exit_code == 0
    assert "Found 1 query" in result.stdout
    assert "11111111" in result.stdout
    assert "pending_approval" in result.stdout


def test_curate_list_rejects_unknown_status(patched_curator) -> None:
    result = runner.invoke(
        app,
        [
            "curate",
            "list",
            "--status",
            "garbage",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 2
    assert "unknown status" in result.stderr.lower()


def test_curate_list_json(patched_curator) -> None:
    patched_curator.list_pending.return_value = [
        PendingQuery(
            query_id="abc",
            query_text="hi",
            status="pending_approval",
            risk_level=None,
            timestamp=None,
            has_fewshot=True,
        )
    ]
    result = runner.invoke(
        app,
        ["curate", "list", "--json", "--uri", "bolt://x", "--password", "pw"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["query_id"] == "abc"
    assert payload[0]["has_fewshot"] is True


# ---------- curate show -----------------------------------------------------


def test_curate_show_missing_query_exits_1(patched_curator) -> None:
    patched_curator.get.return_value = None
    result = runner.invoke(
        app, ["curate", "show", "nope", "--uri", "bolt://x", "--password", "pw"]
    )
    assert result.exit_code == 1
    assert "no query" in result.stderr.lower()


def test_curate_show_renders_detail(patched_curator) -> None:
    patched_curator.get.return_value = QueryDetail(
        query_id="q1",
        query_text="count users",
        status="approved",
        risk_level="low",
        timestamp="2026-04-01T10:00Z",
        user_feedback="helpful",
        fewshot_cypher="MATCH (u:User) RETURN count(u)",
        fewshot_parameters=["tenant"],
    )
    result = runner.invoke(
        app, ["curate", "show", "q1", "--uri", "bolt://x", "--password", "pw"]
    )
    assert result.exit_code == 0
    assert "count users" in result.stdout
    assert "MATCH (u:User)" in result.stdout
    assert "tenant" in result.stdout


# ---------- curate approve --------------------------------------------------


def test_curate_approve_success(patched_curator) -> None:
    patched_curator.approve.return_value = ApprovalResult(
        query_id="q1", fewshot_id="fs-1", fewshot_was_new=True
    )
    result = runner.invoke(
        app,
        [
            "curate",
            "approve",
            "q1",
            "--cypher",
            "RETURN 1",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 0
    assert "created" in result.stdout
    patched_curator.approve.assert_called_once()
    _, kwargs = patched_curator.approve.call_args
    assert kwargs["cypher_template"] == "RETURN 1"
    assert kwargs["parameters"] == []


def test_curate_approve_with_params(patched_curator) -> None:
    patched_curator.approve.return_value = ApprovalResult(
        query_id="q1", fewshot_id="fs-1", fewshot_was_new=False
    )
    result = runner.invoke(
        app,
        [
            "curate",
            "approve",
            "q1",
            "--cypher",
            "MATCH (n {name:$name}) RETURN n",
            "--params",
            "name, role",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 0
    assert "updated" in result.stdout
    _, kwargs = patched_curator.approve.call_args
    assert kwargs["parameters"] == ["name", "role"]


def test_curate_approve_from_file(tmp_path, patched_curator) -> None:
    cypher_file = tmp_path / "template.cypher"
    cypher_file.write_text("MATCH (n) RETURN n LIMIT 5", encoding="utf-8")
    patched_curator.approve.return_value = ApprovalResult(
        query_id="q1", fewshot_id="fs-1", fewshot_was_new=True
    )

    result = runner.invoke(
        app,
        [
            "curate",
            "approve",
            "q1",
            "--cypher",
            f"@{cypher_file}",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 0
    _, kwargs = patched_curator.approve.call_args
    assert kwargs["cypher_template"] == "MATCH (n) RETURN n LIMIT 5"


def test_curate_approve_rejects_missing_file(tmp_path, patched_curator) -> None:
    result = runner.invoke(
        app,
        [
            "curate",
            "approve",
            "q1",
            "--cypher",
            f"@{tmp_path}/nope.cypher",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in (result.stdout + result.stderr).lower()


def test_curate_approve_surfaces_curator_error(patched_curator) -> None:
    patched_curator.approve.side_effect = ValueError("No Query node with id 'bogus'")
    result = runner.invoke(
        app,
        [
            "curate",
            "approve",
            "bogus",
            "--cypher",
            "RETURN 1",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 1
    assert "No Query node" in result.stderr


# ---------- curate reject ---------------------------------------------------


def test_curate_reject_success(patched_curator) -> None:
    result = runner.invoke(
        app,
        [
            "curate",
            "reject",
            "q1",
            "--reason",
            "nonsense",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 0
    assert "nonsense" in result.stdout
    _, kwargs = patched_curator.reject.call_args
    assert kwargs["reason"] == "nonsense"


# ---------- curate edit -----------------------------------------------------


def test_curate_edit_success(patched_curator) -> None:
    result = runner.invoke(
        app,
        [
            "curate",
            "edit",
            "q1",
            "--cypher",
            "RETURN 2",
            "--params",
            "x",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 0
    _, kwargs = patched_curator.edit_fewshot.call_args
    assert kwargs["cypher_template"] == "RETURN 2"
    assert kwargs["parameters"] == ["x"]


# ---------- feedback --------------------------------------------------------


def test_feedback_helpful(patched_curator) -> None:
    result = runner.invoke(
        app, ["feedback", "q1", "--uri", "bolt://x", "--password", "pw"]
    )
    assert result.exit_code == 0
    _, kwargs = patched_curator.submit_feedback.call_args
    assert kwargs["helpful"] is True
    assert "helpful" in result.stdout


def test_feedback_not_helpful(patched_curator) -> None:
    result = runner.invoke(
        app,
        [
            "feedback",
            "q1",
            "--not-helpful",
            "--uri",
            "bolt://x",
            "--password",
            "pw",
        ],
    )
    assert result.exit_code == 0
    _, kwargs = patched_curator.submit_feedback.call_args
    assert kwargs["helpful"] is False


def test_feedback_surfaces_curator_error(patched_curator) -> None:
    patched_curator.submit_feedback.side_effect = ValueError("No Query node with id 'q1'")
    result = runner.invoke(
        app, ["feedback", "q1", "--uri", "bolt://x", "--password", "pw"]
    )
    assert result.exit_code == 1
    assert "No Query node" in result.stderr
