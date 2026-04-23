"""Tier-2 tests for the FastAPI layer in ``booth_retriever.web``.

Pattern mirrors ``tests/unit/test_cli.py``: we build the app with an
injected ``MagicMock`` curator, then assert each route maps correctly
onto a method call. No real Neo4j is touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from booth_retriever.curator import (
    ApprovalResult,
    CuratorStats,
    PendingQuery,
    QueryDetail,
)
from booth_retriever.web import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
def curator_client():
    """Yield ``(TestClient, MagicMock curator)`` with a fresh app each test."""
    mock = MagicMock()
    app = create_app(curator=mock)
    with TestClient(app) as client:
        yield client, mock


# ---------- Health ----------------------------------------------------------


def test_health(curator_client) -> None:
    client, _ = curator_client
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------- Stats -----------------------------------------------------------


def test_stats(curator_client) -> None:
    client, curator = curator_client
    curator.stats.return_value = CuratorStats(
        counts={"approved": 3, "pending_approval": 2, "rejected": 0}
    )
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 5
    assert payload["counts"]["approved"] == 3
    assert payload["counts"]["pending_approval"] == 2


# ---------- List ------------------------------------------------------------


def test_list_pending_default(curator_client) -> None:
    client, curator = curator_client
    curator.list_pending.return_value = [
        PendingQuery(
            query_id="q1",
            query_text="count users",
            status="pending_approval",
            risk_level="low",
            timestamp="2026-04-01T10:00Z",
            has_fewshot=False,
        )
    ]
    resp = client.get("/api/queries")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["has_fewshot"] is False
    curator.list_pending.assert_called_once_with(limit=50)


def test_list_by_status(curator_client) -> None:
    client, curator = curator_client
    curator.list_by_status.return_value = []
    resp = client.get("/api/queries?status=approved&limit=10")
    assert resp.status_code == 200
    assert resp.json() == []
    curator.list_by_status.assert_called_once_with("approved", limit=10)


def test_list_rejects_unknown_status(curator_client) -> None:
    client, _ = curator_client
    resp = client.get("/api/queries?status=garbage")
    assert resp.status_code == 400
    assert "unknown status" in resp.json()["detail"].lower()


def test_list_rejects_nonpositive_limit(curator_client) -> None:
    client, _ = curator_client
    resp = client.get("/api/queries?limit=0")
    assert resp.status_code == 400
    assert "limit" in resp.json()["detail"].lower()


# ---------- Get detail ------------------------------------------------------


def test_get_query_detail(curator_client) -> None:
    client, curator = curator_client
    curator.get.return_value = QueryDetail(
        query_id="q1",
        query_text="count users",
        status="approved",
        risk_level="low",
        timestamp="2026-04-01T10:00Z",
        user_feedback="helpful",
        fewshot_cypher="MATCH (u:User) RETURN count(u)",
        fewshot_parameters=["tenant"],
    )
    resp = client.get("/api/queries/q1")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["query_id"] == "q1"
    assert payload["fewshot_cypher"] == "MATCH (u:User) RETURN count(u)"
    assert payload["fewshot_parameters"] == ["tenant"]


def test_get_query_missing_returns_404(curator_client) -> None:
    client, curator = curator_client
    curator.get.return_value = None
    resp = client.get("/api/queries/nope")
    assert resp.status_code == 404
    assert "no query" in resp.json()["detail"].lower()


# ---------- Approve ---------------------------------------------------------


def test_approve_success(curator_client) -> None:
    client, curator = curator_client
    curator.approve.return_value = ApprovalResult(
        query_id="q1", fewshot_id="fs-1", fewshot_was_new=True
    )
    resp = client.post(
        "/api/queries/q1/approve",
        json={"cypher_template": "RETURN 1"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["fewshot_id"] == "fs-1"
    assert payload["fewshot_was_new"] is True
    _, kwargs = curator.approve.call_args
    assert kwargs["cypher_template"] == "RETURN 1"
    assert kwargs["parameters"] == []


def test_approve_with_params_and_category(curator_client) -> None:
    client, curator = curator_client
    curator.approve.return_value = ApprovalResult(
        query_id="q1", fewshot_id="fs-1", fewshot_was_new=False
    )
    resp = client.post(
        "/api/queries/q1/approve",
        json={
            "cypher_template": "MATCH (u:User {id:$id}) RETURN u",
            "parameters": ["id"],
            "category": "lookup",
        },
    )
    assert resp.status_code == 200
    _, kwargs = curator.approve.call_args
    assert kwargs["parameters"] == ["id"]
    assert kwargs["category"] == "lookup"


def test_approve_verification_failure_is_422(curator_client) -> None:
    client, curator = curator_client
    curator.approve.side_effect = ValueError(
        "cypher_template failed verification: invalid bidirectional relationship: <-[:REL]->"
    )
    resp = client.post(
        "/api/queries/q1/approve",
        json={"cypher_template": "MATCH (a)<-[:REL]->(b) RETURN a"},
    )
    assert resp.status_code == 422
    assert "failed verification" in resp.json()["detail"]


def test_approve_missing_query_is_404(curator_client) -> None:
    client, curator = curator_client
    curator.approve.side_effect = ValueError("No Query node with id 'nope'")
    resp = client.post(
        "/api/queries/nope/approve",
        json={"cypher_template": "RETURN 1"},
    )
    assert resp.status_code == 404


def test_approve_empty_body_fails_validation(curator_client) -> None:
    client, _ = curator_client
    resp = client.post("/api/queries/q1/approve", json={"cypher_template": ""})
    assert resp.status_code == 422


# ---------- Edit ------------------------------------------------------------


def test_edit_success(curator_client) -> None:
    client, curator = curator_client
    resp = client.post(
        "/api/queries/q1/edit",
        json={"cypher_template": "RETURN 2", "parameters": ["x"]},
    )
    assert resp.status_code == 204
    _, kwargs = curator.edit_fewshot.call_args
    assert kwargs["cypher_template"] == "RETURN 2"
    assert kwargs["parameters"] == ["x"]


def test_edit_on_query_without_fewshot_is_404(curator_client) -> None:
    client, curator = curator_client
    curator.edit_fewshot.side_effect = ValueError(
        "Query 'q1' has no linked FewShot; approve it first."
    )
    resp = client.post(
        "/api/queries/q1/edit",
        json={"cypher_template": "RETURN 1"},
    )
    assert resp.status_code == 404


# ---------- Reject ----------------------------------------------------------


def test_reject_with_reason(curator_client) -> None:
    client, curator = curator_client
    resp = client.post(
        "/api/queries/q1/reject", json={"reason": "off-topic"}
    )
    assert resp.status_code == 204
    _, kwargs = curator.reject.call_args
    assert kwargs["reason"] == "off-topic"


def test_reject_without_reason(curator_client) -> None:
    client, curator = curator_client
    resp = client.post("/api/queries/q1/reject", json={})
    assert resp.status_code == 204
    _, kwargs = curator.reject.call_args
    assert kwargs["reason"] is None


def test_reject_missing_query_is_404(curator_client) -> None:
    client, curator = curator_client
    curator.reject.side_effect = ValueError("No Query node with id 'nope'")
    resp = client.post("/api/queries/nope/reject", json={})
    assert resp.status_code == 404


# ---------- Feedback --------------------------------------------------------


def test_feedback_helpful(curator_client) -> None:
    client, curator = curator_client
    resp = client.post("/api/queries/q1/feedback", json={"helpful": True})
    assert resp.status_code == 204
    _, kwargs = curator.submit_feedback.call_args
    assert kwargs["helpful"] is True


def test_feedback_not_helpful(curator_client) -> None:
    client, curator = curator_client
    resp = client.post("/api/queries/q1/feedback", json={"helpful": False})
    assert resp.status_code == 204
    _, kwargs = curator.submit_feedback.call_args
    assert kwargs["helpful"] is False


def test_feedback_missing_query_is_404(curator_client) -> None:
    client, curator = curator_client
    curator.submit_feedback.side_effect = ValueError("No Query node with id 'nope'")
    resp = client.post("/api/queries/nope/feedback", json={"helpful": True})
    assert resp.status_code == 404


# ---------- CORS / misc -----------------------------------------------------


def test_cors_allows_configured_origin() -> None:
    """The configured origin should be echoed back on a preflight."""
    app = create_app(curator=MagicMock(), cors_origins=["http://localhost:5173"])
    with TestClient(app) as client:
        resp = client.options(
            "/api/stats",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
