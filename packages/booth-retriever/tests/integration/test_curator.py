"""Tier-3 integration tests for ``BOOTHCurator`` against a real Neo4j.

Seeds a handful of Query nodes in various statuses, then drives the full
curator API against them. Auto-skips without Docker (see ``conftest.py``).
"""

from __future__ import annotations

import uuid

import pytest

from booth_retriever import BOOTHCurator, init_schema
from booth_retriever.curator import (
    ALL_STATUSES,
    PENDING_STATUSES,
    ApprovalResult,
    PendingQuery,
    QueryDetail,
)

pytestmark = pytest.mark.integration


def _clear_queries(driver) -> None:
    with driver.session() as session:
        session.run(
            "MATCH (n) WHERE n:Query OR n:UserQuestion OR n:FewShot "
            "OR n:CypherAttempt OR n:Response DETACH DELETE n"
        )


def _seed_query(
    driver,
    *,
    query_id: str,
    text: str,
    status: str,
    risk_level: str = "low",
) -> None:
    with driver.session() as session:
        session.run(
            "CREATE (q:Query { "
            "id: $id, text: $text, status: $status, "
            "risk_level: $risk, timestamp: datetime() "
            "})",
            id=query_id,
            text=text,
            status=status,
            risk=risk_level,
        )


@pytest.fixture
def seeded_driver(neo4j_driver):
    init_schema(neo4j_driver, embedding_dimensions=16)
    _clear_queries(neo4j_driver)
    _seed_query(
        neo4j_driver, query_id="pending-1", text="pending one", status="pending_approval"
    )
    _seed_query(
        neo4j_driver, query_id="pending-2", text="pending two", status="pending_approval"
    )
    _seed_query(
        neo4j_driver, query_id="declined-1", text="risky", status="declined",
        risk_level="high",
    )
    _seed_query(
        neo4j_driver, query_id="approved-1", text="approved", status="approved"
    )
    yield neo4j_driver


# ---------- Reads ------------------------------------------------------------


def test_list_pending_returns_pending_statuses(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)

    results = curator.list_pending(limit=50)

    ids = {r.query_id for r in results}
    # Should include pending_approval and declined but NOT approved
    assert ids == {"pending-1", "pending-2", "declined-1"}
    for r in results:
        assert isinstance(r, PendingQuery)
        assert r.status in PENDING_STATUSES


def test_list_by_status_approved(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    results = curator.list_by_status("approved", limit=10)
    assert [r.query_id for r in results] == ["approved-1"]


def test_list_by_status_rejects_unknown(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    with pytest.raises(ValueError, match="Unknown status"):
        curator.list_by_status("garbage")


def test_get_existing_query(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    detail = curator.get("pending-1")
    assert isinstance(detail, QueryDetail)
    assert detail.query_text == "pending one"
    assert detail.fewshot_cypher is None


def test_get_missing_query_returns_none(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    assert curator.get("does-not-exist") is None


def test_stats_counts_by_status(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    stats = curator.stats()
    assert stats.counts["pending_approval"] == 2
    assert stats.counts["declined"] == 1
    assert stats.counts["approved"] == 1
    # Unseen statuses come back as 0 (pre-populated)
    assert stats.counts["rejected"] == 0
    assert stats.total == 4
    assert set(stats.counts.keys()) >= set(ALL_STATUSES)


# ---------- Mutations --------------------------------------------------------


def test_approve_creates_fewshot_end_to_end(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)

    result = curator.approve(
        "pending-1",
        cypher_template="MATCH (u:User) RETURN count(u) AS n",
        parameters=[],
        category="FACTUAL",
    )

    assert isinstance(result, ApprovalResult)
    assert result.fewshot_was_new is True
    assert result.query_id == "pending-1"

    detail = curator.get("pending-1")
    assert detail is not None
    assert detail.status == "approved"
    assert detail.fewshot_cypher == "MATCH (u:User) RETURN count(u) AS n"


def test_approve_is_idempotent(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    first = curator.approve("pending-2", cypher_template="RETURN 1")
    second = curator.approve(
        "pending-2", cypher_template="RETURN 2", parameters=[]
    )

    assert first.fewshot_was_new is True
    assert second.fewshot_was_new is False
    assert first.fewshot_id == second.fewshot_id

    detail = curator.get("pending-2")
    assert detail is not None
    assert detail.fewshot_cypher == "RETURN 2"

    # No duplicate FewShot
    with seeded_driver.session() as session:
        rows = list(
            session.run(
                "MATCH (q:Query {id: 'pending-2'})-[:FEW_SHOT_EXAMPLE]->(fs) "
                "RETURN count(fs) AS n"
            )
        )
    assert rows[0]["n"] == 1


def test_approve_raises_for_missing_query(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.approve(f"missing-{uuid.uuid4()}", cypher_template="RETURN 1")


def test_reject_writes_reason(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    curator.reject("pending-1", reason="off-topic")
    detail = curator.get("pending-1")
    assert detail is not None
    assert detail.status == "rejected"
    assert detail.rejection_reason == "off-topic"


def test_edit_fewshot_requires_existing_fewshot(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    with pytest.raises(ValueError, match="no linked FewShot"):
        curator.edit_fewshot("pending-1", cypher_template="RETURN 2")


def test_edit_fewshot_updates_existing(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    curator.approve("pending-1", cypher_template="RETURN 1")
    curator.edit_fewshot("pending-1", cypher_template="RETURN 99", parameters=["n"])

    detail = curator.get("pending-1")
    assert detail is not None
    assert detail.fewshot_cypher == "RETURN 99"
    assert detail.fewshot_parameters == ["n"]


def test_submit_feedback_helpful_moves_to_pending_approval(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    curator.submit_feedback("approved-1", helpful=True)

    detail = curator.get("approved-1")
    assert detail is not None
    assert detail.status == "pending_approval"
    assert detail.user_feedback == "helpful"


def test_submit_feedback_not_helpful_moves_to_needs_review(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    curator.submit_feedback("pending-1", helpful=False)

    detail = curator.get("pending-1")
    assert detail is not None
    assert detail.status == "needs_review"
    assert detail.user_feedback == "not_helpful"


def test_submit_feedback_raises_for_missing_query(seeded_driver) -> None:
    curator = BOOTHCurator(driver=seeded_driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.submit_feedback(f"missing-{uuid.uuid4()}", helpful=True)
