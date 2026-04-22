"""Tier-3 integration tests for ``BOOTHCurator`` against a real Neo4j.

End-to-end verification of the curation round-trip: list pending queries,
get their detail, approve one with a FewShot template, then use the
retriever to confirm the approved FewShot hits the cache.
"""

from __future__ import annotations

import pytest

from booth_retriever import BOOTHCurator, BOOTHRetriever, init_schema

pytestmark = pytest.mark.integration


class _FakeEmbedder:
    """Deterministic embedder reused across tests in this module."""

    MATCH_VECTOR = [1.0, 0.0] + [0.0] * 1534
    ORTHOGONAL = [0.0, 1.0] + [0.0] * 1534

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}

    def embed_query(self, text: str) -> list[float]:
        return self.mapping.get(text, self.ORTHOGONAL)


@pytest.fixture
def clean_driver(neo4j_driver):
    """A driver pointed at a clean BOOTH schema with no Query nodes."""
    init_schema(neo4j_driver, embedding_dimensions=1536)
    with neo4j_driver.session() as session:
        session.run(
            "MATCH (n) WHERE n:Query OR n:UserQuestion OR n:FewShot "
            "OR n:CypherAttempt OR n:Response DETACH DELETE n"
        )
    yield neo4j_driver


def test_full_round_trip_query_to_approval_to_cache_hit(clean_driver) -> None:
    """The canonical BOOTH flow end to end:

    1. User asks a question - retriever stores it as pending.
    2. Curator lists pending, approves it with a fewshot cypher.
    3. Another user asks a similar question - retriever hits the cache.
    """
    embedder = _FakeEmbedder(
        {
            "how many nodes are there": _FakeEmbedder.MATCH_VECTOR,
            "total number of nodes": _FakeEmbedder.MATCH_VECTOR,
        }
    )
    retriever = BOOTHRetriever(driver=clean_driver, embedder=embedder)
    curator = BOOTHCurator(driver=clean_driver)

    # Step 1: new question -> pending
    first_response = retriever.query("how many nodes are there")
    assert first_response.success is False  # queued for curation
    query_id = first_response.query_id
    assert query_id

    # Step 2: curator workflow
    pending = curator.list_pending()
    assert any(p.query_id == query_id for p in pending)

    detail = curator.get(query_id)
    assert detail is not None
    assert detail.status == "pending_approval"
    assert detail.fewshot_cypher is None

    approval = curator.approve(
        query_id,
        cypher_template="MATCH (n) RETURN count(n) AS total",
    )
    assert approval.fewshot_was_new is True

    # Query should now be in approved state with a FewShot linked
    detail_after = curator.get(query_id)
    assert detail_after.status == "approved"
    assert detail_after.fewshot_cypher == "MATCH (n) RETURN count(n) AS total"

    # Step 3: a semantically-similar question hits the cache
    second_response = retriever.query("total number of nodes")
    assert second_response.success is True
    assert second_response.similar_match is True
    assert second_response.query_id == query_id
    assert second_response.cypher_used == "MATCH (n) RETURN count(n) AS total"


def test_approve_is_idempotent_and_replaces_fewshot(clean_driver) -> None:
    """Re-approving the same query updates the FewShot instead of creating
    a second one."""
    embedder = _FakeEmbedder({"x": _FakeEmbedder.MATCH_VECTOR})
    retriever = BOOTHRetriever(driver=clean_driver, embedder=embedder)
    curator = BOOTHCurator(driver=clean_driver)

    response = retriever.query("x")
    qid = response.query_id

    first = curator.approve(qid, cypher_template="RETURN 1")
    second = curator.approve(qid, cypher_template="RETURN 2", parameters=["p"])

    assert first.fewshot_was_new is True
    assert second.fewshot_was_new is False
    assert first.fewshot_id == second.fewshot_id

    detail = curator.get(qid)
    assert detail.fewshot_cypher == "RETURN 2"
    assert detail.fewshot_parameters == ["p"]

    # Only one FewShot exists on disk
    with clean_driver.session() as session:
        count = session.run(
            "MATCH (:Query {id: $id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "RETURN count(fs) AS n",
            id=qid,
        ).single()["n"]
    assert count == 1


def test_reject_persists_reason_and_status(clean_driver) -> None:
    embedder = _FakeEmbedder()
    retriever = BOOTHRetriever(driver=clean_driver, embedder=embedder)
    curator = BOOTHCurator(driver=clean_driver)

    response = retriever.query("something bogus")
    curator.reject(response.query_id, reason="nonsense question")

    detail = curator.get(response.query_id)
    assert detail.status == "rejected"
    assert detail.rejection_reason == "nonsense question"


def test_reject_raises_on_missing_query(clean_driver) -> None:
    curator = BOOTHCurator(driver=clean_driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.reject("does-not-exist")


def test_edit_fewshot_updates_template_without_touching_status(clean_driver) -> None:
    embedder = _FakeEmbedder({"q": _FakeEmbedder.MATCH_VECTOR})
    retriever = BOOTHRetriever(driver=clean_driver, embedder=embedder)
    curator = BOOTHCurator(driver=clean_driver)

    response = retriever.query("q")
    curator.approve(response.query_id, cypher_template="RETURN 1 AS v")

    curator.edit_fewshot(response.query_id, cypher_template="RETURN 99 AS v")

    detail = curator.get(response.query_id)
    assert detail.status == "approved"  # unchanged
    assert detail.fewshot_cypher == "RETURN 99 AS v"


def test_submit_feedback_updates_status(clean_driver) -> None:
    embedder = _FakeEmbedder({"q": _FakeEmbedder.MATCH_VECTOR})
    retriever = BOOTHRetriever(driver=clean_driver, embedder=embedder)
    curator = BOOTHCurator(driver=clean_driver)

    response = retriever.query("q")
    # Reject first so status != pending_approval, then exercise helpful feedback
    curator.reject(response.query_id, reason="placeholder")

    curator.submit_feedback(response.query_id, helpful=True)
    detail = curator.get(response.query_id)
    assert detail.status == "pending_approval"
    assert detail.user_feedback == "helpful"

    curator.submit_feedback(response.query_id, helpful=False)
    detail = curator.get(response.query_id)
    assert detail.status == "needs_review"
    assert detail.user_feedback == "not_helpful"


def test_stats_counts_queries_by_status(clean_driver) -> None:
    curator = BOOTHCurator(driver=clean_driver)
    with clean_driver.session() as session:
        for i, status in enumerate(
            ["approved", "approved", "pending_approval", "rejected"]
        ):
            session.run(
                "CREATE (q:Query {id: $id, text: 't', status: $status, "
                "timestamp: datetime()})",
                id=f"stats-q-{i}",
                status=status,
            )

    stats = curator.stats()
    assert stats.counts["approved"] == 2
    assert stats.counts["pending_approval"] == 1
    assert stats.counts["rejected"] == 1
    assert stats.total == 4
