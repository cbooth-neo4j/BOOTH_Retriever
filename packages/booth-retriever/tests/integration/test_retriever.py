"""Tier-3 integration test for ``BOOTHRetriever`` end-to-end.

Spins up a real Neo4j via testcontainers, seeds an approved Query with a
FewShot template, and verifies the full flow:

    embed -> vector search -> cache hit -> execute fewshot -> return rows

And the miss path:

    embed -> vector search -> no hit -> store Query + UserQuestion

Uses a deterministic ``_FakeEmbedder`` so the test doesn't need a real
embedding model but the vector index still does its work normally.
"""

from __future__ import annotations

import pytest

from booth_retriever import BOOTHRetriever, init_schema
from booth_retriever.orchestrator import _TOOL_CACHE_HIT, _TOOL_PENDING_REVIEW

pytestmark = pytest.mark.integration


# ---------- Fixtures --------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder that maps known questions to known vectors.

    The fixture seeds one approved Query with ``APPROVED_VECTOR``. Any input
    text mapped to that vector hits the cache; anything else misses.
    """

    APPROVED_VECTOR = [1.0, 0.0] + [0.0] * 1534
    ORTHOGONAL_VECTOR = [0.0, 1.0] + [0.0] * 1534

    def __init__(self, mapping: dict[str, list[float]] | None = None) -> None:
        self.mapping = mapping or {}

    def embed_query(self, text: str) -> list[float]:
        return self.mapping.get(text, self.ORTHOGONAL_VECTOR)


@pytest.fixture
def seeded_driver(neo4j_driver):
    """Initialise schema and seed an approved Query + FewShot."""
    init_schema(neo4j_driver, embedding_dimensions=1536)

    # Wipe any leftover BOOTH-specific data from prior tests. Leave the
    # schema indexes/constraints in place.
    with neo4j_driver.session() as session:
        session.run(
            "MATCH (n) WHERE n:Query OR n:UserQuestion OR n:FewShot "
            "OR n:CypherAttempt OR n:Response DETACH DELETE n"
        )
        session.run(
            "CREATE (q:Query { "
            "id: 'seed-q-1', text: 'count users', "
            "embedding: $embedding, status: 'approved', "
            "timestamp: datetime(), risk_level: 'low' "
            "}) "
            "CREATE (fs:FewShot { "
            "id: 'seed-fs-1', "
            "cypher_template: 'RETURN 42 AS n', "
            "parameters: [] "
            "}) "
            "CREATE (q)-[:FEW_SHOT_EXAMPLE]->(fs)",
            embedding=_FakeEmbedder.APPROVED_VECTOR,
        )
    yield neo4j_driver


# ---------- Tests -----------------------------------------------------------


def test_query_cache_hit_against_real_neo4j(seeded_driver) -> None:
    """A question whose embedding matches the seeded Query executes its FewShot."""
    embedder = _FakeEmbedder({"count users please": _FakeEmbedder.APPROVED_VECTOR})
    retriever = BOOTHRetriever(
        driver=seeded_driver,
        embedder=embedder,
        similarity_threshold=0.90,
    )

    response = retriever.query("count users please")

    assert response.success is True
    assert response.similar_match is True
    assert response.tool_used == _TOOL_CACHE_HIT
    assert response.query_id == "seed-q-1"
    assert response.cypher_used == "RETURN 42 AS n"
    assert response.raw_data == [{"n": 42}]
    assert "42" in response.answer

    # UserQuestion should now exist and be linked to the seeded Query
    with seeded_driver.session() as session:
        rows = list(
            session.run(
                "MATCH (uq:UserQuestion)-[:SIMILAR]->(q:Query {id: 'seed-q-1'}) "
                "RETURN uq.text AS text"
            )
        )
    assert len(rows) == 1
    assert rows[0]["text"] == "count users please"


def test_query_cache_miss_queues_for_curation(seeded_driver) -> None:
    """A question whose embedding does NOT match the seed is stored as pending."""
    embedder = _FakeEmbedder()  # everything maps to ORTHOGONAL_VECTOR
    retriever = BOOTHRetriever(
        driver=seeded_driver,
        embedder=embedder,
        similarity_threshold=0.90,
    )

    response = retriever.query("something totally different")

    assert response.success is False
    assert response.similar_match is False
    assert response.declined is False
    assert response.tool_used == _TOOL_PENDING_REVIEW
    assert response.query_id  # a new id

    with seeded_driver.session() as session:
        rows = list(
            session.run(
                "MATCH (q:Query {id: $id}) "
                "RETURN q.status AS status, q.risk_level AS risk",
                id=response.query_id,
            )
        )
    assert rows[0]["status"] == "pending_approval"
    assert rows[0]["risk"] == "low"


def test_high_risk_cache_miss_is_declined(seeded_driver) -> None:
    embedder = _FakeEmbedder()
    retriever = BOOTHRetriever(driver=seeded_driver, embedder=embedder)

    response = retriever.query("drop everything", is_high_risk=True)

    assert response.declined is True
    assert response.high_risk is True
    assert response.tool_used == _TOOL_PENDING_REVIEW

    with seeded_driver.session() as session:
        rows = list(
            session.run(
                "MATCH (q:Query {id: $id}) RETURN q.status AS status",
                id=response.query_id,
            )
        )
    assert rows[0]["status"] == "declined"


def test_search_returns_retriever_result_shape(seeded_driver) -> None:
    """The spec-compliant ``search()`` method returns a RetrieverResult with
    BOOTH-specific fields tucked into ``metadata``.
    """
    from neo4j_graphrag.types import RetrieverResult

    embedder = _FakeEmbedder({"count users please": _FakeEmbedder.APPROVED_VECTOR})
    retriever = BOOTHRetriever(driver=seeded_driver, embedder=embedder)

    result = retriever.search(query_text="count users please")

    assert isinstance(result, RetrieverResult)
    assert result.metadata["query_id"] == "seed-q-1"
    assert result.metadata["similar_match"] is True
    assert result.metadata["success"] is True
    assert len(result.items) == 1
    assert "42" in result.items[0].content
