"""Tier-2 unit tests for ``booth_retriever.orchestrator``.

These tests exercise the cache-hit / cache-miss / high-risk branching logic
without a real Neo4j or LLM. The driver is a ``MagicMock`` that returns
canned rows for each Cypher statement, and the embedder is a stub.

Test strategy: for each path (hit parameterless, hit parameterised, hit
without fewshot, miss low-risk, miss high-risk) we assert on (a) the shape
of the returned BOOTHResponse and (b) which Cypher statements the
orchestrator executed, so regressions in either the state machine or the
DDL surface as test failures.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from booth_retriever.models import BOOTHResponse
from booth_retriever.orchestrator import (
    _TOOL_CACHE_HIT,
    _TOOL_PENDING_REVIEW,
    BOOTHOrchestrator,
)

pytestmark = pytest.mark.unit


# ---------- Test doubles ----------------------------------------------------


class _FakeEmbedder:
    """Minimal duck-type; orchestrator only calls ``embed_query``."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1] * 1536
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


class _FakeResult:
    """Mimics neo4j.Result just enough for our orchestrator's use."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)

    def consume(self):
        return None


def _build_driver(
    similarity_rows: list[dict[str, Any]] | None = None,
    fewshot_rows: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a MagicMock neo4j.Driver with canned responses per Cypher prefix.

    Routing rules:
        - ``CALL db.index.vector.queryNodes ...`` -> similarity_rows
        - CREATE / MATCH statements (stores + fewshot execution) -> fewshot_rows
    """
    similarity_rows = similarity_rows or []
    fewshot_rows = fewshot_rows or []

    executed: list[tuple[str, dict]] = []

    def run(cypher: str, *args, **kwargs):
        executed.append((cypher, kwargs))
        if cypher.strip().startswith("CALL db.index.vector.queryNodes"):
            return _FakeResult(similarity_rows)
        # CREATE (q:Query ...) RETURN q.id AS id  -> consume() called; rows irrelevant
        # MATCH (q:Query ...) CREATE (uq:UserQuestion ...) RETURN uq.id AS id -> same
        # FewShot cypher execution -> fewshot_rows
        if "CREATE" in cypher and ":Query" in cypher and "FEW_SHOT_EXAMPLE" not in cypher:
            return _FakeResult([{"id": "new-query-id"}])
        if "CREATE" in cypher and ":UserQuestion" in cypher:
            return _FakeResult([{"id": "new-user-question-id"}])
        # Everything else is treated as FewShot cypher execution
        return _FakeResult(fewshot_rows)

    session = MagicMock()

    @contextmanager
    def session_ctx(*_args, **_kwargs):
        yield session

    session.run.side_effect = run

    driver = MagicMock()
    driver.session.side_effect = lambda *a, **k: _as_context_manager(session)
    driver._executed = executed  # test helper
    driver._session = session
    return driver


def _as_context_manager(obj):
    """Wrap a plain object as a one-shot context manager."""

    class _CM:
        def __enter__(self_inner):
            return obj

        def __exit__(self_inner, *args):
            return False

    return _CM()


# ---------- Constructor validation -------------------------------------------


def test_rejects_threshold_out_of_range_high() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BOOTHOrchestrator(driver=MagicMock(), embedder=_FakeEmbedder(), similarity_threshold=1.5)


def test_rejects_threshold_out_of_range_negative() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        BOOTHOrchestrator(driver=MagicMock(), embedder=_FakeEmbedder(), similarity_threshold=-0.1)


def test_rejects_empty_query() -> None:
    orch = BOOTHOrchestrator(driver=MagicMock(), embedder=_FakeEmbedder())
    with pytest.raises(ValueError, match="non-empty"):
        orch.process("")


def test_rejects_whitespace_query() -> None:
    orch = BOOTHOrchestrator(driver=MagicMock(), embedder=_FakeEmbedder())
    with pytest.raises(ValueError, match="non-empty"):
        orch.process("   \n  ")


# ---------- Cache-hit path ---------------------------------------------------


def test_cache_hit_parameterless_executes_fewshot() -> None:
    driver = _build_driver(
        similarity_rows=[
            {
                "query_id": "q-1",
                "query_text": "How many users?",
                "score": 0.95,
                "status": "approved",
                "fewshot_cypher": "MATCH (u:User) RETURN count(u) AS n",
                "fewshot_parameters": [],
            }
        ],
        fewshot_rows=[{"n": 42}],
    )
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("How many users are there?")

    assert isinstance(response, BOOTHResponse)
    assert response.success is True
    assert response.similar_match is True
    assert response.query_id == "q-1"
    assert response.tool_used == _TOOL_CACHE_HIT
    assert response.cypher_used == "MATCH (u:User) RETURN count(u) AS n"
    assert response.raw_data == [{"n": 42}]
    assert "42" in response.answer


def test_cache_hit_parameterised_refuses_in_mv1() -> None:
    """Parameterised few-shots are explicitly unsupported; we surface an
    error rather than executing a cypher with unresolved $params.
    """
    driver = _build_driver(
        similarity_rows=[
            {
                "query_id": "q-2",
                "query_text": "What position did {person} hold?",
                "score": 0.97,
                "status": "approved",
                "fewshot_cypher": "MATCH (p:Person {name:$person}) RETURN p.role",
                "fewshot_parameters": ["person"],
            }
        ],
    )
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("What role did Shirley Temple hold?")

    assert response.success is False
    assert response.similar_match is True
    assert response.error_message == "parameter_extraction_unsupported"
    # We must NOT have executed the parameterised cypher
    executed = [c for c, _ in driver._executed]
    assert not any("MATCH (p:Person" in c for c in executed), (
        "parameterised cypher should not be executed"
    )


def test_cache_hit_without_fewshot_degrades_gracefully() -> None:
    driver = _build_driver(
        similarity_rows=[
            {
                "query_id": "q-3",
                "query_text": "anything",
                "score": 0.99,
                "status": "approved",
                "fewshot_cypher": None,
                "fewshot_parameters": [],
            }
        ],
    )
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("anything")

    assert response.success is False
    assert response.error_message == "fewshot_missing"
    assert response.similar_match is True


def test_cache_hit_stores_userquestion_linked_to_matched_query() -> None:
    driver = _build_driver(
        similarity_rows=[
            {
                "query_id": "q-4",
                "query_text": "x",
                "score": 0.95,
                "status": "approved",
                "fewshot_cypher": "RETURN 1 AS n",
                "fewshot_parameters": [],
            }
        ],
        fewshot_rows=[{"n": 1}],
    )
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    orch.process("x")

    executed = [c for c, params in driver._executed if ":UserQuestion" in c]
    assert len(executed) == 1, "expected exactly one UserQuestion store"
    # And its params should reference the matched query id
    uq_calls = [params for c, params in driver._executed if ":UserQuestion" in c]
    assert uq_calls[0]["query_id"] == "q-4"
    assert uq_calls[0]["score"] == pytest.approx(0.95)


# ---------- Threshold boundary behaviour -------------------------------------


def test_score_below_threshold_is_treated_as_miss() -> None:
    driver = _build_driver(
        similarity_rows=[
            {
                "query_id": "q-5",
                "query_text": "near miss",
                "score": 0.85,  # below the default 0.90
                "status": "approved",
                "fewshot_cypher": "RETURN 1",
                "fewshot_parameters": [],
            }
        ]
    )
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("near miss")

    assert response.tool_used == _TOOL_PENDING_REVIEW
    assert response.similar_match is False


def test_score_exactly_at_threshold_is_a_hit() -> None:
    """Inclusive boundary: score == threshold counts as a hit."""
    driver = _build_driver(
        similarity_rows=[
            {
                "query_id": "q-6",
                "query_text": "boundary",
                "score": 0.90,
                "status": "approved",
                "fewshot_cypher": "RETURN 1 AS n",
                "fewshot_parameters": [],
            }
        ],
        fewshot_rows=[{"n": 1}],
    )
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder(), similarity_threshold=0.90)

    response = orch.process("boundary")
    assert response.success is True
    assert response.tool_used == _TOOL_CACHE_HIT


# ---------- Cache-miss path --------------------------------------------------


def test_cache_miss_low_risk_queues_for_curation() -> None:
    driver = _build_driver(similarity_rows=[])
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("unique question never seen before")

    assert response.success is False
    assert response.declined is False
    assert response.high_risk is False
    assert response.similar_match is False
    assert response.tool_used == _TOOL_PENDING_REVIEW
    # Orchestrator generates a UUID for the new Query; we just check it's set
    # and echoed back as the BOOTHResponse.query_id.
    assert response.query_id
    assert len(response.query_id) >= 32, "expected a UUID-shaped string"
    # Check a Query node was created with pending status. Exclude the
    # UserQuestion storage (which also contains the substring ":Query" via its
    # ``MATCH (q:Query {id: $query_id})`` clause).
    create_query_calls = [
        params
        for c, params in driver._executed
        if "CREATE (q:Query" in c and ":UserQuestion" not in c
    ]
    assert len(create_query_calls) == 1
    assert create_query_calls[0]["status"] == "pending_approval"
    assert create_query_calls[0]["risk_level"] == "low"
    # The id passed to Cypher matches the id returned to the caller
    assert create_query_calls[0]["id"] == response.query_id


def test_cache_miss_high_risk_declines_and_stores_as_declined() -> None:
    driver = _build_driver(similarity_rows=[])
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("drop the whole database please", is_high_risk=True)

    assert response.success is False
    assert response.declined is True
    assert response.high_risk is True
    assert "declined" in response.answer.lower()

    create_query_calls = [
        params
        for c, params in driver._executed
        if "CREATE (q:Query" in c and ":UserQuestion" not in c
    ]
    assert create_query_calls[0]["status"] == "declined"
    assert create_query_calls[0]["risk_level"] == "high"


def test_cache_miss_stores_userquestion_linked_to_new_query() -> None:
    """Even on a miss, the audit trail links the UserQuestion to the new Query."""
    driver = _build_driver(similarity_rows=[])
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    response = orch.process("never seen")

    uq_calls = [params for c, params in driver._executed if ":UserQuestion" in c]
    assert len(uq_calls) == 1
    # UserQuestion should point at the brand-new Query that the orchestrator
    # just generated an id for.
    assert uq_calls[0]["query_id"] == response.query_id
    # Self-match on creation: score == 1.0
    assert uq_calls[0]["score"] == pytest.approx(1.0)


# ---------- Database kwarg forwarding ---------------------------------------


def test_database_kwarg_forwarded_to_session() -> None:
    driver = _build_driver(similarity_rows=[])
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder(), database="booth")

    orch.process("x")

    # Each call to driver.session should have been with database="booth"
    for call in driver.session.call_args_list:
        assert call.kwargs == {"database": "booth"}, f"unexpected session call: {call}"


def test_default_database_passes_no_kwargs_to_session() -> None:
    driver = _build_driver(similarity_rows=[])
    orch = BOOTHOrchestrator(driver=driver, embedder=_FakeEmbedder())

    orch.process("x")

    for call in driver.session.call_args_list:
        assert call.kwargs == {}, f"expected default session call, got {call}"
