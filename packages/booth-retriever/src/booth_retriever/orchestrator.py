"""Core BOOTH retrieval logic.

This module implements the similarity-cache-plus-storage flow that backs
``BOOTHRetriever``. It has no direct dependency on ``neo4j_graphrag``;
the retriever wrapper adapts between this orchestrator's Python API and
the ``neo4j_graphrag.retrievers.base.Retriever`` contract.

Keeping the two layers separate lets us unit-test the orchestrator with a
mocked ``neo4j.Driver`` and a mocked embedder, without pulling in the full
``neo4j_graphrag`` stack.

MV1 scope (what this currently does):
    - Embed the user query via the injected embedder.
    - Run a vector-index similarity search over approved ``Query`` nodes.
    - If the top match is above the threshold, execute the linked
      ``FewShot`` cypher. When an ``llm`` is configured, the raw rows
      are passed to it together with the original question to produce a
      natural-language answer; otherwise we fall back to a minimal
      stringified summary of the rows. Parameter extraction is NOT yet
      implemented; only parameter-free few-shots succeed. Parameterised
      ones surface an explicit "not yet implemented" error without
      executing anything.
    - If there's no match, create a ``Query`` node with ``status =
      'pending_approval'`` and return a placeholder response indicating
      the question has been queued for curation.
    - If the caller flags ``is_high_risk`` on a cache miss, the new
      ``Query`` is created with ``status = 'declined'`` and the caller
      gets a decline message.
    - Every run creates a ``UserQuestion`` node linked via ``SIMILAR`` to
      the matched-or-newly-created ``Query``, preserving the audit trail.

Out of scope for MV1 (tracked as follow-up work):
    - Agentic Text2Cypher fallback on cache miss.
    - LLM-based parameter extraction for parameterised few-shots.
    - ``submit_user_feedback`` and curator reads (``BOOTHCurator``).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .models import BOOTHResponse, SimilarQueryMatch
from .schema import VECTOR_INDEX_NAME

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.embeddings import Embedder
    from neo4j_graphrag.llm.base import LLMInterface


_logger = logging.getLogger(__name__)


# Default number of similarity-search candidates to fetch. We only need the
# top one, but asking for a few lets us log/report near-misses in future.
_VECTOR_TOP_K = 5

_TOOL_CACHE_HIT = "fewshot_cache"
_TOOL_PENDING_REVIEW = "pending_review"

# Cap on how many rows we serialise into the summarisation prompt. FewShot
# cypher can in principle return thousands of rows, but the LLM only needs
# a representative sample to phrase the answer. The full ``raw_data`` is
# still returned on the ``BOOTHResponse`` for callers that need it.
_MAX_ROWS_FOR_LLM = 50

_SUMMARY_SYSTEM_PROMPT = """You are BOOTH's answer-refiner. You receive:

- a user question, and
- the JSON rows returned by a curated Cypher query that was chosen for that
  question.

Write a short, direct natural-language answer to the question, grounded
strictly in the rows you were given. Rules:

1. Use ONLY facts present in the rows. Do not invent values.
2. If the rows are empty, say so plainly ("No matching records were found.").
3. Quote specific names, numbers, and dates from the rows where helpful.
4. Keep it under three sentences unless the data genuinely warrants more.
5. Do not mention Cypher, the database, or how the answer was retrieved.
6. Output prose only — no JSON, no markdown fences, no lists unless the
   question asks for one.
"""


class BOOTHOrchestrator:
    """Internal core of BOOTH. Callers should use ``BOOTHRetriever`` instead.

    Args:
        driver: An open ``neo4j.Driver``.
        embedder: Any embedder exposing ``embed_query(text) -> list[float]``.
            Typically a ``neo4j_graphrag.embeddings.Embedder`` subclass, but
            any duck-typed object works (helpful for unit tests).
        similarity_threshold: Cosine similarity in [0, 1]. Scores at or
            above this trigger the cache-hit path. Defaults to 0.90.
        vector_index_name: Name of the vector index to query. Defaults to
            the one created by ``init_schema``.
        database: Optional Neo4j database name for multi-database setups.
        llm: Optional ``neo4j_graphrag.llm.LLMInterface`` used to refine
            the raw rows returned by an approved FewShot Cypher into a
            natural-language answer. When ``None`` (the default) we fall
            back to a minimal stringified summary of the rows, matching
            the original MV1 behaviour. The ``raw_data`` field on the
            response is unchanged either way.
    """

    def __init__(
        self,
        driver: Driver,
        embedder: Embedder,
        *,
        similarity_threshold: float = 0.90,
        vector_index_name: str = VECTOR_INDEX_NAME,
        database: str | None = None,
        llm: LLMInterface | None = None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold!r}"
            )
        self.driver = driver
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.vector_index_name = vector_index_name
        self.database = database
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        user_query: str,
        *,
        is_high_risk: bool = False,
    ) -> BOOTHResponse:
        """Run the full BOOTH flow for one user question.

        Returns a ``BOOTHResponse``. Does not raise on expected failure
        modes (empty result set, parameterised few-shot, declined
        high-risk query); those surface as fields on the response. Only
        infrastructure problems (driver unreachable, embedder blowing up)
        propagate as exceptions.
        """
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string")

        embedding = self.embedder.embed_query(user_query)
        match = self._search_similar_query(embedding)

        if match is not None and match.score >= self.similarity_threshold:
            return self._handle_cache_hit(user_query, match, is_high_risk)

        return self._handle_cache_miss(
            user_query=user_query,
            embedding=embedding,
            is_high_risk=is_high_risk,
            best_score=match.score if match else 0.0,
        )

    # ------------------------------------------------------------------
    # Internal: similarity lookup
    # ------------------------------------------------------------------

    def _search_similar_query(
        self,
        embedding: list[float],
    ) -> SimilarQueryMatch | None:
        """Return the best-matching approved Query, or None if nothing approved."""
        cypher = (
            "CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) "
            "YIELD node, score "
            "WHERE node.status = 'approved' "
            "OPTIONAL MATCH (node)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "RETURN node.id AS query_id, node.text AS query_text, score, "
            "node.status AS status, fs.cypher_template AS fewshot_cypher, "
            "coalesce(fs.parameters, []) AS fewshot_parameters "
            "ORDER BY score DESC LIMIT 1"
        )
        with self._session() as session:
            row = session.run(
                cypher,
                index_name=self.vector_index_name,
                top_k=_VECTOR_TOP_K,
                embedding=embedding,
            ).single()
        if row is None:
            return None
        return SimilarQueryMatch(
            query_id=row["query_id"],
            query_text=row["query_text"],
            score=float(row["score"]),
            status=row["status"],
            fewshot_cypher=row["fewshot_cypher"],
            fewshot_parameters=list(row["fewshot_parameters"] or []),
        )

    # ------------------------------------------------------------------
    # Internal: cache-hit path
    # ------------------------------------------------------------------

    def _handle_cache_hit(
        self,
        user_query: str,
        match: SimilarQueryMatch,
        is_high_risk: bool,
    ) -> BOOTHResponse:
        """Execute a matched FewShot and return the result."""
        self._store_user_question(
            text=user_query,
            linked_query_id=match.query_id,
            similarity_score=match.score,
            risk_level="high" if is_high_risk else "low",
        )

        if not match.fewshot_cypher:
            # Approved Query with no linked FewShot: shouldn't normally happen
            # but we degrade gracefully.
            return BOOTHResponse(
                success=False,
                answer=(
                    "A similar approved query was found, but it has no linked "
                    "FewShot cypher. Ask a curator to check the database."
                ),
                query_id=match.query_id,
                similar_match=True,
                tool_used=_TOOL_CACHE_HIT,
                error_message="fewshot_missing",
            )

        if match.fewshot_parameters:
            return BOOTHResponse(
                success=False,
                answer=(
                    "A matching parameterised few-shot was found, but automatic "
                    "parameter extraction is not yet supported in this version "
                    "of booth-retriever. Tracked as follow-up work."
                ),
                query_id=match.query_id,
                similar_match=True,
                cypher_used=match.fewshot_cypher,
                tool_used=_TOOL_CACHE_HIT,
                error_message="parameter_extraction_unsupported",
            )

        raw_data = self._execute_fewshot(match.fewshot_cypher)
        answer = self._summarise_rows(user_query=user_query, rows=raw_data)

        return BOOTHResponse(
            success=True,
            answer=answer,
            query_id=match.query_id,
            similar_match=True,
            high_risk=is_high_risk,
            cypher_used=match.fewshot_cypher,
            raw_data=raw_data,
            tool_used=_TOOL_CACHE_HIT,
        )

    def _execute_fewshot(self, cypher: str) -> list[dict[str, Any]]:
        """Execute an approved FewShot Cypher query and return its rows."""
        with self._session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]

    def _summarise_rows(
        self,
        *,
        user_query: str,
        rows: list[dict[str, Any]],
    ) -> str:
        """Turn raw FewShot rows into a natural-language answer.

        When ``self.llm`` is configured we hand the LLM the original
        question plus the rows and let it phrase the answer. If the LLM
        call fails we degrade silently to ``_format_answer_from_rows`` so
        a flaky model never breaks an otherwise-successful retrieval.
        """
        if self.llm is None:
            return self._format_answer_from_rows(rows)

        rows_for_prompt = rows[:_MAX_ROWS_FOR_LLM]
        try:
            rows_json = json.dumps(rows_for_prompt, default=str, indent=2)
        except (TypeError, ValueError) as exc:
            _logger.warning(
                "Falling back to placeholder summary; rows not JSON-serialisable: %s",
                exc,
            )
            return self._format_answer_from_rows(rows)

        truncated_note = (
            f"\n\n(Showing the first {_MAX_ROWS_FOR_LLM} of {len(rows)} rows.)"
            if len(rows) > _MAX_ROWS_FOR_LLM
            else ""
        )
        prompt = (
            f"User question:\n{user_query}\n\n"
            f"Rows returned by the approved Cypher (JSON):\n{rows_json}"
            f"{truncated_note}"
        )

        try:
            response = self.llm.invoke(
                input=prompt,
                system_instruction=_SUMMARY_SYSTEM_PROMPT,
            )
        except Exception as exc:  # noqa: BLE001 - external LLM can fail many ways
            _logger.warning(
                "LLM summarisation failed (%s: %s); falling back to placeholder.",
                type(exc).__name__,
                exc,
            )
            return self._format_answer_from_rows(rows)

        text = getattr(response, "content", None)
        if text is None:
            text = str(response)
        text = text.strip()
        if not text:
            return self._format_answer_from_rows(rows)
        return text

    def _format_answer_from_rows(self, rows: list[dict[str, Any]]) -> str:
        """Minimal stringified summary used when no ``llm`` is configured.

        Single-row/single-column gets the value, otherwise a row-count
        summary. Also used as a graceful fallback when LLM summarisation
        raises.
        """
        if not rows:
            return "The approved query ran successfully but returned no rows."
        if len(rows) == 1 and len(rows[0]) == 1:
            (value,) = rows[0].values()
            return str(value)
        return f"Returned {len(rows)} row(s)."

    # ------------------------------------------------------------------
    # Internal: cache-miss path
    # ------------------------------------------------------------------

    def _handle_cache_miss(
        self,
        *,
        user_query: str,
        embedding: list[float],
        is_high_risk: bool,
        best_score: float,
    ) -> BOOTHResponse:
        """Create a Query node for curation and return a placeholder response.

        For MV1 we do NOT invoke the agent fallback. Customers see a polite
        queued-for-curation message and the query_id they can reference.
        """
        status = "declined" if is_high_risk else "pending_approval"
        new_query_id = self._store_new_query(
            text=user_query,
            embedding=embedding,
            risk_level="high" if is_high_risk else "low",
            status=status,
            best_similarity=best_score,
        )
        self._store_user_question(
            text=user_query,
            linked_query_id=new_query_id,
            similarity_score=1.0,
            risk_level="high" if is_high_risk else "low",
        )

        if is_high_risk:
            return BOOTHResponse(
                success=False,
                declined=True,
                high_risk=True,
                answer=(
                    "This query was flagged as high-risk and has been declined. "
                    "A curator will review it; no results are shown."
                ),
                query_id=new_query_id,
                tool_used=_TOOL_PENDING_REVIEW,
            )

        return BOOTHResponse(
            success=False,
            answer=(
                "This question does not match any approved query yet and has "
                "been queued for curation. Run `booth curate list` (or use "
                "BOOTHCurator) to review and approve."
            ),
            query_id=new_query_id,
            similar_match=False,
            tool_used=_TOOL_PENDING_REVIEW,
            pending_feedback=False,
        )

    # ------------------------------------------------------------------
    # Internal: storage
    # ------------------------------------------------------------------

    def _store_new_query(
        self,
        *,
        text: str,
        embedding: list[float],
        risk_level: str,
        status: str,
        best_similarity: float,
    ) -> str:
        """Create a new Query node for curation and return its id."""
        new_id = str(uuid.uuid4())
        cypher = (
            "CREATE (q:Query { "
            "id: $id, text: $text, embedding: $embedding, "
            "status: $status, risk_level: $risk_level, "
            "timestamp: datetime($timestamp), similarity_matched: false, "
            "best_similarity_on_creation: $best_similarity "
            "}) "
            "RETURN q.id AS id"
        )
        with self._session() as session:
            session.run(
                cypher,
                id=new_id,
                text=text,
                embedding=embedding,
                status=status,
                risk_level=risk_level,
                timestamp=_utcnow_iso(),
                best_similarity=best_similarity,
            ).consume()
        return new_id

    def _store_user_question(
        self,
        *,
        text: str,
        linked_query_id: str,
        similarity_score: float,
        risk_level: str,
    ) -> str:
        """Create a UserQuestion node linked via SIMILAR to its canonical Query."""
        new_id = str(uuid.uuid4())
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "CREATE (uq:UserQuestion { "
            "id: $id, text: $text, timestamp: datetime($timestamp), "
            "risk_level: $risk_level "
            "}) "
            "CREATE (uq)-[:SIMILAR {score: $score}]->(q) "
            "RETURN uq.id AS id"
        )
        with self._session() as session:
            session.run(
                cypher,
                id=new_id,
                text=text,
                timestamp=_utcnow_iso(),
                risk_level=risk_level,
                query_id=linked_query_id,
                score=similarity_score,
            ).consume()
        return new_id

    # ------------------------------------------------------------------
    # Internal: session helper
    # ------------------------------------------------------------------

    def _session(self):
        if self.database is not None:
            return self.driver.session(database=self.database)
        return self.driver.session()


def _utcnow_iso() -> str:
    """Tiny helper; kept module-level so tests can monkeypatch it if needed."""
    return datetime.now(tz=timezone.utc).isoformat()
