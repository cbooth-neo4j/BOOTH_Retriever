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
    - If there's no match and a ``fallback`` is configured, low-risk
      questions are answered by the fallback (typically a vanilla
      ``neo4j-graphrag`` ``HybridRetriever`` + ``GraphRAG``) while still
      creating a ``Query`` node with ``status = 'needs_review'`` so
      curators can later promote a popular fallback answer to a curated
      FewShot. Without a fallback (or if the fallback raises) the
      caller gets a polite queued-for-curation placeholder.
    - If the caller flags ``is_high_risk`` on a cache miss, the new
      ``Query`` is still stored as ``needs_review`` but with
      ``risk_level = 'high'`` and the caller gets a decline message — the
      HybridRetriever fallback is NEVER consulted for high-risk misses.
      A Text2Cypher attempt is recorded against the Query for curation.
    - Every run records a ``UserQuestion`` linked via ``SIMILAR`` to the
      matched-or-newly-created ``Query``, preserving the audit trail. Near-
      duplicate questions (>= 0.99 cosine) collapse onto one node whose
      ``count`` is incremented, rather than creating siblings.

Out of scope for MV1 (tracked as follow-up work):
    - Agentic Text2Cypher fallback on cache miss.
    - LLM-based parameter extraction for parameterised few-shots.
    - ``submit_user_feedback`` and curator reads (``BOOTHCurator``).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .models import BOOTHResponse, SimilarQueryMatch, Text2CypherAttempt
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
_TOOL_FALLBACK = "graphrag.HybridRetriever"

# Cosine similarity at/above which two UserQuestions are considered the "same"
# question and collapsed onto a single node (its ``count`` is incremented)
# instead of creating a near-duplicate. Keeps the provenance graph readable.
_USERQUESTION_DEDUPE_THRESHOLD = 0.99

# Cap on how many attempt rows we persist on the Response node. Declined
# attempts are for curator review, not bulk export, so a representative
# sample keeps the node small and the JSON serialisable.
_MAX_ATTEMPT_ROWS = 50

# Cap on how many rows we serialise into the summarisation prompt. FewShot
# cypher can in principle return thousands of rows, but the LLM only needs
# a representative sample to phrase the answer. The full ``raw_data`` is
# still returned on the ``BOOTHResponse`` for callers that need it.
_MAX_ROWS_FOR_LLM = 50

_SUMMARY_SYSTEM_PROMPT = """You are BOOTH's answer-refiner. You receive:

- a user question, and
- the JSON rows returned by a curated Cypher query that was chosen for that
  question.

Write a clear, direct answer to the question, grounded strictly in the
rows you were given. Rules:

1. Use ONLY facts present in the rows. Do not invent values.
2. If the rows are empty, say so plainly ("No matching records were found.").
3. Reference specific names, numbers, and dates from the rows where helpful.
4. Pick the format that best fits the question:
   - If the answer is a single fact, return one short sentence.
   - If the rows naturally group by a key (e.g. one row per document,
     project, person), use a short ``## Heading`` per group followed by
     a Markdown bullet list of facts. Highlight the most important
     value with ``**bold**``.
   - If the answer is a small list of items, use a Markdown bullet list.
   - Otherwise prose is fine — keep it tight, ideally under five
     sentences.
5. Do NOT wrap names in straight double quotes ("..."). Use the names as
   written in the data, optionally in **bold** for emphasis.
6. Markdown allowed: ``#``/``##``/``###`` headings, ``-`` bullets,
   ``**bold**``, ``*italic*``, and inline `` `code` ``. Do NOT use code
   fences (``` ``` ```), HTML tags, tables, or images.
7. Do not mention Cypher, the database, or how the answer was retrieved.
8. Do not repeat the same fact in both prose and a bullet — pick one.
"""

# Hard ceiling on the characters we send to the summariser. Approved FewShot
# cypher frequently RETURNs whole paths/nodes so the Ask-page popup can draw
# the answer subgraph; those objects are useless to the summariser and, left
# raw, can blow the model's context window (a handful of paths carrying node
# ``embedding`` arrays serialise to >1M chars). We compact graph objects to
# scalar props and then trim rows until the prompt fits comfortably.
_MAX_PROMPT_CHARS = 120_000
# Truncate individual string properties to keep the prompt compact.
_MAX_PROP_STR = 300


def _compact_neo4j_value(value: Any) -> Any:
    """Reduce Neo4j graph objects to compact, LLM-friendly summaries.

    Paths/Nodes/Relationships are returned by FewShot cypher for graph
    visualisation, not for the text answer. For summarisation we keep only the
    human-readable scalar properties (dropping ``embedding`` arrays, long text,
    and other list/dict props) so the prompt stays small. Plain values pass
    through unchanged (with long strings truncated).
    """
    try:
        from neo4j.graph import Node, Path, Relationship
    except Exception:  # pragma: no cover - neo4j is always present at runtime
        Node = Path = Relationship = ()  # type: ignore[assignment]

    if isinstance(value, Path):
        return {
            "nodes": [_compact_neo4j_value(node) for node in value.nodes],
            "relationships": [rel.type for rel in value.relationships],
        }
    if isinstance(value, Node):
        props: dict[str, Any] = {}
        for key, val in value.items():
            if key == "embedding":
                continue
            if val is None or isinstance(val, (str, int, float, bool)):
                if isinstance(val, str) and len(val) > _MAX_PROP_STR:
                    val = val[:_MAX_PROP_STR] + "…"
                props[key] = val
        return {"labels": list(value.labels), "properties": props}
    if isinstance(value, Relationship):
        return {"type": value.type}
    if isinstance(value, list):
        return [_compact_neo4j_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _compact_neo4j_value(val) for key, val in value.items()}
    if isinstance(value, str) and len(value) > _MAX_PROP_STR:
        return value[:_MAX_PROP_STR] + "…"
    return value


def _row_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Compact one result row for inclusion in the summarisation prompt."""
    return {key: _compact_neo4j_value(val) for key, val in row.items()}


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
        fallback: Optional ``Callable[[str], str]`` invoked on a low-risk
            cache miss. Its return value becomes
            ``BOOTHResponse.answer`` and ``tool_used`` is set to
            ``"graphrag.HybridRetriever"``. Typical wiring: a
            ``GraphRAG(retriever=HybridRetriever(...), llm=...)``
            wrapped so that ``fallback(query)`` returns
            ``rag.search(query).answer``. When ``None`` (the default)
            cache misses get the original queued-for-curation
            placeholder. High-risk misses NEVER consult the fallback.
        text2cypher: Optional ``Callable[[str], Text2CypherAttempt]``
            invoked on a high-risk (declined) cache miss. BOOTH still
            declines the answer to the user, but records the generated
            Cypher + its rows as a ``CypherAttempt``/``Response`` pair
            linked to the new Query so curators can review and promote it.
            When ``None`` (the default) declined misses record no attempt.
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
        fallback: Callable[[str], str] | None = None,
        text2cypher: Callable[[str], Text2CypherAttempt] | None = None,
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
        self.fallback = fallback
        self.text2cypher = text2cypher

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
            return self._handle_cache_hit(
                user_query, match, is_high_risk, embedding=embedding
            )

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
        *,
        embedding: list[float] | None = None,
    ) -> BOOTHResponse:
        """Execute a matched FewShot and return the result."""
        self._store_user_question(
            text=user_query,
            linked_query_id=match.query_id,
            similarity_score=match.score,
            risk_level="high" if is_high_risk else "low",
            embedding=embedding,
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

        rows_for_prompt = [_row_for_prompt(row) for row in rows[:_MAX_ROWS_FOR_LLM]]
        try:
            rows_json = json.dumps(rows_for_prompt, default=str, indent=2)
        except (TypeError, ValueError) as exc:
            _logger.warning(
                "Falling back to placeholder summary; rows not JSON-serialisable: %s",
                exc,
            )
            return self._format_answer_from_rows(rows)

        # Even after compaction a graph-heavy result can be large; halve the
        # row sample until the prompt is comfortably within the context window.
        while len(rows_json) > _MAX_PROMPT_CHARS and len(rows_for_prompt) > 1:
            rows_for_prompt = rows_for_prompt[: len(rows_for_prompt) // 2]
            rows_json = json.dumps(rows_for_prompt, default=str, indent=2)

        shown = len(rows_for_prompt)
        truncated_note = (
            f"\n\n(Showing {shown} of {len(rows)} rows.)"
            if len(rows) > shown
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
        """Persist a Query for curation and answer the user.

        Behaviour by branch:
          * High-risk: the new Query is stored as ``needs_review`` (with
            ``risk_level = 'high'``) and the caller gets a decline message.
            The HybridRetriever fallback is never invoked, but a
            Text2Cypher attempt is recorded for curation (see
            ``_run_text2cypher_attempt``).
          * Low-risk + ``self.fallback`` set: store as ``needs_review``
            and return the fallback's answer (typically a vanilla
            ``neo4j-graphrag`` ``HybridRetriever`` + ``GraphRAG``). The
            curator queue still gets a row so a popular fallback answer
            can later be promoted to a curated FewShot.
          * Low-risk, no fallback (or fallback raises): the original
            queued-for-curation placeholder is returned.

        Every miss now lands in the single ``needs_review`` bucket;
        high-risk declines stay distinguishable via ``risk_level`` rather
        than a separate status.
        """
        status = "needs_review"
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
            embedding=embedding,
        )

        if is_high_risk:
            # The user is declined, but we still run a Text2Cypher attempt
            # behind the scenes and persist it for curation.
            self._record_text2cypher_attempt(
                query_id=new_query_id, user_query=user_query
            )
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

        if self.fallback is not None:
            try:
                fallback_answer = self.fallback(user_query)
            except Exception as exc:  # noqa: BLE001 - external retriever can fail
                _logger.warning(
                    "Fallback retriever failed (%s: %s); returning "
                    "queued-for-curation placeholder.",
                    type(exc).__name__,
                    exc,
                )
            else:
                answer = (fallback_answer or "").strip()
                if answer:
                    return BOOTHResponse(
                        success=True,
                        answer=answer,
                        query_id=new_query_id,
                        similar_match=False,
                        tool_used=_TOOL_FALLBACK,
                        pending_feedback=False,
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
        embedding: list[float] | None = None,
    ) -> str:
        """Record a UserQuestion linked via SIMILAR to its canonical Query.

        To keep the provenance graph readable we collapse near-duplicate
        questions: if an existing UserQuestion already linked to this Query
        has an embedding within ``_USERQUESTION_DEDUPE_THRESHOLD`` cosine of
        the incoming one, we bump that node's ``count`` (and refresh its
        ``last_seen`` / ``risk_level``) instead of creating a new node. The
        whole decision is done in a single Cypher statement so concurrent
        asks don't race to create siblings. The embedding is stored on the
        node to make this comparison possible on future asks.

        Returns the id of the node the question was attributed to (existing
        on a dedupe hit, freshly created otherwise).
        """
        new_id = str(uuid.uuid4())
        now = _utcnow_iso()
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            # Best existing duplicate (if any) by cosine over stored embeddings.
            "OPTIONAL MATCH (q)<-[:SIMILAR]-(ex:UserQuestion) "
            "WHERE ex.embedding IS NOT NULL AND $embedding IS NOT NULL "
            "AND vector.similarity.cosine(ex.embedding, $embedding) >= $threshold "
            "WITH q, ex, CASE WHEN ex IS NULL THEN 0.0 "
            "ELSE vector.similarity.cosine(ex.embedding, $embedding) END AS sim "
            "ORDER BY sim DESC LIMIT 1 "
            # Dedupe hit: increment the existing node's counters.
            "FOREACH (_ IN CASE WHEN ex IS NOT NULL THEN [1] ELSE [] END | "
            "  SET ex.count = coalesce(ex.count, 1) + 1, "
            "      ex.last_seen = datetime($timestamp), "
            "      ex.risk_level = $risk_level) "
            # Otherwise create a fresh UserQuestion + SIMILAR edge.
            "FOREACH (_ IN CASE WHEN ex IS NULL THEN [1] ELSE [] END | "
            "  CREATE (uq:UserQuestion { "
            "    id: $id, text: $text, timestamp: datetime($timestamp), "
            "    last_seen: datetime($timestamp), risk_level: $risk_level, "
            "    count: 1, embedding: $embedding }) "
            "  CREATE (uq)-[:SIMILAR {score: $score}]->(q)) "
            "RETURN coalesce(ex.id, $id) AS id"
        )
        with self._session() as session:
            record = session.run(
                cypher,
                id=new_id,
                text=text,
                timestamp=now,
                risk_level=risk_level,
                query_id=linked_query_id,
                score=similarity_score,
                embedding=embedding,
                threshold=_USERQUESTION_DEDUPE_THRESHOLD,
            ).single()
        return record["id"] if record and record["id"] else new_id

    # ------------------------------------------------------------------
    # Internal: Text2Cypher attempt on declined queries
    # ------------------------------------------------------------------

    def _record_text2cypher_attempt(
        self,
        *,
        query_id: str,
        user_query: str,
    ) -> None:
        """Run the injected Text2Cypher agent and persist its attempt.

        Soft-failure by design: no agent, an agent that raises, or a
        storage hiccup must never turn a (correct) decline into a 500.
        Persists ``(q)-[:HAS_ATTEMPT]->(:CypherAttempt)-[:PRODUCED]->(:Response)``.
        """
        if self.text2cypher is None:
            return
        try:
            attempt = self.text2cypher(user_query)
        except Exception as exc:  # noqa: BLE001 - external agent can fail many ways
            _logger.warning(
                "Text2Cypher attempt raised (%s: %s); recording error only.",
                type(exc).__name__,
                exc,
            )
            attempt = Text2CypherAttempt(error=f"{type(exc).__name__}: {exc}")

        rows = attempt.rows or []
        rows_for_store = rows[:_MAX_ATTEMPT_ROWS]
        try:
            rows_json = json.dumps(rows_for_store, default=str)
        except (TypeError, ValueError):
            rows_json = json.dumps(
                [str(row) for row in rows_for_store], default=str
            )

        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "CREATE (ca:CypherAttempt { "
            "id: $attempt_id, cypher: $attempt_cypher, error: $attempt_error, "
            "created: datetime($timestamp) "
            "}) "
            "CREATE (q)-[:HAS_ATTEMPT]->(ca) "
            "CREATE (r:Response { "
            "id: $response_id, rows_json: $rows_json, row_count: $row_count, "
            "created: datetime($timestamp) "
            "}) "
            "CREATE (ca)-[:PRODUCED]->(r)"
        )
        try:
            with self._session() as session:
                session.run(
                    cypher,
                    query_id=query_id,
                    attempt_id=str(uuid.uuid4()),
                    attempt_cypher=attempt.cypher,
                    attempt_error=attempt.error,
                    response_id=str(uuid.uuid4()),
                    rows_json=rows_json,
                    row_count=len(rows),
                    timestamp=_utcnow_iso(),
                ).consume()
        except Exception as exc:  # noqa: BLE001 - storage must not break the decline
            _logger.warning(
                "Failed to persist Text2Cypher attempt for %s (%s: %s).",
                query_id,
                type(exc).__name__,
                exc,
            )

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
