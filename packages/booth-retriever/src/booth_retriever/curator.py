"""``BOOTHCurator`` — Python API for BOOTH's curation workflow.

Exposes the operations the Train-AI page and ``booth curate ...`` CLI need:

    - ``list_pending(limit)``                -> queries awaiting attention
    - ``list_by_status(status, limit)``      -> queries of any one status
    - ``get(query_id)``                      -> full detail (+ FewShot)
    - ``stats()``                            -> counts keyed by status
    - ``approve(query_id, cypher_template=, parameters=)``
    - ``approve(query_id, refinement_agent=, raw_cypher=)`` (LLM-backed)
    - ``reject(query_id, reason=)``
    - ``edit_fewshot(query_id, cypher_template=, parameters=)``
    - ``submit_feedback(query_id, helpful=)``
    - ``delete(query_id)``                   -> permanently remove a query

Design:
    - Mutations are idempotent where the data model allows. Approving an
      already-approved query UPDATES the linked FewShot in place (doesn't
      duplicate it); subsequent approvals just overwrite the cypher.
    - ``approve()`` accepts either a hand-written ``cypher_template`` OR a
      ``refinement_agent`` conforming to ``RefinementProtocol``; the
      curator itself has no LLM dependency.
    - Missing-node detection uses the driver's write counters
      (``properties_set == 0``) rather than a separate SHOW query, so
      mutations remain single-statement where possible.
    - Module-level ``PENDING_STATUSES`` / ``ALL_STATUSES`` constants are
      the single source of truth for the status taxonomy; the CLI and
      tests depend on them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from .verification import verify_cypher

if TYPE_CHECKING:
    from neo4j import Driver


# ---------- Status taxonomy --------------------------------------------------

#: Statuses that the Train-AI page shows by default (anything "not done").
PENDING_STATUSES: tuple[str, ...] = (
    "pending_approval",
    "declined",
    "needs_review",
)

#: Every status a Query node may hold. The CLI validates ``--status`` against
#: this set; the data model treats anything else as unknown.
ALL_STATUSES: tuple[str, ...] = (
    "pending_approval",
    "approved",
    "rejected",
    "declined",
    "needs_review",
)


# ---------- Data classes -----------------------------------------------------


@dataclass
class PendingQuery:
    """Lightweight row for ``list_pending`` / ``list_by_status``."""

    query_id: str
    query_text: str
    status: str
    risk_level: str | None
    timestamp: str
    user_feedback: str | None = None
    has_fewshot: bool = False


@dataclass
class QueryDetail:
    """Full detail returned by ``get()``."""

    query_id: str
    query_text: str
    status: str
    risk_level: str | None
    timestamp: str
    user_feedback: str | None = None
    rejection_reason: str | None = None
    fewshot_cypher: str | None = None
    fewshot_parameters: list[str] = field(default_factory=list)


@dataclass
class CuratorStats:
    """Counts keyed by status, plus a derived total."""

    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass
class ApprovalResult:
    """Structured return from ``approve()`` / ``edit_fewshot()``."""

    query_id: str
    fewshot_id: str
    fewshot_was_new: bool


# ---------- Optional refinement hook ----------------------------------------


@dataclass
class RefinementResult:
    """Structured output of a refinement pass. Also used as the data shape
    returned by ``RefinementAgent.refine``.
    """

    success: bool
    refined_cypher: str | None = None
    parameters: list[str] = field(default_factory=list)
    category: str | None = None
    error: str | None = None


class RefinementProtocol(Protocol):
    """Minimal protocol for anything ``approve(refinement_agent=...)`` accepts."""

    def refine(
        self,
        *,
        original_question: str,
        raw_cypher: str | None = None,
    ) -> RefinementResult:
        ...


# ---------- Curator ---------------------------------------------------------


class BOOTHCurator:
    """Curation-workflow API over a Neo4j driver.

    Args:
        driver: An open ``neo4j.Driver``; caller retains ownership.
        database: Optional Neo4j database for multi-database setups.
    """

    def __init__(self, driver: Driver, *, database: str | None = None) -> None:
        self.driver = driver
        self.database = database

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_pending(self, *, limit: int = 50) -> list[PendingQuery]:
        """Return rows in any ``PENDING_STATUSES``, newest first."""
        _require_positive_limit(limit)
        return self._list(statuses=list(PENDING_STATUSES), limit=limit)

    def list_by_status(self, status: str, *, limit: int = 50) -> list[PendingQuery]:
        """Return rows of exactly one status. Raises on unknown statuses."""
        _require_positive_limit(limit)
        if status not in ALL_STATUSES:
            raise ValueError(
                f"Unknown status {status!r}. Valid: {sorted(ALL_STATUSES)}"
            )
        return self._list(statuses=[status], limit=limit)

    def _list(self, *, statuses: list[str], limit: int) -> list[PendingQuery]:
        cypher = (
            "MATCH (q:Query) "
            "WHERE q.status IN $statuses "
            "OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "RETURN q.id AS query_id, q.text AS query_text, "
            "q.status AS status, coalesce(q.risk_level, 'low') AS risk_level, "
            "toString(q.timestamp) AS timestamp, "
            "q.user_feedback AS user_feedback, "
            "(fs IS NOT NULL) AS has_fewshot "
            "ORDER BY q.timestamp DESC LIMIT $limit"
        )
        with self._session() as session:
            rows = list(session.run(cypher, statuses=statuses, limit=limit))
        return [
            PendingQuery(
                query_id=row["query_id"],
                query_text=row["query_text"],
                status=row["status"],
                risk_level=row["risk_level"],
                timestamp=row["timestamp"] or "",
                user_feedback=row["user_feedback"],
                has_fewshot=bool(row["has_fewshot"]),
            )
            for row in rows
        ]

    def get(self, query_id: str) -> QueryDetail | None:
        """Return the full detail of one query, or None if not found."""
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "RETURN q.id AS query_id, q.text AS query_text, "
            "q.status AS status, coalesce(q.risk_level, 'low') AS risk_level, "
            "toString(q.timestamp) AS timestamp, "
            "q.user_feedback AS user_feedback, "
            "q.rejection_reason AS rejection_reason, "
            "fs.cypher_template AS fewshot_cypher, "
            "coalesce(fs.parameters, []) AS fewshot_parameters"
        )
        with self._session() as session:
            row = session.run(cypher, query_id=query_id).single()
        if row is None:
            return None
        return QueryDetail(
            query_id=row["query_id"],
            query_text=row["query_text"],
            status=row["status"],
            risk_level=row["risk_level"],
            timestamp=row["timestamp"] or "",
            user_feedback=row["user_feedback"],
            rejection_reason=row["rejection_reason"],
            fewshot_cypher=row["fewshot_cypher"],
            fewshot_parameters=list(row["fewshot_parameters"] or []),
        )

    def stats(self) -> CuratorStats:
        """Return counts of Query nodes grouped by status."""
        cypher = (
            "MATCH (q:Query) RETURN q.status AS status, count(q) AS n"
        )
        with self._session() as session:
            rows = list(session.run(cypher))
        counts: dict[str, int] = {s: 0 for s in ALL_STATUSES}
        for row in rows:
            status = row["status"]
            if status is not None:
                counts[status] = counts.get(status, 0) + row["n"]
        return CuratorStats(counts=counts)

    # ------------------------------------------------------------------
    # Mutations: approve / edit
    # ------------------------------------------------------------------

    def approve(
        self,
        query_id: str,
        *,
        cypher_template: str | None = None,
        parameters: list[str] | None = None,
        category: str | None = None,
        refinement_agent: RefinementProtocol | None = None,
        raw_cypher: str | None = None,
        verify: bool = True,
    ) -> ApprovalResult:
        """Approve a Query and attach (or replace) its FewShot template.

        Provide exactly one of ``cypher_template`` or ``refinement_agent``.
        Approving an already-approved Query updates the linked FewShot in
        place rather than creating a duplicate.

        When ``verify=True`` (default), the Cypher template is run through
        ``verify_cypher`` first and rejected with ``ValueError`` if any
        rule fires. Pass ``verify=False`` to skip (useful when porting a
        known-good legacy template).
        """
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        if (cypher_template is None) == (refinement_agent is None):
            raise ValueError(
                "Provide exactly one of cypher_template or refinement_agent."
            )

        if refinement_agent is not None:
            original = self.get(query_id)
            if original is None:
                raise ValueError(f"No Query node with id {query_id!r}")
            refined = refinement_agent.refine(
                original_question=original.query_text,
                raw_cypher=raw_cypher,
            )
            if not refined.success or not refined.refined_cypher:
                raise RuntimeError(
                    f"Refinement agent failed: {refined.error or 'unknown error'}"
                )
            cypher_template = refined.refined_cypher
            parameters = refined.parameters
            category = refined.category or category

        if not cypher_template or not cypher_template.strip():
            raise ValueError("cypher_template must be a non-empty string")

        if verify:
            verification = verify_cypher(cypher_template)
            if not verification.is_valid:
                raise ValueError(
                    "cypher_template failed verification: "
                    + "; ".join(verification.errors)
                )

        parameters = list(parameters or [])
        now = _utcnow_iso()

        with self._session() as session:
            existence = session.run(
                "MATCH (q:Query {id: $query_id}) RETURN q.id AS id",
                query_id=query_id,
            )
            row = existence.single()
            if row is None:
                raise ValueError(f"No Query node with id {query_id!r}")

            existing = session.run(
                "MATCH (q:Query {id: $query_id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
                "RETURN fs.id AS fewshot_id",
                query_id=query_id,
            )
            existing_row = existing.single()

            if existing_row is None:
                fewshot_id = str(uuid.uuid4())
                fewshot_was_new = True
                session.run(
                    "MATCH (q:Query {id: $query_id}) "
                    "CREATE (fs:FewShot { "
                    "id: $fewshot_id, cypher_template: $cypher, "
                    "parameters: $params, category: $category, "
                    "created: datetime($timestamp) "
                    "}) "
                    "CREATE (q)-[:FEW_SHOT_EXAMPLE]->(fs)",
                    query_id=query_id,
                    fewshot_id=fewshot_id,
                    cypher=cypher_template,
                    params=parameters,
                    category=category,
                    timestamp=now,
                )
            else:
                fewshot_id = existing_row["fewshot_id"]
                fewshot_was_new = False
                session.run(
                    "MATCH (fs:FewShot {id: $fewshot_id}) "
                    "SET fs.cypher_template = $cypher, "
                    "fs.parameters = $params, "
                    "fs.category = coalesce($category, fs.category), "
                    "fs.updated = datetime($timestamp)",
                    fewshot_id=fewshot_id,
                    cypher=cypher_template,
                    params=parameters,
                    category=category,
                    timestamp=now,
                )

            session.run(
                "MATCH (q:Query {id: $query_id}) "
                "SET q.status = 'approved', q.approved_at = datetime($timestamp)",
                query_id=query_id,
                timestamp=now,
            )

        return ApprovalResult(
            query_id=query_id,
            fewshot_id=fewshot_id,
            fewshot_was_new=fewshot_was_new,
        )

    def edit_fewshot(
        self,
        query_id: str,
        *,
        cypher_template: str,
        parameters: list[str] | None = None,
        verify: bool = True,
    ) -> None:
        """Edit the Cypher on a Query's already-linked FewShot.

        Raises if the query has no linked FewShot yet (use ``approve``
        first). Unlike ``approve``, this does NOT change the query's
        status. ``verify`` defaults to True; see ``approve`` for details.
        """
        if not cypher_template or not cypher_template.strip():
            raise ValueError("cypher_template must be a non-empty string")

        if verify:
            verification = verify_cypher(cypher_template)
            if not verification.is_valid:
                raise ValueError(
                    "cypher_template failed verification: "
                    + "; ".join(verification.errors)
                )

        parameters = list(parameters or [])
        cypher = (
            "MATCH (q:Query {id: $query_id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "SET fs.cypher_template = $cypher, "
            "fs.parameters = $params, "
            "fs.updated = datetime($timestamp) "
            "RETURN fs.id AS id"
        )
        with self._session() as session:
            row = session.run(
                cypher,
                query_id=query_id,
                cypher=cypher_template,
                params=parameters,
                timestamp=_utcnow_iso(),
            ).single()
        if row is None:
            raise ValueError(
                f"Query {query_id!r} has no linked FewShot; approve it first."
            )

    # ------------------------------------------------------------------
    # Mutations: reject / feedback
    # ------------------------------------------------------------------

    def reject(self, query_id: str, *, reason: str | None = None) -> None:
        """Mark a query as rejected with an optional reason."""
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "SET q.status = 'rejected', "
            "q.rejection_reason = $reason, "
            "q.rejected_at = datetime($timestamp)"
        )
        with self._session() as session:
            summary = session.run(
                cypher,
                query_id=query_id,
                reason=reason,
                timestamp=_utcnow_iso(),
            ).consume()
        if summary.counters.properties_set == 0:
            raise ValueError(f"No Query node with id {query_id!r}")

    def submit_feedback(self, query_id: str, *, helpful: bool) -> None:
        """Record end-user feedback on a retriever response.

        Helpful feedback promotes the Query to ``pending_approval``; a
        thumbs-down moves it to ``needs_review`` so curators can inspect
        what went wrong.
        """
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        status = "pending_approval" if helpful else "needs_review"
        feedback = "helpful" if helpful else "not_helpful"
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "SET q.user_feedback = $feedback, "
            "q.status = $status, "
            "q.feedback_submitted_at = datetime($timestamp)"
        )
        with self._session() as session:
            summary = session.run(
                cypher,
                query_id=query_id,
                feedback=feedback,
                status=status,
                timestamp=_utcnow_iso(),
            ).consume()
        if summary.counters.properties_set == 0:
            raise ValueError(f"No Query node with id {query_id!r}")

    # ------------------------------------------------------------------
    # Mutations: delete
    # ------------------------------------------------------------------

    def delete(self, query_id: str) -> None:
        """Permanently delete a Query and any FewShot it owns.

        Used by the Curator UI's "Delete" affordance to retract a query
        outright (vs. ``reject``, which keeps the node around with a
        ``rejected`` status). Any linked ``FewShot`` is removed in the
        same statement: FewShot nodes are 1:1 owned by their Query in
        BOOTH's data model, so leaving an orphan would just become
        garbage in the few-shot index.

        Raises:
            ValueError: if no Query with this id exists.
        """
        if not query_id:
            raise ValueError("query_id must be a non-empty string")

        with self._session() as session:
            existence = session.run(
                "MATCH (q:Query {id: $query_id}) RETURN q.id AS id",
                query_id=query_id,
            )
            if existence.single() is None:
                raise ValueError(f"No Query node with id {query_id!r}")
            session.run(
                "MATCH (q:Query {id: $query_id}) "
                "OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
                "DETACH DELETE q, fs",
                query_id=query_id,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _session(self):
        if self.database is not None:
            return self.driver.session(database=self.database)
        return self.driver.session()


def _require_positive_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
