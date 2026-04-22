"""``BOOTHCurator`` — the Python API for query curation.

Exposes every operation the ``booth curate`` CLI needs, and everything a
custom curation UI would need to build on top of BOOTH:

    - ``list_pending`` / ``list_by_status`` — paginated reads
    - ``get`` — fetch one query's full detail including any linked FewShot
    - ``approve`` — mark a query as approved and attach (or replace) a
      FewShot cypher template; the retriever will start executing it on
      future cache hits
    - ``reject`` — mark a query as rejected with an optional reason
    - ``edit_fewshot`` — update the cypher template on an already-approved
      query without re-running the approval flow
    - ``submit_feedback`` — record end-user helpful / not-helpful votes;
      helpful votes promote a query to ``pending_approval`` so a human
      curator sees it
    - ``stats`` — counts-by-status for dashboards

Scope notes for MV1:
    - approve() takes an explicit ``cypher_template`` argument; the
      LLM-backed RefinementAgent that auto-generates templates is a
      follow-up. This keeps the curator stand-alone and testable without
      an LLM dependency.
    - FewShot nodes are created with a predictable id so calls are
      idempotent: re-approving a query replaces the existing template
      rather than duplicating it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver


# Statuses we treat as "awaiting a human decision" by default. Order matters
# for some call sites (``pending_approval`` first).
PENDING_STATUSES = (
    "pending_approval",
    "declined",
    "needs_review",
)

ALL_STATUSES = (
    "pending_approval",
    "approved",
    "rejected",
    "declined",
    "needs_review",
)


@dataclass
class PendingQuery:
    """A row from ``list_pending``/``list_by_status``. Summary-only."""

    query_id: str
    query_text: str
    status: str
    risk_level: str | None
    timestamp: str | None
    user_feedback: str | None = None
    has_fewshot: bool = False


@dataclass
class QueryDetail:
    """Full detail of one query, including any linked FewShot template."""

    query_id: str
    query_text: str
    status: str
    risk_level: str | None
    timestamp: str | None
    user_feedback: str | None = None
    rejection_reason: str | None = None
    fewshot_cypher: str | None = None
    fewshot_parameters: list[str] = field(default_factory=list)


@dataclass
class ApprovalResult:
    """Outcome of ``approve()``; distinguishes new vs. updated fewshots."""

    query_id: str
    fewshot_id: str
    fewshot_was_new: bool


@dataclass
class CuratorStats:
    """Counts of queries by status; keys are status strings."""

    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class BOOTHCurator:
    """Curation operations for BOOTH's query corpus.

    Args:
        driver: An open ``neo4j.Driver``.
        database: Optional Neo4j database name for multi-database setups.
    """

    def __init__(self, driver: Driver, *, database: str | None = None) -> None:
        self.driver = driver
        self.database = database

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def list_pending(self, *, limit: int = 50) -> list[PendingQuery]:
        """List queries awaiting human review.

        Includes ``pending_approval``, ``declined``, and ``needs_review``.
        """
        return self._list_by_statuses(PENDING_STATUSES, limit=limit)

    def list_by_status(self, status: str, *, limit: int = 50) -> list[PendingQuery]:
        """List queries with a given status."""
        self._validate_status(status)
        return self._list_by_statuses((status,), limit=limit)

    def get(self, query_id: str) -> QueryDetail | None:
        """Fetch full detail for a single query, or ``None`` if missing."""
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "RETURN q.id AS query_id, q.text AS query_text, "
            "q.status AS status, q.risk_level AS risk_level, "
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
            timestamp=row["timestamp"],
            user_feedback=row["user_feedback"],
            rejection_reason=row["rejection_reason"],
            fewshot_cypher=row["fewshot_cypher"],
            fewshot_parameters=list(row["fewshot_parameters"] or []),
        )

    def stats(self) -> CuratorStats:
        """Return counts-by-status for every status we track."""
        cypher = (
            "MATCH (q:Query) "
            "RETURN q.status AS status, count(*) AS n"
        )
        with self._session() as session:
            rows = list(session.run(cypher))
        counts = {status: 0 for status in ALL_STATUSES}
        for row in rows:
            status = row["status"] or "unknown"
            counts[status] = counts.get(status, 0) + int(row["n"])
        return CuratorStats(counts=counts)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def approve(
        self,
        query_id: str,
        *,
        cypher_template: str,
        parameters: list[str] | None = None,
    ) -> ApprovalResult:
        """Approve a query and attach a FewShot cypher template.

        Idempotent: if the query already has a linked FewShot, its
        ``cypher_template`` and ``parameters`` are replaced in place
        rather than duplicated. This lets curators iterate on templates
        without creating orphaned FewShot nodes.

        Raises:
            ValueError: if ``cypher_template`` is empty or ``query_id``
                doesn't exist.
        """
        if not cypher_template or not cypher_template.strip():
            raise ValueError("cypher_template must be a non-empty string")
        params = list(parameters or [])

        with self._session() as session:
            existing = session.run(
                "MATCH (q:Query {id: $query_id}) RETURN q.id AS id",
                query_id=query_id,
            ).single()
            if existing is None:
                raise ValueError(f"No Query node with id {query_id!r}")

            # Check for existing FewShot.
            fewshot_row = session.run(
                "MATCH (q:Query {id: $query_id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
                "RETURN fs.id AS fewshot_id",
                query_id=query_id,
            ).single()

            if fewshot_row is not None:
                fewshot_id = fewshot_row["fewshot_id"]
                session.run(
                    "MATCH (fs:FewShot {id: $fewshot_id}) "
                    "SET fs.cypher_template = $cypher, "
                    "fs.parameters = $params, "
                    "fs.updated_at = datetime($timestamp)",
                    fewshot_id=fewshot_id,
                    cypher=cypher_template,
                    params=params,
                    timestamp=_utcnow_iso(),
                )
                was_new = False
            else:
                fewshot_id = str(uuid.uuid4())
                session.run(
                    "MATCH (q:Query {id: $query_id}) "
                    "CREATE (fs:FewShot { "
                    "id: $fewshot_id, cypher_template: $cypher, "
                    "parameters: $params, created_at: datetime($timestamp) "
                    "}) "
                    "CREATE (q)-[:FEW_SHOT_EXAMPLE]->(fs)",
                    query_id=query_id,
                    fewshot_id=fewshot_id,
                    cypher=cypher_template,
                    params=params,
                    timestamp=_utcnow_iso(),
                )
                was_new = True

            session.run(
                "MATCH (q:Query {id: $query_id}) "
                "SET q.status = 'approved', q.approved_at = datetime($timestamp) "
                "REMOVE q.rejection_reason",
                query_id=query_id,
                timestamp=_utcnow_iso(),
            )

        return ApprovalResult(
            query_id=query_id,
            fewshot_id=fewshot_id,
            fewshot_was_new=was_new,
        )

    def reject(self, query_id: str, *, reason: str | None = None) -> None:
        """Mark a query as rejected. Optional ``reason`` is stored on the node."""
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
        if summary.counters.properties_set == 0 and summary.counters.labels_added == 0:
            raise ValueError(f"No Query node with id {query_id!r}")

    def edit_fewshot(
        self,
        query_id: str,
        *,
        cypher_template: str,
        parameters: list[str] | None = None,
    ) -> None:
        """Update an existing FewShot template without changing approval status.

        Useful for curators tweaking an approved template. Raises if the
        query has no linked FewShot (use ``approve`` to create one).
        """
        if not cypher_template or not cypher_template.strip():
            raise ValueError("cypher_template must be a non-empty string")
        params = list(parameters or [])

        with self._session() as session:
            row = session.run(
                "MATCH (q:Query {id: $query_id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
                "SET fs.cypher_template = $cypher, "
                "fs.parameters = $params, "
                "fs.updated_at = datetime($timestamp) "
                "RETURN fs.id AS id",
                query_id=query_id,
                cypher=cypher_template,
                params=params,
                timestamp=_utcnow_iso(),
            ).single()
        if row is None:
            raise ValueError(
                f"Query {query_id!r} has no linked FewShot; "
                "use approve() to create one first."
            )

    def submit_feedback(
        self,
        query_id: str,
        *,
        helpful: bool,
    ) -> None:
        """Record end-user feedback on a previously-executed query.

        Semantics match the existing BOOTH app: ``helpful=True`` promotes
        the query to ``pending_approval`` for curator review; ``False``
        marks it ``needs_review`` so curators can inspect poor matches.
        """
        new_status = "pending_approval" if helpful else "needs_review"
        feedback_label = "helpful" if helpful else "not_helpful"
        cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "SET q.status = $status, "
            "q.user_feedback = $feedback, "
            "q.feedback_timestamp = datetime($timestamp)"
        )
        with self._session() as session:
            summary = session.run(
                cypher,
                query_id=query_id,
                status=new_status,
                feedback=feedback_label,
                timestamp=_utcnow_iso(),
            ).consume()
        if summary.counters.properties_set == 0:
            raise ValueError(f"No Query node with id {query_id!r}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _list_by_statuses(
        self,
        statuses: tuple[str, ...],
        *,
        limit: int,
    ) -> list[PendingQuery]:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        cypher = (
            "MATCH (q:Query) WHERE q.status IN $statuses "
            "OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
            "RETURN q.id AS query_id, q.text AS query_text, "
            "q.status AS status, q.risk_level AS risk_level, "
            "toString(q.timestamp) AS timestamp, "
            "q.user_feedback AS user_feedback, "
            "fs IS NOT NULL AS has_fewshot "
            "ORDER BY q.timestamp DESC LIMIT $limit"
        )
        with self._session() as session:
            rows = list(session.run(cypher, statuses=list(statuses), limit=limit))
        return [
            PendingQuery(
                query_id=row["query_id"],
                query_text=row["query_text"],
                status=row["status"],
                risk_level=row["risk_level"],
                timestamp=row["timestamp"],
                user_feedback=row["user_feedback"],
                has_fewshot=bool(row["has_fewshot"]),
            )
            for row in rows
        ]

    def _validate_status(self, status: str) -> None:
        if status not in ALL_STATUSES:
            raise ValueError(
                f"Unknown status {status!r}. "
                f"Valid statuses: {sorted(ALL_STATUSES)}"
            )

    def _session(self):
        if self.database is not None:
            return self.driver.session(database=self.database)
        return self.driver.session()


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
