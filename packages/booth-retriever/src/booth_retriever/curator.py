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

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from .verification import verify_cypher

if TYPE_CHECKING:
    from neo4j import Driver


# ---------- Status taxonomy --------------------------------------------------

#: Statuses retired in the three-state simplification. Anything still carrying
#: one of these (e.g. nodes created before the migration) is treated as
#: ``needs_review`` by ``migrate_statuses`` and the legacy mapping below.
LEGACY_STATUSES: tuple[str, ...] = ("pending_approval", "declined")

#: Statuses that the curation queue shows by default (anything "not done").
#: Collapsed to a single review bucket: high-risk declines and thumbs-down
#: feedback now land in ``needs_review`` and are distinguished by node
#: properties (``risk_level``, ``user_feedback``) rather than separate states.
PENDING_STATUSES: tuple[str, ...] = ("needs_review",)

#: Every status a Query node may hold. The CLI validates ``--status`` against
#: this set; the data model treats anything else as unknown.
ALL_STATUSES: tuple[str, ...] = (
    "needs_review",
    "approved",
    "rejected",
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
    # Distinguishes a plain cached query from a seeded procedure. ``None`` (or
    # any value other than ``"procedural_memory"``) is treated as a query.
    kind: str | None = None


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
    # ``"procedural_memory"`` for seeded multi-step processes, else a query.
    kind: str | None = None
    # Most recent Text2Cypher attempt recorded for a declined query (if any).
    attempt_cypher: str | None = None
    attempt_rows: str | None = None
    attempt_error: str | None = None


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
            "(fs IS NOT NULL) AS has_fewshot, "
            "q.kind AS kind "
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
                kind=row["kind"],
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
            "OPTIONAL MATCH (q)-[:HAS_ATTEMPT]->(ca:CypherAttempt) "
            "OPTIONAL MATCH (ca)-[:PRODUCED]->(r:Response) "
            "WITH q, fs, ca, r ORDER BY ca.created DESC "
            "WITH q, fs, "
            "collect({cypher: ca.cypher, error: ca.error, "
            "rows: r.rows_json})[0] AS attempt "
            "RETURN q.id AS query_id, q.text AS query_text, "
            "q.status AS status, coalesce(q.risk_level, 'low') AS risk_level, "
            "toString(q.timestamp) AS timestamp, "
            "q.user_feedback AS user_feedback, "
            "q.rejection_reason AS rejection_reason, "
            "fs.cypher_template AS fewshot_cypher, "
            "coalesce(fs.parameters, []) AS fewshot_parameters, "
            "q.kind AS kind, "
            "attempt.cypher AS attempt_cypher, "
            "attempt.error AS attempt_error, "
            "attempt.rows AS attempt_rows"
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
            kind=row["kind"],
            attempt_cypher=row["attempt_cypher"],
            attempt_rows=row["attempt_rows"],
            attempt_error=row["attempt_error"],
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
            if status is None:
                continue
            # Fold retired statuses (pending_approval / declined) into the
            # single curation bucket so the obsolete tiles never resurface,
            # even if a legacy node hasn't been migrated yet.
            if status in LEGACY_STATUSES:
                status = "needs_review"
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

        Both helpful and thumbs-down feedback land the Query in the single
        ``needs_review`` curation bucket; the recorded ``user_feedback``
        (``helpful`` / ``not_helpful``) is what lets curators tell the two
        apart, so we no longer fork the status on the feedback signal.
        """
        if not query_id:
            raise ValueError("query_id must be a non-empty string")
        status = "needs_review"
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
    # Migration
    # ------------------------------------------------------------------

    def migrate_statuses(self) -> int:
        """Collapse legacy statuses onto the three-state model.

        Re-labels every Query still carrying a retired status
        (``pending_approval`` / ``declined``) as ``needs_review`` so the
        node lands in the single curation bucket. Idempotent: running it
        again once nothing matches returns ``0``.

        Returns:
            The number of Query nodes that were updated.
        """
        cypher = (
            "MATCH (q:Query) WHERE q.status IN $legacy "
            "SET q.status = 'needs_review' "
            "RETURN count(q) AS migrated"
        )
        with self._session() as session:
            row = session.run(cypher, legacy=list(LEGACY_STATUSES)).single()
        return int(row["migrated"]) if row else 0

    def compact_user_questions(self, *, threshold: float = 0.99) -> int:
        """Collapse near-duplicate ``UserQuestion`` nodes, per Query.

        Repeated asks of effectively the same question pile up identical
        ``UserQuestion`` nodes hanging off one Query, which clutters the
        provenance graph. This merges them: within each Query, questions
        whose embeddings are >= ``threshold`` cosine similar (or, for legacy
        nodes lacking embeddings, whose normalised text is identical) are
        clustered. The earliest node in each cluster is kept, its ``count``
        set to the cluster total, and the rest are deleted.

        Idempotent: a second run with nothing to merge returns ``0``.

        Returns:
            The number of ``UserQuestion`` nodes removed.
        """
        fetch = (
            "MATCH (q:Query)<-[:SIMILAR]-(uq:UserQuestion) "
            "RETURN q.id AS query_id, collect({"
            "id: uq.id, text: uq.text, embedding: uq.embedding, "
            "ts: toString(uq.timestamp), count: coalesce(uq.count, 1)"
            "}) AS questions"
        )
        merge = (
            "MATCH (k:UserQuestion {id: $keeper}) SET k.count = $total "
            "WITH k MATCH (d:UserQuestion) WHERE d.id IN $dupes DETACH DELETE d"
        )
        removed = 0
        with self._session() as session:
            groups = list(session.run(fetch))
            for group in groups:
                for cluster in _cluster_questions(group["questions"], threshold):
                    if len(cluster) < 2:
                        continue
                    keeper = cluster[0]
                    dupes = [c["id"] for c in cluster[1:]]
                    total = sum(int(c["count"] or 1) for c in cluster)
                    session.run(
                        merge, keeper=keeper["id"], dupes=dupes, total=total
                    )
                    removed += len(dupes)
        return removed

    # ------------------------------------------------------------------
    # Graph (for NVL visualisation)
    # ------------------------------------------------------------------

    def get_query_graph(
        self,
        query_id: str,
        *,
        include_answer_subgraph: bool = True,
    ) -> dict | None:
        """Return an NVL-shaped graph for one Query, or None if not found.

        Two layers are merged:

          * **Provenance (always):** the ``UserQuestion -> Query ->
            FewShot`` chain plus any linked ``CypherAttempt`` / ``Response``
            and the procedural-memory chain
            (``HAS_STEP`` / ``NEXT`` / ``USES_AGENT`` / ``USES_TOOL`` and the
            data dimension ``BACKED_BY`` / ``SOURCED_FROM``).
          * **Answer subgraph (optional):** when the Query has a
            parameter-free FewShot, its Cypher is executed and any
            nodes/relationships it returns are merged in so the popup shows
            the actual data that produced the answer.

        Output shape matches ``@neo4j-nvl/base``:
        ``{"nodes": [{"id", "caption", "labels", ...}],
           "relationships": [{"id", "from", "to", "caption"}]}``.
        """
        if not query_id:
            raise ValueError("query_id must be a non-empty string")

        # Depth must clear the longest procedural chain plus its agent/tool
        # tail: a 12-step scenario path then USES_AGENT -> USES_TOOL is ~14
        # hops from the Query, so we allow generous headroom.
        provenance_cypher = (
            "MATCH (q:Query {id: $query_id}) "
            "OPTIONAL MATCH inP = (:UserQuestion)-[:SIMILAR]->(q) "
            "OPTIONAL MATCH outP = (q)-["
            ":FEW_SHOT_EXAMPLE|HAS_ATTEMPT|PRODUCED|HAS_STEP|NEXT"
            "|USES_AGENT|USES_TOOL|BACKED_BY|SOURCED_FROM*1..25]->() "
            "RETURN q AS root, collect(DISTINCT inP) AS in_paths, "
            "collect(DISTINCT outP) AS out_paths"
        )

        builder = _GraphBuilder()
        with self._session() as session:
            row = session.run(provenance_cypher, query_id=query_id).single()
            if row is None:
                return None
            builder.add_node(row["root"])
            for path in (row["in_paths"] or []):
                builder.add_path(path)
            for path in (row["out_paths"] or []):
                builder.add_path(path)

            if include_answer_subgraph:
                fewshot = session.run(
                    "MATCH (q:Query {id: $query_id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
                    "RETURN fs.cypher_template AS cypher, "
                    "coalesce(fs.parameters, []) AS parameters",
                    query_id=query_id,
                ).single()
                # Parameterised templates can't be executed without values.
                if fewshot and fewshot["cypher"] and not fewshot["parameters"]:
                    try:
                        result = session.run(fewshot["cypher"])
                        for record in result:
                            for value in record.values():
                                builder.add_value(value)
                    except Exception:  # noqa: BLE001 - bad FewShot must not 500 the popup
                        pass

        return builder.to_payload()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _session(self):
        if self.database is not None:
            return self.driver.session(database=self.database)
        return self.driver.session()


# ---------- NVL graph builder -----------------------------------------------

#: Node properties we never serialise into the graph payload. Embeddings are
#: ~1536 floats and would bloat the response (and mean nothing to NVL).
_EXCLUDED_NODE_PROPS = frozenset({"embedding"})

#: Order of properties tried when picking a human-readable node caption.
_CAPTION_PROPS = ("text", "name", "caption", "title", "id")


class _GraphBuilder:
    """Accumulates neo4j graph elements and emits an NVL-shaped payload.

    Dedupes nodes and relationships by their Neo4j ``element_id`` so the
    same node appearing on multiple paths (or in the answer subgraph) is
    only emitted once.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._rels: dict[str, dict] = {}

    def add_value(self, value) -> None:
        """Add an arbitrary Cypher value if it's a graph element."""
        from neo4j.graph import Node, Path, Relationship

        if isinstance(value, Node):
            self.add_node(value)
        elif isinstance(value, Relationship):
            self.add_relationship(value)
        elif isinstance(value, Path):
            self.add_path(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.add_value(item)

    def add_path(self, path) -> None:
        if path is None:
            return
        for node in path.nodes:
            self.add_node(node)
        for rel in path.relationships:
            self.add_relationship(rel)

    def add_node(self, node) -> None:
        if node is None:
            return
        key = node.element_id
        if key in self._nodes:
            return
        props = {
            k: v for k, v in dict(node).items() if k not in _EXCLUDED_NODE_PROPS
        }
        labels = list(node.labels)
        self._nodes[key] = {
            "id": key,
            "caption": _node_caption(props, labels),
            "labels": labels,
            "properties": _jsonable_props(props),
        }

    def add_relationship(self, rel) -> None:
        if rel is None:
            return
        key = rel.element_id
        if key in self._rels:
            return
        # Ensure endpoints exist even if a relationship is added in isolation.
        if rel.start_node is not None:
            self.add_node(rel.start_node)
        if rel.end_node is not None:
            self.add_node(rel.end_node)
        self._rels[key] = {
            "id": key,
            "from": rel.start_node.element_id if rel.start_node else None,
            "to": rel.end_node.element_id if rel.end_node else None,
            "caption": rel.type,
        }

    def to_payload(self) -> dict:
        return {
            "nodes": list(self._nodes.values()),
            "relationships": [
                r for r in self._rels.values() if r["from"] and r["to"]
            ],
        }


def _node_caption(props: dict, labels: list[str]) -> str:
    for prop in _CAPTION_PROPS:
        value = props.get(prop)
        if value:
            text = str(value)
            return text if len(text) <= 60 else text[:59] + "\u2026"
    return labels[0] if labels else "node"


def _jsonable_props(props: dict) -> dict:
    """Coerce property values to JSON-safe primitives (datetimes -> str)."""
    safe: dict = {}
    for key, value in props.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [v if isinstance(v, (str, int, float, bool)) else str(v) for v in value]
        else:
            safe[key] = str(value)
    return safe


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if not comparable."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _questions_duplicate(a: dict, b: dict, threshold: float) -> bool:
    """Two UserQuestions are duplicates if cosine-close, else text-identical."""
    ea, eb = a.get("embedding"), b.get("embedding")
    if ea and eb:
        return _cosine(ea, eb) >= threshold
    # Legacy nodes without embeddings: fall back to normalised text equality.
    ta = (a.get("text") or "").strip().casefold()
    tb = (b.get("text") or "").strip().casefold()
    return bool(ta) and ta == tb


def _cluster_questions(questions: list[dict], threshold: float) -> list[list[dict]]:
    """Greedily group duplicate questions; keeper (earliest) is first in each.

    Each input dict carries ``id``, ``text``, ``embedding``, ``ts`` and
    ``count``. Clusters are sorted so the earliest-``ts`` node leads, making
    it the stable "keeper" for merge.
    """
    ordered = sorted(questions, key=lambda q: q.get("ts") or "")
    clusters: list[list[dict]] = []
    claimed: set[str] = set()
    for q in ordered:
        if q["id"] in claimed:
            continue
        cluster = [q]
        claimed.add(q["id"])
        for other in ordered:
            if other["id"] in claimed:
                continue
            if _questions_duplicate(q, other, threshold):
                cluster.append(other)
                claimed.add(other["id"])
        clusters.append(cluster)
    return clusters


def _require_positive_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
