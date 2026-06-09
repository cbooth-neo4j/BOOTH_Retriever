"""Tier-2 unit tests for ``booth_retriever.curator``.

All tests use a MagicMock driver that returns canned responses per Cypher
prefix. No real Neo4j required. End-to-end curator flows are verified
separately in ``tests/integration/test_curator.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from booth_retriever.curator import (
    ALL_STATUSES,
    PENDING_STATUSES,
    ApprovalResult,
    BOOTHCurator,
    CuratorStats,
    PendingQuery,
    QueryDetail,
    _cluster_questions,
    _cosine,
)

pytestmark = pytest.mark.unit


# ---------- UserQuestion compaction helpers ----------------------------------


def test_cosine_identical_and_orthogonal() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # Not comparable -> 0.0 (different length / empty / None).
    assert _cosine([1.0], [1.0, 0.0]) == 0.0
    assert _cosine(None, [1.0]) == 0.0


def test_cluster_questions_merges_near_duplicates_by_embedding() -> None:
    questions = [
        {"id": "a", "text": "q", "embedding": [1.0, 0.0], "ts": "2026-01-01", "count": 1},
        {"id": "b", "text": "q", "embedding": [0.999, 0.001], "ts": "2026-01-02", "count": 1},
        {"id": "c", "text": "other", "embedding": [0.0, 1.0], "ts": "2026-01-03", "count": 1},
    ]
    clusters = _cluster_questions(questions, threshold=0.99)
    # a+b collapse (cosine ~1.0); c stands alone. Earliest (a) leads its cluster.
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]
    big = max(clusters, key=len)
    assert big[0]["id"] == "a"
    assert {q["id"] for q in big} == {"a", "b"}


def test_cluster_questions_text_fallback_without_embeddings() -> None:
    questions = [
        {"id": "a", "text": "Same Question", "embedding": None, "ts": "1", "count": 1},
        {"id": "b", "text": "same question", "embedding": None, "ts": "2", "count": 1},
        {"id": "c", "text": "different", "embedding": None, "ts": "3", "count": 1},
    ]
    clusters = _cluster_questions(questions, threshold=0.99)
    assert sorted(len(c) for c in clusters) == [1, 2]


# ---------- Driver test double -----------------------------------------------


class _FakeResult:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        counters: dict[str, int] | None = None,
    ) -> None:
        self._rows = rows or []
        self._counters = _FakeCounters(**(counters or {}))

    def single(self):
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)

    def consume(self):
        summary = MagicMock()
        summary.counters = self._counters
        return summary


class _FakeCounters:
    def __init__(self, properties_set: int = 0, labels_added: int = 0) -> None:
        self.properties_set = properties_set
        self.labels_added = labels_added


def _build_driver(responses: list):
    """Build a MagicMock driver. ``responses`` is a list matched in order
    against successive session.run() calls. Each entry is either:

        - a ``_FakeResult`` (pre-built)
        - a dict with keys 'rows' and/or 'counters'
        - a callable taking (cypher, kwargs) and returning a ``_FakeResult``
    """
    executed: list[tuple[str, dict]] = []
    iterator = iter(responses)

    # NB: the positional name must not clash with any kwarg the curator
    # passes to session.run() (e.g. ``cypher=...``). Using ``*args`` avoids
    # the clash entirely.
    def run(*args, **kwargs):
        cypher_str = args[0] if args else ""
        executed.append((cypher_str, kwargs))
        try:
            spec = next(iterator)
        except StopIteration:
            raise AssertionError(
                f"Unexpected session.run call {len(executed)}: {cypher_str[:80]!r}"
            ) from None
        if callable(spec):
            return spec(cypher_str, kwargs)
        if isinstance(spec, _FakeResult):
            return spec
        if isinstance(spec, dict):
            return _FakeResult(
                rows=spec.get("rows"),
                counters=spec.get("counters"),
            )
        raise TypeError(f"Unsupported response spec: {spec!r}")

    session = MagicMock()
    session.run.side_effect = run

    def _session_cm(*args, **kwargs):
        @contextmanager
        def _cm():
            yield session

        return _cm()

    driver = MagicMock()
    driver.session.side_effect = _session_cm
    driver._executed = executed
    driver._session = session
    return driver


# ---------- Reads ------------------------------------------------------------


def test_list_pending_queries_uses_pending_statuses() -> None:
    driver = _build_driver(
        [
            {
                "rows": [
                    {
                        "query_id": "q1",
                        "query_text": "count users",
                        "status": "pending_approval",
                        "risk_level": "low",
                        "timestamp": "2026-04-01T10:00:00Z",
                        "user_feedback": None,
                        "has_fewshot": False,
                        "kind": None,
                    },
                    {
                        "query_id": "q2",
                        "query_text": "drop everything",
                        "status": "needs_review",
                        "risk_level": "high",
                        "timestamp": "2026-04-01T09:00:00Z",
                        "user_feedback": None,
                        "has_fewshot": False,
                        "kind": "procedural_memory",
                    },
                ]
            }
        ]
    )
    curator = BOOTHCurator(driver=driver)

    results = curator.list_pending(limit=25)

    assert len(results) == 2
    assert isinstance(results[0], PendingQuery)
    assert results[0].status == "pending_approval"
    assert results[1].status == "needs_review"
    assert results[0].kind is None
    assert results[1].kind == "procedural_memory"

    # Verify the statuses filter was passed correctly
    _cypher, params = driver._executed[0]
    assert set(params["statuses"]) == set(PENDING_STATUSES)
    assert params["limit"] == 25


def test_list_by_status_rejects_unknown_status() -> None:
    driver = _build_driver([])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="Unknown status"):
        curator.list_by_status("garbage")


def test_list_rejects_non_positive_limit() -> None:
    driver = _build_driver([])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="limit must be positive"):
        curator.list_pending(limit=0)


def test_get_returns_none_for_missing_query() -> None:
    driver = _build_driver([{"rows": []}])
    curator = BOOTHCurator(driver=driver)
    assert curator.get("nonexistent") is None


def test_get_returns_detail_with_fewshot() -> None:
    driver = _build_driver(
        [
            {
                "rows": [
                    {
                        "query_id": "q1",
                        "query_text": "count users",
                        "status": "approved",
                        "risk_level": "low",
                        "timestamp": "2026-04-01T10:00:00Z",
                        "user_feedback": "helpful",
                        "rejection_reason": None,
                        "fewshot_cypher": "MATCH (u:User) RETURN count(u)",
                        "fewshot_parameters": [],
                        "kind": None,
                        "attempt_cypher": None,
                        "attempt_error": None,
                        "attempt_rows": None,
                    }
                ]
            }
        ]
    )
    curator = BOOTHCurator(driver=driver)

    detail = curator.get("q1")

    assert isinstance(detail, QueryDetail)
    assert detail.query_id == "q1"
    assert detail.fewshot_cypher == "MATCH (u:User) RETURN count(u)"
    assert detail.fewshot_parameters == []
    assert detail.user_feedback == "helpful"
    assert detail.attempt_cypher is None


def test_get_surfaces_text2cypher_attempt() -> None:
    driver = _build_driver(
        [
            {
                "rows": [
                    {
                        "query_id": "q1",
                        "query_text": "drop everything",
                        "status": "needs_review",
                        "risk_level": "high",
                        "timestamp": "2026-04-01T10:00:00Z",
                        "user_feedback": None,
                        "rejection_reason": None,
                        "fewshot_cypher": None,
                        "fewshot_parameters": [],
                        "kind": None,
                        "attempt_cypher": "MATCH (n) RETURN n LIMIT 5",
                        "attempt_error": None,
                        "attempt_rows": "[{\"n\": 1}]",
                    }
                ]
            }
        ]
    )
    curator = BOOTHCurator(driver=driver)

    detail = curator.get("q1")

    assert detail is not None
    assert detail.attempt_cypher == "MATCH (n) RETURN n LIMIT 5"
    assert detail.attempt_rows == "[{\"n\": 1}]"
    assert detail.attempt_error is None


def test_stats_aggregates_by_status() -> None:
    driver = _build_driver(
        [
            {
                "rows": [
                    {"status": "approved", "n": 12},
                    {"status": "needs_review", "n": 2},
                    # Retired statuses fold into needs_review.
                    {"status": "pending_approval", "n": 3},
                    {"status": "declined", "n": 4},
                    {"status": "rejected", "n": 1},
                ]
            }
        ]
    )
    curator = BOOTHCurator(driver=driver)

    stats = curator.stats()

    assert isinstance(stats, CuratorStats)
    assert stats.counts["approved"] == 12
    assert stats.counts["rejected"] == 1
    # Legacy pending_approval (3) + declined (4) fold into needs_review (2).
    assert stats.counts["needs_review"] == 9
    # Obsolete buckets are not surfaced at all.
    assert "pending_approval" not in stats.counts
    assert "declined" not in stats.counts
    assert stats.total == 22


# ---------- Mutations: approve ----------------------------------------------


def test_approve_rejects_empty_template() -> None:
    driver = _build_driver([])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="non-empty"):
        curator.approve("q1", cypher_template="")


def test_approve_raises_when_query_not_found() -> None:
    driver = _build_driver([{"rows": []}])  # existence check returns nothing
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.approve("nope", cypher_template="RETURN 1")


def test_approve_creates_new_fewshot_when_none_exists() -> None:
    driver = _build_driver(
        [
            {"rows": [{"id": "q1"}]},  # existence check
            {"rows": []},  # fewshot lookup (none exists)
            {"rows": []},  # CREATE FewShot
            {"rows": []},  # SET status = approved
        ]
    )
    curator = BOOTHCurator(driver=driver)

    result = curator.approve(
        "q1",
        cypher_template="MATCH (n) RETURN count(n)",
        parameters=["foo"],
    )

    assert isinstance(result, ApprovalResult)
    assert result.query_id == "q1"
    assert result.fewshot_was_new is True
    assert result.fewshot_id

    # The CREATE FewShot call uses the generated id and the right params
    create_calls = [(c, p) for c, p in driver._executed if "CREATE (fs:FewShot" in c]
    assert len(create_calls) == 1
    _, params = create_calls[0]
    assert params["fewshot_id"] == result.fewshot_id
    assert params["cypher"] == "MATCH (n) RETURN count(n)"
    assert params["params"] == ["foo"]

    # Status update is last
    status_calls = [c for c, _ in driver._executed if "'approved'" in c]
    assert len(status_calls) == 1


def test_approve_replaces_existing_fewshot_in_place() -> None:
    driver = _build_driver(
        [
            {"rows": [{"id": "q1"}]},  # existence check
            {"rows": [{"fewshot_id": "fs-existing"}]},  # fewshot lookup returns one
            {"rows": []},  # SET cypher_template
            {"rows": []},  # SET status
        ]
    )
    curator = BOOTHCurator(driver=driver)

    result = curator.approve(
        "q1",
        cypher_template="MATCH (n) RETURN count(n) AS n2",
    )

    assert result.fewshot_was_new is False
    assert result.fewshot_id == "fs-existing"

    # No CREATE (fs:FewShot should have been executed
    assert not any("CREATE (fs:FewShot" in c for c, _ in driver._executed)
    # A SET on the existing fewshot should have happened
    update_calls = [(c, p) for c, p in driver._executed if "SET fs.cypher_template" in c]
    assert len(update_calls) == 1
    _, params = update_calls[0]
    assert params["fewshot_id"] == "fs-existing"
    assert params["cypher"] == "MATCH (n) RETURN count(n) AS n2"
    assert params["params"] == []


# ---------- Mutations: reject, edit, feedback --------------------------------


def test_reject_with_reason_writes_reason() -> None:
    driver = _build_driver([{"counters": {"properties_set": 3}}])
    curator = BOOTHCurator(driver=driver)

    curator.reject("q1", reason="off-topic")

    _, params = driver._executed[0]
    assert params["query_id"] == "q1"
    assert params["reason"] == "off-topic"


def test_reject_raises_when_no_node_updated() -> None:
    """counters.properties_set == 0 is how we detect a missing Query."""
    driver = _build_driver([{"counters": {"properties_set": 0}}])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.reject("nope")


def test_edit_fewshot_raises_if_no_fewshot_linked() -> None:
    driver = _build_driver([{"rows": []}])  # MATCH returns no rows
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="no linked FewShot"):
        curator.edit_fewshot("q1", cypher_template="RETURN 1")


def test_edit_fewshot_rejects_empty_template() -> None:
    driver = _build_driver([])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="non-empty"):
        curator.edit_fewshot("q1", cypher_template=" ")


def test_edit_fewshot_updates_and_sets_timestamp() -> None:
    driver = _build_driver([{"rows": [{"id": "fs-1"}]}])
    curator = BOOTHCurator(driver=driver)

    curator.edit_fewshot("q1", cypher_template="RETURN 2", parameters=["x"])

    _, params = driver._executed[0]
    assert params["query_id"] == "q1"
    assert params["cypher"] == "RETURN 2"
    assert params["params"] == ["x"]
    assert "timestamp" in params


def test_submit_feedback_helpful_lands_in_needs_review() -> None:
    driver = _build_driver([{"counters": {"properties_set": 3}}])
    curator = BOOTHCurator(driver=driver)

    curator.submit_feedback("q1", helpful=True)

    _, params = driver._executed[0]
    # Three-state model: helpful feedback now also lands in needs_review;
    # the recorded feedback string is what distinguishes it.
    assert params["status"] == "needs_review"
    assert params["feedback"] == "helpful"


def test_submit_feedback_not_helpful_marks_needs_review() -> None:
    driver = _build_driver([{"counters": {"properties_set": 3}}])
    curator = BOOTHCurator(driver=driver)

    curator.submit_feedback("q1", helpful=False)

    _, params = driver._executed[0]
    assert params["status"] == "needs_review"
    assert params["feedback"] == "not_helpful"


def test_submit_feedback_raises_when_no_node_updated() -> None:
    driver = _build_driver([{"counters": {"properties_set": 0}}])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.submit_feedback("nope", helpful=True)


# ---------- Mutations: delete -----------------------------------------------


def test_delete_removes_query_and_linked_fewshot() -> None:
    """Existence check passes, then a DETACH DELETE statement runs."""
    driver = _build_driver(
        [
            {"rows": [{"id": "q1"}]},  # existence check
            {"counters": {}},  # the delete itself
        ]
    )
    curator = BOOTHCurator(driver=driver)

    curator.delete("q1")

    assert len(driver._executed) == 2
    delete_cypher, delete_params = driver._executed[1]
    assert "DETACH DELETE q, fs" in delete_cypher
    assert "FEW_SHOT_EXAMPLE" in delete_cypher
    assert delete_params["query_id"] == "q1"


def test_delete_raises_when_query_missing() -> None:
    driver = _build_driver([{"rows": []}])  # existence check returns no row
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="No Query node"):
        curator.delete("nope")


def test_delete_rejects_empty_id() -> None:
    driver = _build_driver([])
    curator = BOOTHCurator(driver=driver)
    with pytest.raises(ValueError, match="non-empty"):
        curator.delete("")


# ---------- Database kwarg forwarding ---------------------------------------


def test_database_kwarg_forwarded() -> None:
    driver = _build_driver([{"rows": []}])
    curator = BOOTHCurator(driver=driver, database="booth")

    curator.get("q1")

    for call in driver.session.call_args_list:
        assert call.kwargs == {"database": "booth"}


# ---------- Constants regression --------------------------------------------


def test_all_statuses_superset_of_pending_statuses() -> None:
    """ALL_STATUSES must contain every PENDING_STATUS - otherwise
    list_by_status would reject a status used by list_pending."""
    assert set(PENDING_STATUSES).issubset(set(ALL_STATUSES))


def test_three_state_taxonomy() -> None:
    """The simplified model is exactly needs_review / approved / rejected."""
    assert set(ALL_STATUSES) == {"needs_review", "approved", "rejected"}
    assert PENDING_STATUSES == ("needs_review",)


# ---------- Migration -------------------------------------------------------


def test_migrate_statuses_relabels_legacy() -> None:
    driver = _build_driver([{"rows": [{"migrated": 4}]}])
    curator = BOOTHCurator(driver=driver)

    migrated = curator.migrate_statuses()

    assert migrated == 4
    cypher, params = driver._executed[0]
    assert "needs_review" in cypher
    assert set(params["legacy"]) == {"pending_approval", "declined"}


# ---------- Graph (NVL) -----------------------------------------------------


def test_get_query_graph_returns_none_for_missing_query() -> None:
    driver = _build_driver([{"rows": []}])
    curator = BOOTHCurator(driver=driver)
    assert curator.get_query_graph("nope") is None


class _FakeNode:
    """Duck-typed neo4j Node: supports dict(node), .labels, .element_id."""

    def __init__(self, element_id: str, labels: set[str], props: dict) -> None:
        self.element_id = element_id
        self.labels = labels
        self._props = props

    def keys(self):
        return self._props.keys()

    def __getitem__(self, key):
        return self._props[key]


class _FakeRel:
    def __init__(self, element_id, rel_type, start, end) -> None:
        self.element_id = element_id
        self.type = rel_type
        self.start_node = start
        self.end_node = end


class _FakePath:
    def __init__(self, nodes, relationships) -> None:
        self.nodes = nodes
        self.relationships = relationships


def test_get_query_graph_builds_nvl_payload_from_paths() -> None:
    q_node = _FakeNode(
        "1", {"Query"}, {"id": "q1", "text": "count users", "embedding": [0.1, 0.2]}
    )
    fs_node = _FakeNode("2", {"FewShot"}, {"id": "fs1", "cypher_template": "RETURN 1"})
    rel = _FakeRel("10", "FEW_SHOT_EXAMPLE", q_node, fs_node)
    path = _FakePath([q_node, fs_node], [rel])

    row = {"root": q_node, "in_paths": [], "out_paths": [path]}
    # provenance query, then the FewShot lookup (no parameter-free fewshot).
    driver = _build_driver([{"rows": [row]}, {"rows": []}])
    curator = BOOTHCurator(driver=driver)

    graph = curator.get_query_graph("q1")

    assert graph is not None
    node_ids = {n["id"] for n in graph["nodes"]}
    assert node_ids == {"1", "2"}
    # Embedding must be stripped from the payload.
    query_node = next(n for n in graph["nodes"] if "Query" in n["labels"])
    assert "embedding" not in query_node["properties"]
    assert query_node["caption"] == "count users"
    assert len(graph["relationships"]) == 1
    assert graph["relationships"][0]["caption"] == "FEW_SHOT_EXAMPLE"
    assert graph["relationships"][0]["from"] == "1"
    assert graph["relationships"][0]["to"] == "2"
