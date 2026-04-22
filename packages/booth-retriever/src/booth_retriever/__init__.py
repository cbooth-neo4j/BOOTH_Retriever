"""booth-retriever: a self-improving Neo4j retriever for neo4j-graphrag-python.

Public API:
    - BOOTHRetriever:   drop-in neo4j_graphrag Retriever subclass
    - BOOTHCurator:     Python API for listing and approving pending queries
    - BOOTHResponse:    rich response object returned by BOOTHRetriever.query()
    - init_schema:      idempotent DDL bootstrap for BOOTH's own Neo4j schema

These are re-exported here as stubs in the initial scaffold and filled in as
the port from the parent repository progresses. See the plan document at
`.cursor/plans/booth-retriever_pip_package_*.plan.md` in the parent repo.
"""

from __future__ import annotations

from .schema import SchemaInitResult, init_schema

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "BOOTHRetriever",
    "BOOTHCurator",
    "BOOTHResponse",
    "SchemaInitResult",
    "init_schema",
]


class _NotImplementedStub:
    """Placeholder until the real class is ported.

    Raises a clear error on instantiation so smoke tests can still verify
    that the symbol is importable without accidentally succeeding on a
    broken implementation.
    """

    _name: str = "stub"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"{self._name} has not been ported yet. "
            "See packages/booth-retriever/ TODOs in the plan."
        )


class BOOTHRetriever(_NotImplementedStub):
    """Drop-in neo4j_graphrag Retriever with a similarity cache + agent fallback.

    Will subclass ``neo4j_graphrag.retrievers.base.Retriever`` once the
    orchestrator port lands. Exposes ``search()`` (spec-compliant) and
    ``query()`` (rich BOOTHResponse).
    """

    _name = "BOOTHRetriever"


class BOOTHCurator(_NotImplementedStub):
    """Python API for listing, approving, rejecting and editing pending queries."""

    _name = "BOOTHCurator"


class BOOTHResponse:
    """Rich response object returned by BOOTHRetriever.query().

    Kept as a plain class with slots on the stub so smoke tests can construct
    it without triggering the NotImplementedError guard the other classes use.
    Will be replaced by a proper pydantic model during the orchestrator port.
    """

    __slots__ = (
        "success",
        "answer",
        "query_id",
        "similar_match",
        "high_risk",
        "declined",
        "cypher_used",
        "raw_data",
        "error_message",
        "tool_used",
        "pending_feedback",
    )

    def __init__(
        self,
        success: bool = False,
        answer: str = "",
        query_id: str | None = None,
        similar_match: bool = False,
        high_risk: bool = False,
        declined: bool = False,
        cypher_used: str | None = None,
        raw_data: object | None = None,
        error_message: str | None = None,
        tool_used: str | None = None,
        pending_feedback: bool = False,
    ) -> None:
        self.success = success
        self.answer = answer
        self.query_id = query_id
        self.similar_match = similar_match
        self.high_risk = high_risk
        self.declined = declined
        self.cypher_used = cypher_used
        self.raw_data = raw_data
        self.error_message = error_message
        self.tool_used = tool_used
        self.pending_feedback = pending_feedback


# init_schema is re-exported from .schema; see that module for the implementation.
