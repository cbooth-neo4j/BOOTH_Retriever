"""booth-retriever: a self-improving Neo4j retriever for neo4j-graphrag-python.

Public API:
    - BOOTHRetriever:   drop-in neo4j_graphrag Retriever subclass
    - BOOTHCurator:     Python API for listing and approving pending queries
    - BOOTHResponse:    rich response object returned by BOOTHRetriever.query()
    - init_schema:      idempotent DDL bootstrap for BOOTH's own Neo4j schema

See the plan at ``.cursor/plans/booth-retriever_pip_package_*.plan.md`` in
the parent repository for the roadmap. BOOTHCurator is a stub in MV1;
BOOTHRetriever is feature-complete for the cache-hit path and routes cache
misses to the curation queue.
"""

from __future__ import annotations

from .models import BOOTHResponse
from .retriever import BOOTHRetriever
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


class BOOTHCurator(_NotImplementedStub):
    """Python API for listing, approving, rejecting and editing pending queries.

    Stub in MV1. Will be implemented in the curator port.
    """

    _name = "BOOTHCurator"
