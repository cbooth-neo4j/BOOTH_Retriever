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

from .curator import (
    ApprovalResult,
    BOOTHCurator,
    CuratorStats,
    PendingQuery,
    QueryDetail,
)
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
    "ApprovalResult",
    "CuratorStats",
    "PendingQuery",
    "QueryDetail",
]
