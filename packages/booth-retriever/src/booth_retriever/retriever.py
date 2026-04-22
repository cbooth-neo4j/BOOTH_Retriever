"""``BOOTHRetriever`` — the drop-in neo4j-graphrag Retriever.

Wraps ``BOOTHOrchestrator`` in the contract expected by
``neo4j_graphrag.retrievers.base.Retriever``. Exposes both:

    - ``search(query_text)``: spec-compliant. Returns a
      ``neo4j_graphrag.types.RetrieverResult``. Use this when plugging BOOTH
      into a ``neo4j_graphrag`` ``RAG`` / ``GraphRAG`` pipeline that expects
      the standard retriever interface.

    - ``query(query_text, is_high_risk=False)``: the rich path. Returns a
      ``BOOTHResponse`` with extra fields (query_id, cypher_used, declined,
      tool_used, etc.) needed by applications that drive BOOTH directly, such
      as the reference Streamlit app.

Both methods delegate to the same orchestrator; ``search()`` is a thin
adapter that translates the response into neo4j-graphrag's types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import neo4j
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem

from .models import BOOTHResponse
from .orchestrator import BOOTHOrchestrator
from .schema import VECTOR_INDEX_NAME

if TYPE_CHECKING:
    from neo4j import Driver
    from neo4j_graphrag.embeddings import Embedder


class BOOTHRetriever(Retriever):
    """Self-improving Neo4j retriever with a similarity cache of approved queries.

    Typical usage:

        >>> from neo4j import GraphDatabase
        >>> from neo4j_graphrag.embeddings import OpenAIEmbeddings
        >>> from booth_retriever import BOOTHRetriever, init_schema
        >>> driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "pw"))
        >>> init_schema(driver)
        >>> retriever = BOOTHRetriever(
        ...     driver=driver,
        ...     embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
        ... )
        >>> response = retriever.query("How many users are in the system?")
        >>> print(response.answer, response.query_id)

    The first time any given question is asked it will be routed to the
    curation queue (no agent fallback in MV1). Once a curator approves it
    and links a FewShot Cypher template, subsequent similar questions hit
    the cache and execute the approved Cypher directly.

    Args:
        driver: An open ``neo4j.Driver``.
        embedder: Any object implementing ``embed_query(text) -> list[float]``;
            typically a ``neo4j_graphrag.embeddings.Embedder`` subclass.
        similarity_threshold: Minimum cosine similarity for a cache hit.
            Defaults to 0.90.
        neo4j_database: Optional database name for multi-database setups.
        vector_index_name: Name of the vector index on ``Query`` nodes.
            Defaults to the one created by ``init_schema``.
    """

    # Set on the Retriever base class as an ``index_name`` attribute;
    # various neo4j-graphrag helpers refer to it.
    index_name: str

    def __init__(
        self,
        driver: Driver,
        embedder: Embedder,
        *,
        similarity_threshold: float = 0.90,
        neo4j_database: str | None = None,
        vector_index_name: str = VECTOR_INDEX_NAME,
    ) -> None:
        super().__init__(driver=driver, neo4j_database=neo4j_database)
        self.index_name = vector_index_name
        self._orchestrator = BOOTHOrchestrator(
            driver=self.driver,
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            vector_index_name=vector_index_name,
            database=neo4j_database,
        )

    # ------------------------------------------------------------------
    # Rich API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        *,
        is_high_risk: bool = False,
    ) -> BOOTHResponse:
        """Run the full BOOTH flow and return a rich ``BOOTHResponse``."""
        return self._orchestrator.process(query_text, is_high_risk=is_high_risk)

    # ------------------------------------------------------------------
    # neo4j-graphrag Retriever contract
    # ------------------------------------------------------------------

    def get_search_results(
        self,
        query_text: str,
        *,
        is_high_risk: bool = False,
    ) -> RawSearchResult:
        """neo4j-graphrag hook. Runs BOOTH and packages rows + metadata.

        The metadata dict carries every ``BOOTHResponse`` field so callers
        using the standard ``search()`` API still have access to ``query_id``,
        ``cypher_used``, ``declined``, etc. for auditing and curation.
        """
        response = self._orchestrator.process(query_text, is_high_risk=is_high_risk)

        records: list[neo4j.Record] = []
        if isinstance(response.raw_data, list):
            records = [
                neo4j.Record(dict(row)) if not isinstance(row, neo4j.Record) else row
                for row in response.raw_data
            ]

        metadata = {
            "success": response.success,
            "answer": response.answer,
            "query_id": response.query_id,
            "similar_match": response.similar_match,
            "high_risk": response.high_risk,
            "declined": response.declined,
            "cypher_used": response.cypher_used,
            "error_message": response.error_message,
            "tool_used": response.tool_used,
        }
        return RawSearchResult(records=records, metadata=metadata)

    def get_result_formatter(self):
        """Format each row as a RetrieverResultItem.

        Our rows are ``dict``-shaped (not graph nodes) because the FewShot
        Cypher can ``RETURN`` arbitrary projections. We stringify them here
        so they fit the ``RetrieverResultItem.content: str`` contract, and
        pass through the full dict via ``metadata`` for downstream use.
        """

        def _format(record) -> RetrieverResultItem:
            row = dict(record) if not isinstance(record, dict) else record
            return RetrieverResultItem(content=str(row), metadata={"row": row})

        return _format


__all__ = ["BOOTHRetriever", "BOOTHResponse", "RetrieverResult"]
