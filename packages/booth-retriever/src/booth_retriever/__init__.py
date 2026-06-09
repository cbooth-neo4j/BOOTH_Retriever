"""booth-retriever: a self-improving Neo4j retriever for neo4j-graphrag-python.

Public API:
    - BOOTHRetriever:     drop-in neo4j_graphrag Retriever subclass
    - BOOTHCurator:       Python API for listing and approving pending queries
    - RefinementAgent:    LLM-backed agent that produces FewShot templates
    - BOOTHResponse:      rich response object returned by BOOTHRetriever.query()
    - PendingQuery,
      QueryDetail,
      CuratorStats,
      RefinementResult:   data classes used by curator/agent APIs
    - init_schema:        idempotent DDL bootstrap for BOOTH's Neo4j schema

Typical quickstart:

    from neo4j import GraphDatabase
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    from neo4j_graphrag.llm.openai_llm import OpenAILLM
    from booth_retriever import (
        BOOTHRetriever, BOOTHCurator, RefinementAgent, init_schema,
    )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    init_schema(driver)

    retriever = BOOTHRetriever(
        driver=driver,
        embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    print(retriever.query("how many users are there?").answer)

    curator = BOOTHCurator(driver=driver)
    pending = curator.list_pending()
    if pending:
        agent = RefinementAgent(llm=OpenAILLM(model_name="gpt-4o"))
        curator.approve(pending[0].id, refinement_agent=agent)
"""

from __future__ import annotations

from .agents import RefinementAgent, Text2CypherAgent
from .curator import (
    ALL_STATUSES,
    PENDING_STATUSES,
    ApprovalResult,
    BOOTHCurator,
    CuratorStats,
    PendingQuery,
    QueryDetail,
    RefinementResult,
)
from .models import BOOTHResponse, Text2CypherAttempt
from .retriever import BOOTHRetriever
from .schema import SchemaInitResult, init_schema
from .verification import (
    CorrectionResult,
    VerificationResult,
    correct_cypher,
    verify_and_correct,
    verify_cypher,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "ALL_STATUSES",
    "ApprovalResult",
    "BOOTHRetriever",
    "BOOTHCurator",
    "BOOTHResponse",
    "CorrectionResult",
    "CuratorStats",
    "PENDING_STATUSES",
    "PendingQuery",
    "QueryDetail",
    "RefinementAgent",
    "RefinementResult",
    "SchemaInitResult",
    "Text2CypherAgent",
    "Text2CypherAttempt",
    "VerificationResult",
    "correct_cypher",
    "init_schema",
    "verify_and_correct",
    "verify_cypher",
]
