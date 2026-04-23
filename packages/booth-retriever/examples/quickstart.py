"""Minimal end-to-end BOOTH Retriever quickstart.

Usage:

    python packages/booth-retriever/examples/quickstart.py "your question here"

Requires environment variables (read from a ``.env`` in the project root or
the current shell):

    NEO4J_URI        e.g. bolt://localhost:7687
    NEO4J_USER       defaults to "neo4j"
    NEO4J_PASSWORD   required
    NEO4J_DATABASE   optional, for multi-database setups
    OPENAI_API_KEY   required for the OpenAI embedder

This is deliberately small — it exists so users can sanity-check an install
without fighting shell heredoc quoting rules. For richer patterns (GraphRAG
pipelines, curation loops) see the streamlit app next to this file.

Note: the driver is built with ``notifications_disabled_classifications=
["UNRECOGNIZED"]`` to silence the harmless "label/property/relationship not
found" warnings that the Neo4j server emits against a fresh BOOTH install.
Remove that kwarg if you want to see them in your own code.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

from booth_retriever import BOOTHRetriever


def main(question: str) -> int:
    load_dotenv()

    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not password:
        print(
            "Error: NEO4J_URI and NEO4J_PASSWORD must be set (in .env or environment).",
            file=sys.stderr,
        )
        return 2

    user = os.environ.get("NEO4J_USER", "neo4j")
    database = os.environ.get("NEO4J_DATABASE")
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_classifications=["UNRECOGNIZED"],
        )
    except TypeError:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        retriever = BOOTHRetriever(
            driver=driver,
            embedder=OpenAIEmbeddings(model=embedding_model),
            neo4j_database=database,
        )
        response = retriever.query(question)
    finally:
        driver.close()

    print("Answer:      ", response.answer or "(no answer)")
    print("query_id:    ", response.query_id)
    print("similar_hit: ", response.similar_match)
    print("declined:    ", response.declined)
    print("tool_used:   ", response.tool_used)
    if response.cypher_used:
        print("cypher_used: ", response.cypher_used)
    if response.error_message:
        print("error:       ", response.error_message)

    return 0 if response.success else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python quickstart.py "<question>"', file=sys.stderr)
        sys.exit(2)
    sys.exit(main(" ".join(sys.argv[1:])))
