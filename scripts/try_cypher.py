"""Sanity-check a candidate FewShot cypher template before curating it."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

CYPHER = """
MATCH (:BANKING_PARTNER)<-[:HAS_ENTITY]-(:Chunk)-[:HAS_ENTITY]->(req)
WHERE req.entity_type IN [
    'LEGAL_REQUIREMENT',
    'COMPLIANCE_FRAMEWORK',
    'CORPORATE_POLICY'
]
RETURN DISTINCT
    req.name AS requirement,
    req.description AS description,
    req.entity_type AS category
ORDER BY category, requirement
"""


def main() -> int:
    load_dotenv()
    try:
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
            notifications_disabled_classifications=["UNRECOGNIZED"],
        )
    except TypeError:
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        )

    try:
        with driver.session(database=os.environ.get("NEO4J_DATABASE")) as s:
            rows = [dict(r) for r in s.run(CYPHER)]
    finally:
        driver.close()

    print(f"Returned {len(rows)} row(s)\n")
    for i, r in enumerate(rows, 1):
        desc = r["description"] or ""
        if len(desc) > 140:
            desc = desc[:140] + "..."
        print(f"[{i}] ({r['category']}) {r['requirement']}")
        print(f"     {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
