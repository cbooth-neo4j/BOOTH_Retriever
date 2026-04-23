"""Focused graph introspection for the banking-partner question.

Prints:
  * counts of the entity labels that look relevant
  * property keys actually used on each of those labels (with sample values)
  * relationship fan-out from BANKING_PARTNER / LEGAL_REQUIREMENT /
    COMPLIANCE_FRAMEWORK so we can pick a sensible traversal

Keeps output small (no embedding blobs) so it fits in a scrollback and a tool
response.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

TARGET_LABELS = [
    "LEGAL_REQUIREMENT",
    "BANKING_PARTNER",
    "COMPLIANCE_FRAMEWORK",
    "REQUEST_FOR_PROPOSAL",
    "EVALUATION_CRITERIA",
    "FINANCIAL_INSTITUTION",
    "CORPORATE_POLICY",
]

SHORT_PROP_CAP = 120


def _short(value):
    if isinstance(value, str) and len(value) > SHORT_PROP_CAP:
        return value[:SHORT_PROP_CAP] + "..."
    if isinstance(value, list) and len(value) > 4:
        return f"<list len={len(value)}>"
    return value


def _run(session, cypher: str, **params):
    return [dict(r) for r in session.run(cypher, **params)]


def main() -> int:
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE")

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_classifications=["UNRECOGNIZED"],
        )
    except TypeError:
        driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database=database) as s:
            print("=== Target label counts ===")
            for label in TARGET_LABELS:
                c = _run(s, f"MATCH (n:`{label}`) RETURN count(n) AS c")[0]["c"]
                print(f"  {label:<25s} {c}")

            print("\n=== Property keys in use per label (with sample) ===")
            for label in TARGET_LABELS:
                rows = _run(
                    s,
                    f"""
                    MATCH (n:`{label}`)
                    WITH n LIMIT 1
                    RETURN keys(n) AS props, properties(n) AS values
                    """,
                )
                if not rows:
                    print(f"  {label}: (no nodes)")
                    continue
                props = rows[0]["props"]
                values = rows[0]["values"]
                print(f"  {label}:")
                for k in sorted(props):
                    if k in {"embedding", "communities", "co_occurrences"}:
                        continue
                    print(f"      {k} = {_short(values.get(k))!r}")

            print("\n=== Relationship fan-out from BANKING_PARTNER ===")
            rels = _run(
                s,
                """
                MATCH (b:BANKING_PARTNER)-[r]-(m)
                RETURN type(r) AS rel,
                       CASE WHEN startNode(r) = b THEN '->' ELSE '<-' END AS dir,
                       labels(m) AS other_labels,
                       count(*) AS c
                ORDER BY c DESC
                LIMIT 40
                """,
            )
            for row in rels:
                print(f"  (BANKING_PARTNER) {row['dir']} [{row['rel']}] {row['other_labels']}  x{row['c']}")

            print("\n=== Relationship fan-out from LEGAL_REQUIREMENT ===")
            rels = _run(
                s,
                """
                MATCH (l:LEGAL_REQUIREMENT)-[r]-(m)
                RETURN type(r) AS rel,
                       CASE WHEN startNode(r) = l THEN '->' ELSE '<-' END AS dir,
                       labels(m) AS other_labels,
                       count(*) AS c
                ORDER BY c DESC
                LIMIT 40
                """,
            )
            for row in rels:
                print(f"  (LEGAL_REQUIREMENT) {row['dir']} [{row['rel']}] {row['other_labels']}  x{row['c']}")

            print("\n=== Relationship fan-out from COMPLIANCE_FRAMEWORK ===")
            rels = _run(
                s,
                """
                MATCH (c:COMPLIANCE_FRAMEWORK)-[r]-(m)
                RETURN type(r) AS rel,
                       CASE WHEN startNode(r) = c THEN '->' ELSE '<-' END AS dir,
                       labels(m) AS other_labels,
                       count(*) AS c
                ORDER BY c DESC
                LIMIT 40
                """,
            )
            for row in rels:
                print(f"  (COMPLIANCE_FRAMEWORK) {row['dir']} [{row['rel']}] {row['other_labels']}  x{row['c']}")

            print("\n=== Sample BANKING_PARTNER names ===")
            names = _run(
                s,
                """
                MATCH (b:BANKING_PARTNER)
                RETURN coalesce(b.name, b.human_readable_id, b.id) AS label
                ORDER BY label
                LIMIT 15
                """,
            )
            for row in names:
                print(f"  - {row['label']}")

            print("\n=== Sample LEGAL_REQUIREMENT summaries ===")
            lreqs = _run(
                s,
                """
                MATCH (l:LEGAL_REQUIREMENT)
                RETURN coalesce(l.name, l.human_readable_id, l.id) AS label,
                       coalesce(l.description, l.summary, l.original_description) AS detail
                LIMIT 10
                """,
            )
            for row in lreqs:
                print(f"  - {row['label']}: {_short(row['detail'])}")

            print("\n=== How do BANKING_PARTNER and LEGAL_REQUIREMENT connect? ===")
            paths = _run(
                s,
                """
                MATCH p = (b:BANKING_PARTNER)-[*1..3]-(l:LEGAL_REQUIREMENT)
                WITH [rel IN relationships(p) | type(rel)] AS rel_seq,
                     [n IN nodes(p) | labels(n)[0]] AS node_seq
                RETURN node_seq, rel_seq, count(*) AS c
                ORDER BY c DESC
                LIMIT 10
                """,
            )
            if not paths:
                print("  (no direct 1-3 hop path found)")
            for row in paths:
                print(f"  {row['node_seq']}  via  {row['rel_seq']}  x{row['c']}")
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
