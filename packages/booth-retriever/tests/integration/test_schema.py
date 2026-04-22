"""Tier-3 integration tests for init_schema against a real Neo4j.

Requires Docker to be running locally. Tests auto-skip if Docker isn't
available (see tests/integration/conftest.py).
"""

from __future__ import annotations

import pytest

from booth_retriever.schema import _build_schema_objects, init_schema

pytestmark = pytest.mark.integration


def _clear_schema(driver) -> None:
    """Drop every constraint and non-lookup index on the test database.

    Needed between tests because the Neo4j container is session-scoped and
    schema leaks across tests otherwise.
    """
    with driver.session() as session:
        constraints = [r["name"] for r in session.run("SHOW CONSTRAINTS YIELD name")]
        for name in constraints:
            session.run(f"DROP CONSTRAINT {name} IF EXISTS")

        indexes = [
            r["name"]
            for r in session.run(
                "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name"
            )
        ]
        for name in indexes:
            session.run(f"DROP INDEX {name} IF EXISTS")


def test_init_schema_creates_all_objects_on_fresh_database(neo4j_driver) -> None:
    """End-to-end: starting from empty, every BOOTH object gets created."""
    _clear_schema(neo4j_driver)

    result = init_schema(neo4j_driver, embedding_dimensions=1536)

    expected = {obj.name for obj in _build_schema_objects(1536)}
    assert set(result.created) == expected
    assert result.already_existed == []


def test_init_schema_is_idempotent_against_real_neo4j(neo4j_driver) -> None:
    """Running init_schema twice in a row on a fresh DB leaves state unchanged."""
    _clear_schema(neo4j_driver)
    init_schema(neo4j_driver, embedding_dimensions=1536)

    second = init_schema(neo4j_driver, embedding_dimensions=1536)

    expected = {obj.name for obj in _build_schema_objects(1536)}
    assert set(second.already_existed) == expected
    assert second.created == []
    assert second.is_idempotent_rerun


def test_init_schema_actually_creates_vector_index(neo4j_driver) -> None:
    """SHOW INDEXES reports our vector index with the requested dimensions."""
    _clear_schema(neo4j_driver)

    init_schema(neo4j_driver, embedding_dimensions=1536)

    with neo4j_driver.session() as session:
        rows = list(
            session.run(
                "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, "
                "properties, options "
                "WHERE name = 'query_embeddings' "
                "RETURN name, type, labelsOrTypes, properties, options"
            )
        )
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "VECTOR"
    assert row["labelsOrTypes"] == ["Query"]
    assert row["properties"] == ["embedding"]
    # Neo4j normalises the options dict; pick out what we care about.
    options = row["options"]
    index_config = options.get("indexConfig", {})
    assert int(index_config.get("vector.dimensions", 0)) == 1536


def test_init_schema_creates_all_expected_constraints(neo4j_driver) -> None:
    """Every label BOOTH owns has its uniqueness constraint present."""
    _clear_schema(neo4j_driver)
    init_schema(neo4j_driver, embedding_dimensions=1536)

    with neo4j_driver.session() as session:
        names = {r["name"] for r in session.run("SHOW CONSTRAINTS YIELD name RETURN name")}

    expected_constraints = {
        obj.name for obj in _build_schema_objects(1536) if obj.kind == "constraint"
    }
    assert expected_constraints <= names
