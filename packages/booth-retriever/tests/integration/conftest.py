"""Shared fixtures for Tier-3 integration tests.

Integration tests spin up an ephemeral Neo4j container via ``testcontainers``.
They are automatically skipped if Docker is not available on the host so that
``pytest`` without flags still works on a fresh laptop.
"""

from __future__ import annotations

import pytest

try:
    from testcontainers.neo4j import Neo4jContainer
    _TESTCONTAINERS_AVAILABLE = True
except ImportError:
    _TESTCONTAINERS_AVAILABLE = False


@pytest.fixture(scope="session")
def neo4j_container():
    """Session-scoped Neo4j container. Skips the whole session if Docker isn't available."""
    if not _TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not installed; install with pip install -e '.[dev]'")

    try:
        with Neo4jContainer("neo4j:5.20") as neo4j:
            yield neo4j
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not start Neo4j container (is Docker running?): {exc}")


@pytest.fixture
def neo4j_driver(neo4j_container):
    """A fresh Neo4j driver bound to the session-scoped container."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        neo4j_container.get_connection_url(),
        auth=("neo4j", neo4j_container.NEO4J_ADMIN_PASSWORD),
    )
    try:
        yield driver
    finally:
        driver.close()
