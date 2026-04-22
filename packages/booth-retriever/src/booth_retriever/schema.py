"""Neo4j schema bootstrap for BOOTH.

BOOTH owns a handful of node labels (``Query``, ``UserQuestion``, ``FewShot``,
``Tool``, ``CypherAttempt``, ``Response``) on top of the customer's existing
knowledge graph. ``init_schema`` creates the indexes and constraints needed
for those labels.

All DDL statements use ``IF NOT EXISTS`` so calling this function is
idempotent: running it a second time is a no-op and returns
``already_existed=True`` for each object. Integration tests rely on that.

Design notes:
    - Embedding dimensions are a constructor arg. Different embedders use
      different sizes (OpenAI text-embedding-3-small = 1536,
      text-embedding-3-large = 3072, Cohere v3 = 1024, etc.). We default
      to 1536 to match the existing BOOTH deployment but customers MUST
      override this if they use a different embedder.
    - Vector similarity function is fixed to ``cosine``; swapping it would
      require reindexing.
    - The ``database`` kwarg targets a specific Neo4j database for
      multi-database Aura/Enterprise setups. None uses the driver default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver


VECTOR_INDEX_NAME = "query_embeddings"
VECTOR_INDEX_LABEL = "Query"
VECTOR_INDEX_PROPERTY = "embedding"
VECTOR_SIMILARITY_FUNCTION = "cosine"


@dataclass
class SchemaObject:
    """A single DDL object (constraint or index) that init_schema manages."""

    kind: str          # "constraint" | "index" | "vector_index"
    name: str          # Neo4j object name for SHOW CONSTRAINTS / SHOW INDEXES
    label: str         # node label this object is attached to
    cypher: str        # the DDL statement (idempotent via IF NOT EXISTS)


@dataclass
class SchemaInitResult:
    """Structured report of what init_schema did.

    Returned so callers (and tests) can assert exactly which objects were
    freshly created vs. already present. An idempotent second call should
    report every object as ``already_existed``.
    """

    created: list[str] = field(default_factory=list)
    already_existed: list[str] = field(default_factory=list)
    embedding_dimensions: int = 0

    @property
    def is_first_run(self) -> bool:
        return len(self.created) > 0 and len(self.already_existed) == 0

    @property
    def is_idempotent_rerun(self) -> bool:
        return len(self.created) == 0 and len(self.already_existed) > 0


def _build_schema_objects(embedding_dimensions: int) -> list[SchemaObject]:
    """Return the full list of BOOTH-owned schema objects to bootstrap.

    Kept as a pure function so it can be unit-tested without a live driver.
    """
    objects: list[SchemaObject] = [
        SchemaObject(
            kind="constraint",
            name="query_id_unique",
            label="Query",
            cypher=(
                "CREATE CONSTRAINT query_id_unique IF NOT EXISTS "
                "FOR (q:Query) REQUIRE q.id IS UNIQUE"
            ),
        ),
        SchemaObject(
            kind="constraint",
            name="user_question_id_unique",
            label="UserQuestion",
            cypher=(
                "CREATE CONSTRAINT user_question_id_unique IF NOT EXISTS "
                "FOR (u:UserQuestion) REQUIRE u.id IS UNIQUE"
            ),
        ),
        SchemaObject(
            kind="constraint",
            name="fewshot_id_unique",
            label="FewShot",
            cypher=(
                "CREATE CONSTRAINT fewshot_id_unique IF NOT EXISTS "
                "FOR (f:FewShot) REQUIRE f.id IS UNIQUE"
            ),
        ),
        SchemaObject(
            kind="constraint",
            name="tool_name_unique",
            label="Tool",
            cypher=(
                "CREATE CONSTRAINT tool_name_unique IF NOT EXISTS "
                "FOR (t:Tool) REQUIRE t.name IS UNIQUE"
            ),
        ),
        SchemaObject(
            kind="constraint",
            name="cypher_attempt_id_unique",
            label="CypherAttempt",
            cypher=(
                "CREATE CONSTRAINT cypher_attempt_id_unique IF NOT EXISTS "
                "FOR (c:CypherAttempt) REQUIRE c.id IS UNIQUE"
            ),
        ),
        SchemaObject(
            kind="constraint",
            name="response_id_unique",
            label="Response",
            cypher=(
                "CREATE CONSTRAINT response_id_unique IF NOT EXISTS "
                "FOR (r:Response) REQUIRE r.id IS UNIQUE"
            ),
        ),
        SchemaObject(
            kind="index",
            name="query_status_idx",
            label="Query",
            cypher=(
                "CREATE INDEX query_status_idx IF NOT EXISTS "
                "FOR (q:Query) ON (q.status)"
            ),
        ),
        SchemaObject(
            kind="index",
            name="query_timestamp_idx",
            label="Query",
            cypher=(
                "CREATE INDEX query_timestamp_idx IF NOT EXISTS "
                "FOR (q:Query) ON (q.timestamp)"
            ),
        ),
        SchemaObject(
            kind="vector_index",
            name=VECTOR_INDEX_NAME,
            label=VECTOR_INDEX_LABEL,
            cypher=(
                f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
                f"FOR (q:{VECTOR_INDEX_LABEL}) ON (q.{VECTOR_INDEX_PROPERTY}) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {embedding_dimensions}, "
                f"`vector.similarity_function`: '{VECTOR_SIMILARITY_FUNCTION}'"
                "}}"
            ),
        ),
    ]
    return objects


def _existing_object_names(session, kind: str) -> set[str]:
    """Return the set of Neo4j-managed objects of the given kind that exist."""
    if kind == "constraint":
        result = session.run("SHOW CONSTRAINTS YIELD name RETURN name")
    else:
        # Both "index" and "vector_index" live in SHOW INDEXES
        result = session.run("SHOW INDEXES YIELD name RETURN name")
    return {record["name"] for record in result}


def init_schema(
    driver: Driver,
    *,
    embedding_dimensions: int = 1536,
    database: str | None = None,
) -> SchemaInitResult:
    """Idempotently create BOOTH's Neo4j schema (constraints + indexes).

    Args:
        driver: An open ``neo4j.Driver``. The caller retains ownership and
            is responsible for closing it.
        embedding_dimensions: Vector dimensionality of the embedder you plan
            to use. Must match your embedder's output size; the vector index
            cannot be resized without a rebuild. Common values: OpenAI
            text-embedding-3-small = 1536, text-embedding-3-large = 3072,
            Cohere embed-english-v3 = 1024.
        database: Optional Neo4j database name for multi-database setups.
            Defaults to the driver's default database.

    Returns:
        SchemaInitResult listing which objects were freshly created and
        which already existed. An idempotent second call should return
        ``is_idempotent_rerun == True``.

    Raises:
        ValueError: If ``embedding_dimensions`` is not a positive integer.
        neo4j.exceptions.ClientError: If the connected Neo4j version does
            not support vector indexes (requires 5.11+).
    """
    if not isinstance(embedding_dimensions, int) or embedding_dimensions <= 0:
        raise ValueError(
            f"embedding_dimensions must be a positive integer, got "
            f"{embedding_dimensions!r}"
        )

    objects = _build_schema_objects(embedding_dimensions)
    result = SchemaInitResult(embedding_dimensions=embedding_dimensions)

    session_kwargs: dict = {}
    if database is not None:
        session_kwargs["database"] = database

    with driver.session(**session_kwargs) as session:
        existing_constraints = _existing_object_names(session, "constraint")
        existing_indexes = _existing_object_names(session, "index")

        for obj in objects:
            pre_existing = (
                existing_constraints if obj.kind == "constraint" else existing_indexes
            )
            if obj.name in pre_existing:
                result.already_existed.append(obj.name)
                continue

            # IF NOT EXISTS makes this race-safe even if another process
            # created the object between our SHOW and our CREATE.
            session.run(obj.cypher)
            result.created.append(obj.name)

    return result
