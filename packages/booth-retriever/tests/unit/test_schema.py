"""Tier-2 unit tests for booth_retriever.schema.

These tests exercise the pure logic in ``_build_schema_objects`` and the
idempotency / error paths of ``init_schema`` using a mocked ``neo4j.Driver``.
No real database is involved.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from booth_retriever.schema import (
    VECTOR_INDEX_NAME,
    SchemaInitResult,
    _build_schema_objects,
    init_schema,
)

pytestmark = pytest.mark.unit


# ---------- _build_schema_objects: pure logic -------------------------------


def test_schema_objects_cover_all_booth_labels() -> None:
    """Every BOOTH-owned label has at least one schema object managing it."""
    required_labels = {
        "Query",
        "UserQuestion",
        "FewShot",
        "Tool",
        "CypherAttempt",
        "Response",
    }
    labels_with_objects = {obj.label for obj in _build_schema_objects(1536)}
    missing = required_labels - labels_with_objects
    assert not missing, f"missing schema objects for labels: {missing}"


def test_schema_objects_all_have_if_not_exists() -> None:
    """Every DDL statement is idempotent via IF NOT EXISTS."""
    for obj in _build_schema_objects(1536):
        assert "IF NOT EXISTS" in obj.cypher, f"{obj.name}: missing IF NOT EXISTS"


def test_schema_object_names_are_unique() -> None:
    """No two schema objects share a name; we assert on them in tests."""
    names = [obj.name for obj in _build_schema_objects(1536)]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate schema object names: {duplicates}"


@pytest.mark.parametrize("dims", [512, 1024, 1536, 3072])
def test_vector_index_embeds_requested_dimensions(dims: int) -> None:
    """The vector index DDL uses the dimensions argument verbatim."""
    objects = _build_schema_objects(dims)
    vector_objs = [o for o in objects if o.kind == "vector_index"]
    assert len(vector_objs) == 1
    assert f"`vector.dimensions`: {dims}" in vector_objs[0].cypher


def test_vector_index_uses_cosine_similarity() -> None:
    """Cosine is the only similarity function we support."""
    objects = _build_schema_objects(1536)
    vector_objs = [o for o in objects if o.kind == "vector_index"]
    assert "`vector.similarity_function`: 'cosine'" in vector_objs[0].cypher


def test_vector_index_name_matches_module_constant() -> None:
    """The constant used by the retriever and the DDL name agree."""
    objects = _build_schema_objects(1536)
    vector_objs = [o for o in objects if o.kind == "vector_index"]
    assert vector_objs[0].name == VECTOR_INDEX_NAME


# ---------- init_schema: argument validation --------------------------------


def test_init_schema_rejects_zero_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        init_schema(driver=MagicMock(), embedding_dimensions=0)


def test_init_schema_rejects_negative_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        init_schema(driver=MagicMock(), embedding_dimensions=-1)


def test_init_schema_rejects_non_integer_dimensions() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        init_schema(driver=MagicMock(), embedding_dimensions="1536")  # type: ignore[arg-type]


# ---------- init_schema: idempotency and driver interaction ------------------


def _make_mock_driver(
    existing_constraints: list[str] | None = None,
    existing_indexes: list[str] | None = None,
) -> MagicMock:
    """Build a mock driver whose SHOW queries return the given rows.

    The mock's ``session().__enter__`` returns a session whose ``session.run``
    returns different row sets depending on which SHOW query was issued.
    ``session.run`` is a MagicMock so tests can assert the exact DDL that
    was executed.
    """
    existing_constraints = existing_constraints or []
    existing_indexes = existing_indexes or []

    def fake_run(cypher: str, *args, **kwargs):
        if cypher.startswith("SHOW CONSTRAINTS"):
            return iter([{"name": n} for n in existing_constraints])
        if cypher.startswith("SHOW INDEXES"):
            return iter([{"name": n} for n in existing_indexes])
        # A CREATE statement - return an empty result
        return iter([])

    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session.run.side_effect = fake_run

    driver = MagicMock()
    driver.session.return_value = session
    return driver


def test_init_schema_fresh_database_creates_every_object() -> None:
    """On a blank DB, every schema object ends up in ``created``."""
    driver = _make_mock_driver()

    result = init_schema(driver, embedding_dimensions=1536)

    assert isinstance(result, SchemaInitResult)
    assert result.already_existed == []
    assert result.created, "expected at least one object to be created"
    assert result.is_first_run
    assert result.embedding_dimensions == 1536


def test_init_schema_is_idempotent_on_rerun() -> None:
    """If SHOW returns all our objects, the second call creates nothing."""
    all_names = [obj.name for obj in _build_schema_objects(1536)]
    constraint_names = [
        obj.name for obj in _build_schema_objects(1536) if obj.kind == "constraint"
    ]
    index_names = [
        obj.name for obj in _build_schema_objects(1536) if obj.kind != "constraint"
    ]

    driver = _make_mock_driver(
        existing_constraints=constraint_names,
        existing_indexes=index_names,
    )

    result = init_schema(driver, embedding_dimensions=1536)

    assert result.created == []
    assert sorted(result.already_existed) == sorted(all_names)
    assert result.is_idempotent_rerun


def test_init_schema_only_creates_missing_objects() -> None:
    """Partial-state DBs get only the missing objects created."""
    driver = _make_mock_driver(
        existing_constraints=["query_id_unique"],
        existing_indexes=["query_embeddings"],
    )

    result = init_schema(driver, embedding_dimensions=1536)

    assert "query_id_unique" in result.already_existed
    assert "query_embeddings" in result.already_existed
    assert "query_id_unique" not in result.created
    assert "query_embeddings" not in result.created
    # Some other object must have been created
    assert result.created


def test_init_schema_passes_database_kwarg_to_session() -> None:
    """The optional ``database`` arg is forwarded to driver.session()."""
    driver = _make_mock_driver()

    init_schema(driver, embedding_dimensions=1536, database="booth")

    driver.session.assert_called_with(database="booth")


def test_init_schema_default_omits_database_kwarg() -> None:
    """When database=None we don't pass it; use the driver default."""
    driver = _make_mock_driver()

    init_schema(driver, embedding_dimensions=1536)

    driver.session.assert_called_with()


def test_init_schema_executes_vector_index_ddl_with_requested_dims() -> None:
    """The CREATE VECTOR INDEX statement carries the requested dimensions."""
    driver = _make_mock_driver()

    init_schema(driver, embedding_dimensions=3072)

    session = driver.session.return_value
    executed = [call.args[0] for call in session.run.call_args_list]
    vector_ddl = [c for c in executed if "CREATE VECTOR INDEX" in c]
    assert len(vector_ddl) == 1
    assert "`vector.dimensions`: 3072" in vector_ddl[0]
