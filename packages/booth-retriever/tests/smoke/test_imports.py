"""Tier-1 smoke tests: the package imports and the public surface exists.

These tests must stay fast (< 1s total), pure-Python, and have zero external
dependencies on Neo4j or an LLM provider. If any of these fail, the package
is DOA and nothing else is worth running.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.smoke


def test_package_imports() -> None:
    """The top-level package imports cleanly."""
    mod = importlib.import_module("booth_retriever")
    assert mod.__version__


def test_public_api_exports_are_present() -> None:
    """Every symbol declared in __all__ is actually exported."""
    import booth_retriever

    for name in booth_retriever.__all__:
        assert hasattr(booth_retriever, name), f"booth_retriever.{name} missing"


def test_boothresponse_is_constructible() -> None:
    """BOOTHResponse is a plain data object and can be constructed."""
    from booth_retriever import BOOTHResponse

    r = BOOTHResponse(success=True, answer="hello", query_id="abc")
    assert r.success is True
    assert r.answer == "hello"
    assert r.query_id == "abc"


def test_remaining_stub_classes_raise_until_ported() -> None:
    """Stubs for un-ported classes fail loudly, not silently.

    BOOTHRetriever is now a real class so it's excluded; the instantiation
    test for it lives in tests/unit/test_retriever.py where we can pass a
    real driver fixture.
    """
    from booth_retriever import BOOTHCurator

    with pytest.raises(NotImplementedError, match="BOOTHCurator"):
        BOOTHCurator()  # type: ignore[call-arg]


def test_boothretriever_class_is_a_retriever_subclass() -> None:
    """BOOTHRetriever is a real neo4j-graphrag Retriever subclass."""
    from neo4j_graphrag.retrievers.base import Retriever

    from booth_retriever import BOOTHRetriever

    assert issubclass(BOOTHRetriever, Retriever)


def test_init_schema_is_a_function() -> None:
    """init_schema is importable and callable."""
    from booth_retriever import init_schema

    assert callable(init_schema)


def test_schema_init_result_is_exported() -> None:
    """SchemaInitResult is part of the public surface."""
    from booth_retriever import SchemaInitResult

    r = SchemaInitResult()
    assert r.created == []
    assert r.already_existed == []
