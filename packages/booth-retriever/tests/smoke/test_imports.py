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


def test_boothretriever_class_is_a_retriever_subclass() -> None:
    """BOOTHRetriever is a real neo4j-graphrag Retriever subclass."""
    from neo4j_graphrag.retrievers.base import Retriever

    from booth_retriever import BOOTHRetriever

    assert issubclass(BOOTHRetriever, Retriever)


def test_boothcurator_class_is_constructable_with_mock_driver() -> None:
    """BOOTHCurator is now a real class; smoke-check it accepts a driver arg."""
    from unittest.mock import MagicMock

    from booth_retriever import BOOTHCurator

    curator = BOOTHCurator(driver=MagicMock(), database="foo")
    assert curator.database == "foo"
    for method in (
        "list_pending",
        "list_by_status",
        "get",
        "stats",
        "approve",
        "reject",
        "edit_fewshot",
        "submit_feedback",
    ):
        assert hasattr(curator, method), f"BOOTHCurator missing method {method!r}"


def test_refinement_agent_is_constructable_with_mock_llm() -> None:
    """RefinementAgent accepts any duck-typed LLM interface."""
    from unittest.mock import MagicMock

    from booth_retriever import RefinementAgent

    agent = RefinementAgent(llm=MagicMock())
    assert hasattr(agent, "refine")


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
