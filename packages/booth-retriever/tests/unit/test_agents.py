"""Tier-2 unit tests for ``booth_retriever.agents.RefinementAgent``.

Uses a canned-response fake LLM (no network). Exercises:

    - Successful single-shot refinement with a clean JSON response
    - Tolerance of ```json fenced output
    - Tolerance of surrounding prose
    - Parameter / cypher consistency check (mismatch -> error)
    - Malformed JSON -> error
    - LLM-side failure (success: false) -> error
    - Exception from the LLM interface -> error (not raised)
    - Empty question -> error
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from booth_retriever.agents import RefinementAgent
from booth_retriever.curator import RefinementResult

pytestmark = pytest.mark.unit


@dataclass
class _FakeLLMResponse:
    """Mirrors neo4j_graphrag.llm.types.LLMResponse enough for our agent."""

    content: str


class _FakeLLM:
    """Returns ``canned`` from .invoke(). Optionally raises instead."""

    def __init__(self, canned: str = "", exc: Exception | None = None) -> None:
        self.canned = canned
        self.exc = exc
        self.calls: list[dict] = []

    def invoke(self, **kwargs) -> _FakeLLMResponse:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return _FakeLLMResponse(content=self.canned)


def test_refine_happy_path() -> None:
    llm = _FakeLLM(
        canned=(
            '{"success": true, '
            '"refined_cypher": "MATCH (p:Person {name: $person_name}) RETURN p.role", '
            '"parameters": ["person_name"], '
            '"category": "PERSON_ATTRIBUTE"}'
        )
    )
    agent = RefinementAgent(llm=llm)

    result = agent.refine(
        original_question="What role did Bob hold?",
        raw_cypher="MATCH (p:Person {name: 'Bob'}) RETURN p.role",
    )

    assert isinstance(result, RefinementResult)
    assert result.success is True
    assert result.refined_cypher == "MATCH (p:Person {name: $person_name}) RETURN p.role"
    assert result.parameters == ["person_name"]
    assert result.category == "PERSON_ATTRIBUTE"
    assert result.error is None

    # The prompt includes the raw cypher
    assert len(llm.calls) == 1
    assert "Bob" in llm.calls[0]["input"]


def test_refine_without_raw_cypher_uses_placeholder() -> None:
    llm = _FakeLLM(
        canned=(
            '{"success": true, "refined_cypher": "RETURN $n AS n", '
            '"parameters": ["n"], "category": "FACTUAL"}'
        )
    )
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="how many?")

    assert result.success is True
    assert "(none - write one from scratch)" in llm.calls[0]["input"]


def test_refine_accepts_json_fenced_output() -> None:
    """LLMs often wrap JSON in ```json fences. The agent should strip them."""
    llm = _FakeLLM(
        canned=(
            "Here's the result:\n```json\n"
            '{"success": true, "refined_cypher": "RETURN 1 AS n", '
            '"parameters": [], "category": "FACTUAL"}\n'
            "```\nHope that helps!"
        )
    )
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="count")
    assert result.success is True
    assert result.refined_cypher == "RETURN 1 AS n"
    assert result.parameters == []


def test_refine_rejects_malformed_json() -> None:
    llm = _FakeLLM(canned="this is not JSON at all")
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False
    assert result.error is not None
    assert "parse" in result.error.lower() or "json" in result.error.lower()


def test_refine_surfaces_llm_reported_failure() -> None:
    llm = _FakeLLM(canned='{"success": false, "error": "question too ambiguous"}')
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False
    assert "ambiguous" in (result.error or "")


def test_refine_rejects_missing_refined_cypher() -> None:
    llm = _FakeLLM(canned='{"success": true, "parameters": []}')
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False
    assert "refined_cypher" in (result.error or "")


def test_refine_rejects_parameter_mismatch_declared_extra() -> None:
    """Declared parameter that doesn't appear in the cypher is a bug."""
    llm = _FakeLLM(
        canned=(
            '{"success": true, "refined_cypher": "RETURN 1", '
            '"parameters": ["unused"], "category": "FACTUAL"}'
        )
    )
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False
    assert "parameters mismatch" in (result.error or "")


def test_refine_rejects_parameter_mismatch_cypher_extra() -> None:
    """$param in the cypher not listed in parameters is also a bug."""
    llm = _FakeLLM(
        canned=(
            '{"success": true, "refined_cypher": "RETURN $unlisted AS x", '
            '"parameters": [], "category": "FACTUAL"}'
        )
    )
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False
    assert "parameters mismatch" in (result.error or "")


def test_refine_rejects_non_string_parameters() -> None:
    llm = _FakeLLM(
        canned=(
            '{"success": true, "refined_cypher": "RETURN 1", '
            '"parameters": [1, 2, 3], "category": "FACTUAL"}'
        )
    )
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False


def test_refine_wraps_llm_exceptions() -> None:
    llm = _FakeLLM(exc=RuntimeError("rate limit exceeded"))
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="x")
    assert result.success is False
    assert "rate limit" in (result.error or "")


def test_refine_rejects_empty_question() -> None:
    llm = MagicMock()
    agent = RefinementAgent(llm=llm)

    result = agent.refine(original_question="   \n  ")
    assert result.success is False
    # And the LLM was never called
    llm.invoke.assert_not_called()
