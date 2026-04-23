"""Tier-2 unit tests for ``booth_retriever.verification``.

Pure-string logic, no fixtures needed.
"""

from __future__ import annotations

import pytest

from booth_retriever.verification import (
    correct_cypher,
    verify_and_correct,
    verify_cypher,
)

pytestmark = pytest.mark.unit


# ---------- verify_cypher: failures -----------------------------------------


@pytest.mark.parametrize("value", [None, "", "   ", "\n\t "])
def test_verify_flags_empty(value: str | None) -> None:
    result = verify_cypher(value)
    assert result.is_valid is False
    assert any("empty" in e.lower() for e in result.errors)


def test_verify_flags_no_cypher_keywords() -> None:
    result = verify_cypher("this is just prose, nothing to see here")
    assert result.is_valid is False
    assert any("keyword" in e.lower() for e in result.errors)


def test_verify_flags_unmatched_square_brackets() -> None:
    result = verify_cypher("MATCH (a)-[:REL]-(b RETURN a")
    assert result.is_valid is False
    assert any("parenthes" in e.lower() for e in result.errors) or any(
        "square" in e.lower() for e in result.errors
    )


def test_verify_flags_unmatched_parens() -> None:
    result = verify_cypher("MATCH (a RETURN a")
    assert result.is_valid is False
    assert any("parenthes" in e.lower() for e in result.errors)


def test_verify_flags_invalid_bidirectional_rel() -> None:
    result = verify_cypher("MATCH (a)<-[:KNOWS]->(b) RETURN a, b")
    assert result.is_valid is False
    assert any("bidirectional" in e.lower() for e in result.errors)


def test_verify_flags_invalid_reverse_rel() -> None:
    result = verify_cypher("MATCH (a)>-[:KNOWS]-<(b) RETURN a, b")
    assert result.is_valid is False
    assert any("reverse" in e.lower() for e in result.errors)


def test_verify_flags_match_without_return() -> None:
    result = verify_cypher("MATCH (a)")
    assert result.is_valid is False
    assert any("RETURN" in e for e in result.errors)


def test_verify_flags_hanging_where() -> None:
    result = verify_cypher("MATCH (a) WHERE RETURN a")
    assert result.is_valid is False
    assert any("hanging WHERE" in e for e in result.errors)


def test_verify_flags_trailing_where_with_no_condition() -> None:
    result = verify_cypher("MATCH (a) WHERE ")
    assert result.is_valid is False
    assert any("hanging WHERE" in e for e in result.errors)


# ---------- verify_cypher: passes -------------------------------------------


def test_verify_passes_minimal_return() -> None:
    result = verify_cypher("RETURN 1 AS n")
    assert result.is_valid is True


def test_verify_passes_directed_relationship() -> None:
    result = verify_cypher(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"
    )
    assert result.is_valid is True


def test_verify_passes_undirected_relationship() -> None:
    result = verify_cypher(
        "MATCH (a:Person)-[:KNOWS]-(b:Person) RETURN a.name, b.name"
    )
    assert result.is_valid is True


def test_verify_passes_reverse_directed() -> None:
    """<-[:REL]- (left-directed) is legal; only <-[:REL]-> is a bug."""
    result = verify_cypher("MATCH (a)<-[:KNOWS]-(b) RETURN a, b")
    assert result.is_valid is True


def test_verify_passes_delete_without_return() -> None:
    result = verify_cypher("MATCH (a:Stale) DELETE a")
    assert result.is_valid is True


def test_verify_passes_merge_without_return() -> None:
    result = verify_cypher(
        "MERGE (a:Person {id: $id}) SET a.updated = datetime()"
    )
    assert result.is_valid is True


def test_verify_passes_valid_where() -> None:
    result = verify_cypher(
        "MATCH (a:Person) WHERE a.age > 18 RETURN a.name"
    )
    assert result.is_valid is True


def test_verify_long_literal_surfaces_as_warning_not_error() -> None:
    result = verify_cypher(
        "MATCH (a) WHERE a.text = "
        "'this is a very long hardcoded string literal right here' "
        "RETURN a"
    )
    assert result.is_valid is True
    assert any("parameters" in w.lower() for w in result.warnings)


def test_verify_reports_multiple_errors() -> None:
    """Verification keeps going after the first error; tests can rely on it."""
    result = verify_cypher("prose no cypher (with unmatched paren")
    assert result.is_valid is False
    assert len(result.errors) >= 2


# ---------- correct_cypher ---------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_correct_noop_on_empty(value: str | None) -> None:
    result = correct_cypher(value)
    assert result.was_corrected is False
    assert result.corrected_cypher == (value or "")


def test_correct_strips_markdown_fences() -> None:
    result = correct_cypher("```cypher\nMATCH (a) RETURN a\n```")
    assert result.was_corrected is True
    assert result.corrected_cypher == "MATCH (a) RETURN a"
    assert "markdown" in result.corrections[0]


def test_correct_fixes_bidirectional_rel() -> None:
    result = correct_cypher("MATCH (a)<-[:KNOWS]->(b) RETURN a")
    assert result.was_corrected is True
    assert "<-[:KNOWS]->" not in result.corrected_cypher
    assert "-[:KNOWS]-" in result.corrected_cypher


def test_correct_fixes_reverse_rel() -> None:
    result = correct_cypher("MATCH (a)>-[:KNOWS]-<(b) RETURN a")
    assert result.was_corrected is True
    assert ">-[:KNOWS]-<" not in result.corrected_cypher
    assert "-[:KNOWS]-" in result.corrected_cypher


def test_correct_adds_missing_colon() -> None:
    result = correct_cypher("MATCH (a)-[KNOWS]-(b) RETURN a")
    assert result.was_corrected is True
    assert "-[:KNOWS]-" in result.corrected_cypher


def test_correct_adds_missing_relationship_brackets() -> None:
    result = correct_cypher("MATCH (a)-(KNOWS)-(b) RETURN a")
    assert result.was_corrected is True
    assert "-[:KNOWS]-" in result.corrected_cypher


def test_correct_trims_whitespace() -> None:
    result = correct_cypher("   MATCH (a) RETURN a   \n")
    assert result.was_corrected is True
    assert result.corrected_cypher == "MATCH (a) RETURN a"


def test_correct_leaves_valid_cypher_unchanged() -> None:
    result = correct_cypher("MATCH (a)-[:KNOWS]->(b) RETURN a.name")
    assert result.was_corrected is False
    assert result.corrections == []


# ---------- verify_and_correct ----------------------------------------------


def test_verify_and_correct_roundtrips_fixable_query() -> None:
    verif, corrected = verify_and_correct(
        "```cypher\nMATCH (a)-[KNOWS]-(b) RETURN a\n```"
    )
    assert verif.is_valid is True
    assert "-[:KNOWS]-" in corrected
    assert "```" not in corrected


def test_verify_and_correct_still_flags_unfixable() -> None:
    """Correction can't add a missing RETURN; verification still fails."""
    verif, _ = verify_and_correct("MATCH (a)")
    assert verif.is_valid is False
