"""Pure-logic Cypher verification and correction.

No Neo4j connection required — these checks operate on the query string.
Three things live in this module:

    - ``verify_cypher(cypher)``     -> ``VerificationResult``
      Rule-based checks: empty/None, missing Cypher keyword, unbalanced
      brackets, invalid bidirectional relationship syntax, hanging WHERE.

    - ``correct_cypher(cypher)``    -> ``CorrectionResult``
      Rule-based fixes for the errors above (strip markdown fences,
      collapse ``<-[:REL]->`` to undirected, add missing relationship
      colons).

    - ``verify_and_correct(cypher)`` -> ``(VerificationResult, str)``
      Convenience helper: apply corrections, re-verify, return the final
      verification result plus the corrected text.

BOOTHCurator uses ``verify_cypher`` in ``approve()`` so hand-written or
LLM-generated templates are checked before being stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CYPHER_KEYWORDS = (
    "MATCH",
    "CREATE",
    "MERGE",
    "DELETE",
    "SET",
    "RETURN",
    "WITH",
    "UNWIND",
    "CALL",
)

_HANGING_WHERE_FOLLOWERS = frozenset({"RETURN", "WITH", "ORDER", "LIMIT", "MATCH"})


@dataclass
class VerificationResult:
    """Rule-based verification outcome for a single Cypher string.

    Attributes:
        is_valid: True iff no rule fired. When False, ``errors`` lists every
            rule that flagged the query. An empty ``errors`` list on a
            False result would be a bug.
        errors: Short human-readable error messages, in the order they
            were discovered. One per rule that fired.
        warnings: Soft signals (e.g. "query has no LIMIT") that are not
            failures but may be worth surfacing.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    """Result of attempting rule-based corrections.

    ``corrected_cypher`` is always populated (equals ``original`` if no
    fixes applied). ``corrections`` lists the fixes applied in order.
    """

    corrected_cypher: str
    original_cypher: str
    corrections: list[str] = field(default_factory=list)

    @property
    def was_corrected(self) -> bool:
        return self.corrected_cypher != self.original_cypher


# ---------- Verification -----------------------------------------------------


def verify_cypher(cypher: str | None) -> VerificationResult:
    """Run all rule-based checks and return a VerificationResult.

    Checks, in order:
        1. Empty / None query.
        2. No Cypher keyword present at all.
        3. Unmatched square brackets ``[`` / ``]``.
        4. Unmatched parentheses ``(`` / ``)``.
        5. Invalid bidirectional relationship ``<-[:REL]->``.
        6. Invalid reverse relationship ``>-[:REL]-<``.
        7. MATCH without RETURN or writing clause.
        8. Hanging WHERE clause (followed immediately by RETURN/WITH/etc).

    Never raises; all findings come back as ``errors``.
    """
    if cypher is None or not cypher.strip():
        return VerificationResult(is_valid=False, errors=["empty Cypher query"])

    errors: list[str] = []
    warnings: list[str] = []

    upper = cypher.upper()
    if not any(kw in upper for kw in _CYPHER_KEYWORDS):
        errors.append(f"no Cypher keyword found (expected one of {_CYPHER_KEYWORDS})")

    if cypher.count("[") != cypher.count("]"):
        errors.append("unmatched square brackets")

    if cypher.count("(") != cypher.count(")"):
        errors.append("unmatched parentheses")

    for match in re.findall(r"<-\[[^\[\]]*\]->", cypher):
        errors.append(f"invalid bidirectional relationship: {match}")
    for match in re.findall(r">-\[[^\[\]]*\]-<", cypher):
        errors.append(f"invalid reverse relationship: {match}")

    if (
        "MATCH" in upper
        and "RETURN" not in upper
        and "DELETE" not in upper
        and "SET" not in upper
        and "CREATE" not in upper
        and "MERGE" not in upper
    ):
        errors.append("MATCH clause without RETURN or writing clause")

    for part in re.split(r"\bWHERE\b", cypher, flags=re.IGNORECASE)[1:]:
        tokens = part.strip().split()
        if not tokens:
            errors.append("hanging WHERE clause (no condition)")
            continue
        first = tokens[0].upper().rstrip(",()")
        if first in _HANGING_WHERE_FOLLOWERS:
            errors.append(f"hanging WHERE clause: followed immediately by {first}")

    # Soft warnings: unparameterised large queries with hardcoded string
    # literals are candidates for parameterisation.
    if re.search(r"['\"][^'\"]{20,}['\"]", cypher):
        warnings.append("query contains long string literal(s); consider parameters")

    return VerificationResult(
        is_valid=not errors, errors=errors, warnings=warnings
    )


# ---------- Correction -------------------------------------------------------


def correct_cypher(cypher: str | None) -> CorrectionResult:
    """Apply safe rule-based fixes for the issues ``verify_cypher`` detects.

    Only fixes things we can do deterministically without changing query
    semantics. Specifically:

        - Strip ```cypher``` / ``` fences (common when LLMs don't follow
          instructions).
        - Collapse ``<-[:REL]->`` / ``>-[:REL]-<`` to undirected ``-[:REL]-``.
        - Add missing ``:`` in relationship types (e.g. ``-[REL]-`` -> ``-[:REL]-``).
        - Add missing relationship brackets (``-(REL)-`` -> ``-[:REL]-``).
        - Trim surrounding whitespace.

    Does NOT try to fix deeper structural problems (missing RETURN,
    unbalanced brackets). Those surface via ``verify_cypher`` as errors.
    """
    original = cypher or ""
    if not original.strip():
        return CorrectionResult(corrected_cypher=original, original_cypher=original)

    working = original
    applied: list[str] = []

    if "```" in working:
        working = re.sub(r"```(?:cypher|Cypher|CYPHER)?\s*", "", working)
        working = re.sub(r"\s*```", "", working)
        applied.append("stripped markdown fences")

    if re.search(r"<-\[[^\[\]]*\]->", working):
        working = re.sub(r"<-(\[[^\[\]]*\])->", r"-\1-", working)
        applied.append("collapsed invalid bidirectional relationship")

    if re.search(r">-\[[^\[\]]*\]-<", working):
        working = re.sub(r">-(\[[^\[\]]*\])-<", r"-\1-", working)
        applied.append("collapsed invalid reverse relationship")

    if re.search(r"-\(\s*[A-Z_][A-Z_0-9]*\s*\)-", working):
        working = re.sub(
            r"-\(\s*([A-Z_][A-Z_0-9]*)\s*\)-", r"-[:\1]-", working
        )
        applied.append("added relationship brackets")

    if re.search(r"-\[([A-Z_][A-Z_0-9]*)\]", working):
        working = re.sub(r"-\[([A-Z_][A-Z_0-9]*)\]", r"-[:\1]", working)
        applied.append("added missing colon on relationship type")

    trimmed = working.strip()
    if trimmed != working:
        applied.append("trimmed surrounding whitespace")
        working = trimmed

    return CorrectionResult(
        corrected_cypher=working, original_cypher=original, corrections=applied
    )


def verify_and_correct(cypher: str | None) -> tuple[VerificationResult, str]:
    """Apply corrections first, then verify. Returns (result, corrected_cypher).

    Useful when you have LLM-generated Cypher that might need trivial
    clean-up before validation.
    """
    correction = correct_cypher(cypher)
    verification = verify_cypher(correction.corrected_cypher)
    return verification, correction.corrected_cypher
