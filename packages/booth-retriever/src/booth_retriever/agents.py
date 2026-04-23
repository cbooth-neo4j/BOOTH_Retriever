"""Agent components for booth-retriever.

MV1 ships a single agent: ``RefinementAgent``. Its job is to take a raw
Cypher query + the natural-language question it answers, and produce a
parameterised template that can be reused for similar questions.

Design choices vs. the legacy agent in the parent repo:
    - Pluggable LLM via ``neo4j_graphrag.llm.LLMInterface``. Any provider
      that implements the interface (OpenAI, Anthropic, Ollama, VertexAI,
      ...) works without code changes.
    - No ``deepagents`` / ``langgraph`` dependency. The legacy agent used
      a tool-calling loop to test refinements against a live database; we
      offload that responsibility to the curator (who can execute the
      refined template and reject bad results). Single-shot LLM call here
      keeps the surface small and the cost predictable.
    - Strict JSON output contract. The prompt asks for a specific JSON
      shape; we parse it and surface parse failures as ``success=False``
      on the ``RefinementResult``.

An LLM-backed agent is inherently non-deterministic. Tests pass a fake
``LLMInterface`` that returns canned JSON so the ``RefinementAgent``'s
parsing and validation logic is exercised without network calls.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from .curator import RefinementResult

if TYPE_CHECKING:
    from neo4j_graphrag.llm.base import LLMInterface


_SYSTEM_PROMPT = """You convert a concrete Cypher query into a parameterised \
template that can be reused for similar natural-language questions.

INPUT
-----
- original_question: the user's question in natural language.
- raw_cypher: a Cypher query that correctly answers it (may be None; in \
that case write a new template from scratch that would answer the question).

OUTPUT (STRICT JSON, no prose around it)
----------------------------------------
{
  "success": true,
  "refined_cypher": "MATCH (p:Person {name: $person_name}) RETURN p.role",
  "parameters": ["person_name"],
  "category": "PERSON_ATTRIBUTE"
}

OR on failure:
{
  "success": false,
  "error": "short explanation"
}

RULES
-----
1. Replace hardcoded literals that came from the question with Cypher \
parameters named snake_case (e.g. $person_name, $film_title, $year).
2. Keep structural literals (labels, relationship types, LIMIT values) \
unchanged.
3. The "parameters" array MUST list every $name referenced in \
"refined_cypher", in the order they first appear.
4. Pick one CATEGORY from: PERSON_ATTRIBUTE, PERSON_ROLE, WORK_CREATOR, \
WORK_PARTICIPANT, LOCATION_ATTRIBUTE, TEMPORAL, RELATIONSHIP, FACTUAL, \
MULTI_HOP. Use FACTUAL if unsure.
5. Output ONLY the JSON object. No backticks, no commentary, no prefix.
"""


class RefinementAgent:
    """Produce a parameterised FewShot template from a raw Cypher + question.

    Args:
        llm: Any ``neo4j_graphrag.llm.LLMInterface`` implementation. The
            agent only calls ``llm.invoke(input=...)`` and reads the
            ``.content`` attribute off the result, so test doubles are easy.
    """

    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    def refine(
        self,
        *,
        original_question: str,
        raw_cypher: str | None = None,
    ) -> RefinementResult:
        """Run the refinement LLM call and return a structured result.

        Never raises; LLM or parse errors come back as
        ``RefinementResult(success=False, error=...)``.
        """
        if not original_question or not original_question.strip():
            return RefinementResult(
                success=False, error="original_question must not be empty"
            )

        prompt = self._build_prompt(original_question, raw_cypher)
        try:
            response = self._llm.invoke(input=prompt, system_instruction=_SYSTEM_PROMPT)
            raw_text = getattr(response, "content", None)
            if raw_text is None:
                raw_text = str(response)
        except Exception as exc:  # noqa: BLE001 - external LLM can fail in many ways
            return RefinementResult(
                success=False,
                error=f"LLM invocation failed: {type(exc).__name__}: {exc}",
            )

        parsed = _extract_json(raw_text)
        if parsed is None:
            return RefinementResult(
                success=False,
                error=f"Could not parse JSON from LLM output: {raw_text[:200]!r}",
            )

        if not parsed.get("success"):
            return RefinementResult(
                success=False,
                error=str(parsed.get("error") or "LLM reported failure"),
            )

        refined_cypher = parsed.get("refined_cypher")
        if not refined_cypher or not isinstance(refined_cypher, str):
            return RefinementResult(
                success=False, error="LLM output missing 'refined_cypher' string"
            )

        parameters = parsed.get("parameters") or []
        if not isinstance(parameters, list) or not all(
            isinstance(p, str) for p in parameters
        ):
            return RefinementResult(
                success=False,
                error="LLM output 'parameters' must be a list of strings",
            )

        # Sanity check: every declared parameter should actually appear in
        # the cypher. Missing ones mean the LLM hallucinated; extra $foo in
        # cypher without a matching declaration is also a bug.
        referenced = set(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", refined_cypher))
        declared = set(parameters)
        if declared != referenced:
            return RefinementResult(
                success=False,
                error=(
                    f"parameters mismatch: declared={sorted(declared)} "
                    f"referenced_in_cypher={sorted(referenced)}"
                ),
            )

        return RefinementResult(
            success=True,
            refined_cypher=refined_cypher,
            parameters=parameters,
            category=parsed.get("category"),
        )

    def _build_prompt(self, question: str, raw_cypher: str | None) -> str:
        lines = [f"original_question: {question}"]
        if raw_cypher:
            lines.append(f"raw_cypher:\n```cypher\n{raw_cypher}\n```")
        else:
            lines.append("raw_cypher: (none - write one from scratch)")
        lines.append("")
        lines.append("Return ONLY the JSON object described in the system instructions.")
        return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from ``text``. Tolerates ```json fences."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else None

    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
