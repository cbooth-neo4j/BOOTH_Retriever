"""Shared data models for booth_retriever.

Kept separate from ``__init__`` so tests and internal modules can import
them without triggering the public-API imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimilarQueryMatch:
    """A hit returned by the similarity cache lookup."""

    query_id: str
    query_text: str
    score: float
    status: str
    fewshot_cypher: str | None = None
    fewshot_parameters: list[str] = field(default_factory=list)


@dataclass
class BOOTHResponse:
    """Rich response object returned by ``BOOTHRetriever.query()``.

    Mirrors the shape of ``src/booth_orchestrator.BOOTHResponse`` in the
    parent repo for drop-in compatibility with the Streamlit reference app.
    """

    success: bool = False
    answer: str = ""
    query_id: str | None = None
    similar_match: bool = False
    high_risk: bool = False
    declined: bool = False
    cypher_used: str | None = None
    raw_data: Any | None = None
    error_message: str | None = None
    tool_used: str | None = None
    pending_feedback: bool = False
