"""Pydantic request models for the BOOTH web API.

Response bodies are plain ``dict`` payloads shaped from the existing
``booth_retriever.curator`` dataclasses (see ``api._pending_to_dict`` and
friends), so the wire format stays stable even if the dataclasses pick up
non-serialisable fields later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApproveRequest(BaseModel):
    """Body for ``POST /api/queries/{id}/approve``."""

    cypher_template: str = Field(..., min_length=1)
    parameters: list[str] = Field(default_factory=list)
    category: str | None = None


class EditRequest(BaseModel):
    """Body for ``POST /api/queries/{id}/edit``."""

    cypher_template: str = Field(..., min_length=1)
    parameters: list[str] = Field(default_factory=list)


class RejectRequest(BaseModel):
    """Body for ``POST /api/queries/{id}/reject``."""

    reason: str | None = None


class FeedbackRequest(BaseModel):
    """Body for ``POST /api/queries/{id}/feedback``."""

    helpful: bool
