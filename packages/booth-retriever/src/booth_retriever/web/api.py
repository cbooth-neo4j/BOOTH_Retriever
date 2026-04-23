"""FastAPI app exposing BOOTH curation over HTTP.

The public surface mirrors the ``booth curate ...`` CLI (see
``booth_retriever.cli``); see ``__init__.py`` in this package for the
top-level usage story. Every route is a thin wrapper over a single
``BOOTHCurator`` method. We intentionally do not expose arbitrary Cypher
execution here; only the curation operations that are safe to surface to a
browser client.

The FastAPI / uvicorn imports are deferred to module import time but raise
a helpful error if the ``web`` extra hasn't been installed.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
except ImportError as exc:  # pragma: no cover - exercised only when extra missing
    raise ImportError(
        "booth_retriever.web requires the 'web' extra. Install with:\n"
        "    pip install 'booth-retriever[web]'"
    ) from exc

from ..curator import (
    ALL_STATUSES,
    ApprovalResult,
    BOOTHCurator,
    CuratorStats,
    PendingQuery,
    QueryDetail,
)
from ..models import BOOTHResponse
from ..retriever import BOOTHRetriever
from .schemas import (
    ApproveRequest,
    AskRequest,
    EditRequest,
    FeedbackRequest,
    RejectRequest,
)


# ---------------------------------------------------------------------------
# Bootstrap helpers — mirrors ``booth_retriever.cli`` so the two entrypoints
# behave identically for env loading and driver construction.
# ---------------------------------------------------------------------------


def _load_env_file() -> None:
    """Load ``.env`` from the current working directory (walking up)."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dep
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


class _RetrieverUnavailable(RuntimeError):
    """Raised when we can't build a ``BOOTHRetriever`` (e.g. no API key).

    Translated to HTTP 503 in ``get_retriever`` so the browser client can
    surface a clean error instead of a 500 + stack trace.
    """


def _default_retriever_factory(curator: BOOTHCurator) -> BOOTHRetriever:
    """Build a ``BOOTHRetriever`` using the curator's driver and OpenAI embeddings.

    Kept as a module-level function (instead of inline in ``get_retriever``)
    so tests can monkeypatch it. Reuses the driver already opened for
    curation — one driver per process is plenty.

    Raises:
        _RetrieverUnavailable: if ``OPENAI_API_KEY`` is missing or the
            ``neo4j-graphrag[openai]`` extra isn't installed.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise _RetrieverUnavailable(
            "OPENAI_API_KEY is not set. The Ask page needs an embedder; set "
            "OPENAI_API_KEY in the environment (or a .env file) and retry."
        )
    try:
        from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise _RetrieverUnavailable(
            "OpenAI embeddings unavailable. Install with:\n"
            "    pip install 'booth-retriever[web]'"
        ) from exc

    embedder = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    database = os.environ.get("NEO4J_DATABASE")
    return BOOTHRetriever(
        driver=curator.driver,
        embedder=embedder,
        neo4j_database=database,
    )


def _default_driver_factory():
    """Build a real ``neo4j.Driver`` from environment variables.

    Silences ``UNRECOGNIZED`` notifications to match the CLI's behaviour; on
    older drivers the kwarg raises ``TypeError`` and we fall back.
    """
    from neo4j import GraphDatabase

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD is required (set in the environment or a .env "
            "file in the working directory)."
        )
    try:
        return GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_classifications=["UNRECOGNIZED"],
        )
    except TypeError:  # pragma: no cover - exercised only on pre-5.21 drivers
        return GraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    curator: BOOTHCurator | None = None,
    retriever: BOOTHRetriever | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        curator: Pre-built ``BOOTHCurator``. Supplying this skips the default
            Neo4j driver construction in the lifespan — the primary hook
            tests use to inject a ``MagicMock``.
        retriever: Pre-built ``BOOTHRetriever`` used by ``POST /api/ask``.
            Optional because the retriever needs an embedder (and therefore
            an ``OPENAI_API_KEY`` by default); we lazily build one on the
            first ``/api/ask`` call so curator-only deployments aren't
            forced to configure OpenAI.
        cors_origins: Origins permitted by the CORS middleware. Defaults to
            ``$BOOTH_CORS_ORIGINS`` (comma-separated) or the Vite dev server
            at ``http://localhost:5173``.
    """
    _load_env_file()

    origins = cors_origins or [
        o.strip()
        for o in os.environ.get("BOOTH_CORS_ORIGINS", "http://localhost:5173").split(",")
        if o.strip()
    ]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.driver = None
        if getattr(app.state, "curator", None) is None:
            driver = _default_driver_factory()
            app.state.driver = driver
            database = os.environ.get("NEO4J_DATABASE")
            app.state.curator = BOOTHCurator(driver=driver, database=database)
        try:
            yield
        finally:
            if app.state.driver is not None:
                app.state.driver.close()

    app = FastAPI(
        title="BOOTH Curator API",
        version="0.0.1",
        summary="REST layer over booth_retriever.BOOTHCurator.",
        lifespan=lifespan,
    )
    app.state.curator = curator
    app.state.retriever = retriever

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    def get_curator(request: Request) -> BOOTHCurator:
        cur: BOOTHCurator | None = getattr(request.app.state, "curator", None)
        if cur is None:
            raise HTTPException(status_code=503, detail="Curator not initialized")
        return cur

    def get_retriever(request: Request) -> BOOTHRetriever:
        """Return the app's retriever, building it lazily on first use.

        The curator-only path is intentionally decoupled: we don't want to
        require ``OPENAI_API_KEY`` just to curate. The retriever is built
        against the driver already owned by the curator so we don't open a
        second Neo4j connection.
        """
        existing: BOOTHRetriever | None = getattr(request.app.state, "retriever", None)
        if existing is not None:
            return existing
        cur = get_curator(request)
        try:
            retr = _default_retriever_factory(cur)
        except _RetrieverUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        request.app.state.retriever = retr
        return retr

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @app.get("/api/stats")
    def stats(curator: BOOTHCurator = Depends(get_curator)) -> dict[str, Any]:
        s: CuratorStats = curator.stats()
        return {"total": s.total, "counts": s.counts}

    # ------------------------------------------------------------------
    # List queries
    # ------------------------------------------------------------------

    @app.get("/api/queries")
    def list_queries(
        status: str | None = None,
        limit: int = 50,
        curator: BOOTHCurator = Depends(get_curator),
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise HTTPException(status_code=400, detail="limit must be positive")
        if status is not None and status not in ALL_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown status {status!r}. "
                    f"Valid: {sorted(ALL_STATUSES)}"
                ),
            )
        rows: list[PendingQuery] = (
            curator.list_by_status(status, limit=limit)
            if status
            else curator.list_pending(limit=limit)
        )
        return [_pending_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Query detail
    # ------------------------------------------------------------------

    @app.get("/api/queries/{query_id}")
    def get_query(
        query_id: str,
        curator: BOOTHCurator = Depends(get_curator),
    ) -> dict[str, Any]:
        detail = curator.get(query_id)
        if detail is None:
            raise HTTPException(
                status_code=404, detail=f"No query with id {query_id!r}"
            )
        return _detail_to_dict(detail)

    # ------------------------------------------------------------------
    # Approve
    # ------------------------------------------------------------------

    @app.post("/api/queries/{query_id}/approve")
    def approve(
        query_id: str,
        body: ApproveRequest,
        curator: BOOTHCurator = Depends(get_curator),
    ) -> dict[str, Any]:
        try:
            result: ApprovalResult = curator.approve(
                query_id,
                cypher_template=body.cypher_template,
                parameters=body.parameters,
                category=body.category,
            )
        except ValueError as exc:
            raise _map_curator_value_error(exc) from None
        return {
            "query_id": result.query_id,
            "fewshot_id": result.fewshot_id,
            "fewshot_was_new": result.fewshot_was_new,
        }

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    @app.post("/api/queries/{query_id}/edit", status_code=204)
    def edit(
        query_id: str,
        body: EditRequest,
        curator: BOOTHCurator = Depends(get_curator),
    ) -> Response:
        try:
            curator.edit_fewshot(
                query_id,
                cypher_template=body.cypher_template,
                parameters=body.parameters,
            )
        except ValueError as exc:
            raise _map_curator_value_error(exc) from None
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------

    @app.post("/api/queries/{query_id}/reject", status_code=204)
    def reject(
        query_id: str,
        body: RejectRequest,
        curator: BOOTHCurator = Depends(get_curator),
    ) -> Response:
        try:
            curator.reject(query_id, reason=body.reason)
        except ValueError as exc:
            raise _map_curator_value_error(exc) from None
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    @app.post("/api/queries/{query_id}/feedback", status_code=204)
    def feedback(
        query_id: str,
        body: FeedbackRequest,
        curator: BOOTHCurator = Depends(get_curator),
    ) -> Response:
        try:
            curator.submit_feedback(query_id, helpful=body.helpful)
        except ValueError as exc:
            raise _map_curator_value_error(exc) from None
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # Ask — run the full BOOTH flow end-to-end.
    # ------------------------------------------------------------------

    @app.post("/api/ask")
    def ask(
        body: AskRequest,
        retriever: BOOTHRetriever = Depends(get_retriever),
    ) -> dict[str, Any]:
        resp = retriever.query(body.query_text, is_high_risk=body.is_high_risk)
        return _response_to_dict(resp)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pending_to_dict(p: PendingQuery) -> dict[str, Any]:
    return {
        "query_id": p.query_id,
        "query_text": p.query_text,
        "status": p.status,
        "risk_level": p.risk_level,
        "timestamp": p.timestamp,
        "user_feedback": p.user_feedback,
        "has_fewshot": p.has_fewshot,
    }


def _response_to_dict(r: BOOTHResponse) -> dict[str, Any]:
    """Flatten a ``BOOTHResponse`` into a JSON-safe payload.

    ``raw_data`` is intentionally dropped: FewShot Cypher can RETURN
    arbitrary projections (datetimes, nodes, relationships) that aren't
    guaranteed to serialise. Callers who need row data should use
    ``BOOTHRetriever.search()`` with neo4j-graphrag's result types.
    """
    return {
        "success": r.success,
        "answer": r.answer,
        "query_id": r.query_id,
        "similar_match": r.similar_match,
        "high_risk": r.high_risk,
        "declined": r.declined,
        "cypher_used": r.cypher_used,
        "tool_used": r.tool_used,
        "error_message": r.error_message,
        "pending_feedback": r.pending_feedback,
    }


def _detail_to_dict(d: QueryDetail) -> dict[str, Any]:
    return {
        "query_id": d.query_id,
        "query_text": d.query_text,
        "status": d.status,
        "risk_level": d.risk_level,
        "timestamp": d.timestamp,
        "user_feedback": d.user_feedback,
        "rejection_reason": d.rejection_reason,
        "fewshot_cypher": d.fewshot_cypher,
        "fewshot_parameters": d.fewshot_parameters,
    }


def _map_curator_value_error(exc: ValueError) -> HTTPException:
    """Translate a curator ``ValueError`` into the right HTTP status code.

    * ``"cypher_template failed verification: ..."`` -> 422 (unprocessable)
    * ``"No Query node with id '...'"`` / has-no-linked-FewShot -> 404
    * anything else -> 400
    """
    msg = str(exc)
    low = msg.lower()
    if "failed verification" in low:
        return HTTPException(status_code=422, detail=msg)
    if "no query node" in low or "no linked fewshot" in low:
        return HTTPException(status_code=404, detail=msg)
    return HTTPException(status_code=400, detail=msg)


# ---------------------------------------------------------------------------
# Module-level app for ``uvicorn booth_retriever.web:app``.
# ---------------------------------------------------------------------------


app = create_app()
