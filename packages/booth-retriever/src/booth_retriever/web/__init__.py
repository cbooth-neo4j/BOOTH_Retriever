"""FastAPI layer over ``BOOTHCurator`` and ``BOOTHRetriever``.

Opt-in install::

    pip install 'booth-retriever[web]'

Run the dev server::

    uvicorn booth_retriever.web:app --reload

This module exposes the same operations as the ``booth curate ...`` CLI so
the static TypeScript UI in ``packages/booth-retriever-ui`` (or any other
HTTP client) can drive the curation workflow without embedding Neo4j
credentials in the browser.
"""

from __future__ import annotations

from .api import app, create_app

__all__ = ["app", "create_app"]
