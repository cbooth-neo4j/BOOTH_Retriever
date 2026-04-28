"""Minimal Streamlit front-end for BOOTH Retriever.

Demonstrates the full loop with as little code as possible:

    1. Ask a question       (BOOTHRetriever.query)
    2. Give feedback        (BOOTHCurator.submit_feedback)
    3. Curate pending       (BOOTHCurator.list_pending / approve / reject)

Run from this directory:

    streamlit run app.py

Environment variables (see ``.env.example`` in this folder):
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD   (required)
    - OPENAI_API_KEY                          (required for embeddings)
    - EMBEDDING_MODEL                         (default: text-embedding-3-small)
    - EMBEDDING_DIMENSIONS                    (default: 1536)

This app is a *reference implementation*. In production you will want
paginated tables, auth, and proper error surfacing; this file keeps
everything on one page for readability.
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from booth_retriever import BOOTHCurator, BOOTHRetriever

load_dotenv()

st.set_page_config(page_title="BOOTH Example", page_icon="🧠", layout="wide")


@st.cache_resource
def get_driver():
    """Long-lived Neo4j driver, shared across reruns."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )


@st.cache_resource
def get_embedder():
    """Embedder resolved from env. Swap this out to use a different provider."""
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    )


@st.cache_resource
def get_retriever() -> BOOTHRetriever:
    return BOOTHRetriever(driver=get_driver(), embedder=get_embedder())


@st.cache_resource
def get_curator() -> BOOTHCurator:
    return BOOTHCurator(driver=get_driver())


st.title("🧠 BOOTH Retriever demo")
st.caption(
    "A one-file Streamlit app that consumes `booth-retriever`. "
    "Cache-hit path, feedback, and inline curation."
)

ask_tab, curate_tab = st.tabs(["Ask", "Curate"])


# -----------------------------------------------------------------------------
# Ask tab
# -----------------------------------------------------------------------------

with ask_tab:
    question = st.text_input(
        "Your question", placeholder="How many users are in the system?"
    )
    is_high_risk = st.checkbox(
        "High-risk", value=False
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Running BOOTH..."):
                resp = get_retriever().query(question, is_high_risk=is_high_risk)

            if resp.success:
                st.success(resp.answer)
            elif resp.declined:
                st.error(resp.answer)
            else:
                st.info(resp.answer)

            with st.expander("Response metadata"):
                st.json(
                    {
                        "query_id": resp.query_id,
                        "similar_match": resp.similar_match,
                        "high_risk": resp.high_risk,
                        "declined": resp.declined,
                        "tool_used": resp.tool_used,
                        "cypher_used": resp.cypher_used,
                        "error_message": resp.error_message,
                    }
                )

            if resp.query_id and not resp.declined:
                st.session_state["last_query_id"] = resp.query_id

    if "last_query_id" in st.session_state:
        st.divider()
        st.subheader("Feedback on the last answer")
        col1, col2 = st.columns(2)
        if col1.button("👍 Helpful", use_container_width=True):
            get_curator().submit_feedback(
                st.session_state["last_query_id"], helpful=True
            )
            st.toast("Marked helpful. Query is now pending curator approval.")
        if col2.button("👎 Not helpful", use_container_width=True):
            get_curator().submit_feedback(
                st.session_state["last_query_id"], helpful=False
            )
            st.toast("Marked for human review.")


# -----------------------------------------------------------------------------
# Curate tab
# -----------------------------------------------------------------------------

with curate_tab:
    curator = get_curator()
    stats = curator.stats()
    cols = st.columns(5)
    for col, status in zip(cols, sorted(stats.counts.keys()), strict=False):
        col.metric(status, stats.counts[status])

    st.divider()
    pending = curator.list_pending(limit=25)
    if not pending:
        st.info("Nothing pending. Ask a few questions in the other tab first.")
    else:
        for row in pending:
            with st.expander(
                f"[{row.status}] {row.query_text[:80]} "
                f"(risk={row.risk_level}, fewshot={'yes' if row.has_fewshot else 'no'})"
            ):
                st.code(row.query_id, language=None)
                detail = curator.get(row.query_id)
                if detail is None:
                    st.warning("Query vanished; refresh.")
                    continue

                cypher = st.text_area(
                    "FewShot Cypher template",
                    value=detail.fewshot_cypher
                    or "MATCH (n) RETURN count(n) AS n",
                    key=f"cypher-{row.query_id}",
                )
                params = st.text_input(
                    "Parameters (comma-separated)",
                    value=",".join(detail.fewshot_parameters),
                    key=f"params-{row.query_id}",
                )
                left, right = st.columns(2)
                if left.button("Approve", key=f"approve-{row.query_id}"):
                    try:
                        curator.approve(
                            row.query_id,
                            cypher_template=cypher,
                            parameters=[
                                p.strip() for p in params.split(",") if p.strip()
                            ],
                        )
                        st.success("Approved. Reload to refresh.")
                    except ValueError as e:
                        st.error(f"Approval failed: {e}")
                reason = right.text_input(
                    "Rejection reason (optional)",
                    key=f"reason-{row.query_id}",
                    label_visibility="collapsed",
                    placeholder="Rejection reason (optional)",
                )
                if right.button("Reject", key=f"reject-{row.query_id}"):
                    curator.reject(row.query_id, reason=reason or None)
                    st.warning("Rejected. Reload to refresh.")
