# BOOTH Streamlit example

A minimal single-file Streamlit app that consumes the `booth-retriever` package.
It's ~200 lines of Python and shows the end-to-end loop:

- Ask a question (`BOOTHRetriever.query`)
- Thumbs-up / thumbs-down feedback (`BOOTHCurator.submit_feedback`)
- Review pending queries, approve or reject them (`BOOTHCurator.list_pending` / `approve` / `reject`)

Customers should treat this as a reference, not a product. For production UIs
you'll want auth, pagination, proper error surfacing, etc.

## Prerequisites

- A running Neo4j 5.11+ with BOOTH's schema bootstrapped (`booth init`).
- An OpenAI API key (the example uses OpenAI embeddings; swap the embedder if
  you're using a different provider).

## Install

```bash
cd packages/booth-retriever/examples/streamlit_app
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and fill in:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your password>
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

## Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

1. In the **Ask** tab, type a question. On a blank database every question
   will miss the cache and be queued for curation.
2. Switch to the **Curate** tab. You'll see the queued question; paste in a
   Cypher template that answers it, then click **Approve**.
3. Back in the **Ask** tab, ask a similar question. This time you'll hit the
   cache and see the answer from the approved template.
