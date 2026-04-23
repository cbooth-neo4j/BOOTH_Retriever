# booth-retriever

A self-improving Neo4j retriever for [`neo4j-graphrag-python`](https://neo4j.com/docs/neo4j-graphrag-python/current/).

BOOTH plugs into `neo4j-graphrag` as a drop-in `Retriever` subclass. For each
natural-language question it:

1. **Embeds** the question and searches a similarity cache of approved
   `Query` nodes (instant, cheap).
2. If the top match is above a configurable threshold, **executes the linked
   FewShot Cypher template** and returns the rows.
3. If nothing is above threshold, **queues the question for curation** and
   returns a polite "pending review" response. A human approves it with a
   Cypher template (or points an LLM at it), and from then on similar
   questions hit the cache.

The cache is domain-tuned by your curators; it gets better and cheaper every
time one of them approves a template.

---

## Status (v0.0.1)

Alpha. Developed in-tree at `packages/booth-retriever/`. Will be extracted to
its own repo at `v0.1.0`.

What's implemented today:

- `BOOTHRetriever`: cache-hit + cache-miss + high-risk paths. Parameter-less
  few-shots execute; parameterised templates return a descriptive error
  (automatic parameter extraction is tracked as follow-up).
- `BOOTHCurator`: `list_*`, `get`, `stats`, `approve`, `reject`, `edit_fewshot`,
  `submit_feedback`. Idempotent.
- `RefinementAgent`: single-shot LLM call to turn a raw Cypher + question into
  a parameterised template. No `deepagents` dependency.
- `verify_cypher` / `correct_cypher`: pure-logic Cypher lint. Used by the
  curator to validate templates before accepting.
- `init_schema`: idempotent DDL bootstrap.
- `booth` CLI: `init`, `curate list/show/approve/reject/edit`, `feedback`,
  `stats`, with `--json` everywhere for piping.
- Streamlit reference app at `examples/streamlit_app/`.

What's NOT implemented yet (deliberate; tracked as future work):

- Agentic Text2Cypher fallback on cache miss. MV1 just queues for curation.
- LLM-based parameter extraction for parameterised templates at retrieval
  time. The curator stores templates fine; the retriever refuses to run them
  with unresolved `$vars` until extraction is added.
- LLM-based answer summarisation. MV1 stringifies rows directly.

---

## Install

```bash
# once published
pip install booth-retriever

# from source during the beta
git clone https://github.com/<org>/BOOTH_Retriever.git
cd BOOTH_Retriever/packages/booth-retriever
pip install -e ".[dev]"
```

Minimum dependencies: Python 3.10+, Neo4j 5.11+ (for vector indexes),
`neo4j-graphrag>=1.0`.

---

## Quick start

```python
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from booth_retriever import BOOTHRetriever, BOOTHCurator, init_schema

driver = GraphDatabase.driver(
    "bolt://localhost:7687", auth=("neo4j", "password")
)

init_schema(driver, embedding_dimensions=1536)

retriever = BOOTHRetriever(
    driver=driver,
    embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    similarity_threshold=0.90,
)

# First time a question is asked -> queued for curation
response = retriever.query("How many users are there?")
print(response.answer)          # "...queued for curation..."
print(response.query_id)         # uuid for later feedback/curation

# Curate it
curator = BOOTHCurator(driver=driver)
curator.approve(
    response.query_id,
    cypher_template="MATCH (u:User) RETURN count(u) AS n",
    parameters=[],
)

# Ask a similar question -> cache hit
response = retriever.query("How many users do we have?")
print(response.answer)          # "42"
print(response.cypher_used)     # the approved template
```

---

## Retriever API

`BOOTHRetriever` implements both methods you might want:

### `retriever.query(text, *, is_high_risk=False) -> BOOTHResponse`

The rich path. Returns a `BOOTHResponse` dataclass with:

| field            | type           | meaning                                                |
| ---------------- | -------------- | ------------------------------------------------------ |
| `success`        | `bool`         | Answer is a real result, not a placeholder.            |
| `answer`         | `str`          | Human-readable answer (or "queued for curation").      |
| `query_id`       | `str \| None`  | Use for feedback / curation calls.                     |
| `similar_match`  | `bool`         | True iff a cache hit.                                  |
| `high_risk`      | `bool`         | Echo of the request flag.                              |
| `declined`       | `bool`         | Query was auto-declined as high-risk.                  |
| `cypher_used`    | `str \| None`  | The FewShot template that ran (if any).                |
| `raw_data`       | `list[dict]`   | Raw rows from the FewShot.                             |
| `tool_used`      | `str`          | `"fewshot_cache"` or `"pending_review"`.               |
| `error_message`  | `str \| None`  | Populated on graceful failures (eg. unresolved params).|

### `retriever.search(query_text) -> RetrieverResult`

Spec-compliant `neo4j-graphrag` signature. Returns a `RetrieverResult` where
the row data lives in `items` and every BOOTH-specific field above is carried
through `metadata`. Use this when plugging into a `neo4j_graphrag.RAG`
pipeline.

---

## Curator API

```python
curator = BOOTHCurator(driver=driver)

# Read
curator.list_pending(limit=50)            # pending_approval + declined + needs_review
curator.list_by_status("approved")
curator.get(query_id)                     # full detail + linked FewShot
curator.stats()                           # counts by status

# Write
curator.approve(
    query_id,
    cypher_template="MATCH (n) RETURN count(n) AS n",
    parameters=[],
    category="FACTUAL",
)

# Approve via an LLM-backed RefinementAgent
from booth_retriever import RefinementAgent
from neo4j_graphrag.llm.openai_llm import OpenAILLM
agent = RefinementAgent(llm=OpenAILLM(model_name="gpt-4o"))
curator.approve(query_id, refinement_agent=agent, raw_cypher="...")

curator.reject(query_id, reason="duplicate of q-1234")
curator.edit_fewshot(query_id, cypher_template="...", parameters=[...])
curator.submit_feedback(query_id, helpful=True)   # -> pending_approval
curator.submit_feedback(query_id, helpful=False)  # -> needs_review
```

Approving an already-approved query updates the FewShot in place (no
duplicates). `approve` / `edit_fewshot` run the template through
`verify_cypher` before accepting — pass `verify=False` to skip.

---

## CLI

Every command accepts connection options as flags or environment variables
(`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`).

```bash
booth --version
booth init --dimensions 1536             # idempotent; safe to rerun

booth stats --json

booth curate list                         # all pending
booth curate list --status approved
booth curate show <query_id>
booth curate show <query_id> --json

booth curate approve <query_id> \
    --cypher "MATCH (u:User) RETURN count(u) AS n"
booth curate approve <query_id> \
    --cypher @path/to/template.cypher \
    --params "name,role"
booth curate reject <query_id> --reason "off-topic"
booth curate edit <query_id> --cypher "..."

booth feedback <query_id> --helpful
booth feedback <query_id> --not-helpful
```

Use `--json` on read commands to pipe into `jq`.

---

## Testing

Four tiers, from fastest to slowest:

| Tier | Marker         | Needs                   | Duration |
| ---- | -------------- | ----------------------- | -------- |
| 1    | `smoke`        | Python only             | < 1s     |
| 2    | `unit`         | Python only             | 2-3s     |
| 3    | `integration`  | Docker (testcontainers) | ~60s     |
| 4    | manual (`eval/`) | Real Neo4j + LLM      | minutes  |

```bash
pip install -e ".[dev]"

pytest -m smoke                    # every push, CI, pre-commit
pytest -m unit                     # every push, CI
pytest -m "smoke or unit"          # recommended dev loop
pytest -m integration              # scheduled CI + before releases
pytest                             # everything

ruff check .                       # lint
```

The integration tests auto-skip if Docker isn't running.

### Evaluation harness

See `eval/README.md`. Not part of pytest; run manually:

```bash
python eval/run_eval.py --testset eval/testsets/starter.yaml
```

---

## Pluggable backends

- **Embedder**: any `neo4j_graphrag.embeddings.Embedder`, or any duck-typed
  object with `embed_query(text) -> list[float]`. Tests pass plain fakes.
- **LLM (for `RefinementAgent`)**: any `neo4j_graphrag.llm.LLMInterface`.
- **Vector index**: customisable via the `vector_index_name` kwarg on
  `BOOTHRetriever`, default `"query_embeddings"` matches `init_schema`.

---

## Architecture

```
BOOTHRetriever (public, subclass of neo4j_graphrag.retrievers.base.Retriever)
     │
     └── BOOTHOrchestrator (internal; has no neo4j_graphrag dep for testability)
              │
              ├── embedder.embed_query(...)
              ├── driver.session().run(vector_search_cypher)   # cache lookup
              ├── [hit]  -> execute FewShot cypher, return rows
              └── [miss] -> create Query node with status = pending_approval
                             + UserQuestion node for audit

BOOTHCurator (public)
     └── driver-backed CRUD over Query, FewShot, UserQuestion nodes
          └── uses verify_cypher() to lint templates before storing

RefinementAgent (public, optional)
     └── wraps an LLMInterface; turns raw_cypher + question into a
         parameterised template + parameter list
```

---

## License

Proprietary (for now).
