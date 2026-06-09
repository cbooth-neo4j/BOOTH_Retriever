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
  `submit_feedback`, `migrate_statuses`, `get_query_graph`. Idempotent.
- `RefinementAgent`: single-shot LLM call to turn a raw Cypher + question into
  a parameterised template. No `deepagents` dependency.
- `Text2CypherAgent`: generate-AND-execute Cypher (wraps
  `neo4j_graphrag.retrievers.Text2CypherRetriever`). On a high-risk decline the
  orchestrator records the attempt (`CypherAttempt` + `Response`) against the
  Query for curation; the end user still only sees the decline message.
- `verify_cypher` / `correct_cypher`: pure-logic Cypher lint. Used by the
  curator to validate templates before accepting.
- `init_schema`: idempotent DDL bootstrap.
- `booth` CLI: `init`, `curate list/show/approve/reject/edit`, `feedback`,
  `stats`, with `--json` everywhere for piping.
- Streamlit reference app at `examples/streamlit_app/`.
- Procedural-memory seed at `examples/seed_procedural_memory.py`: creates an
  approved `Query` that is a **command** ("reconcile the APAC custody payment
  break with UUID …") whose steps form a chain with a data dimension:

  ```
  (Query)-[:HAS_STEP]->(Step)-[:NEXT]->(Step)-> …
          (Step)-[:USES_AGENT]->(Agent)-[:USES_TOOL]->(Tool)
          (Tool)-[:BACKED_BY]->(DataProduct)-[:SOURCED_FROM]->(System)
  ```

  Given a break UUID the agent works it end to end: retrieve the break from the
  Transaction Lifecycle Management (TLM) system, map it to its transaction flow,
  gather wider context from the two core banking systems (Citi, Vanguard),
  **classify** it (AI judgement) as "Payment in Live" vs "Return of Funds", then
  — for Payment in Live, the branch expanded here — run due diligence, make the
  decision, annotate the scenario back into OpsFlow and TLM, and apply the break
  age rule. The classify step is the divergence; its label dynamically selects
  the substeps (Return of Funds is a single stub branch). Each `Step` is tagged
  `deterministic`/`ai_judgement`; `Tool`s are MCP-registry-style callable
  interfaces with intent-revealing descriptions and a `data`/`capability`
  category; and each data `Tool` is backed by a governed `DataProduct` carrying
  the data-readiness scenario (1–4), pipeline `status`, owner, freshness and
  entitlements, sourced from one of the four `System`s (TLM, Citi, Vanguard,
  OpsFlow). The Ask page answers the command and the NVL popup (and the
  Curate-page process graph) visualise the whole procedure including which
  systems/data products each step depends on.
- Vanilla-TS web UI at `packages/booth-retriever-ui/`: a Curate page and an
  Ask page. The Ask page renders the graph behind each answer (provenance
  chain + answer subgraph) in a right-side NVL popup via
  `GET /api/queries/{id}/graph`. The Curate page tags every row as a
  **query** or a **process** (the latter for procedural-memory `Query` nodes,
  i.e. `kind = "procedural_memory"`) and, for processes only, mounts the same
  NVL graph inline in the detail pane so the whole procedure is reviewable.
  Both surfaces use the `d3Force` layout (main-thread, no web worker) so the
  graph auto-fits instead of stacking every node at the origin.

### Query status model

Three states (simplified from the earlier five):

- `needs_review` — the single curation queue. Absorbs the old
  `pending_approval` and `declined` states; high-risk declines and
  thumbs-down feedback land here too and are distinguished by node
  properties (`risk_level`, `user_feedback`) rather than a separate status.
- `approved` — has a linked, verified FewShot and serves the cache.
- `rejected` — explicitly turned down by a curator.

Migrating an existing graph: `booth migrate-statuses` (or
`BOOTHCurator.migrate_statuses()`) re-labels any legacy
`pending_approval`/`declined` nodes to `needs_review`.

**Deduplicating questions:** every ask records a `UserQuestion` linked
`:SIMILAR` to its canonical `Query`. To stop identical re-asks piling up,
a new node is only created when no existing one for that Query is `>= 0.99`
cosine similar (the incoming embedding is stored on the node for the
comparison); otherwise the existing node's `count` is bumped. Clean up a
pre-existing pile-up with `booth compact-questions` (or
`BOOTHCurator.compact_user_questions()`), which merges duplicates per Query
(cosine for embedded nodes, exact-text fallback for legacy ones).

What's NOT implemented yet (deliberate; tracked as future work):

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
curator.list_pending(limit=50)            # the needs_review queue
curator.list_by_status("approved")
curator.get(query_id)                     # full detail + linked FewShot + any Text2Cypher attempt
curator.stats()                           # counts by status
curator.get_query_graph(query_id)         # NVL-shaped provenance + answer subgraph

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
curator.submit_feedback(query_id, helpful=True)   # -> needs_review (user_feedback="helpful")
curator.submit_feedback(query_id, helpful=False)  # -> needs_review (user_feedback="not_helpful")
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

booth migrate-statuses                    # collapse legacy statuses
booth compact-questions                   # merge duplicate UserQuestions
booth compact-questions --threshold 0.97
```

Use `--json` on read commands to pipe into `jq`.

---

## Curator UI (web)

A browser-based version of the CLI's curate flow ships as an opt-in extra,
designed to be embeddable in a Neo4j dashboard (NeoDash iframe report, or any
similar host):

- **Backend** — a small FastAPI layer at `booth_retriever.web` that wraps
  `BOOTHCurator`. Browser clients never see Neo4j credentials.
- **Frontend** — a static TypeScript + Vite app in the sibling
  `packages/booth-retriever-ui/` package. No framework; vanilla DOM.

### Install + run the backend

```bash
pip install -e ".[web]"

# Reads NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE from the
# environment or a .env file in CWD (same rules as the `booth` CLI).
uvicorn booth_retriever.web:app --reload
# -> http://localhost:8000
```

Routes (all under `/api`, JSON throughout):

| Method | Path                              | Curator call                    |
| ------ | --------------------------------- | ------------------------------- |
| GET    | `/api/health`                     | —                               |
| GET    | `/api/stats`                      | `stats()`                       |
| GET    | `/api/queries?status=&limit=`     | `list_pending` / `list_by_status` |
| GET    | `/api/queries/{id}`               | `get()` (404 if missing)        |
| GET    | `/api/queries/{id}/graph`         | `get_query_graph()` (NVL payload) |
| POST   | `/api/queries/{id}/approve`       | `approve()`                     |
| POST   | `/api/queries/{id}/reject`        | `reject()`                      |
| POST   | `/api/queries/{id}/edit`          | `edit_fewshot()`                |
| POST   | `/api/queries/{id}/feedback`      | `submit_feedback()`             |

Error mapping: Cypher verification failures surface as `422` with the
verifier's message in `detail`; missing queries as `404`; other curator
`ValueError`s as `400`.

CORS defaults to `http://localhost:5173` (the Vite dev server). Override by
setting `BOOTH_CORS_ORIGINS` to a comma-separated list, or pass
`cors_origins=[...]` to `create_app()`.

### Run the frontend

See [`packages/booth-retriever-ui/README.md`](../booth-retriever-ui/README.md)
for the dev workflow and production build. TL;DR:

```bash
cd packages/booth-retriever-ui
npm install
npm run dev    # -> http://localhost:5173
```

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
              └── [miss] -> create Query node with status = needs_review
                             + UserQuestion node for audit
                             + [high-risk] Text2Cypher attempt saved
                               (CypherAttempt + Response) for curation

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
