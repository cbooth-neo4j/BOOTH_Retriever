# BOOTH Retriever - Productisation Brief

**Purpose:** Share context on what BOOTH is today and the proposed plan to ship it as a pip-installable retriever that plugs into Neo4j's `neo4j-graphrag-python` library.

---

## 1. What BOOTH is today

**BOOTH = Bounded Orchestration Of Text Handling.**

A Streamlit app (this repo) that answers natural-language questions against a Neo4j knowledge graph and **learns from human curation**. It's effectively two systems glued together:

1. A **self-improving retriever** - similarity cache of approved queries with an agentic Text2Cypher fallback.
2. A **curation workflow** - humans approve/reject the fallback's output; approvals get refined into parameterised Cypher and cached for next time.

### Runtime flow

```
User prompt
   -> embed
   -> HNSW vector search over approved Query nodes in Neo4j
   -> if similarity >= 0.90 : run the linked FewShot Cypher directly  (~2 LLM calls)
   -> else if high-risk     : decline to user, run agent in background for review
   -> else                  : run Deep Agent Text2Cypher
   -> store everything (UserQuestion, attempts, response) for curation
```

### Curation flow (Train AI page)

```
Pending queries
   -> human approves
   -> RefinementAgent consolidates multi-step exploration into ONE parameterised Cypher
   -> creates Query + FewShot nodes
   -> future similar questions hit the cache instead of the agent
```

### Key modules in this repo

- `src/booth_orchestrator.py` - main workflow (~970 lines)
- `src/agents/booth_agent.py` + `agentic_retriever.py` - Deep Agent Text2Cypher wrapper
- `src/agents/refinement_agent.py` - consolidates agent runs into parameterised Cypher
- `src/neo4j_client.py` - all graph ops (vector search, curation mutations)
- `src/cypher_verification.py` / `cypher_correction.py` - iterative refinement loop (rule-based / CyVer / execution / LLM), based on Neo4j research
- `pages/1_Train_AI.py` - curator UI
- `build_graph/` - KG construction pipeline (entity discovery, chunking)

### Neo4j data model (BOOTH-owned nodes)

- `UserQuestion` - verbatim audit trail, no embedding
- `Query` - canonical pattern, embedded, status = approved | pending | rejected | declined
- `FewShot` - approved parameterised Cypher, linked via `FEW_SHOT_EXAMPLE`
- `Tool`, `CypherAttempt`, `Response` - tool recommendations + audit trail

---

## 2. Proposal: `pip install booth-retriever`

**Goal:** Let any customer who already has a Neo4j KG drop BOOTH in as a retriever, with a clean path for curating approved queries.

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Relationship to `neo4j-graphrag-python` | **Depend on it**, subclass its `Retriever` base class | No fork maintenance; automatic upstream upgrades; works inside existing GraphRAG pipelines |
| Repo strategy | **Subfolder-first, split later** at `packages/booth-retriever/` in this repo; extract to its own GitHub repo at v0.1.0 | Fastest iteration while the API is moving; one PR can touch both the package and the example app. Customers can already `pip install` from the GitHub subdirectory during the beta |
| Package scope | **Retriever + curation only** | Customer brings their own KG, their own embedder, their own LLM. `build_graph/` stays as a demo, not shipped |
| LLM / embedder | **Fully pluggable** via `neo4j_graphrag.llm.LLMInterface` and `neo4j_graphrag.embeddings.Embedder` | Customers already using OpenAI / Anthropic / Vertex / Ollama through GraphRAG get zero-config support |
| Search API | **Both** `search()` (spec-compliant, returns `RetrieverResult`) **and** `query()` (rich, returns `BOOTHResponse`) | Spec compliance for pipeline use; rich API for apps that want query_id + cypher + metadata |
| Curation UX | **CLI-first** (`booth curate ...`) on top of a Python API | Ops-friendly, scriptable, no opinionated UI. Customers build their own front-end if they want one |
| Self-building auto-curator agent | **Deferred** | CLI + Python API first; revisit once the core is stable |

### Proposed package layout

```
booth-retriever/
  pyproject.toml          # deps: neo4j-graphrag, deepagents (optional extra), typer
  src/booth_retriever/
    retriever.py          # BOOTHRetriever(Retriever)  -> drop-in class
    orchestrator.py       # similarity-cache + agent-fallback logic
    curator.py            # BOOTHCurator: list/approve/reject/edit/feedback
    schema.py             # init_schema(driver) - idempotent DDL
    models.py             # Pydantic: Query, FewShot, BOOTHResponse
    agents/               # refinement + Text2Cypher (deepagents optional)
    verification/         # rule_based / cyver / execution / llm_based
    correction/           # rule_based / llm_based
    cli.py                # Typer app
  examples/streamlit_app/ # current app.py + pages/, now consuming the package
```

### What the customer writes

```python
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from booth_retriever import BOOTHRetriever, BOOTHCurator, init_schema

driver = GraphDatabase.driver(URI, auth=AUTH)
init_schema(driver)  # creates Query/FewShot/UserQuestion indexes, idempotent

retriever = BOOTHRetriever(
    driver=driver,
    embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    llm=OpenAILLM(model_name="gpt-4o"),
    similarity_threshold=0.90,
)

response = retriever.query("How many users are in the system?")
# -> BOOTHResponse(success, answer, query_id, cypher_used, tool_used, ...)
```

And for curation, the operator runs:

```
booth curate list --status pending
booth curate show <query_id>
booth curate approve <query_id>   # triggers RefinementAgent -> creates FewShot
booth curate reject <query_id> --reason "..."
booth feedback <query_id> --helpful
booth stats                       # cache hit rate, counts by status
```

### The "curation UI elephant", addressed

Today's Train AI Streamlit page is a front-end over the same mutations we'll expose via `BOOTHCurator`. Three concentric rings for the customer:

1. **Python API** (`BOOTHCurator`) - the core. Everything else is a thin wrapper.
2. **CLI** (`booth curate ...`) - what ships in v1. Ops-friendly, good for CI and scripts.
3. **Customer's own UI** - they build it on `BOOTHCurator`. We publish the Streamlit app as an `examples/` reference so they see a working implementation.

Later rings we explicitly deferred: REST server, MCP server for AI-agent-driven curation, self-building auto-curator. All become easy once (1) is clean.

---

## 3. Migration path

- Port `src/booth_orchestrator.py` -> `orchestrator.py`; swap direct OpenAI calls for `LLMInterface` / `Embedder`.
- Port `src/neo4j_client.py` - retrieval methods into `orchestrator.py`, curation methods into `curator.py`.
- Port `cypher_verification.py` + `cypher_correction.py` into `verification/` and `correction/` with a registry for env-based selection.
- Port `src/agents/*` as-is, gate deepagents behind `pip install booth-retriever[agent]`.
- Drop `build_graph/`, `scripts/`, `app.py`, `pages/` from the package; park them in `examples/`.

## 4. Open questions for discussion

- **Sync vs. async**: `neo4j-graphrag` supports both. Ship sync only for v1?
- **FewShot versioning**: track which FewShot version answered which UserQuestion so we can A/B and rollback?
- **Label namespacing**: BOOTH owns `Query`, `FewShot`, `UserQuestion`, etc. Risk of collision with customer labels - document clearly, or prefix with `BOOTH__Query`?

---

*Prepared as a pre-read. Full plan lives in `.cursor/plans/booth-retriever_pip_package_*.plan.md`.*
