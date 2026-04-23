# BOOTH Retriever

**Bounded Orchestration Of Text Handling** — a self-improving Neo4j retriever
that answers natural-language questions by matching them against a
human-curated cache of approved Cypher templates. Cache miss? The question is
queued for a human curator; once approved, every similar question from then on
is an instant, cheap cache hit.

All state — embeddings, queries, curation status, approved templates — lives
in your Neo4j graph. No secondary store, no SaaS dependency.

```
question → embed → vector search over approved Query nodes
                        ├── hit  → run linked FewShot template, return rows
                        └── miss → queue for curator, return "pending review"
```

See [`docs/architecture.md`](./docs/architecture.md) for the full system
diagram and lifecycle.

---

## Repository layout

This is a monorepo. Each component has its own README with install and
usage details:

| Path                                  | What it is                                                                 | Docs |
| ------------------------------------- | -------------------------------------------------------------------------- | ---- |
| `packages/booth-retriever/`           | The installable Python package: `BOOTHRetriever`, `BOOTHCurator`, `RefinementAgent`, the `booth` CLI, and the FastAPI layer at `booth_retriever.web`. This is the actively developed code path. | [README](./packages/booth-retriever/README.md) |
| `packages/booth-retriever-ui/`        | Static Vite + TypeScript UI. Two pages — **Curate** (triage pending questions, approve templates) and **Ask** (single-turn retriever playground) — backed by the FastAPI layer above. | [README](./packages/booth-retriever-ui/README.md) |
| `packages/booth-retriever/examples/streamlit_app/` | Streamlit reference app that exercises the same package, useful as a single-process demo. | [README](./packages/booth-retriever/examples/streamlit_app/README.md) |
| `build_graph/`                        | Scripts for seeding a demo graph from CSVs.                                | [README](./build_graph/README.md) |
| `eval/` (under `packages/booth-retriever/`) | Offline evaluation harness. Not part of pytest; run manually.        | [README](./packages/booth-retriever/eval/README.md) |
| `app.py`, `src/`, `pages/`, `retriever.py`, `tools.py`, `requirements.txt` | Legacy MV1 Streamlit prototype. Kept for reference; new work lives under `packages/`. | — |

---

## Quick start — pip package + web UI

Two terminals: one for FastAPI, one for Vite. Run both from the repo root
with a Python venv active.

**Prerequisites:** Python 3.10+, Neo4j 5.11+ (vector-index capable), an
OpenAI API key, and Node 18+ for the UI.

**1. Install the package and bring up the API:**

```bash
pip install -e "packages/booth-retriever[web]"

# .env in the repo root (or environment variables) must set:
#   NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
#   OPENAI_API_KEY  (only needed for the Ask page / retriever)
booth init --dimensions 1536        # idempotent schema bootstrap
uvicorn booth_retriever.web:app --reload   # -> http://localhost:8000
```

**2. In a second terminal, bring up the UI:**

```bash
cd packages/booth-retriever-ui
npm install
npm run dev                          # -> http://localhost:5173
```

Then open:

- `http://localhost:5173/` — **Curate** page (stats, pending queries, approve / reject / edit templates).
- `http://localhost:5173/ask.html` — **Ask** page (single-turn query against the retriever; shows cache hits, declined high-risk questions, errors).

Vite proxies `/api/*` to the FastAPI layer, so the browser never deals with
CORS in dev.

### Library usage

```python
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from booth_retriever import BOOTHRetriever, BOOTHCurator, init_schema

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
init_schema(driver, embedding_dimensions=1536)

retriever = BOOTHRetriever(
    driver=driver,
    embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    similarity_threshold=0.90,
)

resp = retriever.query("How many users are there?")
if not resp.similar_match:
    curator = BOOTHCurator(driver=driver)
    curator.approve(
        resp.query_id,
        cypher_template="MATCH (u:User) RETURN count(u) AS n",
        parameters=[],
    )
```

The full API surface (retriever, curator, CLI, FastAPI routes) is documented
in [`packages/booth-retriever/README.md`](./packages/booth-retriever/README.md).

---

## Legacy Streamlit prototype

The original MV1 lives at the repo root (`app.py`, `src/`, `pages/`,
`retriever.py`, `tools.py`, `requirements.txt`). It is being replaced by
`packages/booth-retriever` and will eventually be removed, but works today:

```bash
pip install -r requirements.txt
cp env.example .env        # fill in OPENAI_API_KEY + Neo4j credentials
python scripts/init_neo4j.py
streamlit run app.py       # -> http://localhost:8501
```

The root-level `src/` — `llm_client.py`, `neo4j_client.py`,
`booth_orchestrator.py`, `cypher_verification.py`, `cypher_correction.py`,
`agents/refinement_agent.py`, `logger.py` — contains the original
implementation. Most of this logic has been ported into
`packages/booth-retriever/src/booth_retriever/` with tests; new features
should go there, not here.

---

## Iterative refinement for Text2Cypher

BOOTH's Cypher verification / correction loop is based on
[Neo4j's research on iterative refinement for Text2Cypher](https://neo4j.com/blog/developer/iterative-refinement-for-text2cypher/).
That work benchmarked four verifiers (rule-based, CyVer, execution-based,
LLM-based) and found LLM-based was slowest and had the highest
false-positive rate, while rule-based was ~600× faster. BOOTH therefore
defaults to `RULE_BASED,EXECUTION_BASED`, but all four are configurable.

```
1. Generate Cypher (LLM + schema + few-shot examples)
   ↓
2. Verify (RULE_BASED → CYVER → EXECUTION_BASED → LLM_BASED, in order)
   ↓
3. Valid?
   ├─ Yes → Execute → Success?
   │          ├─ Yes → return results
   │          └─ No  → correction loop
   └─ No  → correction loop

Correction loop:
4. Correct (RULE_BASED → LLM_BASED)
   ↓
5. Iteration < MAX_CYPHER_RETRIES?
   ├─ Yes → back to step 2
   └─ No  → decline and log for human review
```

**Stop conditions:** successful execution, all verifiers passing, or
`MAX_CYPHER_RETRIES` (default 3) exhausted.

**Verifier options** (runs in listed order, stops at first failure):

- `RULE_BASED` — regex/AST checks (~1.5 ms): relationship direction, brackets, basic syntax.
- `CYVER` — structural Cypher validation: RETURN clauses, WHERE conditions, property access.
- `EXECUTION_BASED` — dry-run against the database (slower, most accurate).
- `LLM_BASED` — LLM-judged intent match (slowest, most comprehensive).

**Corrector options** (runs in listed order, stops at first success):

- `RULE_BASED` — auto-fix common issues: relationship direction, brackets, markdown fences.
- `LLM_BASED` — schema-aware correction driven by the verifier's error message.

Most corrections succeed in the first iteration; the loop exists for edge
cases.

---

## Configuration

A single `.env` in the repo root serves both the package and the legacy
app. The package reads it automatically via `python-dotenv`; the CLI
walks up from the current working directory.

```bash
# Required
OPENAI_API_KEY=sk-...
NEO4J_PASSWORD=your-password

# Optional (defaults shown)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_DATABASE=neo4j
OPENAI_CHAT_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_MODEL=text-embedding-3-small        # used by booth_retriever.web
SIMILARITY_THRESHOLD=0.90                     # 0-1, higher = stricter cache hit
MAX_CYPHER_RETRIES=3

# Iterative refinement
CYPHER_VERIFIERS=RULE_BASED,EXECUTION_BASED   # RULE_BASED, CYVER, EXECUTION_BASED, LLM_BASED
CYPHER_CORRECTORS=RULE_BASED,LLM_BASED        # RULE_BASED, LLM_BASED
USE_VERIFICATION_METADATA=true

# FastAPI / web UI
BOOTH_CORS_ORIGINS=http://localhost:5173      # comma-separated
```

---

## Neo4j database dumps

BOOTH's graph is a normal Neo4j database — dump and restore it like any other.

**Create a dump (Neo4j Desktop):** stop the database → click the `…` menu →
**Dump** → pick a location.

**CLI:**

```bash
bin/neo4j-admin database dump neo4j --to-path=/path/to/dumps/
```

**Restore (Neo4j Desktop):** create a new empty database → stop it → `…` →
**Restore** / **Load from a dump** → pick the `.dump` file → start.

---

## MCP (Model Context Protocol) integration

Configuration for Cursor / Claude to run direct Neo4j queries lives in
`mcp.json`. Ensure Neo4j is running at `bolt://localhost:7687`, update
credentials in `mcp.json` if needed, and MCP tools become available in
Cursor for exploring the graph.

[Setup walkthrough (YouTube)](https://www.youtube.com/watch?v=UilGH0j73rI)

---

## Testing

The package has a four-tier suite; full details in
[`packages/booth-retriever/README.md`](./packages/booth-retriever/README.md).
From the package directory:

```bash
cd packages/booth-retriever
pip install -e ".[dev]"
pytest -m smoke              # < 1s,  every push
pytest -m unit               # 2-3s,  every push
pytest -m "smoke or unit"    # recommended dev loop
pytest -m integration        # ~60s,  needs Docker (testcontainers)
pytest                       # everything
ruff check .
```

UI tests (from `packages/booth-retriever-ui`):

```bash
npm test          # vitest, one-shot
npm run build     # type-check + production bundle
```

---

## Logging

The legacy Streamlit app writes structured logs to `logs/booth_YYYYMMDD.log`
(auto-created, daily-rotated). Console shows `INFO+`, file shows `DEBUG+`.

```bash
tail -f logs/booth_$(date +%Y%m%d).log
grep -i "error" logs/booth_*.log
grep "query_id: abc-123" logs/booth_*.log
```

The new package uses the standard Python `logging` module; configure it in
your host application.

---

## License

Proprietary (for now).
