# booth-retriever

> **Bounded Orchestration Of Text Handling** — a self-improving Neo4j retriever for the [`neo4j-graphrag-python`](https://neo4j.com/docs/neo4j-graphrag-python/current/) library.

BOOTH plugs into `neo4j-graphrag-python` as a drop-in `Retriever`. It answers natural-language questions against a Neo4j knowledge graph by:

1. Looking up a similarity cache of **approved** queries (instant path, no LLM Cypher generation).
2. Falling back to agentic Text2Cypher when nothing in the cache matches.
3. Storing every run for **human curation**. Approvals refine the multi-step agent output into one parameterised Cypher query and grow the cache.

The result is a retriever that gets faster and cheaper over time as more queries get human-approved.

## Status

Alpha. Currently developed in-tree at `packages/booth-retriever/` alongside the reference Streamlit app that consumes it. Will be extracted into its own repository at `v0.1.0`.

## Install (dev / beta)

```bash
# from PyPI (once released)
pip install booth-retriever

# direct from the in-tree subfolder during the beta
pip install "git+https://github.com/<org>/BOOTH_Retriever.git#subdirectory=packages/booth-retriever"

# with the optional agentic Text2Cypher backend
pip install "booth-retriever[agent]"

# for local development
git clone https://github.com/<org>/BOOTH_Retriever.git
cd BOOTH_Retriever/packages/booth-retriever
pip install -e ".[dev,agent]"
```

## Quick start

```python
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from booth_retriever import BOOTHRetriever, init_schema

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
init_schema(driver)   # idempotent - creates BOOTH's own indexes

retriever = BOOTHRetriever(
    driver=driver,
    embedder=OpenAIEmbeddings(model="text-embedding-3-small"),
    llm=OpenAILLM(model_name="gpt-4o"),
    similarity_threshold=0.90,
)

response = retriever.query("How many users are in the system?")
print(response.answer)
```

## Curation

```bash
booth init                                   # runs init_schema from env vars
booth curate list --status pending
booth curate show <query_id>
booth curate approve <query_id>              # triggers refinement, creates FewShot
booth curate reject <query_id> --reason "..."
booth feedback <query_id> --helpful
booth stats
```

All CLI commands are thin wrappers around `BOOTHCurator`. See `docs/` in the parent repo for the full curation flow.

## Testing

```bash
pip install -e ".[dev]"

pytest -m smoke                              # < 1s, runs everywhere
pytest -m unit                               # mocked Neo4j + LLM
pytest -m integration                        # needs Docker, spins up Neo4j via testcontainers
pytest                                       # everything
```

See the parent repo's `.cursor/plans/` for the full testing strategy.

## License

Proprietary (for now).
