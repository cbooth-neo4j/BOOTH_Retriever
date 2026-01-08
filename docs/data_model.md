# BOOTH Data Model (v3)

## Overview

The BOOTH system uses a graph-based data model in Neo4j to track user questions, approved queries, and their execution patterns. This enables:

1. **Instant execution** - When a similar question is asked, run the approved Cypher directly (no LLM inference)
2. **Question clustering** - Many verbatim questions map to the same canonical Query pattern
3. **Continuous learning** - Every approved query improves the system's coverage

---

## Node Types

### `UserQuestion`
Individual questions asked by users. This is the **verbatim audit trail** - lightweight, no embedding.

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | UUID |
| `text` | STRING | Original question text (verbatim, as entered) |
| `timestamp` | DATETIME | When the question was asked |
| `risk_level` | STRING | `low`, `high` |

**Note:** UserQuestions do NOT store embeddings. They link to Query nodes via SIMILAR relationship.

### `Query`
The **canonical question pattern** with embedding. This is what we match against for similarity search.

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | UUID |
| `text` | STRING | Canonical question text |
| `embedding` | LIST<FLOAT> | Vector embedding for similarity search |
| `timestamp` | DATETIME | When the query was created |
| `status` | STRING | `pending_approval`, `approved`, `rejected`, `declined` |
| `risk_level` | STRING | `low`, `high` |
| `similarity_matched` | BOOLEAN | Whether created from a similarity match |

### `FewShot`
The approved Cypher query that answers questions matching a Query pattern.

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | UUID |
| `cypher_template` | STRING | Cypher query (can include `$param` syntax) |
| `parameters` | LIST<STRING> | Parameter names used (e.g., `["person_name", "film_title"]`) |
| `example_values` | STRING | JSON of example parameter values |
| `created_at` | DATETIME | When created |

### `Tool`
Represents retrieval tools/methods available in the system.

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | UUID |
| `name` | STRING | Tool name (e.g., `agentic_text2cypher`) |
| `description` | STRING | What the tool does |

### `CypherAttempt` (audit)
Stores individual Cypher attempts during agentic exploration.

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | UUID |
| `cypher_text` | STRING | The actual Cypher executed |
| `attempt_number` | INTEGER | Attempt number in sequence |
| `success` | BOOLEAN | Whether execution succeeded |
| `timestamp` | DATETIME | When the attempt was made |

### `Response` (audit)
Stores responses from successful queries.

| Property | Type | Description |
|----------|------|-------------|
| `id` | STRING | UUID |
| `result_data` | STRING | JSON of raw query results |
| `summary` | STRING | LLM-generated answer summary |
| `timestamp` | DATETIME | When the response was generated |

---

## Relationships

```
                                    ┌─────────────────┐
                                    │      Query      │
                                    │  ─────────────  │
                                    │  text           │
                                    │  embedding      │──────[query_embeddings index]
                                    │  status         │
                                    └────────┬────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
  ┌───────────────────┐          ┌─────────────────┐            ┌─────────────────┐
  │   UserQuestion    │          │     FewShot     │            │      Tool       │
  │   ─────────────   │          │  ────────────── │            │  ────────────── │
  │   text (verbatim) │◄─────────│  cypher_template│            │  name           │
  │   timestamp       │ SIMILAR  │  parameters     │            │  description    │
  │   risk_level      │ (score)  │  example_values │            └─────────────────┘
  └───────────────────┘          └─────────────────┘                    ▲
                                         ▲                              │
                                         │                              │
                                FEW_SHOT_EXAMPLE                   USES_TOOL
                                         │                              │
                                         └──────────────────────────────┘
```

### Relationship Types

| Relationship | From | To | Properties | Description |
|-------------|------|-----|------------|-------------|
| `SIMILAR` | UserQuestion | Query | `score: FLOAT` | Links verbatim question to matched pattern |
| `FEW_SHOT_EXAMPLE` | Query | FewShot | - | Links query pattern to approved Cypher |
| `USES_TOOL` | Query | Tool | `recommended: BOOL` | Recommends tool for this query type |
| `GENERATED` | Query | CypherAttempt | - | Audit: queries generated during exploration |
| `PRODUCED` | CypherAttempt | Response | - | Audit: responses from attempts |

---

## Flow

### 1. New Question Arrives
```
User asks: "What government position did Shirley Temple hold?"
           ↓
       [Embed question]
           ↓
       [Vector search Query nodes]
           ↓
    ┌──────┴──────┐
    │             │
 MATCH         NO MATCH
    │             │
    ▼             ▼
[Store         [Store Query + 
UserQuestion]   UserQuestion]
    │             │
    ▼             ▼
[Use FewShot]  [Run Agent]
```

### 2. Similar Question Found (Instant Path)
```cypher
// 1. Vector search finds similar approved Query
CALL db.index.vector.queryNodes('query_embeddings', 5, $embedding)
YIELD node, score
WHERE node.status = 'approved' AND score >= 0.90

// 2. Store UserQuestion linked via SIMILAR
CREATE (uq:UserQuestion {text: $verbatim_text, timestamp: datetime()})
WITH uq
MATCH (q:Query {id: $matched_query_id})
CREATE (uq)-[:SIMILAR {score: $score}]->(q)

// 3. Execute the FewShot cypher
MATCH (q:Query {id: $matched_query_id})-[:FEW_SHOT_EXAMPLE]->(fs:FewShot)
RETURN fs.cypher_template
```

### 3. No Match (Agentic Path)
```cypher
// 1. Create new Query (becomes canonical if approved)
CREATE (q:Query {
    text: $question, 
    embedding: $embedding, 
    status: 'pending_approval'
})

// 2. Store UserQuestion linked to new Query
CREATE (uq:UserQuestion {text: $verbatim_text})
CREATE (uq)-[:SIMILAR {score: 1.0}]->(q)

// 3. Run agent, store attempts
// ... agent execution ...

// 4. On approval, create FewShot
CREATE (fs:FewShot {cypher_template: $refined_cypher})
CREATE (q)-[:FEW_SHOT_EXAMPLE]->(fs)
```

---

## Indexes

```cypher
// Vector index for Query similarity matching
CREATE VECTOR INDEX query_embeddings IF NOT EXISTS
FOR (q:Query) ON (q.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

// Status index for filtering
CREATE INDEX query_status_idx IF NOT EXISTS
FOR (q:Query) ON (q.status);

// Tool name index
CREATE INDEX tool_name_idx IF NOT EXISTS
FOR (t:Tool) ON (t.name);
```

---

## Key Design Decisions

1. **UserQuestion has no embedding** - It's just an audit trail. We match against Query embeddings.

2. **Query is the canonical pattern** - Multiple UserQuestions cluster around the same Query via SIMILAR.

3. **FewShot is the executable** - The approved Cypher that answers this question type.

4. **Tool recommendations** - Each Query can recommend which tool works best for it.

5. **Verbatim matching** - New questions match against existing verbatim Query texts (high similarity), not abstract parameterized templates.

---

## Migration from v2

If you have existing `QueryTemplate` nodes:
```cypher
// Convert QueryTemplate to Query pattern
MATCH (qt:QueryTemplate)
CREATE (q:Query {
    id: qt.id,
    text: qt.template,
    embedding: qt.embedding,
    status: qt.status,
    timestamp: qt.created_at
})
WITH qt, q
MATCH (qt)-[:FEW_SHOT_EXAMPLE]->(fs:FewShotCypher)
CREATE (q)-[:FEW_SHOT_EXAMPLE]->(f:FewShot {
    cypher_template: fs.cypher_template,
    parameters: fs.parameters
})
```
