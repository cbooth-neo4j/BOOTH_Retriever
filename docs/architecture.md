# BOOTH Retriever Architecture

## System Flow Diagram (Mermaid)

```mermaid
flowchart TD
    A[User Prompt] --> B[Embed user prompt]
    B --> C[HNSW against stored queries]
    C --> D{Over ~90% Similar?}
    
    D -->|Yes| E[Retrieve linked few-shot prompt]
    E --> F[LLM Summary]
    F --> G[Output Answer]
    
    D -->|No| H{High risk query?}
    
    H -->|No| I[Text2Cypher with Iterative Refinement]
    H -->|Yes| I2[Run Text2Cypher in Background]
    
    I --> J[Generate Cypher]
    I2 --> J
    
    J --> K[Verify Cypher]
    K --> L{Valid?}
    
    L -->|Yes| M[Execute Cypher]
    M --> N{Execution Success?}
    
    N -->|Yes| X{High Risk?}
    X -->|No| F
    X -->|Yes| Q[Decline to User & Store Results]
    
    N -->|No| O{Max iterations reached?}
    
    L -->|No| O
    
    O -->|No| P[Correct Cypher]
    P --> J
    
    O -->|Yes| Q
    Q --> R[Human Review in Train AI]
    R --> S{Human Approval?}
    
    S -->|Yes| T[Add to few-shot library]
    T --> C
    
    S -->|No| U[Reject Query]
    
    G --> V[End]
    Q --> V
    
    style I fill:#e1f5ff
    style I2 fill:#ffe1e1
    style K fill:#ffe1e1
    style P fill:#fff4e1
    style Q fill:#ffe1e1
```

## Verification Techniques (in Iterative Refinement)

1. **Rule-Based Relation Direction**: Fast regex checks for correct relationship direction syntax
2. **CyVer-Based Verification**: Cypher syntax validation using CyVer library
3. **Execution-Based**: Try executing against database (slower but accurate)
4. **LLM-Based**: Use LLM to validate if Cypher matches intent (slowest, most comprehensive)

## Correction Techniques

1. **Rule-Based**: Fix common relation direction errors automatically
2. **LLM-Based**: Use LLM to correct Cypher based on verification feedback

## Key Components

- **`src/booth_orchestrator.py`**: Main workflow controller with iterative refinement loop
- **`src/cypher_verification.py`**: Verification techniques for Cypher validation
- **`src/cypher_correction.py`**: Correction techniques for Cypher refinement
- **`src/llm_client.py`**: OpenAI integration for embeddings, generation, and correction
- **`src/neo4j_client.py`**: Neo4j operations and Cypher execution
- **`app.py`**: Streamlit query interface
- **`pages/1_Train_AI.py`**: Human curation interface (Train AI page)

## High-Risk Query Handling

When a query is marked as high-risk:
1. The query is declined to the user (results not shown)
2. The full text2cypher pipeline runs in the background
3. All attempts, results, and summaries are stored in Neo4j
4. Curators can review what would have happened in the Train AI page
5. Safe queries can be approved to become few-shot examples

## Stop Criteria

- Max 3 iterations (configurable)
- Early stop if valid Cypher is generated and executes successfully

