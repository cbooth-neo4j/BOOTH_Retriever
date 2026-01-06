# BOOTH Retriever

**Bounded Orchestration Of Text Handling** - An intelligent retrieval system that learns to answer natural language questions about your Neo4j database through semantic similarity search and human-in-the-loop curation.

## Overview

BOOTH uses vector embeddings to match queries against approved examples, generates Cypher queries from natural language, and improves over time through human curation. All data—embeddings, queries, and results—live in your Neo4j database.

### How It Works
See [architecture.md](./docs/architecture.md) for a system architecture diagram and workflow details.

## Quick Start

### Prerequisites
- Python 3.8+
- Neo4j 5.11+ (running with vector index support)
- OpenAI API key

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials

# 3. Initialize database
python scripts/init_neo4j.py

# 4. Launch application
streamlit run app.py
```

Visit `http://localhost:8501` and start querying!

## Using Neo4j Database Dumps

### Creating a Database Dump (Neo4j Desktop)

To share your BOOTH database or create a backup:

1. **Open Neo4j Desktop** and stop your database
2. **Click on your database** in the Projects pane
3. **Click the "..." menu** (three dots) next to your database
4. **Select "Dump"** from the dropdown menu
5. **Choose a location** to save the dump file (e.g., `booth_database.dump`)

Alternatively, use the Neo4j admin tool from the command line:

```bash
# Navigate to your Neo4j installation directory
cd /path/to/neo4j

# Create a dump
bin/neo4j-admin database dump neo4j --to-path=/path/to/dumps/
```

### Importing a Database Dump (Neo4j Desktop)

1. **Open Neo4j Desktop**
2. **Create a new database** in your project (or select an existing empty database)
3. **Stop the database** if it's running
4. **Click the "..." menu** next to the database
5. **Select "Restore"** or "Load from a dump"
6. **Browse to your dump file** (e.g., `booth_database.dump`)
7. **Confirm the import** - this will replace the current database
8. **Start the database**

## Using Neo4j with MCP (Model Context Protocol)

BOOTH includes MCP integration for direct Neo4j queries through Cursor/Claude. Configuration is in `mcp.json`.

**Quick setup:**
1. Ensure Neo4j is running locally (bolt://localhost:7687)
2. Update credentials in `mcp.json` if needed
3. MCP tools will be available in Cursor for querying your graph

📺 [Watch setup guide](https://www.youtube.com/watch?v=UilGH0j73rI)

## Architecture

See `docs/architecture.md` for the full Mermaid diagram of the system flow.

### Iterative Refinement for Text2Cypher

BOOTH implements an advanced iterative refinement loop based on [Neo4j's research](https://neo4j.com/blog/developer/iterative-refinement-for-text2cypher/). Their research tested all four verifiers (Rule-based, CyVer, Execution, LLM) and found LLM-based was slowest and flagged 15-20x more invalids (many false positives), while rule-based was 600x faster. We default to `RULE_BASED,EXECUTION_BASED` for optimal speed/accuracy balance, but you can use all four verifiers like the research by setting `CYPHER_VERIFIERS=RULE_BASED,CYVER,EXECUTION_BASED,LLM_BASED`.

```
1. Generate Cypher query (LLM with schema + few-shot examples)
   ↓
2. Verify query (RULE_BASED → CYVER → EXECUTION_BASED → LLM_BASED)
   ↓
3. Is Valid?
   ├─ Yes → Execute → Success? 
   │         ├─ Yes → Return results ✓
   │         └─ No → Correction loop
   └─ No → Correction loop
   
Correction loop:
4. Correct query (RULE_BASED → LLM_BASED)
   ↓
5. Iteration < MAX_RETRIES?
   ├─ Yes → Back to step 2 (Verify)
   └─ No → Decline and log for human review
```

**Stop Criteria**:
- Max 3 iterations (configurable via `MAX_CYPHER_RETRIES`)
- Early stop on successful execution
- All verifiers pass OR no corrections can be applied


### Key Components

- **`src/llm_client.py`**: OpenAI embeddings, GPT-4 for Cypher generation and summarization
- **`src/neo4j_client.py`**: Vector search, query storage, Cypher execution, curation operations
- **`src/booth_orchestrator.py`**: Main workflow controller with iterative refinement loop
- **`src/cypher_verification.py`**: Cypher verification techniques (rule-based, CyVer, execution, LLM)
- **`src/cypher_correction.py`**: Cypher correction techniques (rule-based, LLM)
- **`app.py`**: Streamlit query interface
- **`pages/1_Curation.py`**: Human curation interface

## Configuration

Edit `.env` to customize:

```bash
# Required
OPENAI_API_KEY=sk-your-key-here
NEO4J_PASSWORD=your-password

# Optional (with defaults)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
OPENAI_CHAT_MODEL=gpt-4                          # OpenAI model for chat/completions
OPENAI_EMBEDDING_MODEL=text-embedding-3-small    # OpenAI model for embeddings
SIMILARITY_THRESHOLD=0.90          # 0-1, higher = stricter matching
MAX_CYPHER_RETRIES=3               # Max iterations for verification-correction loop

# Iterative Refinement Configuration
CYPHER_VERIFIERS=RULE_BASED,EXECUTION_BASED  # Comma-separated: RULE_BASED, CYVER, EXECUTION_BASED, LLM_BASED
CYPHER_CORRECTORS=RULE_BASED,LLM_BASED       # Comma-separated: RULE_BASED, LLM_BASED
USE_VERIFICATION_METADATA=true               # Include verification details in correction
```

### Verification & Correction Options

**Verifiers** (executed in order until failure):
- `RULE_BASED`: Fast regex checks (~1.5ms) - relationship direction, brackets, basic syntax
- `CYVER`: Cypher syntax validation - RETURN clauses, WHERE conditions, property access
- `EXECUTION_BASED`: Try executing against database (slower, most accurate)
- `LLM_BASED`: Use LLM to validate intent match (slowest, most comprehensive)

**Correctors** (executed in order until success):
- `RULE_BASED`: Fix common syntax errors - relationship direction, brackets, markdown
- `LLM_BASED`: Intelligent correction based on error feedback and schema

**Performance Tips**:
- Use `RULE_BASED` for speed (~600x faster than LLM)
- Add `EXECUTION_BASED` for accuracy (catches runtime errors)
- Add `LLM_BASED` for complex validation (costs more, slower)
- Most corrections happen in first iteration; loop helps edge cases

## Project Structure

```
BOOTH_Retriever/
├── src/
│   ├── booth_orchestrator.py   # Main workflow controller
│   ├── llm_client.py            # OpenAI integration
│   ├── neo4j_client.py          # Neo4j operations
│   └── logger.py                # Logging configuration
├── scripts/
│   └── init_neo4j.py            # Database initialization
├── pages/
│   └── 1_Curation.py            # Curation interface
├── logs/                        # Log files (auto-created)
├── app.py                       # Main Streamlit app
├── requirements.txt             # Dependencies
└── env.example                  # Config template
```

## Logging

BOOTH includes comprehensive logging to help you validate operations and troubleshoot issues:

### Log Files
- Located in `logs/` directory (auto-created)
- Named by date: `booth_YYYYMMDD.log`
- Automatically rotated daily

### Log Levels
- **Console**: INFO and above (less verbose)
- **File**: DEBUG and above (detailed)

### What's Logged
- Query submissions and processing
- Similarity searches and matches
- Cypher generation attempts and execution
- Database operations (stores, retrieves)
- API calls to OpenAI
- Approval/rejection actions in curation
- Errors with full stack traces

### Example Log Entry
```
2025-11-20 14:32:15 - INFO - Processing query: 'List all users...' (high_risk=False)
2025-11-20 14:32:15 - DEBUG - Generating embedding for user query
2025-11-20 14:32:16 - INFO - Found 2 similar queries (best score: 0.9234)
2025-11-20 14:32:17 - INFO - Generated Cypher query: MATCH (u:User) RETURN u.name...
2025-11-20 14:32:17 - INFO - Query completed successfully (query_id: abc-123)
```

### Viewing Logs
```bash
# View today's log in real-time
tail -f logs/booth_$(date +%Y%m%d).log

# Search for errors
grep -i "error" logs/booth_*.log

# View specific query ID
grep "query_id: abc-123" logs/booth_*.log
```