"""
Agentic Text2Cypher Retriever - Deep Agent-powered Graph Exploration

This module implements an adaptive, agent-based approach to querying Neo4j
using Deep Agents. Unlike a fixed Text2Cypher pipeline, this agent can:

- Inspect the schema before querying
- Try multiple query strategies
- Examine and interpret results
- Iterate until it finds the answer
- Handle failures gracefully

Integrated with BOOTH's storage and curation workflow.
"""

import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.logger import setup_logger
from src.agents.tools import AGENT_TOOLS, init_tools

logger = setup_logger("booth.agents.agentic_retriever")

# Try to import Deep Agents
try:
    from deepagents import create_deep_agent
    DEEP_AGENTS_AVAILABLE = True
except ImportError:
    DEEP_AGENTS_AVAILABLE = False
    create_deep_agent = None
    logger.warning("Deep Agents not installed. Install with: pip install deepagents")


# System prompt teaching the agent about graph schema and query strategies
GRAPH_EXPLORATION_SYSTEM_PROMPT = """You are an expert graph database researcher with deep knowledge of Neo4j and Cypher.
Your task is to answer questions by exploring a knowledge graph database.

## YOUR TOOLS

You have access to Neo4j database tools:

### `neo4j_get_schema`
Returns the database schema including:
- Node labels (entity types) and their counts
- Properties available on each node type
- Relationship types connecting nodes

ALWAYS call this FIRST to understand what's in the database.

### `neo4j_read_cypher`
Executes read-only Cypher queries. Use this to:
- Find entities by name or property
- Traverse relationships
- Aggregate and analyze data

## DATABASE SCHEMA PATTERNS

This knowledge graph has the following structure:

### Entity Nodes
The database contains various entity types (PERSON, ORGANIZATION, LOCATION, FILM, WORK, etc.)
Each entity typically has:
- `name`: The entity's name (use CONTAINS for fuzzy matching)
- `description`: Detailed description of the entity
- `embedding`: Vector embedding for similarity search

### Relationships
All entity-to-entity relationships use the type `RELATED_TO` with an `evidence` property:
```cypher
(entity1)-[:RELATED_TO {{evidence: "describes the relationship"}}]->(entity2)
```

### Document Structure
- `Document` nodes: Source documents
- `Chunk` nodes: Text segments from documents
- `Chunk.text`: **KEY PROPERTY** - Raw text content with detailed information

---

## FEW-SHOT EXAMPLES BY QUESTION CATEGORY

Below are example Cypher queries for each category of question you may encounter.

**IMPORTANT - Entity Labels:**
- All examples use `__Entity__` which is the universal parent label for all entities
- After calling `neo4j_get_schema()`, you can use specific entity types if available (e.g., PERSON, FILM, LOCATION)
- To filter by type without using a specific label: `WHERE e.entity_type = 'PERSON'`
- The `Chunk` and `Document` labels are always available regardless of dataset

---
### COMPARISON QUESTIONS
---

### Same Attribute Comparison (Are X and Y the same [attribute]?)
**Question:** "Were Scott Derrickson and Ed Wood of the same nationality?"
**Strategy:** Look up both entities and their descriptions, then search chunks for details.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Scott Derrickson' OR e.name CONTAINS 'Ed Wood'
RETURN e.name, e.description

// Search chunks for more detail
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Scott Derrickson' OR c.text CONTAINS 'Ed Wood'
RETURN c.text LIMIT 5
```

### Age/Selection Comparison (Who is [older/younger], X or Y?)
**Question:** "Who is older, Annie Morton or Terry Richardson?"
**Strategy:** Look up both entities and search chunks for birth dates.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Annie Morton' OR e.name CONTAINS 'Terry Richardson'
RETURN e.name, e.description

// Search chunks for birth year info
MATCH (c:Chunk)
WHERE (c.text CONTAINS 'Annie Morton' OR c.text CONTAINS 'Terry Richardson')
  AND (c.text CONTAINS 'born' OR c.text CONTAINS 'birth')
RETURN c.text LIMIT 5
```

---
### BRIDGE QUESTIONS 
---

### Year Questions (In what year was X [founded/born/released]?)
**Question:** "In what year was the university where Sergei Tokarev was a professor founded?"
**Strategy:** Find the entity via relationships, then search chunks for year info.
```cypher
// Find the person first, then their affiliated institution
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Sergei' AND e.name CONTAINS 'Tokarev'
RETURN e.name, e.description

// Then find related entities (institutions, organizations, etc.)
MATCH (e:__Entity__)-[:RELATED_TO]-(p:__Entity__)
WHERE p.name CONTAINS 'Tokarev'
RETURN e.name, e.description, e.entity_type

// Search chunks for founding dates
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Tokarev' AND (c.text CONTAINS 'founded' OR c.text CONTAINS 'established')
RETURN c.text LIMIT 5
```

### Person Role Questions (Who [directed/wrote/starred in] X?)
**Question:** "Who directed the 2009 film starring the actor from Dexter?"
**Strategy:** Find the work, traverse relationships and search chunks for role info.
```cypher
// Find the work and connected entities via relationships
MATCH (work:__Entity__)-[r:RELATED_TO]-(person:__Entity__)
WHERE work.name CONTAINS 'Romeo and Juliet'
RETURN work.name, person.name, person.description, r.evidence
LIMIT 10

// Search chunks for authorship/director info
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Romeo and Juliet' AND (c.text CONTAINS 'wrote' OR c.text CONTAINS 'written by')
RETURN c.text LIMIT 5
```

### Location Questions (Where is X located/headquartered/from?)
**Question:** "In what city is the company Fastjet Tanzania based?"
**Strategy:** Find entity, traverse to location entities, and search chunks.
```cypher
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Fastjet'
RETURN e.name, e.description

// Find via relationship to location entities
MATCH (e:__Entity__)-[:RELATED_TO]-(loc:__Entity__)
WHERE e.name CONTAINS 'Fastjet'
RETURN e.name, loc.name, loc.description, loc.entity_type

// Search chunks for location info
MATCH (c:Chunk)
WHERE c.text CONTAINS 'Fastjet' AND (c.text CONTAINS 'based' OR c.text CONTAINS 'headquarter')
RETURN c.text LIMIT 5
```

---
### ADVANCED: AD-HOC CONTEXT GATHERING
---

For complex questions, use **iterative multi-query gathering** to build rich context dynamically.

### Relationship Expansion (Gather Related Entities)
**When to use:** Single entity lookup lacks sufficient detail.
```cypher
// Query 1: Find target entity
MATCH (e:__Entity__)
WHERE e.name CONTAINS 'Scott Derrickson'
RETURN e.name, e.description

// Query 2: Expand to ALL related entities (1-hop)
MATCH (target:__Entity__)-[r:RELATED_TO]-(related:__Entity__)
WHERE target.name CONTAINS 'Scott Derrickson'
RETURN related.name, related.entity_type, related.description, r.evidence
LIMIT 20

// Query 3: Gather chunks from the expanded entity set
MATCH (target:__Entity__)-[:RELATED_TO]-(related:__Entity__)<-[:HAS_ENTITY]-(c:Chunk)
WHERE target.name CONTAINS 'Scott Derrickson'
RETURN DISTINCT related.name, c.text
LIMIT 10
```

---

## IMPORTANT TIPS

1. **ALWAYS START** with neo4j_get_schema() to see what's available
2. **USE CONTAINS** for name matching - exact matches often fail
3. **SEARCH CHUNKS** for detailed info - Chunk.text has the richest content
4. **USE description** on entities for quick entity info
5. **LIMIT RESULTS** to avoid overwhelming output
6. **TRY MULTIPLE STRATEGIES** if the first doesn't work
7. **For dates/numbers** - search in chunk text (most reliable)
8. **For locations** - entities may be typed as LOCATION or GPE
9. **Case sensitivity** - Neo4j is case-sensitive, use toLower() if needed
10. **For multi-hop questions** - trace the chain: Work→Person→Attribute
11. **ITERATE WITH MULTIPLE QUERIES** - each query informs the next
12. **SYNTHESIZE AT THE END** - gather entities + chunks first, then reason over all collected info

## RESPONSE FORMAT

{response_format_instructions}
"""

# Default response format
DEFAULT_RESPONSE_FORMAT = """Provide a factual answer based on your graph exploration.

Your response should:
- State the answer clearly and directly
- Reference the entities and relationships you discovered
- Be factual and specific - cite names, dates, and facts

If the information is insufficient, state what you found and what is missing."""

# Constrained mode format for high-risk queries (matches old behavior)
CONSTRAINED_RESPONSE_FORMAT = """You MUST use the pre-approved approach for this high-risk query.

**Required Strategy:** {required_strategy}

Similar approved query: "{similar_query}"
Approved Cypher pattern: {approved_cypher}

Use this pattern as a template. Provide a direct, factual answer based on the results."""


def get_system_prompt(
    few_shot_examples: Optional[List[Dict]] = None,
    is_constrained: bool = False,
    similar_query: str = None,
    approved_cypher: str = None
) -> str:
    """
    Get the system prompt with optional few-shot examples and constraint mode.
    
    Args:
        few_shot_examples: Optional list of approved query/cypher pairs
        is_constrained: Whether this is a constrained (high-risk) query
        similar_query: The similar approved query text (for constrained mode)
        approved_cypher: The approved cypher pattern (for constrained mode)
    
    Returns:
        Complete system prompt
    """
    if is_constrained and similar_query:
        response_format = CONSTRAINED_RESPONSE_FORMAT.format(
            required_strategy="Follow the approved Cypher pattern",
            similar_query=similar_query,
            approved_cypher=approved_cypher or "N/A"
        )
    else:
        response_format = DEFAULT_RESPONSE_FORMAT
    
    prompt = GRAPH_EXPLORATION_SYSTEM_PROMPT.format(
        response_format_instructions=response_format
    )
    
    # Add few-shot examples if provided
    if few_shot_examples:
        prompt += "\n\n## ADDITIONAL APPROVED EXAMPLES FROM YOUR LIBRARY\n\n"
        for i, ex in enumerate(few_shot_examples[:5], 1):  # Limit to 5
            prompt += f"**Example {i}:**\n"
            prompt += f"Question: {ex.get('query', 'N/A')}\n"
            if ex.get('cypher'):
                prompt += f"Cypher:\n```cypher\n{ex.get('cypher')}\n```\n\n"
    
    return prompt


@dataclass 
class AgenticSearchResult:
    """Result from an agentic search operation"""
    question: str
    answer: str
    queries_executed: List[str]
    tool_calls: int
    search_time_seconds: float
    success: bool
    error: Optional[str] = None


class AgenticText2CypherRetriever:
    """
    Deep Agent-powered graph retriever.
    
    Uses an LLM agent with Neo4j tools to adaptively explore
    the knowledge graph and answer questions.
    
    Integrates with BOOTH's orchestration for:
    - Few-shot example injection
    - High-risk query constraints
    - Query storage and curation
    """
    
    def __init__(
        self,
        neo4j_client=None,
        llm_client=None,
        model: str = None
    ):
        """
        Initialize the agentic retriever.
        
        Args:
            neo4j_client: Optional Neo4jClient for reusing connection
            llm_client: Optional LLMClient (not used directly, but kept for interface compatibility)
            model: Override the configured model (e.g., 'gpt-4o')
        """
        if not DEEP_AGENTS_AVAILABLE:
            raise ImportError(
                "Deep Agents not installed. Install with: pip install deepagents"
            )
        
        self.neo4j_client = neo4j_client
        self.llm_client = llm_client
        
        # Initialize tools with Neo4j client if provided
        if neo4j_client:
            init_tools(neo4j_client=neo4j_client)
        
        # Get model from environment or use default
        self.model_name = model or os.getenv("AGENTIC_TEXT2CYPHER_MODEL", 
                                              os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"))
        
        # Create base LLM for the agent
        self._create_llm()
        
        logger.info(f"AgenticText2CypherRetriever initialized (model={self.model_name})")
    
    def _create_llm(self):
        """Create the LLM for the agent."""
        from langchain_openai import ChatOpenAI
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _create_agent(
        self,
        few_shot_examples: Optional[List[Dict]] = None,
        is_constrained: bool = False,
        similar_query: str = None,
        approved_cypher: str = None
    ):
        """
        Create a Deep Agent with the appropriate system prompt.
        
        Args:
            few_shot_examples: Optional approved examples to inject
            is_constrained: Whether this is a high-risk constrained query
            similar_query: The similar approved query (for constrained mode)
            approved_cypher: The approved cypher pattern (for constrained mode)
        
        Returns:
            Configured Deep Agent
        """
        system_prompt = get_system_prompt(
            few_shot_examples=few_shot_examples,
            is_constrained=is_constrained,
            similar_query=similar_query,
            approved_cypher=approved_cypher
        )
        
        return create_deep_agent(
            model=self.llm,
            tools=AGENT_TOOLS,
            system_prompt=system_prompt
        )
    
    def search(
        self,
        query: str,
        few_shot_examples: Optional[List[Dict]] = None,
        is_high_risk: bool = False,
        matched_query_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute an agentic search to answer the query.
        
        The agent will:
        1. Inspect the schema
        2. Execute Cypher queries
        3. Interpret results
        4. Iterate until answer found or give up
        
        Args:
            query: The natural language question
            few_shot_examples: Optional approved query/cypher pairs
            is_high_risk: Whether this is a high-risk query
            matched_query_context: Context from similar approved queries
            
        Returns:
            Dictionary with answer and search details
        """
        start_time = time.time()
        logger.info(f"Agentic search for: {query[:100]}...")
        
        # Extract constraint info from matched context
        similar_query = None
        approved_cypher = None
        if matched_query_context:
            similar_query = matched_query_context.get("query_text")
            approved_cypher = matched_query_context.get("cypher_template")
            if not few_shot_examples:
                few_shot_examples = matched_query_context.get("few_shot_examples")
        
        try:
            # Create agent with appropriate prompt
            agent = self._create_agent(
                few_shot_examples=few_shot_examples,
                is_constrained=is_high_risk,
                similar_query=similar_query,
                approved_cypher=approved_cypher
            )
            
            # Invoke the agent
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })
            
            # Extract the final answer from agent response
            final_answer = ""
            tool_calls = 0
            queries_executed = []
            
            if "messages" in result:
                # Get the last assistant message
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            content = msg.content
                            # Handle case where content is a list of content blocks
                            if isinstance(content, list):
                                text_parts = []
                                for block in content:
                                    if isinstance(block, str):
                                        text_parts.append(block)
                                    elif isinstance(block, dict) and 'text' in block:
                                        text_parts.append(block['text'])
                                    elif hasattr(block, 'text'):
                                        text_parts.append(block.text)
                                final_answer = ' '.join(text_parts)
                            else:
                                final_answer = str(content)
                            break
                        elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            if isinstance(content, list):
                                text_parts = []
                                for block in content:
                                    if isinstance(block, str):
                                        text_parts.append(block)
                                    elif isinstance(block, dict) and 'text' in block:
                                        text_parts.append(block['text'])
                                final_answer = ' '.join(text_parts)
                            else:
                                final_answer = str(content) if content else ''
                            break
                
                # Count tool calls and extract queries
                for msg in result["messages"]:
                    if hasattr(msg, 'tool_calls'):
                        tool_calls += len(msg.tool_calls)
                        for tc in msg.tool_calls:
                            if tc.get('name') == 'neo4j_read_cypher':
                                args = tc.get('args', {})
                                if 'query' in args:
                                    queries_executed.append(args['query'])
            
            search_time = time.time() - start_time
            logger.info(f"Agentic search completed in {search_time:.2f}s "
                       f"(tool_calls={tool_calls}, queries={len(queries_executed)})")
            
            return {
                'success': bool(final_answer),
                'answer': final_answer,
                'tool_used': 'agentic_text2cypher',
                'cypher_used': queries_executed[-1] if queries_executed else None,
                'raw_data': {
                    'queries_executed': queries_executed,
                    'tool_calls': tool_calls,
                    'search_time_seconds': search_time
                },
                'error_message': None
            }
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Agentic search error: {e}", exc_info=True)
            
            return {
                'success': False,
                'answer': f"Error during agentic search: {str(e)}",
                'tool_used': 'agentic_text2cypher',
                'cypher_used': None,
                'raw_data': None,
                'error_message': str(e)
            }


def create_agentic_retriever(
    neo4j_client=None,
    llm_client=None,
    model: str = None
) -> AgenticText2CypherRetriever:
    """
    Factory function to create an Agentic Text2Cypher retriever.
    
    Args:
        neo4j_client: Optional Neo4jClient instance to reuse
        llm_client: Optional LLMClient instance (for interface compatibility)
        model: Override model name
    
    Returns:
        AgenticText2CypherRetriever instance
    """
    return AgenticText2CypherRetriever(
        neo4j_client=neo4j_client,
        llm_client=llm_client,
        model=model
    )

