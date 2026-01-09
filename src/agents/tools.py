"""
Neo4j Agent Tools - Direct Neo4j driver wrappers for Deep Agents

This module provides Python function wrappers around Neo4j operations,
mirroring the functionality of the Neo4j MCP server tools:
- get_schema: Introspect labels, relationship types, property keys
- read_cypher: Execute read-only Cypher queries

These tools are designed for use with Deep Agents (create_deep_agent).
"""

import os
from typing import Dict, Any, List, Optional

from src.logger import setup_logger

logger = setup_logger("booth.agents.tools")

# Neo4j configuration from environment
NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DB = os.environ.get('NEO4J_DATABASE', 'neo4j')


class Neo4jAgentTools:
    """
    Neo4j tools for Deep Agent graph exploration.
    
    Provides the same capabilities as the Neo4j MCP server but as
    Python functions callable by an LLM agent.
    """
    
    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None,
        neo4j_client = None
    ):
        """
        Initialize Neo4j tools with connection parameters.
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            username: Database username (defaults to NEO4J_USER env var)
            password: Database password (defaults to NEO4J_PASSWORD env var)
            database: Database name (defaults to NEO4J_DATABASE env var)
            neo4j_client: Optional existing Neo4jClient instance to reuse
        """
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DB
        self._neo4j_client = neo4j_client
        self._driver = None
        
        if not self._neo4j_client and not all([self.uri, self.username, self.password]):
            raise ValueError("Neo4j connection parameters required: uri, username, password")
        
        logger.info(f"Neo4j Agent Tools initialized for database: {self.database}")
    
    def _get_driver(self):
        """Get or create a Neo4j driver connection."""
        if self._neo4j_client:
            return self._neo4j_client.driver
        
        if self._driver is None:
            import neo4j
            self._driver = neo4j.GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
        return self._driver
    
    def close(self):
        """Close the driver connection if we own it."""
        if self._driver is not None and self._neo4j_client is None:
            self._driver.close()
            self._driver = None
    
    def get_schema(self) -> str:
        """
        Get the Neo4j database schema including node labels, relationship types,
        and property keys.
        
        Returns:
            String representation of the database schema
        """
        logger.debug("Fetching Neo4j schema")
        
        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                # Get node labels and their properties
                labels_result = session.run("""
                    CALL db.labels() YIELD label
                    RETURN collect(label) as labels
                """)
                labels = labels_result.single()["labels"]
                
                # Get relationship types
                rel_result = session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    RETURN collect(relationshipType) as types
                """)
                rel_types = rel_result.single()["types"]
                
                # Get property keys
                props_result = session.run("""
                    CALL db.propertyKeys() YIELD propertyKey
                    RETURN collect(propertyKey) as keys
                """)
                prop_keys = props_result.single()["keys"]
                
                # Get sample node counts per label
                label_counts = {}
                for label in labels:
                    # Skip internal BOOTH labels
                    if label in ['Query', 'CypherAttempt', 'Response']:
                        continue
                    try:
                        count_result = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count")
                        label_counts[label] = count_result.single()["count"]
                    except:
                        label_counts[label] = 0
                
                # Get sample properties per label (from first few nodes)
                label_properties = {}
                for label in labels:
                    if label in ['Query', 'CypherAttempt', 'Response']:
                        continue
                    try:
                        sample = session.run(f"""
                            MATCH (n:`{label}`) 
                            RETURN keys(n) as props 
                            LIMIT 1
                        """)
                        record = sample.single()
                        if record:
                            label_properties[label] = record["props"]
                    except:
                        label_properties[label] = []
                
                # Format schema output
                schema_parts = []
                schema_parts.append("=== NEO4J DATABASE SCHEMA ===\n")
                
                schema_parts.append("NODE LABELS (with counts):")
                for label in sorted(labels):
                    if label in ['Query', 'CypherAttempt', 'Response']:
                        continue  # Skip internal BOOTH labels
                    props = label_properties.get(label, [])
                    props_str = ", ".join(props[:10])  # First 10 properties
                    if len(props) > 10:
                        props_str += f"... (+{len(props)-10} more)"
                    schema_parts.append(f"  - {label}: {label_counts.get(label, 0)} nodes")
                    if props_str:
                        schema_parts.append(f"    Properties: {props_str}")
                
                schema_parts.append("\nRELATIONSHIP TYPES:")
                for rel_type in sorted(rel_types):
                    # Skip internal BOOTH relationships
                    if rel_type in ['GENERATED', 'PRODUCED', 'SIMILAR_TO', 'FEW_SHOT_PROMPT']:
                        continue
                    schema_parts.append(f"  - {rel_type}")
                
                schema_parts.append("\nKEY PROPERTY KEYS:")
                # Show most relevant properties (excluding internal ones)
                internal_props = ['embedding', 'status', 'timestamp', 'risk_level', 
                                 'similarity_matched', 'attempt_number', 'cypher_text',
                                 'result_data', 'summary', 'error_message', 'rejection_reason']
                relevant_props = [p for p in prop_keys if p not in internal_props and not p.startswith('_')][:30]
                schema_parts.append(f"  {', '.join(sorted(relevant_props))}")
                
                schema = "\n".join(schema_parts)
                logger.debug(f"Schema extracted: {len(schema)} characters")
                return schema
                
        except Exception as e:
            error_msg = f"Error fetching schema: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def read_cypher(self, query: str, params: Dict[str, Any] = None) -> str:
        """
        Execute a read-only Cypher query against the Neo4j database.
        
        Write operations (CREATE, MERGE, DELETE, SET) will be rejected.
        
        Args:
            query: The Cypher query to execute
            params: Optional parameters for the query
            
        Returns:
            String representation of query results
        """
        logger.debug(f"Executing Cypher query: {query[:100]}...")
        
        # Check for write operations
        query_upper = query.upper().strip()
        write_keywords = ['CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'DROP', 'DETACH']
        
        for keyword in write_keywords:
            # Check if keyword appears as a standalone word (not in comments)
            if f' {keyword} ' in f' {query_upper} ' or query_upper.startswith(keyword):
                error_msg = f"Write operations not allowed in read_cypher. Found '{keyword}' clause."
                logger.warning(error_msg)
                return f"ERROR: {error_msg}"
        
        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                records = list(result)
                
                if not records:
                    return "Query returned no results."
                
                # Format results
                output_parts = []
                output_parts.append(f"Query returned {len(records)} result(s):\n")
                
                for i, record in enumerate(records[:50], 1):  # Limit to 50 results
                    record_dict = dict(record)
                    # Format each record nicely
                    formatted = {}
                    for key, value in record_dict.items():
                        if hasattr(value, '__dict__'):
                            # Neo4j Node or Relationship
                            if hasattr(value, 'labels'):
                                # Node
                                formatted[key] = {
                                    'labels': list(value.labels),
                                    'properties': dict(value)
                                }
                            elif hasattr(value, 'type'):
                                # Relationship
                                formatted[key] = {
                                    'type': value.type,
                                    'properties': dict(value)
                                }
                            else:
                                formatted[key] = str(value)
                        else:
                            formatted[key] = value
                    output_parts.append(f"[{i}] {formatted}")
                
                if len(records) > 50:
                    output_parts.append(f"\n... and {len(records) - 50} more results (truncated)")
                
                output = "\n".join(output_parts)
                logger.debug(f"Query returned {len(records)} results")
                return output
                
        except Exception as e:
            error_msg = f"Cypher error: {str(e)}"
            logger.warning(error_msg)
            return f"ERROR: {error_msg}"


# Singleton tools instance
_tools_instance = None


def _get_tools(neo4j_client=None) -> Neo4jAgentTools:
    """Get or create the singleton tools instance."""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = Neo4jAgentTools(neo4j_client=neo4j_client)
    return _tools_instance


def init_tools(neo4j_client=None):
    """Initialize tools with an optional Neo4j client."""
    global _tools_instance
    _tools_instance = Neo4jAgentTools(neo4j_client=neo4j_client)
    return _tools_instance


def neo4j_get_schema() -> str:
    """
    Get the Neo4j database schema.
    
    Returns information about:
    - Node labels and their counts
    - Properties on each node type
    - Relationship types
    - Property keys
    
    Use this FIRST to understand what's in the database before writing queries.
    
    Returns:
        String describing the database schema
    """
    return _get_tools().get_schema()


def neo4j_read_cypher(query: str, params: dict = None) -> str:
    """
    Execute a read-only Cypher query against the Neo4j graph database.
    
    Use this to:
    - Find specific entities by name or property
    - Traverse relationships between entities
    - Aggregate and analyze graph data
    - Search for patterns in the knowledge graph
    
    IMPORTANT TIPS:
    - Always check schema first with neo4j_get_schema()
    - Use CONTAINS for fuzzy text matching: WHERE n.name CONTAINS 'term'
    - For entities, check the 'description' property for entity info
    - Search Chunk.text for detailed information - this has the richest content
    - RELATED_TO relationships have 'evidence' property with relationship details
    - LIMIT your results (e.g., LIMIT 10) to avoid overwhelming output
    - Use params dict to pass parameters for parameterized queries
    
    Args:
        query: A valid Cypher query (read-only operations only)
        params: Optional dict of parameters for parameterized queries 
                (e.g., {"film_title": "Kiss and Tell", "person_name": "Shirley Temple"})
        
    Returns:
        Query results as formatted string, or error message
    """
    # Check for parameterized queries without params
    import re
    param_matches = re.findall(r'\$(\w+)', query)
    if param_matches and not params:
        unique_params = list(dict.fromkeys(param_matches))  # Preserve order, remove duplicates
        return (
            f"ERROR: Query contains parameters {unique_params} but no params dict was provided.\n"
            f"ACTION REQUIRED: Look at the original queries in your task to find the actual values.\n"
            f"Then call: neo4j_read_cypher(query, {{\"{unique_params[0]}\": \"actual_value_from_original_query\"}})\n"
            f"Example: If original query had 'Kiss and Tell', use: {{\"film_title\": \"Kiss and Tell\"}}"
        )
    
    return _get_tools().read_cypher(query, params)


# Tool registry for Deep Agent
AGENT_TOOLS = [
    neo4j_get_schema,
    neo4j_read_cypher
]

# Tools for refinement agent (schema provided in prompt - no get_schema)
REFINEMENT_TOOLS = [
    neo4j_read_cypher
]


def get_schema() -> str:
    """Fetch schema - for use by orchestrating code, not agents."""
    return _get_tools().get_schema()


def neo4j_hybrid_search(
    query_text: str,
    vector_index_name: str = "chunk-embeddings",
    fulltext_index_name: str = "chunk-fulltext",
    top_k: int = 5
) -> str:
    """
    Perform a hybrid search combining vector similarity and full-text search.
    
    This tool searches for relevant Chunk nodes using both:
    - Vector similarity (semantic search via embeddings)
    - Full-text search (keyword matching)
    
    The results are combined and ranked by relevance.
    
    NOTE: This requires both vector and fulltext indexes to exist in Neo4j.
    If indexes don't exist, the search will fail gracefully with an error message.
    
    Args:
        query_text: The search query text
        vector_index_name: Name of the vector index (default: "chunk-embeddings")
        fulltext_index_name: Name of the fulltext index (default: "chunk-fulltext")
        top_k: Number of results to return (default: 5)
        
    Returns:
        String representation of search results with chunks and their text, or error message
    """
    logger.debug(f"Performing hybrid search for: {query_text[:100]}...")
    
    try:
        # Try to import neo4j-graphrag
        try:
            from neo4j_graphrag.retrievers import HybridRetriever
            from neo4j_graphrag.embeddings import OpenAIEmbeddings
            from neo4j import GraphDatabase
        except ImportError:
            return (
                "ERROR: neo4j-graphrag not installed. "
                "Install with: pip install neo4j-graphrag"
            )
        
        # Get driver from tools instance
        tools = _get_tools()
        driver = tools._get_driver()
        
        # Create embedder
        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Create hybrid retriever
        retriever = HybridRetriever(
            driver,
            vector_index_name,
            fulltext_index_name,
            embedder
        )
        
        # Perform search
        results = retriever.search(query_text=query_text, top_k=top_k)
        
        if not results or not results.results:
            return f"Hybrid search returned no results for: {query_text}"
        
        # Format results
        output_parts = []
        output_parts.append(f"Hybrid search returned {len(results.results)} result(s):\n")
        
        for i, result in enumerate(results.results[:top_k], 1):
            node = result.node
            score = getattr(result, 'score', None)
            
            # Extract text from chunk
            text = node.get('text', '') if isinstance(node, dict) else getattr(node, 'text', '')
            chunk_id = node.get('id', '') if isinstance(node, dict) else getattr(node, 'id', '')
            
            output_parts.append(f"[{i}] Chunk ID: {chunk_id}")
            if score is not None:
                output_parts.append(f"    Score: {score:.4f}")
            output_parts.append(f"    Text: {text[:200]}...")
            output_parts.append("")
        
        output = "\n".join(output_parts)
        logger.debug(f"Hybrid search returned {len(results.results)} results")
        return output
        
    except Exception as e:
        error_msg = f"Hybrid search error: {str(e)}"
        logger.warning(error_msg, exc_info=True)
        return f"ERROR: {error_msg}"


# Tool registry for Deep Agent
AGENT_TOOLS = [
    neo4j_get_schema,
    neo4j_read_cypher,
    neo4j_hybrid_search
]

# Tools for refinement agent (schema provided in prompt - no get_schema, but includes hybrid search)
REFINEMENT_TOOLS = [
    neo4j_read_cypher,
    neo4j_hybrid_search
]