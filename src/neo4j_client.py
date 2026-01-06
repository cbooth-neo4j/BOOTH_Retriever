"""Unified Neo4j client for vector search, storage, and query execution."""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from neo4j import GraphDatabase

from src.logger import setup_logger

logger = setup_logger("booth.neo4j_client")


class Neo4jClient:
    """Unified client for Neo4j operations in BOOTH system."""
    
    def __init__(
        self, 
        uri: Optional[str] = None, 
        user: Optional[str] = None, 
        password: Optional[str] = None
    ):
        """Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI. If not provided, uses NEO4J_URI env var.
            user: Neo4j username. If not provided, uses NEO4J_USER env var.
            password: Neo4j password. If not provided, uses NEO4J_PASSWORD env var.
        """
        logger.debug("Initializing Neo4jClient")
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not self.password:
            logger.error("Neo4j password not provided")
            raise ValueError("Neo4j password is required. Set NEO4J_PASSWORD environment variable.")
        
        logger.info(f"Connecting to Neo4j at {self.uri} as user {self.user}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))
        logger.info(f"Neo4jClient initialized (similarity_threshold={self.similarity_threshold})")
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            logger.debug("Closing Neo4j driver connection")
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def get_database_schema(self) -> str:
        """Get a text description of the Neo4j database schema.
        
        Returns:
            String description of node labels, relationship types, and properties.
        """
        logger.debug("Fetching database schema")
        with self.driver.session() as session:
            # Get node labels and their properties - fetch raw data to avoid deprecated array indexing
            node_result = session.run("""
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, nodeLabels, propertyName, propertyTypes
                RETURN nodeLabels, propertyName, propertyTypes
            """)
            
            # Group properties by node label
            node_properties = {}
            for record in node_result:
                labels = record['nodeLabels']
                prop_name = record['propertyName']
                prop_types = record['propertyTypes']
                
                # Handle propertyTypes - it might be a list or other format
                if isinstance(prop_types, list) and prop_types:
                    prop_type = prop_types[0]
                else:
                    prop_type = str(prop_types)
                
                if labels:
                    label = labels[0]
                    if label not in node_properties:
                        node_properties[label] = []
                    node_properties[label].append(f"{prop_name}: {prop_type}")
            
            # Get relationship types and their properties
            rel_result = session.run("""
                CALL db.schema.relTypeProperties()
                YIELD relType, propertyName, propertyTypes
                RETURN relType, propertyName, propertyTypes
            """)
            
            # Group properties by relationship type
            rel_properties = {}
            for record in rel_result:
                rel_type = record['relType']
                prop_name = record['propertyName']
                prop_types = record['propertyTypes']
                
                # Handle propertyTypes - it might be a list or other format
                if isinstance(prop_types, list) and prop_types:
                    prop_type = prop_types[0]
                else:
                    prop_type = str(prop_types)
                
                if rel_type not in rel_properties:
                    rel_properties[rel_type] = []
                rel_properties[rel_type].append(f"{prop_name}: {prop_type}")
            
            logger.debug(f"Retrieved {len(node_properties)} node types and {len(rel_properties)} relationship types")
            
            schema_parts = ["Database Schema:\n"]
            
            # Format node information
            schema_parts.append("Node Labels:")
            for label, props in node_properties.items():
                if label not in ['Query', 'CypherAttempt', 'Response']:
                    schema_parts.append(f"  {label}: {', '.join(props)}")
            
            # Format relationship information
            schema_parts.append("\nRelationship Types:")
            for rel_type, props in rel_properties.items():
                if props:
                    schema_parts.append(f"  {rel_type}: {', '.join(props)}")
                else:
                    schema_parts.append(f"  {rel_type}")
            
            schema = "\n".join(schema_parts)
            logger.debug(f"Database schema retrieved (length: {len(schema)} chars)")
            return schema
    
    def find_similar_queries(
        self, 
        embedding: List[float], 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar approved queries using vector similarity search.
        
        Args:
            embedding: Query embedding vector.
            k: Number of similar queries to return.
            
        Returns:
            List of similar query dictionaries with scores.
        """
        logger.debug(f"Searching for similar queries (k={k}, threshold={self.similarity_threshold})")
        with self.driver.session() as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('query_embeddings', $k, $embedding)
                    YIELD node, score
                    WHERE node.status = 'approved' AND score >= $threshold
                    RETURN node.id as id,
                           node.text as text,
                           node.timestamp as timestamp,
                           score
                    ORDER BY score DESC
                """, k=k, embedding=embedding, threshold=self.similarity_threshold)
                
                similar_queries = [dict(record) for record in result]
                logger.info(f"Found {len(similar_queries)} similar approved queries")
                if similar_queries:
                    logger.debug(f"Top match score: {similar_queries[0]['score']:.4f}")
                return similar_queries
            except Exception as e:
                # If vector index doesn't exist or other error, return empty
                logger.warning(f"Vector search failed: {e}")
                return []
    
    def get_few_shot_examples(self, query_ids: List[str]) -> List[Dict[str, str]]:
        """Retrieve few-shot examples for similar queries.
        
        Args:
            query_ids: List of Query node IDs.
            
        Returns:
            List of dicts with 'query' and 'cypher' keys.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Query)-[:FEW_SHOT_PROMPT]->(c:CypherAttempt)
                WHERE q.id IN $query_ids
                RETURN q.text as query, c.cypher_text as cypher
                LIMIT 5
            """, query_ids=query_ids)
            
            return [{"query": record["query"], "cypher": record["cypher"]} 
                    for record in result]
    
    def store_query(
        self, 
        text: str, 
        embedding: List[float], 
        status: str = "pending_approval",
        similarity_matched: bool = False,
        risk_level: str = "low"
    ) -> str:
        """Store a new query in the database.
        
        Args:
            text: Query text.
            embedding: Query embedding vector.
            status: Query status (pending_approval, approved, rejected, declined).
            similarity_matched: Whether this query matched a similar one.
            risk_level: Risk assessment (low, high).
            
        Returns:
            The generated query ID.
        """
        query_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.debug(f"Storing query (status={status}, risk_level={risk_level}, similarity_matched={similarity_matched})")
        with self.driver.session() as session:
            session.run("""
                CREATE (q:Query {
                    id: $id,
                    text: $text,
                    embedding: $embedding,
                    timestamp: $timestamp,
                    status: $status,
                    similarity_matched: $similarity_matched,
                    risk_level: $risk_level
                })
            """, id=query_id, text=text, embedding=embedding, 
                timestamp=timestamp, status=status, 
                similarity_matched=similarity_matched, risk_level=risk_level)
        
        logger.info(f"Stored query with ID: {query_id}")
        return query_id
    
    def store_cypher_attempt(
        self, 
        query_id: str,
        cypher_text: str,
        attempt_number: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> str:
        """Store a Cypher query attempt.
        
        Args:
            query_id: Parent Query node ID.
            cypher_text: The generated Cypher query.
            attempt_number: Attempt number (1, 2, 3...).
            success: Whether the query executed successfully.
            error_message: Error message if query failed.
            
        Returns:
            The generated CypherAttempt ID.
        """
        attempt_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.debug(f"Storing Cypher attempt #{attempt_number} for query {query_id} (success={success})")
        with self.driver.session() as session:
            session.run("""
                MATCH (q:Query {id: $query_id})
                CREATE (c:CypherAttempt {
                    id: $id,
                    cypher_text: $cypher_text,
                    attempt_number: $attempt_number,
                    success: $success,
                    error_message: $error_message,
                    timestamp: $timestamp
                })
                CREATE (q)-[:GENERATED]->(c)
            """, query_id=query_id, id=attempt_id, cypher_text=cypher_text,
                attempt_number=attempt_number, success=success, 
                error_message=error_message, timestamp=timestamp)
        
        logger.info(f"Stored Cypher attempt with ID: {attempt_id}")
        return attempt_id
    
    def store_response(
        self, 
        cypher_attempt_id: str,
        result_data: Any,
        summary: str
    ) -> str:
        """Store a response from a successful query.
        
        Args:
            cypher_attempt_id: Parent CypherAttempt node ID.
            result_data: The query result data (will be converted to JSON).
            summary: Natural language summary of results.
            
        Returns:
            The generated Response ID.
        """
        response_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Convert result_data to JSON string
        if isinstance(result_data, str):
            result_json = result_data
        else:
            result_json = json.dumps(result_data, default=str)
        
        logger.debug(f"Storing response for Cypher attempt {cypher_attempt_id}")
        with self.driver.session() as session:
            session.run("""
                MATCH (c:CypherAttempt {id: $cypher_attempt_id})
                CREATE (r:Response {
                    id: $id,
                    result_data: $result_data,
                    summary: $summary,
                    timestamp: $timestamp
                })
                CREATE (c)-[:PRODUCED]->(r)
            """, cypher_attempt_id=cypher_attempt_id, id=response_id,
                result_data=result_json, summary=summary, timestamp=timestamp)
        
        logger.info(f"Stored response with ID: {response_id}")
        return response_id
    
    def execute_cypher(self, cypher_query: str) -> Tuple[bool, Any, Optional[str]]:
        """Execute a Cypher query against the database.
        
        Args:
            cypher_query: The Cypher query to execute.
            
        Returns:
            Tuple of (success, result_data, error_message).
        """
        logger.debug(f"Executing Cypher query: {cypher_query[:200]}...")
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query)
                data = [dict(record) for record in result]
                logger.info(f"Cypher query executed successfully (returned {len(data)} records)")
                return True, data, None
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Cypher query execution failed: {error_msg}")
                return False, None, error_msg
    
    def link_similar_queries(self, query_id: str, similar_query_id: str, score: float):
        """Create a SIMILAR_TO relationship between queries.
        
        Args:
            query_id: The new query ID.
            similar_query_id: The similar query ID found.
            score: Similarity score.
        """
        logger.debug(f"Linking query {query_id} to similar query {similar_query_id} (score={score:.4f})")
        with self.driver.session() as session:
            session.run("""
                MATCH (q1:Query {id: $query_id})
                MATCH (q2:Query {id: $similar_query_id})
                CREATE (q1)-[:SIMILAR_TO {score: $score}]->(q2)
            """, query_id=query_id, similar_query_id=similar_query_id, score=score)
        logger.debug("Linked similar queries successfully")
    
    def get_pending_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get queries pending approval for curation.
        
        Args:
            limit: Maximum number of queries to return.
            
        Returns:
            List of query dictionaries with their cypher attempts and responses.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Query)
                WHERE q.status IN ['pending_approval', 'declined', 'rejected']
                OPTIONAL MATCH (q)-[:GENERATED]->(c:CypherAttempt)
                WHERE c.success = true
                OPTIONAL MATCH (c)-[:PRODUCED]->(r:Response)
                RETURN q.id as query_id,
                       q.text as query_text,
                       q.timestamp as timestamp,
                       q.status as status,
                       q.risk_level as risk_level,
                       c.id as cypher_id,
                       c.cypher_text as cypher_text,
                       r.result_data as result_data,
                       r.summary as summary
                ORDER BY q.timestamp DESC
                LIMIT $limit
            """, limit=limit)
            
            return [dict(record) for record in result]
    
    def approve_query(self, query_id: str, cypher_id: str):
        """Approve a query and promote it to a few-shot example.
        
        Args:
            query_id: The Query node ID to approve.
            cypher_id: The CypherAttempt node ID to use as few-shot example.
        """
        logger.info(f"Approving query {query_id} with Cypher attempt {cypher_id}")
        with self.driver.session() as session:
            # Update query status
            session.run("""
                MATCH (q:Query {id: $query_id})
                SET q.status = 'approved'
            """, query_id=query_id)
            logger.debug(f"Updated query status to 'approved'")
            
            # Create FEW_SHOT_PROMPT relationship
            session.run("""
                MATCH (q:Query {id: $query_id})
                MATCH (c:CypherAttempt {id: $cypher_id})
                CREATE (q)-[:FEW_SHOT_PROMPT]->(c)
            """, query_id=query_id, cypher_id=cypher_id)
            logger.info(f"Query approved and promoted to few-shot example")
    
    def reject_query(self, query_id: str, reason: Optional[str] = None):
        """Reject a query.
        
        Args:
            query_id: The Query node ID to reject.
            reason: Optional rejection reason.
        """
        logger.info(f"Rejecting query {query_id}" + (f" with reason: {reason}" if reason else ""))
        with self.driver.session() as session:
            if reason:
                session.run("""
                    MATCH (q:Query {id: $query_id})
                    SET q.status = 'rejected', q.rejection_reason = $reason
                """, query_id=query_id, reason=reason)
            else:
                session.run("""
                    MATCH (q:Query {id: $query_id})
                    SET q.status = 'rejected'
                """, query_id=query_id)
        logger.info(f"Query {query_id} rejected successfully")
    
    def decline_high_risk_query(self, query_id: str):
        """Mark a query as declined due to high risk.
        
        Args:
            query_id: The Query node ID to decline.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (q:Query {id: $query_id})
                SET q.status = 'declined', q.risk_level = 'high'
            """, query_id=query_id)

