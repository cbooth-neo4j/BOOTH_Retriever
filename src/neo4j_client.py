"""Unified Neo4j client for vector search, storage, and query execution.

Data model:
- UserQuestion: Individual user questions (verbatim)
- Query: Canonical question patterns with embeddings for similarity matching
- FewShot: Approved parameterized Cypher queries linked to Query nodes
- CypherAttempt/Response: Audit trail

See docs/data_model.md for full schema documentation.
"""

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
        # Suppress noisy Neo4j warnings about non-existent properties (they're expected)
        self.driver = GraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password),
            notifications_min_severity="OFF"  # Suppress all notification warnings
        )
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
        
        Returns Query nodes with their linked FewShot examples and Tool recommendations.
        This is the main matching endpoint - new questions match against Query embeddings.
        
        Args:
            embedding: Query embedding vector.
            k: Number of similar queries to return.
            
        Returns:
            List of similar query dictionaries with:
            - id, text, score: Query info
            - few_shot_cypher: Approved Cypher template (REQUIRED - approved queries must have FewShot)
            - few_shot_params: Parameter names for the template
            - tool_name, tool_description: Recommended tool (if exists)
            
        Note: Only returns approved Query nodes that have a FewShot linked via FEW_SHOT_EXAMPLE.
              This enforces the data integrity constraint that approved queries must have FewShot.
        """
        logger.debug(f"Searching for similar queries (k={k}, threshold={self.similarity_threshold})")
        with self.driver.session() as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('query_embeddings', $k, $embedding)
                    YIELD node, score
                    WHERE node.status = 'approved' AND score >= $threshold
                    MATCH (node)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot)
                    OPTIONAL MATCH (node)-[:USES_TOOL]->(tool:Tool)
                    RETURN node.id as id,
                           node.text as text,
                           node.timestamp as timestamp,
                           score,
                           fs.cypher_template as few_shot_cypher,
                           fs.parameters as few_shot_params,
                           tool.name as tool_name,
                           tool.description as tool_description
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
    
    def store_few_shot_for_query(
        self,
        query_id: str,
        cypher_template: str,
        parameters: Optional[List[str]] = None,
        example_values: Optional[Dict[str, str]] = None
    ) -> str:
        """Store a FewShot example linked to a Query.
        
        This is the approved Cypher pattern that answers this type of question.
        Can be parameterized (with $param syntax) or concrete.
        
        Args:
            query_id: Query node ID to link to
            cypher_template: The Cypher query (can include $params)
            parameters: List of parameter names if parameterized
            example_values: Example parameter values for documentation
            
        Returns:
            Generated FewShot ID
        """
        few_shot_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.debug(f"Storing FewShot for query {query_id}")
        with self.driver.session() as session:
            session.run("""
                MATCH (q:Query {id: $query_id})
                CREATE (fs:FewShot {
                    id: $id,
                    cypher_template: $cypher_template,
                    parameters: $param_list,
                    example_values: $example_values,
                    created_at: $created_at
                })
                CREATE (q)-[:FEW_SHOT_EXAMPLE]->(fs)
            """, {
                "query_id": query_id,
                "id": few_shot_id,
                "cypher_template": cypher_template,
                "param_list": parameters or [],
                "example_values": json.dumps(example_values) if example_values else None,
                "created_at": timestamp
            })
        
        logger.info(f"Stored FewShot with ID: {few_shot_id}")
        return few_shot_id
    
    def get_or_create_tool(
        self,
        name: str,
        description: Optional[str] = None
    ) -> str:
        """Get or create a Tool node.
        
        Tools represent the retrieval methods (agentic_text2cypher, hybrid_retriever, etc.)
        
        Args:
            name: Tool name (e.g., 'agentic_text2cypher')
            description: Tool description
            
        Returns:
            Tool node ID
        """
        with self.driver.session() as session:
            result = session.run("""
                MERGE (t:Tool {name: $name})
                ON CREATE SET t.id = $id, t.description = $description
                RETURN t.id as id
            """, name=name, id=str(uuid.uuid4()), description=description)
            
            record = result.single()
            return record['id']
    
    def link_query_to_tool(
        self,
        query_id: str,
        tool_name: str,
        is_recommended: bool = True
    ):
        """Link a Query to a Tool via USES_TOOL relationship.
        
        Args:
            query_id: Query node ID
            tool_name: Tool name to link to
            is_recommended: Whether this is the recommended tool for this query type
        """
        logger.debug(f"Linking query {query_id} to tool {tool_name}")
        with self.driver.session() as session:
            session.run("""
                MATCH (q:Query {id: $query_id})
                MATCH (t:Tool {name: $tool_name})
                MERGE (q)-[r:USES_TOOL]->(t)
                SET r.recommended = $is_recommended
            """, query_id=query_id, tool_name=tool_name, is_recommended=is_recommended)
    
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
                WHERE q.status IN ['pending_approval', 'declined', 'rejected', 'needs_review', 'needs_human_support', 'approved']
                OPTIONAL MATCH (q)-[:GENERATED]->(c:CypherAttempt)
                WHERE c.success = true
                OPTIONAL MATCH (c)-[:PRODUCED]->(r:Response)
                RETURN q.id as query_id,
                       q.text as query_text,
                       q.timestamp as timestamp,
                       q.status as status,
                       q.risk_level as risk_level,
                       q.user_feedback as user_feedback,
                       q.refinement_error as refinement_error,
                       c.id as cypher_id,
                       c.cypher_text as cypher_text,
                       c.refined_cypher as refined_cypher,
                       c.category as category,
                       c.parameters as parameters,
                       c.refinement_success as refinement_success,
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

    # =========================================================================
    # DATA MODEL - UserQuestion, Query, FewShot
    # =========================================================================
    
    def ensure_template_indexes(self):
        """Create indexes for the data model if they don't exist.
        
        Note: Query embeddings index is created in ensure_indexes().
        This method is kept for backward compatibility.
        """
        logger.info("Ensuring indexes exist (Query embeddings index created in ensure_indexes())")
        # Query embeddings index is handled in ensure_indexes()
        # No additional indexes needed here
    
    def store_user_question(
        self,
        text: str,
        risk_level: str = "low",
        matched_query_id: Optional[str] = None,
        similarity_score: Optional[float] = None
    ) -> str:
        """Store a new user question (verbatim, for audit trail).
        
        UserQuestions are lightweight - just the verbatim text and timestamp.
        They link to Query nodes via SIMILAR relationship for clustering.
        
        Args:
            text: Original question text (verbatim)
            risk_level: Risk level (low, high)
            matched_query_id: If matched to existing Query, link via SIMILAR
            similarity_score: Similarity score for the SIMILAR relationship
            
        Returns:
            Generated UserQuestion ID
        """
        question_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.debug(f"Storing UserQuestion (risk_level={risk_level}, matched={matched_query_id is not None})")
        with self.driver.session() as session:
            # Create the UserQuestion node
            session.run("""
                CREATE (uq:UserQuestion {
                    id: $id,
                    text: $text,
                    timestamp: $timestamp,
                    risk_level: $risk_level
                })
            """, id=question_id, text=text, timestamp=timestamp, risk_level=risk_level)
            
            # If matched to a Query, create SIMILAR relationship
            if matched_query_id and similarity_score:
                session.run("""
                    MATCH (uq:UserQuestion {id: $uq_id})
                    MATCH (q:Query {id: $q_id})
                    CREATE (uq)-[:SIMILAR {score: $score}]->(q)
                """, uq_id=question_id, q_id=matched_query_id, score=similarity_score)
                logger.debug(f"Linked UserQuestion to Query {matched_query_id} (score={similarity_score:.4f})")
        
        logger.info(f"Stored UserQuestion with ID: {question_id}")
        return question_id
    
    def find_similar_templates(
        self,
        embedding: List[float],
        k: int = 5,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Find similar Query nodes with FewShot examples for refinement context.
        
        This is a compatibility method that uses Query nodes instead of QueryTemplate.
        
        Args:
            embedding: Query embedding vector
            k: Number of similar queries to return
            threshold: Similarity threshold (uses default if not provided)
            
        Returns:
            List of similar query dictionaries with scores and FewShot info
        """
        threshold = threshold or self.similarity_threshold
        logger.debug(f"Searching for similar queries for refinement context (k={k}, threshold={threshold})")
        
        with self.driver.session() as session:
            try:
                result = session.run("""
                    CALL db.index.vector.queryNodes('query_embeddings', $k, $embedding)
                    YIELD node, score
                    WHERE score >= $threshold AND node.status = 'approved'
                    OPTIONAL MATCH (node)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot)
                    RETURN node.id as id,
                           node.text as template,
                           node.category as category,
                           fs.parameters as parameters,
                           fs.cypher_template as cypher_template,
                           fs.parameters as cypher_parameters,
                           score
                    ORDER BY score DESC
                """, k=k, embedding=embedding, threshold=threshold)
                
                templates = [dict(record) for record in result]
                logger.info(f"Found {len(templates)} similar approved queries for refinement context")
                if templates:
                    logger.debug(f"Top match score: {templates[0]['score']:.4f}")
                return templates
            except Exception as e:
                logger.warning(f"Query vector search for refinement context failed: {e}")
                return []
    
    def get_existing_categories(self) -> List[str]:
        """Get list of existing question categories from Query nodes.
        
        Returns:
            List of category names in use
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Query)
                WHERE q.category IS NOT NULL
                RETURN DISTINCT q.category as category
                ORDER BY category
            """)
            return [record['category'] for record in result]
    
    def update_user_question_status(
        self,
        question_id: str,
        status: str,
        extracted_params: Optional[Dict[str, str]] = None
    ):
        """Update status of a UserQuestion.
        
        Args:
            question_id: UserQuestion ID
            status: New status
            extracted_params: Optional extracted parameters
        """
        logger.debug(f"Updating UserQuestion {question_id} status to {status}")
        with self.driver.session() as session:
            if extracted_params:
                params_json = json.dumps(extracted_params)
                session.run("""
                    MATCH (uq:UserQuestion {id: $question_id})
                    SET uq.status = $status, uq.extracted_params = $params_json
                """, question_id=question_id, status=status, params_json=params_json)
            else:
                session.run("""
                    MATCH (uq:UserQuestion {id: $question_id})
                    SET uq.status = $status
                """, question_id=question_id, status=status)
    
    def get_pending_user_questions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get UserQuestions pending approval.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of pending questions with their agentic results
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (uq:UserQuestion)
                WHERE uq.status IN ['pending', 'needs_human_support']
                OPTIONAL MATCH (uq)-[:GENERATED]->(c:CypherAttempt)
                WHERE c.success = true
                OPTIONAL MATCH (c)-[:PRODUCED]->(r:Response)
                RETURN uq.id as question_id,
                       uq.text as question_text,
                       uq.timestamp as timestamp,
                       uq.status as status,
                       uq.risk_level as risk_level,
                       c.id as cypher_id,
                       c.cypher_text as cypher_text,
                       r.result_data as result_data,
                       r.summary as summary
                ORDER BY uq.timestamp DESC
                LIMIT $limit
            """, limit=limit)
            
            return [dict(record) for record in result]
    
    def get_questions_needing_human_support(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get questions where refinement failed and needs human intervention.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of questions needing human support
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (uq:UserQuestion)
                WHERE uq.status = 'needs_human_support'
                OPTIONAL MATCH (uq)-[:GENERATED]->(c:CypherAttempt)
                WHERE c.success = true
                OPTIONAL MATCH (c)-[:PRODUCED]->(r:Response)
                RETURN uq.id as question_id,
                       uq.text as question_text,
                       uq.timestamp as timestamp,
                       uq.risk_level as risk_level,
                       uq.refinement_error as refinement_error,
                       c.cypher_text as cypher_text,
                       r.result_data as result_data,
                       r.summary as summary
                ORDER BY uq.timestamp DESC
                LIMIT $limit
            """, limit=limit)
            
            return [dict(record) for record in result]
    
    def find_similar_template_for_new_question(
        self,
        embedding: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching template for a new question.
        
        This is the main entry point for the "instant execution" flow.
        
        Args:
            embedding: Question embedding
            
        Returns:
            Best matching template with cypher, or None if no match above threshold
        """
        templates = self.find_similar_templates(embedding, k=1)
        if templates:
            return templates[0]
        return None
    
    # ===========================================
    # Development/Testing Cleanup Methods
    # ===========================================
    
    def delete_all_rejected_queries(self) -> int:
        """Delete all rejected queries and their related nodes.
        
        Returns:
            Number of Query nodes deleted
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Query {status: 'rejected'})
                OPTIONAL MATCH (q)-[:GENERATED]->(ca:CypherAttempt)
                OPTIONAL MATCH (ca)-[:PRODUCED]->(r:Response)
                OPTIONAL MATCH (uq:UserQuestion)-[:SIMILAR]->(q)
                OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot)
                WITH q, collect(DISTINCT ca) as attempts, collect(DISTINCT r) as responses, 
                     collect(DISTINCT uq) as questions, collect(DISTINCT fs) as fewshots
                DETACH DELETE q
                FOREACH (ca IN attempts | DETACH DELETE ca)
                FOREACH (r IN responses | DETACH DELETE r)
                FOREACH (uq IN questions | DETACH DELETE uq)
                FOREACH (fs IN fewshots | DETACH DELETE fs)
                RETURN count(q) as deleted_count
            """)
            record = result.single()
            return record["deleted_count"] if record else 0
    
    def delete_all_booth_data(self) -> Dict[str, int]:
        """Delete all BOOTH-related nodes (UserQuestion, Query, FewShot, Tool, CypherAttempt, Response).
        
        This does NOT delete the entire database - only nodes defined in the BOOTH data model.
        
        Returns:
            Dictionary with counts of deleted nodes by type
        """
        deleted_counts = {}
        
        with self.driver.session() as session:
            # Delete in order to handle relationships properly
            # First delete leaf nodes, then work up
            
            # Response nodes
            result = session.run("MATCH (r:Response) DETACH DELETE r RETURN count(r) as count")
            deleted_counts['Response'] = result.single()["count"]
            
            # CypherAttempt nodes
            result = session.run("MATCH (ca:CypherAttempt) DETACH DELETE ca RETURN count(ca) as count")
            deleted_counts['CypherAttempt'] = result.single()["count"]
            
            # FewShot nodes (v3 model)
            result = session.run("MATCH (fs:FewShot) DETACH DELETE fs RETURN count(fs) as count")
            deleted_counts['FewShot'] = result.single()["count"]
            
            # FewShotCypher nodes (v2 legacy)
            result = session.run("MATCH (fsc:FewShotCypher) DETACH DELETE fsc RETURN count(fsc) as count")
            deleted_counts['FewShotCypher'] = result.single()["count"]
            
            # QueryTemplate nodes (v2 legacy)
            result = session.run("MATCH (qt:QueryTemplate) DETACH DELETE qt RETURN count(qt) as count")
            deleted_counts['QueryTemplate'] = result.single()["count"]
            
            # UserQuestion nodes
            result = session.run("MATCH (uq:UserQuestion) DETACH DELETE uq RETURN count(uq) as count")
            deleted_counts['UserQuestion'] = result.single()["count"]
            
            # Query nodes
            result = session.run("MATCH (q:Query) DETACH DELETE q RETURN count(q) as count")
            deleted_counts['Query'] = result.single()["count"]
            
            # Tool nodes
            result = session.run("MATCH (t:Tool) DETACH DELETE t RETURN count(t) as count")
            deleted_counts['Tool'] = result.single()["count"]
        
        return deleted_counts

