"""Main BOOTH orchestrator implementing the complete flow logic."""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from src.llm_client import LLMClient
from src.neo4j_client import Neo4jClient
from src.cypher_verification import CypherVerifier, VerifierType
from src.cypher_correction import CypherCorrector, CorrectionType
from src.logger import setup_logger

logger = setup_logger("booth.orchestrator")


@dataclass
class BOOTHResponse:
    """Response from BOOTH system."""
    success: bool
    answer: str
    query_id: Optional[str] = None
    similar_match: bool = False
    high_risk: bool = False
    declined: bool = False
    cypher_used: Optional[str] = None
    raw_data: Optional[Any] = None
    error_message: Optional[str] = None


class BOOTHOrchestrator:
    """
    Orchestrates the BOOTH (Bounded Orchestration Of Text Handling) workflow.
    
    Workflow:
    1. Embed user query
    2. Check for similar approved queries (>90% similarity)
       - If match found: Use cached few-shot prompt -> Generate cypher -> Execute
    3. If no match: User marks as high-risk or safe
       - If high-risk: Decline and store
       - If safe: Generate cypher with retry logic
    4. Store results for human curation
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        neo4j_client: Optional[Neo4jClient] = None
    ):
        """Initialize the BOOTH orchestrator.
        
        Args:
            llm_client: OpenAI client instance. If None, creates new one.
            neo4j_client: Neo4j client instance. If None, creates new one.
        """
        self.llm_client = llm_client or LLMClient()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.max_retries = int(os.getenv("MAX_CYPHER_RETRIES", "3"))
        
        # Initialize verification and correction components
        self.verifier = CypherVerifier(
            neo4j_client=self.neo4j_client,
            llm_client=self.llm_client
        )
        self.corrector = CypherCorrector(
            llm_client=self.llm_client,
            neo4j_client=self.neo4j_client
        )
        
        # Configure verification and correction methods from environment
        self.verifier_types = self._parse_verifier_types()
        self.correction_types = self._parse_correction_types()
        self.use_verification_metadata = os.getenv("USE_VERIFICATION_METADATA", "true").lower() == "true"
    
    def _parse_verifier_types(self) -> List[VerifierType]:
        """Parse verifier types from environment variables."""
        verifier_config = os.getenv("CYPHER_VERIFIERS", "RULE_BASED,EXECUTION_BASED")
        verifier_names = [v.strip().upper() for v in verifier_config.split(",")]
        verifiers = []
        
        for name in verifier_names:
            try:
                verifiers.append(VerifierType[name])
            except KeyError:
                logger.warning(f"Unknown verifier type: {name}")
        
        if not verifiers:
            logger.warning("No valid verifiers configured, using RULE_BASED as default")
            verifiers = [VerifierType.RULE_BASED]
        
        logger.info(f"Configured verifiers: {[v.value for v in verifiers]}")
        return verifiers
    
    def _parse_correction_types(self) -> List[CorrectionType]:
        """Parse correction types from environment variables."""
        correction_config = os.getenv("CYPHER_CORRECTORS", "RULE_BASED,LLM_BASED")
        correction_names = [c.strip().upper() for c in correction_config.split(",")]
        correctors = []
        
        for name in correction_names:
            try:
                correctors.append(CorrectionType[name])
            except KeyError:
                logger.warning(f"Unknown correction type: {name}")
        
        if not correctors:
            logger.warning("No valid correctors configured, using RULE_BASED as default")
            correctors = [CorrectionType.RULE_BASED]
        
        logger.info(f"Configured correctors: {[c.value for c in correctors]}")
        return correctors
    
    def process_query(
        self,
        user_query: str,
        is_high_risk: bool = False
    ) -> BOOTHResponse:
        """Process a user query through the BOOTH workflow.
        
        Args:
            user_query: Natural language query from user.
            is_high_risk: Whether user marked this as high-risk.
            
        Returns:
            BOOTHResponse with results.
        """
        logger.info(f"Processing query: '{user_query[:100]}...' (high_risk={is_high_risk})")
        
        try:
            # Step 1: Embed user query
            logger.debug("Generating embedding for user query")
            embedding = self.llm_client.get_embedding(user_query)
            logger.debug(f"Embedding generated successfully (dimension: {len(embedding)})")
            
            # Step 2: Check for similar approved queries (HNSW search)
            logger.debug("Searching for similar approved queries")
            similar_queries = self.neo4j_client.find_similar_queries(embedding, k=5)
            
            if similar_queries:
                # High similarity match found - use few-shot prompt
                logger.info(f"Found {len(similar_queries)} similar queries (best score: {similar_queries[0]['score']:.4f})")
                return self._handle_similar_match(user_query, embedding, similar_queries)
            
            logger.info("No similar queries found above threshold")
            
            # Step 3: No similar match - check risk level
            if is_high_risk:
                # User marked as high-risk - decline and store
                logger.warning(f"Query marked as high-risk by user, declining")
                return self._handle_high_risk_query(user_query, embedding)
            
            # Step 4: Safe query - proceed with text-to-cypher
            logger.info("Processing as new query (no similar match, safe)")
            return self._handle_new_query(user_query, embedding)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return BOOTHResponse(
                success=False,
                answer=f"An error occurred: {str(e)}",
                error_message=str(e)
            )
    
    def _handle_similar_match(
        self,
        user_query: str,
        embedding: List[float],
        similar_queries: List[Dict]
    ) -> BOOTHResponse:
        """Handle query with high similarity match."""
        # Get the best match
        best_match = similar_queries[0]
        logger.info(f"Using similar match (score: {best_match['score']:.4f}, text: '{best_match['text'][:50]}...')")
        
        # Store the new query with similarity match
        logger.debug("Storing new query with similarity match flag")
        query_id = self.neo4j_client.store_query(
            text=user_query,
            embedding=embedding,
            status="pending_approval",
            similarity_matched=True
        )
        logger.info(f"Stored query with ID: {query_id}")
        
        # Link to similar query
        logger.debug(f"Linking to similar query: {best_match['id']}")
        self.neo4j_client.link_similar_queries(
            query_id, 
            best_match['id'], 
            best_match['score']
        )
        
        # Get few-shot examples from similar queries
        similar_ids = [q['id'] for q in similar_queries]
        logger.debug(f"Retrieving few-shot examples for {len(similar_ids)} similar queries")
        few_shot_examples = self.neo4j_client.get_few_shot_examples(similar_ids)
        logger.info(f"Retrieved {len(few_shot_examples)} few-shot examples")
        
        # Generate and execute cypher with few-shot examples
        return self._generate_and_execute_cypher(
            query_id,
            user_query,
            few_shot_examples=few_shot_examples,
            similarity_matched=True
        )
    
    def _handle_high_risk_query(
        self,
        user_query: str,
        embedding: List[float]
    ) -> BOOTHResponse:
        """Handle query marked as high-risk.
        
        Runs the full text2cypher iterative refinement pipeline in the background
        (including verification, correction, and execution) but declines showing
        results to the user. All attempts and results are stored for review in
        the Train AI page, giving curators full context about what would have happened.
        """
        logger.warning(f"Declining high-risk query to user, but running text2cypher pipeline for review: '{user_query[:100]}...'")
        
        # Store query as declined
        query_id = self.neo4j_client.store_query(
            text=user_query,
            embedding=embedding,
            status="declined",
            risk_level="high"
        )
        logger.info(f"High-risk query stored with ID: {query_id} (status: declined)")
        
        # Run the full text2cypher pipeline in background (for review purposes)
        # This generates, verifies, corrects, and executes the query but results
        # are not shown to the user - only stored for curator review
        pipeline_result = None
        try:
            logger.info("Running full text2cypher pipeline for high-risk query (results hidden from user)")
            
            # Run the complete pipeline as if it were a normal query
            pipeline_result = self._generate_and_execute_cypher(
                query_id=query_id,
                user_query=user_query,
                few_shot_examples=None,
                similarity_matched=False
            )
            
            logger.info(f"Pipeline completed for high-risk query: success={pipeline_result.success}")
            if pipeline_result.success:
                logger.info("Text2cypher succeeded - results stored for review (not shown to user)")
            else:
                logger.info("Text2cypher failed - attempts stored for review")
                
        except Exception as e:
            logger.error(f"Error running text2cypher pipeline for high-risk query: {str(e)}", exc_info=True)
            # Store the error
            self.neo4j_client.store_cypher_attempt(
                query_id=query_id,
                cypher_text="",
                attempt_number=1,
                success=False,
                error_message=f"Pipeline error: {str(e)}"
            )
        
        # Return declined response to user (don't show pipeline results)
        # But store the cypher_used so it can be referenced if needed
        return BOOTHResponse(
            success=False,
            answer="This query has been declined due to high risk. It has been logged for review.",
            query_id=query_id,
            high_risk=True,
            declined=True,
            cypher_used=pipeline_result.cypher_used if pipeline_result else None
        )
    
    def _handle_new_query(
        self,
        user_query: str,
        embedding: List[float]
    ) -> BOOTHResponse:
        """Handle new query with no similar matches."""
        logger.debug("Handling new query (no similarity match)")
        
        # Store query
        query_id = self.neo4j_client.store_query(
            text=user_query,
            embedding=embedding,
            status="pending_approval",
            similarity_matched=False
        )
        logger.info(f"Stored new query with ID: {query_id}")
        
        # Generate and execute cypher without few-shot examples
        logger.debug("Proceeding to generate Cypher without few-shot examples")
        return self._generate_and_execute_cypher(
            query_id,
            user_query,
            few_shot_examples=None,
            similarity_matched=False
        )
    
    def _generate_and_execute_cypher(
        self,
        query_id: str,
        user_query: str,
        few_shot_examples: Optional[List[Dict]] = None,
        similarity_matched: bool = False
    ) -> BOOTHResponse:
        """Generate Cypher query and execute with iterative refinement loop.
        
        Implements the verification-correction loop as described in:
        https://neo4j.com/blog/developer/iterative-refinement-for-text2cypher/
        
        Loop:
        1. Generate Cypher query
        2. Verify using configured verifiers
        3. If valid -> Execute
        4. If invalid or execution fails -> Correct and retry
        5. Repeat until max iterations or success
        """
        # Get database schema
        logger.debug("Retrieving database schema")
        schema = self.neo4j_client.get_database_schema()
        logger.debug("Database schema retrieved")
        
        cypher_query = None
        error_feedback = None
        
        for iteration in range(1, self.max_retries + 1):
            logger.info(f"=== Iteration {iteration}/{self.max_retries} ===")
            
            # STEP 1: Generate or Correct Cypher query
            if iteration == 1:
                # First iteration: Generate new query
                logger.info("Generating initial Cypher query")
                try:
                    logger.debug(f"Generating Cypher (with {len(few_shot_examples) if few_shot_examples else 0} few-shot examples)")
                    cypher_query = self.llm_client.generate_cypher(
                        user_query=user_query,
                        schema=schema,
                        few_shot_examples=few_shot_examples,
                        error_feedback=None
                    )
                    logger.info(f"Generated Cypher: {cypher_query[:200]}{'...' if len(cypher_query) > 200 else ''}")
                    
                except Exception as e:
                    error_message = f"Failed to generate Cypher: {str(e)}"
                    logger.error(f"Cypher generation failed: {error_message}", exc_info=True)
                    
                    # Store failed attempt
                    self.neo4j_client.store_cypher_attempt(
                        query_id=query_id,
                        cypher_text="",
                        attempt_number=iteration,
                        success=False,
                        error_message=error_message
                    )
                    error_feedback = error_message
                    continue
            else:
                # Subsequent iterations: Correct based on previous error
                logger.info(f"Attempting correction (iteration {iteration})")
                correction_result = self.corrector.correct(
                    cypher_query=cypher_query,
                    user_query=user_query,
                    error_message=error_feedback,
                    correction_types=self.correction_types,
                    schema=schema
                )
                
                if correction_result.was_corrected:
                    logger.info(f"Correction applied: {correction_result.correction_description}")
                    cypher_query = correction_result.corrected_cypher
                    logger.debug(f"Corrected Cypher: {cypher_query[:200]}{'...' if len(cypher_query) > 200 else ''}")
                else:
                    logger.warning("No correction could be applied, retrying with LLM regeneration")
                    # If correction failed, try regenerating with error feedback
                    try:
                        cypher_query = self.llm_client.generate_cypher(
                            user_query=user_query,
                            schema=schema,
                            few_shot_examples=few_shot_examples,
                            error_feedback=error_feedback
                        )
                        logger.info(f"Regenerated Cypher: {cypher_query[:200]}{'...' if len(cypher_query) > 200 else ''}")
                    except Exception as e:
                        error_message = f"Failed to regenerate Cypher: {str(e)}"
                        logger.error(error_message, exc_info=True)
                        continue
            
            # STEP 2: Verify Cypher query
            logger.info("Verifying Cypher query")
            verification_result = self.verifier.verify(
                cypher_query=cypher_query,
                user_query=user_query,
                verifier_types=self.verifier_types,
                use_metadata=self.use_verification_metadata
            )
            
            if not verification_result.is_valid:
                logger.warning(f"Verification failed: {verification_result.error_message}")
                error_feedback = verification_result.error_message
                
                # Store failed verification attempt
                self.neo4j_client.store_cypher_attempt(
                    query_id=query_id,
                    cypher_text=cypher_query,
                    attempt_number=iteration,
                    success=False,
                    error_message=f"Verification failed ({verification_result.verifier_type.value}): {verification_result.error_message}"
                )
                
                # Continue to next iteration for correction
                continue
            
            logger.info("Verification passed")
            
            # STEP 3: Execute Cypher query
            logger.debug(f"Executing verified Cypher query (iteration {iteration})")
            success, result_data, execution_error = self.neo4j_client.execute_cypher(cypher_query)
            
            # Store attempt
            cypher_attempt_id = self.neo4j_client.store_cypher_attempt(
                query_id=query_id,
                cypher_text=cypher_query,
                attempt_number=iteration,
                success=success,
                error_message=execution_error
            )
            logger.debug(f"Stored Cypher attempt with ID: {cypher_attempt_id}")
            
            if success:
                logger.info(f"✓ Cypher execution successful (returned {len(result_data) if isinstance(result_data, list) else 1} records)")
                
                # Generate natural language summary
                logger.debug("Generating natural language summary")
                result_json = json.dumps(result_data, default=str)
                summary = self.llm_client.generate_summary(user_query, result_json)
                logger.debug(f"Summary generated: {summary[:100]}...")
                
                # Store response
                self.neo4j_client.store_response(
                    cypher_attempt_id=cypher_attempt_id,
                    result_data=result_data,
                    summary=summary
                )
                logger.info(f"Query completed successfully after {iteration} iteration(s) (query_id: {query_id})")
                
                return BOOTHResponse(
                    success=True,
                    answer=summary,
                    query_id=query_id,
                    similar_match=similarity_matched,
                    cypher_used=cypher_query,
                    raw_data=result_data
                )
            else:
                # Execution failed - prepare error feedback for next iteration
                logger.warning(f"Cypher execution failed (iteration {iteration}): {execution_error}")
                error_feedback = execution_error
        
        # All iterations exhausted
        logger.error(f"All {self.max_retries} iterations exhausted for query_id: {query_id}")
        return BOOTHResponse(
            success=False,
            answer=f"Failed to generate a valid query after {self.max_retries} iterations with verification and correction. This has been logged for review.",
            query_id=query_id,
            similar_match=similarity_matched,
            error_message=error_feedback
        )
    
    def get_pending_queries_for_curation(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get queries pending approval for human curation.
        
        Args:
            limit: Maximum number of queries to return.
            
        Returns:
            List of query dictionaries.
        """
        return self.neo4j_client.get_pending_queries(limit)
    
    def approve_query(self, query_id: str, cypher_id: str):
        """Approve a query for use as few-shot example.
        
        Args:
            query_id: Query ID to approve.
            cypher_id: CypherAttempt ID to use as example.
        """
        logger.info(f"Approving query {query_id} with cypher {cypher_id}")
        self.neo4j_client.approve_query(query_id, cypher_id)
        logger.info(f"Query {query_id} approved successfully")
    
    def reject_query(self, query_id: str, reason: Optional[str] = None):
        """Reject a query.
        
        Args:
            query_id: Query ID to reject.
            reason: Optional rejection reason.
        """
        logger.info(f"Rejecting query {query_id}" + (f" (reason: {reason})" if reason else ""))
        self.neo4j_client.reject_query(query_id, reason)
        logger.info(f"Query {query_id} rejected successfully")
    
    def close(self):
        """Close connections."""
        if self.neo4j_client:
            self.neo4j_client.close()

