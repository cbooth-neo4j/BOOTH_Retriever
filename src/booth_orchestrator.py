"""Main BOOTH orchestrator implementing the complete flow logic.

Data model (see docs/data_model.md):
- UserQuestion: Individual user questions (verbatim)
- Query: Canonical question patterns with embeddings for similarity matching
- FewShot: Approved parameterized Cypher queries linked to Query nodes

Flow:
1. Check for similar approved Query nodes (>90% similarity)
   - If match with FewShot: Extract params, run parameterized Cypher (2 LLM calls)
2. If no match, run agentic exploration
3. On approval: Trigger refinement agent to create FewShot linked to Query
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from src.llm_client import LLMClient
from src.neo4j_client import Neo4jClient
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
    tool_used: Optional[str] = None  # Track which retriever was used
    pending_feedback: bool = False  # Whether user feedback is pending


class BOOTHOrchestrator:
    """
    Orchestrates the BOOTH (Bounded Orchestration Of Text Handling) workflow.
    
    Workflow:
    1. Embed user query
    2. Check for similar approved queries (>90% similarity)
       - If match found: Use cached few-shot prompt -> Agent query
    3. If no match: User marks as high-risk or safe
       - If high-risk: Decline and store, but run agent in background for review
       - If safe: Run agent with exploratory mode
    4. Store results for human curation
    
    Uses the Agentic Text2Cypher retriever (Deep Agent) for query execution.
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
        
        # Initialize BOOTH Agent
        self.agent = None
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the BOOTH Agent."""
        try:
            from src.agents.booth_agent import BOOTHAgent
            self.agent = BOOTHAgent(
                neo4j_client=self.neo4j_client,
                llm_client=self.llm_client
            )
            logger.info("BOOTH Agent initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to initialize BOOTH Agent: {e}")
            logger.error("Install Deep Agents with: pip install deepagents")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize BOOTH Agent: {e}")
            raise
    
    def process_query(
        self,
        user_query: str,
        is_high_risk: bool = False
    ) -> BOOTHResponse:
        """Process a user query through the BOOTH workflow.
        
        Flow (v3 - simplified):
        1. Embed user query
        2. Search for similar approved Query nodes (verbatim matching)
           - If match: Store UserQuestion linked via SIMILAR, use FewShot cypher
        3. If no match and high-risk: Decline, run agent in background
        4. If no match and safe: Create new Query, run agent
        
        UserQuestions cluster around Query nodes via SIMILAR relationships.
        Query nodes have the embeddings, FewShot examples, and Tool recommendations.
        
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
            
            # Step 2: Search for similar approved Query nodes
            logger.debug("Searching for similar approved queries")
            similar_queries = self.neo4j_client.find_similar_queries(embedding, k=5)
            
            if similar_queries:
                # High similarity match found - use the approved Query's FewShot
                best_match = similar_queries[0]
                logger.info(f"Found similar query! (score: {best_match['score']:.4f})")
                return self._handle_similar_match(
                    user_query, embedding, similar_queries, is_high_risk
                )
            
            logger.info("No similar approved queries found above threshold")
            
            # Step 3: No similar match - check risk level
            if is_high_risk:
                # User marked as high-risk - decline and store
                logger.warning(f"Query marked as high-risk by user, declining")
                return self._handle_high_risk_query(user_query, embedding)
            
            # Step 4: Safe query - proceed with agent in exploratory mode
            logger.info("Processing as new query (no similar match, safe)")
            return self._handle_new_query(user_query, embedding)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return BOOTHResponse(
                success=False,
                answer=f"An error occurred: {str(e)}",
                error_message=str(e)
            )
    
    def _extract_parameters(
        self,
        user_query: str,
        template: str,
        parameters: List[str]
    ) -> Optional[Dict[str, str]]:
        """Extract parameter values from user query using the template.
        
        Args:
            user_query: The actual user question
            template: The parameterized template (e.g., "What {attribute} did {person} hold?")
            parameters: List of parameter names
            
        Returns:
            Dict of parameter values or None if extraction failed
        """
        if not parameters:
            return {}
        
        prompt = f"""Extract parameter values from this question based on the template.

Template: {template}
Parameters needed: {', '.join(parameters)}
User question: {user_query}

Return a JSON object with the parameter values. Example:
{{"person": "Shirley Temple", "attribute": "government position"}}

Only return the JSON object, nothing else."""

        try:
            response = self.llm_client.client.chat.completions.create(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', result_text)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(result_text)
            
        except Exception as e:
            logger.warning(f"Parameter extraction failed: {e}")
            return None
    
    def _extract_cypher_parameters(
        self,
        user_query: str,
        cypher_template: str,
        parameter_names: List[str],
        matched_query_text: str
    ) -> Optional[Dict[str, Any]]:
        """Extract parameter values from user query for Cypher parameters.
        
        This is a simpler extraction focused on entity names and values needed
        for Cypher queries. Used for high-risk queries with similar matches.
        
        Args:
            user_query: The actual user question
            cypher_template: The parameterized Cypher query
            parameter_names: List of parameter names (e.g., ["entity1", "entity2"])
            matched_query_text: The original query text that this template was derived from
            
        Returns:
            Dict of parameter values or None if extraction failed
        """
        if not parameter_names:
            return {}
        
        prompt = f"""Extract parameter values from the user question to fill in the Cypher query parameters.

Original similar question: {matched_query_text}
Cypher query template: {cypher_template}
Parameters needed: {', '.join(parameter_names)}
User question: {user_query}

Your task: Extract the entity names, values, or search terms from the user question that correspond to each parameter.

For example, if the Cypher has $entity1 and $entity2, and the question asks about "Badly Drawn Boy" and "Wolf Alice", extract:
{{"entity1": "Badly Drawn Boy", "entity2": "Wolf Alice"}}

For array parameters (like $search_terms), extract as an array:
{{"search_terms": ["term1", "term2"]}}

Return ONLY a JSON object with the parameter values. No explanation, just the JSON."""

        try:
            # Use the simple responses.create() API (cleaner than chat.completions)
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
            response = self.llm_client.client.responses.create(
                model=model,
                input=prompt
            )
            result_text = response.output_text.strip()
            
            # Parse JSON from response (handle both single and multi-line JSON)
            import re
            # Try to find JSON object (handles nested objects and arrays)
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(result_text)
            
        except Exception as e:
            logger.warning(f"Cypher parameter extraction failed: {e}")
            return None
    
    def _execute_parameterized_cypher(
        self,
        cypher_template: str,
        params: Dict[str, str]
    ) -> tuple:
        """Execute a parameterized Cypher query.
        
        Args:
            cypher_template: Cypher with $param placeholders
            params: Parameter values to substitute
            
        Returns:
            Tuple of (success, result_data, error_message)
        """
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run(cypher_template, **params)
                data = [dict(record) for record in result]
                return True, data, None
        except Exception as e:
            return False, None, str(e)
    
    def _summarize_results(
        self,
        user_query: str,
        result_data: List[Dict]
    ) -> str:
        """Generate a natural language summary of query results."""
        if not result_data:
            return "No results found for this query."
        
        # Format results for summarization
        results_text = json.dumps(result_data[:10], indent=2, default=str)
        
        prompt = f"""Based on these database results, provide a concise answer to the question.

Question: {user_query}

Results:
{results_text}

Provide a direct, factual answer based on the data. Be specific and cite relevant details."""

        try:
            # Use the simple responses.create() API (cleaner than chat.completions)
            model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
            response = self.llm_client.client.responses.create(
                model=model,
                input=prompt
            )
            return response.output_text.strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return f"Found {len(result_data)} results. Raw data: {results_text[:500]}..."
    
    def _handle_similar_match(
        self,
        user_query: str,
        embedding: List[float],
        similar_queries: List[Dict],
        is_high_risk: bool
    ) -> BOOTHResponse:
        """Handle query with high similarity match.
        
        New flow (v3):
        1. Store UserQuestion linked to matched Query via SIMILAR
        2. If Query has FewShot cypher:
           - For high-risk queries: Use simple pipeline (extract params → execute → summarize) - 2 LLM calls
           - For low-risk queries: Try instant execution, fall back to agent if needed
        3. Fall back to agent with context if no FewShot or execution fails
        """
        best_match = similar_queries[0]
        logger.info(f"Similar match found (score: {best_match['score']:.4f}, text: '{best_match['text'][:50]}...')")
        
        # Store UserQuestion linked to the matched Query via SIMILAR
        user_question_id = self.neo4j_client.store_user_question(
            text=user_query,
            risk_level="high" if is_high_risk else "low",
            matched_query_id=best_match['id'],
            similarity_score=best_match['score']
        )
        logger.info(f"Stored UserQuestion {user_question_id} linked to Query {best_match['id']}")
        
        # Check if matched Query has a FewShot cypher for instant execution
        few_shot_cypher = best_match.get('few_shot_cypher')
        few_shot_params = best_match.get('few_shot_params', [])
        
        logger.debug(f"FewShot check: cypher={few_shot_cypher is not None}, params={few_shot_params}")
        
        if few_shot_cypher:
            # For high-risk queries with a match, use the simpler pipeline (2 LLM calls)
            if is_high_risk:
                logger.info("HIGH-RISK SIMPLE PATH: Extract params → Execute → Summarize (2 LLM calls)")
                
                # Step 1: Extract parameters from user query (1st LLM call)
                if few_shot_params:
                    # Parse parameter names if stored as JSON string
                    if isinstance(few_shot_params, str):
                        try:
                            parameter_names = json.loads(few_shot_params)
                        except:
                            parameter_names = few_shot_params if isinstance(few_shot_params, list) else []
                    else:
                        parameter_names = few_shot_params
                    
                    extracted_params = self._extract_cypher_parameters(
                        user_query=user_query,
                        cypher_template=few_shot_cypher,
                        parameter_names=parameter_names,
                        matched_query_text=best_match['text']
                    )
                    
                    if not extracted_params:
                        logger.warning("Parameter extraction failed, falling back to agent")
                        return self._fallback_to_agent(user_query, embedding, similar_queries, best_match, is_high_risk, user_question_id)
                else:
                    # No parameters needed
                    extracted_params = {}
                
                # Step 2: Execute the parameterized Cypher
                success, result_data, error = self._execute_parameterized_cypher(
                    cypher_template=few_shot_cypher,
                    params=extracted_params
                )
                
                if not success:
                    logger.warning(f"Cypher execution failed: {error}, falling back to agent")
                    return self._fallback_to_agent(user_query, embedding, similar_queries, best_match, is_high_risk, user_question_id)
                
                # Step 3: Summarize results (2nd LLM call)
                summary = self._summarize_results(user_query, result_data)
                
                logger.info(f"Simple pipeline successful (using approved query {best_match['id']}, 2 LLM calls)")
                
                # Since we're using an already approved query's template, use the existing approved Query ID
                # Don't create a new Query node that needs approval - the template is already approved
                approved_query_id = best_match['id']
                
                # Store the Cypher attempt linked to the existing approved Query
                cypher_attempt_id = self.neo4j_client.store_cypher_attempt(
                    query_id=approved_query_id,
                    cypher_text=few_shot_cypher,
                    attempt_number=1,
                    success=True,
                    error_message=None
                )
                
                # Store the response
                self.neo4j_client.store_response(
                    cypher_attempt_id=cypher_attempt_id,
                    result_data=json.dumps(result_data) if result_data else None,
                    summary=summary
                )
                
                # High-risk queries using approved query templates are safe to show
                # since they're using a pre-approved pattern
                return BOOTHResponse(
                    success=True,
                    answer=summary,
                    query_id=approved_query_id,  # Use the approved query ID
                    similar_match=True,
                    high_risk=is_high_risk,
                    declined=False,  # Using approved template - safe to show
                    cypher_used=few_shot_cypher,
                    raw_data=result_data,
                    tool_used="few_shot_simple",
                    pending_feedback=True  # Still want feedback for tracking
                )
            else:
                # Low-risk: Try instant execution first
                logger.info("INSTANT PATH: Using FewShot cypher from matched Query")
                
                # Execute the FewShot cypher directly (may have no params)
                success, result_data, error = self._execute_parameterized_cypher(
                    cypher_template=few_shot_cypher,
                    params={}
                )
                
                if success:
                    # Generate summary from results
                    summary = self._summarize_results(user_query, result_data)
                    
                    logger.info(f"Instant execution successful (query_id: {user_question_id})")
                    return BOOTHResponse(
                        success=True,
                        answer=summary,
                        query_id=user_question_id,
                        similar_match=True,
                        high_risk=is_high_risk,
                        declined=False,
                        cypher_used=few_shot_cypher,
                        raw_data=result_data,
                        tool_used="few_shot_instant",
                        pending_feedback=True  # Still want user feedback
                    )
                else:
                    logger.warning(f"FewShot execution failed: {error}, falling back to agent")
        
        # If we reach here, FewShot should exist (data integrity constraint)
        # If it doesn't, that's a data integrity issue - log error and fall back to agent
        if not few_shot_cypher:
            logger.error(f"Data integrity issue: Approved Query {best_match['id']} has no FewShot! This should not happen.")
            logger.error(f"Query text: '{best_match['text'][:100]}...', similarity: {best_match.get('score', 0.0):.4f}")
            # Fall back to agent as last resort
            return self._fallback_to_agent(user_query, embedding, similar_queries, best_match, is_high_risk, user_question_id)
        
        # This should never be reached if FewShot exists (handled above)
        # But keeping as safety net
        logger.warning(f"Unexpected state: FewShot check passed but no FewShot found")
        return self._fallback_to_agent(user_query, embedding, similar_queries, best_match, is_high_risk, user_question_id)
    
    def _fallback_to_agent(
        self,
        user_query: str,
        embedding: List[float],
        similar_queries: List[Dict],
        best_match: Dict,
        is_high_risk: bool,
        user_question_id: str
    ) -> BOOTHResponse:
        """Fall back to agentic pipeline when simple path fails."""
        logger.info("Using agent with few-shot context from similar queries")
        
        # For agent path, we still need a Query node to track the cypher attempts
        # Create one linked to the UserQuestion
        query_id = self.neo4j_client.store_query(
            text=user_query,
            embedding=embedding,
            status="pending_approval",
            similarity_matched=True
        )
        
        # Build context for the agent
        few_shot_examples = []
        few_shot_cypher = best_match.get('few_shot_cypher')
        for q in similar_queries:
            if q.get('few_shot_cypher'):
                few_shot_examples.append({
                    "query": q['text'],
                    "cypher": q['few_shot_cypher']
                })
        
        matched_context = {
            "query_id": best_match['id'],
            "query_text": best_match['text'],
            "similarity_score": best_match['score'],
            "cypher_template": few_shot_cypher,
            "few_shot_examples": few_shot_examples
        }
        
        # Invoke the agent
        agent_result = self.agent.invoke(
            user_query=user_query,
            query_id=query_id,
            is_high_risk=is_high_risk,
            matched_query_context=matched_context
        )
        
        # Store the results
        return self._process_agent_result(
            agent_result, query_id, is_high_risk, similarity_matched=True
        )
    
    def _handle_high_risk_query(
        self,
        user_query: str,
        embedding: List[float]
    ) -> BOOTHResponse:
        """Handle query marked as high-risk.
        
        IMMEDIATELY declines to the user, then schedules background agent work.
        This ensures the user is not left waiting while the agent runs.
        """
        logger.warning(f"Declining high-risk query to user: '{user_query[:100]}...'")
        
        # Store query as declined FIRST
        query_id = self.neo4j_client.store_query(
            text=user_query,
            embedding=embedding,
            status="declined",
            risk_level="high"
        )
        logger.info(f"High-risk query stored with ID: {query_id} (status: declined)")
        
        # Schedule background processing (agent + refinement) in a separate thread
        # so the user gets their decline message immediately
        import threading
        
        def _background_processing():
            """Run agent and refinement in background thread."""
            try:
                logger.info(f"Background processing started for high-risk query {query_id}")
                
                agent_result = self.agent.invoke(
                    user_query=user_query,
                    query_id=query_id,
                    is_high_risk=True,
                    matched_query_context=None
                )
                
                # Store the agent results for review
                cypher_used = agent_result.get("cypher_used")
                cypher_attempt_id = None
                if cypher_used or agent_result.get("tool_used"):
                    cypher_attempt_id = self.neo4j_client.store_cypher_attempt(
                        query_id=query_id,
                        cypher_text=cypher_used or f"Tool: {agent_result.get('tool_used')}",
                        attempt_number=1,
                        success=agent_result.get("success", False),
                        error_message=agent_result.get("error_message")
                    )
                    
                    if agent_result.get("raw_data"):
                        self.neo4j_client.store_response(
                            cypher_attempt_id=cypher_attempt_id,
                            result_data=agent_result.get("raw_data"),
                            summary=agent_result.get("answer", "")
                        )
                
                logger.info(f"Agent completed for high-risk query {query_id}: success={agent_result.get('success')}")
                
                # Automatically run refinement to create parameterized template
                if agent_result.get("answer") and cypher_attempt_id:
                    self._run_background_refinement(
                        query_id=query_id,
                        cypher_id=cypher_attempt_id,
                        user_query=user_query,
                        agent_result=agent_result,
                        embedding=embedding
                    )
                    
            except Exception as e:
                logger.error(f"Background processing error for query {query_id}: {str(e)}", exc_info=True)
                try:
                    self.neo4j_client.store_cypher_attempt(
                        query_id=query_id,
                        cypher_text="",
                        attempt_number=1,
                        success=False,
                        error_message=f"Agent error: {str(e)}"
                    )
                except Exception as store_err:
                    logger.error(f"Failed to store error for query {query_id}: {store_err}")
        
        # Start background thread - user doesn't wait for this
        thread = threading.Thread(target=_background_processing, daemon=True)
        thread.start()
        logger.info(f"Background processing thread started for query {query_id}")
        
        # Return declined response to user IMMEDIATELY
        # Be explicit that no similar query was found - this is important feedback
        return BOOTHResponse(
            success=False,
            answer="⚠️ **No similar approved query found.** This query has been declined due to high risk. The system is generating a response in the background for review in the Train AI page.",
            query_id=query_id,
            high_risk=True,
            declined=True,
            cypher_used=None,  # Not available yet - running in background
            tool_used="agentic_text2cypher"
        )
    
    def _run_background_refinement(
        self,
        query_id: str,
        cypher_id: str,
        user_query: str,
        agent_result: Dict[str, Any],
        embedding: List[float]
    ):
        """Run refinement agent in background to create parameterized template.
        
        This runs after a high-risk query's agent completes. Creates a refined
        template ready for curator review, rather than just storing raw attempts.
        """
        try:
            from src.agents.refinement_agent import create_refinement_agent
            
            logger.info(f"Running background refinement for declined query {query_id}")
            
            refinement_agent = create_refinement_agent(
                neo4j_client=self.neo4j_client
            )
            
            # Get existing categories and similar templates for context
            existing_categories = self.neo4j_client.get_existing_categories()
            similar_templates = self.neo4j_client.find_similar_templates(embedding, k=3, threshold=0.7)
            
            # Extract queries executed from raw_data
            queries_executed = []
            raw_data = agent_result.get("raw_data", {})
            if isinstance(raw_data, dict):
                queries_executed = raw_data.get("queries_executed", [])
            
            # Run refinement
            result = refinement_agent.refine(
                original_question=user_query,
                multi_step_queries=queries_executed,
                approved_answer=agent_result.get("answer", ""),
                existing_categories=existing_categories,
                similar_templates=similar_templates
            )
            
            if result.success:
                logger.info(f"Background refinement successful for query {query_id}")
                
                # Use the original verbatim question text for similarity matching (not parameterized)
                # Get the original question text from the Query node
                with self.neo4j_client.driver.session() as session:
                    query_record = session.run("""
                        MATCH (q:Query {id: $query_id})
                        RETURN q.text as question_text
                    """, query_id=query_id).single()
                    
                    if not query_record:
                        logger.error(f"Could not find Query node {query_id} for background refinement")
                        return
                    
                    verbatim_question = query_record['question_text']
                
                # Store the FewShot linked to the existing Query node
                few_shot_id = self.neo4j_client.store_few_shot_for_query(
                    query_id=query_id,
                    cypher_template=result.refined_cypher,
                    parameters=result.parameters,
                    example_values={"category": result.category} if result.category else None
                )
                
                # Update the cypher attempt with refined info
                self._update_cypher_with_refinement(cypher_id, result)
                
                logger.info(f"Background refinement stored: few_shot_id={few_shot_id}, category={result.category}")
            else:
                logger.warning(f"Background refinement failed for query {query_id}: {result.error}")
                # Mark as needs human support so curator knows refinement failed
                self._mark_needs_human_support(query_id, result.error or "Refinement failed")
                
        except ImportError as e:
            logger.warning(f"Refinement agent not available for background refinement: {e}")
        except Exception as e:
            logger.error(f"Background refinement error for query {query_id}: {e}", exc_info=True)
    
    def _update_cypher_with_refinement(self, cypher_id: str, refinement_result):
        """Update a CypherAttempt with refinement results."""
        try:
            # Serialize parameters list to JSON string for Neo4j storage
            parameters_json = json.dumps(refinement_result.parameters) if refinement_result.parameters else "[]"
            
            with self.neo4j_client.driver.session() as session:
                # Use dict to avoid 'parameters' kwarg conflict with Neo4j
                session.run("""
                    MATCH (c:CypherAttempt {id: $cypher_id})
                    SET c.refined_cypher = $refined_cypher,
                        c.category = $category,
                        c.parameters = $param_json,
                        c.refinement_success = true
                """, {
                    "cypher_id": cypher_id,
                    "refined_cypher": refinement_result.refined_cypher,
                    "category": refinement_result.category,
                    "param_json": parameters_json
                })
        except Exception as e:
            logger.warning(f"Could not update cypher with refinement: {e}")
    
    def _handle_new_query(
        self,
        user_query: str,
        embedding: List[float]
    ) -> BOOTHResponse:
        """Handle new query with no similar matches using BOOTH Agent."""
        logger.info("Handling new query with BOOTH Agent (exploratory mode)")
        
        # Store query
        query_id = self.neo4j_client.store_query(
            text=user_query,
            embedding=embedding,
            status="pending_approval",
            similarity_matched=False
        )
        logger.info(f"Stored new query with ID: {query_id}")
        
        # Invoke the agent without context (exploratory mode)
        agent_result = self.agent.invoke(
            user_query=user_query,
            query_id=query_id,
            is_high_risk=False,
            matched_query_context=None
        )
        
        # Store the results
        return self._process_agent_result(
            agent_result, query_id, is_high_risk=False, similarity_matched=False
        )
    
    def _process_agent_result(
        self,
        agent_result: Dict[str, Any],
        query_id: str,
        is_high_risk: bool,
        similarity_matched: bool
    ) -> BOOTHResponse:
        """Process the result from the BOOTH Agent and store appropriately."""
        
        success = agent_result.get("success", False)
        answer = agent_result.get("answer", "No answer generated")
        tool_used = agent_result.get("tool_used", "agentic_text2cypher")
        cypher_used = agent_result.get("cypher_used")
        raw_data = agent_result.get("raw_data")
        error_message = agent_result.get("error_message")
        
        # Store the cypher attempt if we have one
        if cypher_used or tool_used:
            cypher_attempt_id = self.neo4j_client.store_cypher_attempt(
                query_id=query_id,
                cypher_text=cypher_used or f"Tool: {tool_used}",
                attempt_number=1,
                success=success,
                error_message=error_message
            )
            
            # Store the response if successful
            if success and raw_data:
                self.neo4j_client.store_response(
                    cypher_attempt_id=cypher_attempt_id,
                    result_data=raw_data,
                    summary=answer
                )
        
        # Update query with tool used
        self._update_query_tool_used(query_id, tool_used)
        
        logger.info(f"Agent completed (success={success}, tool={tool_used}, query_id={query_id})")
        
        return BOOTHResponse(
            success=success,
            answer=answer,
            query_id=query_id,
            similar_match=similarity_matched,
            high_risk=is_high_risk,
            declined=False,
            cypher_used=cypher_used,
            raw_data=raw_data,
            error_message=error_message,
            tool_used=tool_used,
            pending_feedback=success and not is_high_risk  # Low-risk successful queries need feedback
        )
    
    def _update_query_tool_used(self, query_id: str, tool_used: str):
        """Update a query with the tool that was used."""
        try:
            with self.neo4j_client.driver.session() as session:
                session.run("""
                    MATCH (q:Query {id: $query_id})
                    SET q.tool_used = $tool_used
                """, query_id=query_id, tool_used=tool_used)
        except Exception as e:
            logger.warning(f"Could not update query tool_used: {e}")
    
    def submit_user_feedback(
        self,
        query_id: str,
        is_helpful: bool
    ) -> bool:
        """
        Submit user feedback for a query response.
        
        For low-risk queries, user feedback determines whether the query
        is stored for final approval (helpful) or logged for review (not helpful).
        
        Args:
            query_id: The query ID to submit feedback for.
            is_helpful: Whether the user found the response helpful.
        
        Returns:
            True if feedback was recorded successfully.
        """
        logger.info(f"User feedback for query {query_id}: helpful={is_helpful}")
        
        try:
            with self.neo4j_client.driver.session() as session:
                if is_helpful:
                    # Mark for final approval - curator will review
                    session.run("""
                        MATCH (q:Query {id: $query_id})
                        SET q.status = 'pending_approval',
                            q.user_feedback = 'helpful',
                            q.feedback_timestamp = datetime()
                    """, query_id=query_id)
                    logger.info(f"Query {query_id} marked for final approval (user rated helpful)")
                else:
                    # Log for review - helps improve prompts/tools
                    session.run("""
                        MATCH (q:Query {id: $query_id})
                        SET q.status = 'needs_review',
                            q.user_feedback = 'not_helpful',
                            q.feedback_timestamp = datetime()
                    """, query_id=query_id)
                    logger.info(f"Query {query_id} logged for review (user rated not helpful)")
            
            return True
        except Exception as e:
            logger.error(f"Failed to record user feedback: {e}", exc_info=True)
            return False
    
    def get_pending_queries_for_curation(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get queries pending approval for human curation.
        
        Args:
            limit: Maximum number of queries to return.
            
        Returns:
            List of query dictionaries.
        """
        return self.neo4j_client.get_pending_queries(limit)
    
    def approve_query(self, query_id: str, cypher_id: str):
        """Approve a query for use as few-shot example (legacy method).
        
        Args:
            query_id: Query ID to approve.
            cypher_id: CypherAttempt ID to use as example.
        """
        logger.info(f"Approving query {query_id} with cypher {cypher_id}")
        self.neo4j_client.approve_query(query_id, cypher_id)
        logger.info(f"Query {query_id} approved successfully")
    
    def approve_query_with_refinement(
        self,
        query_id: str,
        cypher_id: str,
        trigger_refinement: bool = True
    ) -> Dict[str, Any]:
        """Approve a query and optionally trigger the refinement agent.
        
        This is the v2 approval flow that creates parameterized templates.
        
        Args:
            query_id: Query ID to approve
            cypher_id: CypherAttempt ID with the successful result
            trigger_refinement: Whether to run the refinement agent
            
        Returns:
            Dict with refinement result information
        """
        logger.info(f"Approving query {query_id} with refinement")
        
        # First, mark the query as approved (legacy behavior for compatibility)
        self.neo4j_client.approve_query(query_id, cypher_id)
        
        if not trigger_refinement:
            return {"success": True, "refinement_skipped": True}
        
        # Get the query details for refinement
        query_details = self._get_query_details_for_refinement(query_id, cypher_id)
        if not query_details:
            logger.warning(f"Could not get query details for refinement: {query_id}")
            return {"success": False, "error": "Could not retrieve query details"}
        
        # Run the refinement agent
        try:
            from src.agents.refinement_agent import create_refinement_agent
            
            refinement_agent = create_refinement_agent(
                neo4j_client=self.neo4j_client
            )
            
            # Get existing categories and similar templates for context
            existing_categories = self.neo4j_client.get_existing_categories()
            
            # Get embedding for the question
            embedding = self.llm_client.get_embedding(query_details['question_text'])
            similar_templates = self.neo4j_client.find_similar_templates(embedding, k=3, threshold=0.7)
            
            # Run refinement
            result = refinement_agent.refine(
                original_question=query_details['question_text'],
                multi_step_queries=query_details['queries_executed'],
                approved_answer=query_details['approved_answer'],
                existing_categories=existing_categories,
                similar_templates=similar_templates
            )
            
            if result.success:
                logger.info(f"Refinement successful for query {query_id} (retrieval_type={result.retrieval_type})")
                
                # Handle hybrid_search retrieval type
                if result.retrieval_type == "hybrid_search":
                    # For hybrid search, we don't create a Cypher template
                    # Just mark the query as approved with hybrid search recommendation
                    logger.info(f"Query {query_id} uses hybrid search - no Cypher template needed")
                    return {
                        "success": True,
                        "retrieval_type": "hybrid_search",
                        "category": result.category,
                        "message": "Query can be answered with hybrid search - no Cypher template created"
                    }
                
                # Handle Cypher template creation
                # Use the original verbatim question text for similarity matching (not parameterized)
                # Get the original question text from the Query node
                with self.neo4j_client.driver.session() as session:
                    query_record = session.run("""
                        MATCH (q:Query {id: $query_id})
                        RETURN q.text as question_text
                    """, query_id=query_id).single()
                    
                    if not query_record:
                        logger.error(f"Could not find Query node {query_id}")
                        return {"success": False, "error": "Query node not found"}
                    
                    verbatim_question = query_record['question_text']
                
                # Store the FewShot linked to the existing Query node
                few_shot_id = self.neo4j_client.store_few_shot_for_query(
                    query_id=query_id,
                    cypher_template=result.refined_cypher,
                    parameters=result.parameters,
                    example_values={"category": result.category} if result.category else None
                )
                
                return {
                    "success": True,
                    "retrieval_type": "cypher",
                    "few_shot_id": few_shot_id,
                    "category": result.category,
                    "refined_cypher": result.refined_cypher,
                    "parameters": result.parameters,
                    "refinement_attempts": result.attempts
                }
            else:
                logger.warning(f"Refinement failed for query {query_id}: {result.error}")
                
                # Mark as needing human support
                self._mark_needs_human_support(query_id, result.error)
                
                return {
                    "success": False,
                    "needs_human_support": True,
                    "error": result.error,
                    "partial_cypher": result.refined_cypher
                }
                
        except ImportError as e:
            logger.error(f"Refinement agent not available: {e}")
            return {"success": False, "error": "Refinement agent not available"}
        except Exception as e:
            logger.error(f"Refinement error: {e}", exc_info=True)
            self._mark_needs_human_support(query_id, str(e))
            return {"success": False, "error": str(e)}
    
    def _get_query_details_for_refinement(
        self,
        query_id: str,
        cypher_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get query details needed for refinement."""
        try:
            with self.neo4j_client.driver.session() as session:
                result = session.run("""
                    MATCH (q:Query {id: $query_id})
                    OPTIONAL MATCH (q)-[:GENERATED]->(c:CypherAttempt {id: $cypher_id})
                    OPTIONAL MATCH (c)-[:PRODUCED]->(r:Response)
                    RETURN q.text as question_text,
                           r.result_data as result_data,
                           r.summary as approved_answer
                """, query_id=query_id, cypher_id=cypher_id)
                
                record = result.single()
                if not record:
                    return None
                
                # Parse the result_data to get queries_executed
                result_data = record['result_data']
                queries_executed = []
                
                if result_data:
                    try:
                        data = json.loads(result_data)
                        queries_executed = data.get('queries_executed', [])
                    except json.JSONDecodeError:
                        pass
                
                return {
                    'question_text': record['question_text'],
                    'approved_answer': record['approved_answer'] or '',
                    'queries_executed': queries_executed
                }
        except Exception as e:
            logger.error(f"Error getting query details: {e}")
            return None
    
    def _mark_needs_human_support(self, query_id: str, error: str):
        """Mark a query as needing human support after refinement failure."""
        try:
            with self.neo4j_client.driver.session() as session:
                session.run("""
                    MATCH (q:Query {id: $query_id})
                    SET q.status = 'needs_human_support',
                        q.refinement_error = $error
                """, query_id=query_id, error=error)
        except Exception as e:
            logger.error(f"Could not mark query as needs_human_support: {e}")
    
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
