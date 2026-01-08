"""BOOTH Agent - Wrapper around Agentic Text2Cypher for BOOTH orchestration."""

import os
from typing import Optional, Dict, Any, List

from src.agents.agentic_retriever import (
    AgenticText2CypherRetriever,
    create_agentic_retriever,
    DEEP_AGENTS_AVAILABLE
)
from src.logger import setup_logger

logger = setup_logger("booth.agents.booth_agent")


class BOOTHAgent:
    """
    BOOTH Agent wrapper around the Agentic Text2Cypher retriever.
    
    The agent operates in two modes:
    - Exploratory (low-risk): Agent freely explores the graph
    - Constrained (high-risk): Agent follows pre-approved patterns
    
    This wrapper maintains compatibility with BOOTH's orchestrator
    while delegating actual query execution to the Deep Agent.
    """
    
    def __init__(
        self,
        neo4j_client,
        llm_client,
        model_name: Optional[str] = None,
        temperature: float = 0.0  # kept for interface compatibility
    ):
        """
        Initialize the BOOTH Agent.
        
        Args:
            neo4j_client: Neo4j client for database operations.
            llm_client: LLM client for embeddings (agent uses its own LLM).
            model_name: OpenAI model to use. Defaults to AGENTIC_TEXT2CYPHER_MODEL 
                       or OPENAI_CHAT_MODEL env var.
            temperature: Temperature for LLM responses (kept for compatibility).
        """
        self.neo4j_client = neo4j_client
        self.llm_client = llm_client
        
        # Determine model name
        model_name = model_name or os.getenv(
            "AGENTIC_TEXT2CYPHER_MODEL",
            os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
        )
        
        # Initialize the agentic retriever
        if not DEEP_AGENTS_AVAILABLE:
            raise ImportError(
                "Deep Agents not installed. Install with: pip install deepagents"
            )
        
        self.retriever = create_agentic_retriever(
            neo4j_client=neo4j_client,
            llm_client=llm_client,
            model=model_name
        )
        
        logger.info(f"BOOTHAgent initialized (model={model_name})")
    
    def invoke(
        self,
        user_query: str,
        query_id: str,
        is_high_risk: bool = False,
        matched_query_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Invoke the BOOTH Agent with a user query.
        
        Args:
            user_query: The natural language query from the user.
            query_id: Unique identifier for this query (for tracking).
            is_high_risk: Whether this is a high-risk query.
            matched_query_context: Context from similar approved queries.
        
        Returns:
            Dictionary with agent results including:
            - success: Whether the query was answered successfully
            - answer: The final answer text
            - tool_used: Which tool was used ('agentic_text2cypher')
            - cypher_used: The last Cypher query executed
            - raw_data: Raw data from the retriever
            - error_message: Error message if failed
        """
        logger.info(f"BOOTH Agent invoked (query_id={query_id}, high_risk={is_high_risk})")
        
        # Extract few-shot examples from context
        few_shot_examples = None
        if matched_query_context:
            few_shot_examples = matched_query_context.get("few_shot_examples")
        
        try:
            # Delegate to the agentic retriever
            result = self.retriever.search(
                query=user_query,
                few_shot_examples=few_shot_examples,
                is_high_risk=is_high_risk,
                matched_query_context=matched_query_context
            )
            
            logger.info(f"BOOTH Agent completed (success={result['success']}, query_id={query_id})")
            return result
            
        except Exception as e:
            logger.error(f"BOOTH Agent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "answer": f"An error occurred while processing your query: {str(e)}",
                "tool_used": "agentic_text2cypher",
                "cypher_used": None,
                "raw_data": None,
                "error_message": str(e)
            }
    
    def stream(
        self,
        user_query: str,
        query_id: str,
        is_high_risk: bool = False,
        matched_query_context: Optional[Dict] = None
    ):
        """
        Stream the BOOTH Agent execution.
        
        Note: Deep Agents don't currently support streaming in the same way
        as LangGraph. This method invokes the agent and yields the final result.
        
        Args:
            user_query: The natural language query from the user.
            query_id: Unique identifier for this query.
            is_high_risk: Whether this is a high-risk query.
            matched_query_context: Context from similar approved queries.
        
        Yields:
            Final state update with results.
        """
        logger.info(f"BOOTH Agent streaming (query_id={query_id}, high_risk={is_high_risk})")
        
        # For now, just invoke and yield the result
        # Deep Agents streaming could be added in the future
        result = self.invoke(
            user_query=user_query,
            query_id=query_id,
            is_high_risk=is_high_risk,
            matched_query_context=matched_query_context
        )
        
        yield {"agent": result}


def create_booth_agent(neo4j_client, llm_client, **kwargs) -> BOOTHAgent:
    """
    Factory function to create a BOOTH Agent.
    
    Args:
        neo4j_client: Neo4j client instance.
        llm_client: LLM client instance.
        **kwargs: Additional arguments passed to BOOTHAgent.
    
    Returns:
        Configured BOOTHAgent instance.
    """
    return BOOTHAgent(neo4j_client, llm_client, **kwargs)
