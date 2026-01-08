"""
Refinement Agent - Consolidates multi-step agentic queries into single parameterized Cypher

When a user approves a query (thumbs up), this agent:
1. Reviews the multi-step queries that were executed
2. Consolidates them into a single optimized Cypher query
3. Parameterizes entity names so the query works for similar questions
4. Verifies the refined query produces the same answer
5. Auto-categorizes the question type

Uses the Deep Agent architecture with Neo4j tools for iteration and testing.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from src.logger import setup_logger
from src.agents.tools import REFINEMENT_TOOLS, init_tools, get_schema

logger = setup_logger("booth.agents.refinement")

# Try to import LangChain agent (simpler than Deep Agents for bounded tasks)
try:
    from langgraph.prebuilt import create_react_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    create_react_agent = None
    logger.warning("LangGraph not installed. Install with: pip install langgraph")


# Question categories for auto-classification
QUESTION_CATEGORIES = [
    ("PERSON_ATTRIBUTE", "Questions about a person's attributes (nationality, birth date, etc.)"),
    ("PERSON_ROLE", "Questions about roles/positions a person held"),
    ("WORK_CREATOR", "Questions about who created a work (directed, wrote, composed)"),
    ("WORK_PARTICIPANT", "Questions about who participated in a work (starred, performed)"),
    ("LOCATION_COMPARISON", "Questions comparing locations of two or more things"),
    ("LOCATION_ATTRIBUTE", "Questions about where something is located"),
    ("TEMPORAL", "Questions about when something happened (dates, years)"),
    ("RELATIONSHIP", "Questions about how entities are related"),
    ("FACTUAL", "Simple fact lookup questions"),
    ("MULTI_HOP", "Questions requiring traversal across multiple relationships"),
]


REFINEMENT_SYSTEM_PROMPT = """Combine multiple Cypher queries into ONE simple, parameterized query.

## GOAL
Create a REUSABLE query template that works for ANY similar question.

## DATABASE SCHEMA (provided - DO NOT call neo4j_get_schema)
{schema}

## TASK
1. Analyze the queries that answered the question
2. Write ONE Cypher combining the key logic
3. Replace entity names with $param_name
4. Test ONCE: neo4j_read_cypher(query, params_dict)
5. Output JSON

## TOOL: neo4j_read_cypher(query, params)
Example: neo4j_read_cypher("MATCH (f:FILM) WHERE f.name CONTAINS $title RETURN f", {{"title": "Kiss"}})
- ALWAYS pass params dict when using $variables
- Test ONCE, then output result

## CATEGORIES
{categories}

## OUTPUT (JSON only)
Success: {{"success": true, "refined_cypher": "MATCH...", "parameters": ["param_name"], "question_template": "What...?", "category": "CATEGORY", "test_params": {{"param": "value"}}}}
Failure: {{"success": false, "reason": "why", "needs_human_support": true}}

## RULES
- Simple: MATCH → WHERE → RETURN → LIMIT
- ONE test, then output JSON
- DO NOT call neo4j_get_schema (schema provided above)
"""


@dataclass
class RefinementResult:
    """Result from the refinement process"""
    success: bool
    refined_cypher: Optional[str] = None
    parameters: Optional[List[str]] = None
    question_template: Optional[str] = None
    category: Optional[str] = None
    test_params: Optional[Dict[str, str]] = None
    attempts: int = 0
    error: Optional[str] = None
    needs_human_support: bool = False


class RefinementAgent:
    """
    Agent that refines multi-step agentic queries into single parameterized Cypher.
    
    Uses Deep Agent architecture to iterate and test until the refined query
    produces equivalent results to the original multi-step execution.
    """
    
    MAX_ATTEMPTS = 5
    
    def __init__(
        self,
        neo4j_client=None,
        model: str = None
    ):
        """
        Initialize the refinement agent.
        
        Args:
            neo4j_client: Optional Neo4jClient for reusing connection
            model: Override the configured model
        """
        if not AGENT_AVAILABLE:
            raise ImportError(
                "LangGraph not installed. Install with: pip install langgraph"
            )
        
        self.neo4j_client = neo4j_client
        
        # Initialize tools with Neo4j client if provided
        if neo4j_client:
            init_tools(neo4j_client=neo4j_client)
        
        # Get model from environment - no fallback, must be explicitly set
        self.model_name = model or os.getenv("REFINEMENT_AGENT_MODEL")
        if not self.model_name:
            raise ValueError(
                "REFINEMENT_AGENT_MODEL environment variable must be set. "
                "Example: REFINEMENT_AGENT_MODEL=gpt-4o"
            )
        
        # Create LLM
        self._create_llm()
        
        logger.info(f"RefinementAgent initialized (model={self.model_name})")
    
    def _create_llm(self):
        """Create the LLM for the agent."""
        from langchain_openai import ChatOpenAI
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _get_system_prompt(self, schema: str, existing_categories: List[str] = None) -> str:
        """Get the system prompt with schema and category information."""
        category_list = "\n".join([
            f"- **{cat}**: {desc}" for cat, desc in QUESTION_CATEGORIES
        ])
        
        # Use string replacement to avoid issues with JSON curly braces
        prompt = REFINEMENT_SYSTEM_PROMPT.replace("{categories}", category_list)
        prompt = prompt.replace("{schema}", schema)
        
        if existing_categories:
            prompt += f"\n\nEXISTING CATEGORIES IN USE: {', '.join(existing_categories)}"
        
        return prompt
    
    def _create_agent(self, schema: str, existing_categories: List[str] = None):
        """Create a simple ReAct agent for refinement (no planning overhead)."""
        system_prompt = self._get_system_prompt(schema, existing_categories)
        
        # Use simple ReAct agent - direct tool loop without planning/subagents
        return create_react_agent(
            model=self.llm,
            tools=REFINEMENT_TOOLS,  # Only read_cypher - no get_schema
            state_modifier=system_prompt  # System prompt for the agent
        )
    
    def _build_refinement_prompt(
        self,
        original_question: str,
        multi_step_queries: List[str],
        approved_answer: str,
        similar_templates: List[Dict] = None
    ) -> str:
        """Build the user prompt for the refinement task."""
        
        queries_text = "\n\n".join([
            f"**Query {i+1}:**\n```cypher\n{q}\n```"
            for i, q in enumerate(multi_step_queries)
        ])
        
        prompt = f"""## REFINEMENT TASK

**Original Question:**
{original_question}

**Multi-Step Queries Executed:**
{queries_text}

**Approved Answer:**
{approved_answer}

## YOUR TASK

1. Analyze these queries and the answer
2. Create a SINGLE refined Cypher query that retrieves the key information
3. Parameterize entity names (use $param_name syntax)
4. Test your query to verify it supports the answer
5. Categorize the question type
6. Return the JSON result format specified in your instructions
"""
        
        if similar_templates:
            prompt += "\n\n## SIMILAR EXISTING TEMPLATES\n\nThese templates already exist for similar questions:\n"
            for t in similar_templates[:3]:
                prompt += f"\n- Category: {t.get('category', 'N/A')}\n"
                prompt += f"  Template: {t.get('template', 'N/A')}\n"
                prompt += f"  Cypher: {t.get('cypher', 'N/A')[:200]}...\n"
        
        return prompt
    
    def _extract_json_result(self, agent_response: str) -> Optional[Dict]:
        """Extract JSON result from agent response."""
        # Try to find JSON block in response
        json_match = re.search(r'```json\s*(.*?)\s*```', agent_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        try:
            # Find the first { and last }
            start = agent_response.find('{')
            end = agent_response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(agent_response[start:end+1])
        except json.JSONDecodeError:
            pass
        
        return None
    
    def refine(
        self,
        original_question: str,
        multi_step_queries: List[str],
        approved_answer: str,
        existing_categories: List[str] = None,
        similar_templates: List[Dict] = None
    ) -> RefinementResult:
        """
        Refine a multi-step query into a single parameterized Cypher.
        
        Args:
            original_question: The user's original question
            multi_step_queries: List of Cypher queries executed by the agent
            approved_answer: The approved answer text
            existing_categories: List of existing category names
            similar_templates: Similar existing templates for reference
            
        Returns:
            RefinementResult with the refined query or error information
        """
        start_time = time.time()
        logger.info(f"Starting refinement for: {original_question[:100]}...")
        logger.debug(f"Multi-step queries to refine: {len(multi_step_queries)}")
        
        try:
            # Fetch schema upfront - no need for agent to call get_schema
            logger.info("Fetching database schema for refinement...")
            schema = get_schema()
            
            # Create agent with schema injected
            agent = self._create_agent(schema, existing_categories)
            
            # Build the refinement prompt
            user_prompt = self._build_refinement_prompt(
                original_question=original_question,
                multi_step_queries=multi_step_queries,
                approved_answer=approved_answer,
                similar_templates=similar_templates
            )
            
            # Log inputs for debugging
            logger.info(f"Refinement input - {len(multi_step_queries)} queries to consolidate")
            if not multi_step_queries:
                logger.warning("No queries provided to refine! Agent may have nothing to work with.")
            else:
                for i, q in enumerate(multi_step_queries[:2]):  # Log first 2 queries
                    logger.info(f"  Query {i+1}: {q[:150]}...")
            logger.info(f"Refinement input - Answer preview: {approved_answer[:150]}...")
            
            # Invoke the agent with strict iteration limit
            # Max 5 tool calls as per prompt, plus some buffer = limit 12
            try:
                from langgraph.errors import GraphRecursionError
            except ImportError:
                GraphRecursionError = Exception  # Fallback if not available
            
            try:
                # Stream to capture intermediate steps
                tool_calls_seen = []
                final_result = None
                
                for event in agent.stream(
                    {"messages": [{"role": "user", "content": user_prompt}]},
                    config={"recursion_limit": 12},  # ~5 tool calls max
                    stream_mode="values"
                ):
                    # Log tool calls and results as they happen
                    if "messages" in event:
                        for msg in event["messages"]:
                            # Log tool calls
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                                    tool_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                    tool_calls_seen.append(tool_name)
                                    
                                    # Show query details
                                    if tool_name == 'neo4j_read_cypher':
                                        query = str(tool_args.get('query', ''))[:80]
                                        has_params = bool(tool_args.get('params'))
                                        logger.info(f"Tool call [{len(tool_calls_seen)}]: {query}... (params={has_params})")
                                    else:
                                        logger.info(f"Tool call [{len(tool_calls_seen)}]: {tool_name}")
                            
                            # Log tool results (especially errors)
                            if hasattr(msg, 'type') and msg.type == 'tool':
                                content = str(getattr(msg, 'content', ''))[:150]
                                if 'ERROR' in content or 'error' in content.lower():
                                    logger.warning(f"Tool error: {content}")
                    final_result = event
                
                result = final_result
                logger.info(f"Refinement completed with {len(tool_calls_seen)} tool calls: {tool_calls_seen}")
                
            except GraphRecursionError as e:
                # Agent hit iteration limit - log what it was doing
                elapsed = time.time() - start_time
                logger.warning(f"Refinement agent hit iteration limit after {elapsed:.2f}s")
                logger.warning(f"Tool calls before limit: {tool_calls_seen if 'tool_calls_seen' in dir() else 'unknown'}")
                
                return RefinementResult(
                    success=False,
                    attempts=len(tool_calls_seen) if 'tool_calls_seen' in dir() else 10,
                    error="Agent hit iteration limit. Query may be too complex for automatic refinement.",
                    needs_human_support=True
                )
            
            # Extract the final response and log agent's reasoning
            final_response = ""
            tool_calls_count = 0
            tool_call_details = []
            
            if "messages" in result:
                # Log each message for debugging
                for i, msg in enumerate(result["messages"]):
                    msg_type = getattr(msg, 'type', 'unknown')
                    
                    # Log tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown')
                            tool_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                            tool_calls_count += 1
                            
                            # Summarize the tool call
                            if tool_name == 'neo4j_read_cypher':
                                query_preview = str(tool_args.get('query', ''))[:100]
                                has_params = 'params' in tool_args and tool_args['params']
                                tool_call_details.append(f"  [{tool_calls_count}] {tool_name}: {query_preview}... (params={has_params})")
                            else:
                                tool_call_details.append(f"  [{tool_calls_count}] {tool_name}")
                    
                    # Log tool results (errors especially)
                    if msg_type == 'tool' and hasattr(msg, 'content'):
                        content = str(msg.content)[:200]
                        if 'ERROR' in content or 'error' in content.lower():
                            logger.debug(f"  Tool error: {content}")
                
                # Get final AI response
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content:
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = [
                                    b.get('text', '') if isinstance(b, dict) else str(b)
                                    for b in content
                                ]
                                final_response = ' '.join(text_parts)
                            else:
                                final_response = str(content)
                            break
            
            elapsed = time.time() - start_time
            logger.info(f"Refinement agent completed in {elapsed:.2f}s (tool_calls={tool_calls_count})")
            
            # Log tool call summary
            if tool_call_details:
                logger.info("Refinement tool calls:")
                for detail in tool_call_details:
                    logger.info(detail)
            
            # Log final response preview
            logger.info(f"Final response preview: {final_response[:200]}...")
            
            # Parse the JSON result
            json_result = self._extract_json_result(final_response)
            
            if not json_result:
                logger.warning("Could not extract JSON result from agent response")
                return RefinementResult(
                    success=False,
                    attempts=tool_calls_count,
                    error="Could not extract JSON result from agent response",
                    needs_human_support=True
                )
            
            if json_result.get('success', False):
                return RefinementResult(
                    success=True,
                    refined_cypher=json_result.get('refined_cypher'),
                    parameters=json_result.get('parameters', []),
                    question_template=json_result.get('question_template'),
                    category=json_result.get('category'),
                    test_params=json_result.get('test_params', {}),
                    attempts=tool_calls_count
                )
            else:
                return RefinementResult(
                    success=False,
                    refined_cypher=json_result.get('partial_cypher'),
                    attempts=tool_calls_count,
                    error=json_result.get('reason', 'Unknown error'),
                    needs_human_support=json_result.get('needs_human_support', True)
                )
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Refinement error after {elapsed:.2f}s: {e}", exc_info=True)
            
            return RefinementResult(
                success=False,
                attempts=0,
                error=str(e),
                needs_human_support=True
            )


def create_refinement_agent(
    neo4j_client=None,
    model: str = None
) -> RefinementAgent:
    """
    Factory function to create a Refinement Agent.
    
    Args:
        neo4j_client: Optional Neo4jClient instance to reuse
        model: Override model name
        
    Returns:
        RefinementAgent instance
    """
    return RefinementAgent(
        neo4j_client=neo4j_client,
        model=model
    )

