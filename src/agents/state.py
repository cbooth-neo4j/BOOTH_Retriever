"""State schema for BOOTH Agent."""

from typing import TypedDict, Optional, List, Any


class BOOTHAgentState(TypedDict):
    """
    State schema for the BOOTH Agent.
    
    Attributes:
        user_query: The original natural language query from the user.
        is_high_risk: Whether the query is marked as high-risk by the user.
        matched_query_context: Context from similar approved queries (few-shot examples
            and suggested retriever). None if no similar match found.
        final_answer: The final summarized answer to return to the user.
        query_id: Unique identifier for this query (for storage/tracking).
        tool_used: Which retriever tool was used (e.g., "agentic_text2cypher").
        user_rating: User feedback on the response ("success" | "failure" | None).
        raw_data: Raw data returned from the retriever (for storage/review).
        cypher_used: The Cypher query used (if available).
        error_message: Error message if the query failed.
    """
    user_query: str
    is_high_risk: bool
    matched_query_context: Optional[dict]
    final_answer: Optional[str]
    query_id: str
    tool_used: Optional[str]
    user_rating: Optional[str]
    raw_data: Optional[Any]
    cypher_used: Optional[str]
    error_message: Optional[str]


class MatchedQueryContext(TypedDict):
    """
    Context from a similar approved query.
    
    Attributes:
        query_id: ID of the matched query.
        query_text: Text of the matched query.
        similarity_score: Similarity score (0-1).
        cypher_template: The approved Cypher query (if available).
        few_shot_examples: List of few-shot examples from similar queries.
    """
    query_id: str
    query_text: str
    similarity_score: float
    cypher_template: Optional[str]
    few_shot_examples: Optional[List[dict]]
