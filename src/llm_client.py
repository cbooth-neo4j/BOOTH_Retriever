"""OpenAI client for embeddings and LLM completions."""

import os
from typing import List, Optional
from openai import OpenAI
import numpy as np

from src.logger import setup_logger

logger = setup_logger("booth.llm_client")


class LLMClient:
    """Handles interactions with OpenAI API for embeddings and completions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
        """
        logger.debug("Initializing LLMClient")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")
        logger.info(f"LLMClient initialized (embedding_model={self.embedding_model}, chat_model={self.chat_model})")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string.
        
        Args:
            text: Input text to embed.
            
        Returns:
            List of floats representing the embedding vector.
        """
        logger.debug(f"Generating embedding for text (length: {len(text)} chars)")
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            logger.debug(f"Embedding generated successfully (dimension: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def chat_completion(
        self, 
        messages: List[dict], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Get a chat completion from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens in response (will be converted to max_completion_tokens for newer models).
            temperature: Sampling temperature (0.0-2.0). If None, uses model default.
            
        Returns:
            The assistant's response text.
        """
        logger.debug(f"Requesting chat completion ({len(messages)} messages)")
        try:
            # Build kwargs, only including parameters if they're not None
            # Newer OpenAI models use max_completion_tokens instead of max_tokens
            kwargs = {
                "model": self.chat_model,
                "messages": messages,
            }
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens
            if temperature is not None:
                kwargs["temperature"] = temperature
            
            response = self.client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content
            logger.debug(f"Chat completion received (length: {len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"Failed to get chat completion: {str(e)}", exc_info=True)
            raise Exception(f"Failed to get chat completion: {str(e)}")
    
    def generate_cypher(
        self, 
        user_query: str, 
        schema: str, 
        few_shot_examples: Optional[List[dict]] = None,
        error_feedback: Optional[str] = None
    ) -> str:
        """Generate a Cypher query from natural language.
        
        Args:
            user_query: Natural language query from user.
            schema: Neo4j database schema description.
            few_shot_examples: Optional list of example query/cypher pairs.
            error_feedback: Optional feedback from previous failed attempt.
            
        Returns:
            Generated Cypher query string.
        """
        logger.info(f"Generating Cypher for query: '{user_query[:100]}...'")
        logger.debug(f"Using {len(few_shot_examples) if few_shot_examples else 0} few-shot examples")
        if error_feedback:
            logger.debug(f"Retry with error feedback: {error_feedback[:100]}...")
        
        system_message = f"""You are an expert at converting natural language questions into Neo4j Cypher queries.

Database Schema:
{schema}

Generate ONLY the Cypher query without any explanation or markdown formatting."""

        user_message = f"Natural language question: {user_query}"
        
        # Add few-shot examples if provided
        messages = [{"role": "system", "content": system_message}]
        
        if few_shot_examples:
            logger.debug(f"Adding {len(few_shot_examples)} few-shot examples to prompt")
            for example in few_shot_examples:
                messages.append({"role": "user", "content": f"Question: {example['query']}"})
                messages.append({"role": "assistant", "content": example['cypher']})
        
        # Add error feedback if this is a retry
        if error_feedback:
            user_message += f"\n\nPrevious attempt failed with error: {error_feedback}\nPlease fix the query."
        
        messages.append({"role": "user", "content": user_message})
        
        cypher = self.chat_completion(messages)
        
        # Clean up the response
        cypher = cypher.strip()
        # Remove markdown code blocks if present
        if cypher.startswith("```"):
            logger.debug("Removing markdown code block formatting from Cypher response")
            lines = cypher.split("\n")
            cypher = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher
        
        logger.info(f"Generated Cypher query (length: {len(cypher)} chars)")
        return cypher.strip()
    
    def generate_summary(self, query: str, result_data: str) -> str:
        """Generate a natural language summary of query results.
        
        Args:
            query: Original user query.
            result_data: JSON string of query results.
            
        Returns:
            Natural language summary.
        """
        logger.debug(f"Generating summary for query: '{query[:100]}...'")
        logger.debug(f"Result data length: {len(result_data)} chars")
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes database query results in natural language."
            },
            {
                "role": "user",
                "content": f"""User asked: "{query}"

Query results:
{result_data}

Provide a clear, concise natural language answer to the user's question based on these results."""
            }
        ]
        
        summary = self.chat_completion(messages)
        logger.info(f"Summary generated successfully (length: {len(summary)} chars)")
        return summary

