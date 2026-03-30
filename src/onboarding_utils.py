"""Helper utilities for the onboarding workflow.

Provides functions for:
- Text extraction and sampling from files
- Entity extraction with context
- Test question generation
- Neo4j connection validation
- Environment file updates
"""

import os
import re
import json
import csv
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from io import StringIO

from src.logger import setup_logger
from src.llm_client import LLMClient
from src.neo4j_client import Neo4jClient

logger = setup_logger("booth.onboarding")


def extract_sample_text_from_files(files: List[Any], max_chars: int = 10000) -> str:
    """Extract sample text from uploaded Streamlit files.
    
    Args:
        files: List of Streamlit UploadedFile objects
        max_chars: Maximum characters to extract per file
        
    Returns:
        Combined sample text from all files
    """
    logger.info(f"Extracting sample text from {len(files)} files")
    samples = []
    
    for file in files:
        try:
            # Reset file pointer to beginning
            file.seek(0)
            
            # Determine file type
            if file.name.endswith('.pdf'):
                # For PDF, use PyPDF2
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_parts = []
                    # Sample first few pages
                    for page_num in range(min(3, len(pdf_reader.pages))):
                        text_parts.append(pdf_reader.pages[page_num].extract_text())
                    text = "\n".join(text_parts)[:max_chars]
                    samples.append(text)
                except Exception as e:
                    logger.warning(f"Failed to extract PDF text from {file.name}: {e}")
                    
            elif file.name.endswith('.txt'):
                # For text files, read directly
                text = file.read().decode('utf-8', errors='ignore')[:max_chars]
                samples.append(text)
                
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
    
    combined = "\n\n".join(samples)
    logger.info(f"Extracted {len(combined)} characters from {len(samples)} files")
    return combined


def extract_entities_for_review(
    text: str, 
    entity_types: List[str],
    llm_client: Optional[LLMClient] = None,
    max_entities_per_type: int = 50
) -> Dict[str, List[Dict[str, str]]]:
    """Extract entities from text with context for user review.
    
    Args:
        text: Source text to extract entities from
        entity_types: List of entity type labels to extract
        llm_client: LLM client for entity extraction
        max_entities_per_type: Maximum entities to extract per type
        
    Returns:
        Dict mapping entity type to list of entity dicts with 'name' and 'context'
    """
    logger.info(f"Extracting entities for {len(entity_types)} types")
    
    if llm_client is None:
        llm_client = LLMClient()
    
    # Chunk text for processing
    max_chunk_size = 3000
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)][:10]
    
    results = {entity_type: [] for entity_type in entity_types}
    seen_entities = {entity_type: set() for entity_type in entity_types}
    
    for chunk_idx, chunk in enumerate(chunks):
        try:
            # Build prompt for entity extraction
            prompt = f"""Extract entities from the following text. For each entity type, list the entity names you find.

Entity types to extract: {', '.join(entity_types)}

Text:
{chunk}

Return a JSON object with entity types as keys and lists of entity names as values. Example:
{{
  "PERSON": ["John Smith", "Jane Doe"],
  "ORGANIZATION": ["ACME Corp"],
  "LOCATION": ["New York"]
}}

Only return the JSON object, nothing else."""

            response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse JSON response
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    entities = json.loads(json_match.group())
                    
                    # Process each entity type
                    for entity_type in entity_types:
                        if entity_type in entities:
                            for entity_name in entities[entity_type]:
                                # Deduplicate
                                entity_key = entity_name.lower().strip()
                                if entity_key not in seen_entities[entity_type]:
                                    # Find context (surrounding text)
                                    context = _extract_context(chunk, entity_name)
                                    
                                    results[entity_type].append({
                                        'name': entity_name,
                                        'context': context
                                    })
                                    seen_entities[entity_type].add(entity_key)
                                    
                                    # Limit entities per type
                                    if len(results[entity_type]) >= max_entities_per_type:
                                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {e}")
                    
        except Exception as e:
            logger.error(f"Error extracting entities from chunk {chunk_idx}: {e}")
    
    # Log results
    for entity_type, entities in results.items():
        logger.info(f"Extracted {len(entities)} {entity_type} entities")
    
    return results


def _extract_context(text: str, entity_name: str, context_chars: int = 100) -> str:
    """Extract surrounding context for an entity mention."""
    try:
        # Find entity in text (case-insensitive)
        pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
        match = pattern.search(text)
        
        if match:
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            context = text[start:end].strip()
            
            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
                
            return context
    except Exception as e:
        logger.warning(f"Error extracting context: {e}")
    
    return entity_name  # Fallback to just the entity name


def generate_test_questions(
    neo4j_client: Neo4jClient,
    llm_client: LLMClient,
    n_questions: int = 20
) -> List[Dict[str, str]]:
    """Generate test questions from the knowledge graph.
    
    Args:
        neo4j_client: Neo4j client for querying the graph
        llm_client: LLM client for question generation
        n_questions: Number of questions to generate
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    logger.info(f"Generating {n_questions} test questions from graph")
    
    # Sample entities from the graph
    with neo4j_client.driver.session() as session:
        # Get diverse entity sample
        result = session.run("""
            MATCH (e:__Entity__)
            WITH e, rand() as r
            ORDER BY r
            LIMIT $limit
            RETURN e.name as name, e.entity_type as type, e.description as description
        """, limit=n_questions * 2)
        
        entities = [dict(record) for record in result]
    
    if not entities:
        logger.warning("No entities found in graph for question generation")
        return []
    
    logger.info(f"Sampled {len(entities)} entities for question generation")
    
    questions = []
    question_templates = [
        "What is {name}?",
        "Describe {name}.",
        "What do you know about {name}?",
        "Tell me about {name}.",
        "What are the key facts about {name}?"
    ]
    
    for i, entity in enumerate(entities[:n_questions]):
        try:
            # Generate question
            template = question_templates[i % len(question_templates)]
            question = template.format(name=entity['name'])
            
            # Generate answer using entity description or by querying chunks
            if entity.get('description'):
                answer_prompt = f"""Based on this information, provide a concise answer to the question.

Information: {entity['description']}

Question: {question}

Answer (1-2 sentences):"""
            else:
                answer_prompt = f"""Generate a brief, factual answer to this question about {entity['name']} (a {entity['type']}).

Question: {question}

Answer (1-2 sentences):"""
            
            answer = llm_client.chat_completion(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.3,
                max_tokens=150
            ).strip()
            
            questions.append({
                'question': question,
                'answer': answer,
                'entity': entity['name'],
                'entity_type': entity.get('type', 'UNKNOWN')
            })
            
        except Exception as e:
            logger.error(f"Error generating question for entity {entity.get('name')}: {e}")
    
    logger.info(f"Generated {len(questions)} test questions")
    return questions


def validate_neo4j_connection(uri: str, username: str, password: str) -> Tuple[bool, Optional[str]]:
    """Test Neo4j connection with provided credentials.
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    logger.info(f"Validating Neo4j connection to {uri}")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection with a simple query
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        
        driver.close()
        logger.info("Neo4j connection validated successfully")
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Neo4j connection validation failed: {error_msg}")
        return False, error_msg


def update_env_file(key: str, value: str, env_path: str = ".env") -> bool:
    """Update or add a key-value pair in the .env file.
    
    Args:
        key: Environment variable key
        value: Environment variable value
        env_path: Path to .env file
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Updating .env file with key: {key}")
    
    try:
        env_file = Path(env_path)
        
        # Read existing content
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
        
        # Find and update or append
        key_found = False
        new_lines = []
        
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                new_lines.append(line)
        
        if not key_found:
            new_lines.append(f"{key}={value}\n")
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(new_lines)
        
        logger.info(f"Successfully updated {key} in .env file")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update .env file: {e}")
        return False


def parse_test_set_csv(csv_file: Any) -> Tuple[List[Dict[str, str]], List[str]]:
    """Parse uploaded CSV test set file.
    
    Args:
        csv_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (questions list, errors list)
    """
    logger.info(f"Parsing test set CSV: {csv_file.name}")
    
    questions = []
    errors = []
    
    try:
        # Read CSV content
        csv_file.seek(0)
        content = csv_file.read().decode('utf-8')
        
        # Parse CSV
        csv_reader = csv.DictReader(StringIO(content))
        
        # Validate headers
        if 'question' not in csv_reader.fieldnames or 'expected_answer' not in csv_reader.fieldnames:
            # Try alternative column names
            fieldnames = csv_reader.fieldnames
            question_col = None
            answer_col = None
            
            for field in fieldnames:
                field_lower = field.lower()
                if 'question' in field_lower and not question_col:
                    question_col = field
                elif 'answer' in field_lower and not answer_col:
                    answer_col = field
            
            if not question_col or not answer_col:
                errors.append("CSV must have 'question' and 'expected_answer' columns (or similar)")
                return questions, errors
        else:
            question_col = 'question'
            answer_col = 'expected_answer'
        
        # Parse rows
        for idx, row in enumerate(csv_reader, start=1):
            try:
                question = row.get(question_col, '').strip()
                answer = row.get(answer_col, '').strip()
                
                if not question:
                    errors.append(f"Row {idx}: Missing question")
                    continue
                
                if not answer:
                    errors.append(f"Row {idx}: Missing answer")
                    continue
                
                questions.append({
                    'question': question,
                    'expected_answer': answer,
                    'row_number': idx
                })
                
            except Exception as e:
                errors.append(f"Row {idx}: Parse error - {str(e)}")
        
        logger.info(f"Parsed {len(questions)} questions from CSV ({len(errors)} errors)")
        
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        errors.append(f"Failed to parse CSV: {str(e)}")
    
    return questions, errors


def save_test_set_to_csv(questions: List[Dict[str, str]], output_path: str) -> bool:
    """Save test questions to CSV file.
    
    Args:
        questions: List of question dicts with 'question' and 'answer' keys
        output_path: Path to save CSV file
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving {len(questions)} questions to CSV: {output_path}")
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'expected_answer'])
            writer.writeheader()
            
            for q in questions:
                writer.writerow({
                    'question': q.get('question', ''),
                    'expected_answer': q.get('answer', q.get('expected_answer', ''))
                })
        
        logger.info(f"Successfully saved test set to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return False

