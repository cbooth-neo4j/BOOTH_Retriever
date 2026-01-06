"""Cypher correction techniques for iterative refinement."""

import re
from typing import Optional
from enum import Enum

from src.logger import setup_logger

logger = setup_logger("booth.correction")


class CorrectionType(Enum):
    """Types of Cypher correctors."""
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"


class CorrectionResult:
    """Result from Cypher correction."""
    
    def __init__(
        self, 
        corrected_cypher: str,
        was_corrected: bool,
        correction_type: Optional[CorrectionType] = None,
        correction_description: Optional[str] = None
    ):
        self.corrected_cypher = corrected_cypher
        self.was_corrected = was_corrected
        self.correction_type = correction_type
        self.correction_description = correction_description
    
    def __repr__(self):
        return f"CorrectionResult(corrected={self.was_corrected}, type={self.correction_type})"


class CypherCorrector:
    """Main corrector class coordinating multiple correction techniques."""
    
    def __init__(self, llm_client=None, neo4j_client=None):
        """Initialize corrector with optional clients.
        
        Args:
            llm_client: LLM client for LLM-based correction
            neo4j_client: Neo4j client for schema context
        """
        self.llm_client = llm_client
        self.neo4j_client = neo4j_client
    
    def correct(
        self,
        cypher_query: str,
        user_query: str,
        error_message: Optional[str],
        correction_types: list[CorrectionType],
        schema: Optional[str] = None
    ) -> CorrectionResult:
        """Correct Cypher query using specified correction techniques.
        
        Executes correctors in order: RULE_BASED -> LLM_BASED
        Returns first successful correction.
        
        Args:
            cypher_query: The Cypher query to correct
            user_query: Original natural language query for context
            error_message: Error message from verification
            correction_types: List of correction types to use
            schema: Optional database schema for context
            
        Returns:
            CorrectionResult with corrected query
        """
        logger.debug(f"Starting correction with {len(correction_types)} correctors: {[c.value for c in correction_types]}")
        
        # Define execution order
        execution_order = [
            CorrectionType.RULE_BASED,
            CorrectionType.LLM_BASED
        ]
        
        # Filter to only requested correctors in correct order
        ordered_correctors = [c for c in execution_order if c in correction_types]
        
        for correction_type in ordered_correctors:
            logger.debug(f"Running {correction_type.value} correction")
            
            if correction_type == CorrectionType.RULE_BASED:
                result = self._rule_based_correction(cypher_query, error_message)
            elif correction_type == CorrectionType.LLM_BASED:
                result = self._llm_based_correction(cypher_query, user_query, error_message, schema)
            else:
                logger.warning(f"Unknown correction type: {correction_type}")
                continue
            
            if result.was_corrected:
                logger.info(f"Correction successful with {correction_type.value}: {result.correction_description}")
                return result
        
        logger.info("No corrections applied")
        return CorrectionResult(
            corrected_cypher=cypher_query,
            was_corrected=False
        )
    
    def _rule_based_correction(
        self, 
        cypher_query: str, 
        error_message: Optional[str]
    ) -> CorrectionResult:
        """Apply rule-based corrections for common issues.
        
        Focuses on fixing relationship direction errors and other simple syntax issues.
        
        Args:
            cypher_query: The Cypher query to correct
            error_message: Error message from verification
            
        Returns:
            CorrectionResult
        """
        logger.debug("Running rule-based correction")
        
        # Handle None or empty query
        if cypher_query is None or not cypher_query:
            logger.warning("Cannot apply rule-based correction to None or empty query")
            return CorrectionResult(
                corrected_cypher=cypher_query or "",
                was_corrected=False,
                correction_type=CorrectionType.RULE_BASED
            )
        
        original_query = cypher_query
        corrections_made = []
        
        # Fix 1: Invalid bidirectional relationship <-[:REL]->
        # Should be either <- or -> or just -
        if re.search(r'<-\[.*?\]->', cypher_query):
            logger.debug("Fixing bidirectional relationship")
            # Replace with undirected
            cypher_query = re.sub(r'<-(\[.*?\])->', r'-\1-', cypher_query)
            corrections_made.append("Fixed bidirectional relationship to undirected")
        
        # Fix 2: Invalid reverse relationship >-[:REL]-<
        # Should be either <- or -> or just -
        if re.search(r'>-\[.*?\]-<', cypher_query):
            logger.debug("Fixing reverse relationship")
            # Replace with undirected
            cypher_query = re.sub(r'>-(\[.*?\])-<', r'-\1-', cypher_query)
            corrections_made.append("Fixed reverse relationship to undirected")
        
        # Fix 3: Missing relationship brackets
        # Pattern: -(type)- should be -[:type]-
        if re.search(r'-\([A-Z_]+\)-', cypher_query):
            logger.debug("Fixing missing relationship brackets")
            cypher_query = re.sub(r'-\(([A-Z_]+)\)-', r'-[:\1]-', cypher_query)
            corrections_made.append("Added relationship brackets")
        
        # Fix 4: Wrong bracket type for relationships
        # Pattern: -[type]- should be -[:type]-
        if re.search(r'-\[([A-Z_][A-Z_]*)\]', cypher_query) and not re.search(r'-\[:[A-Z_]', cypher_query):
            logger.debug("Adding colon to relationship types")
            cypher_query = re.sub(r'-\[([A-Z_][A-Z_]*[^:])\]', r'-[:\1]', cypher_query)
            corrections_made.append("Added colon to relationship type")
        
        # Fix 5: Remove markdown code block markers if present
        if '```' in cypher_query:
            logger.debug("Removing markdown code blocks")
            cypher_query = re.sub(r'```cypher\s*', '', cypher_query)
            cypher_query = re.sub(r'```\s*', '', cypher_query)
            corrections_made.append("Removed markdown formatting")
        
        # Fix 6: Remove leading/trailing whitespace
        cypher_query = cypher_query.strip()
        
        if cypher_query != original_query:
            logger.info(f"Rule-based corrections applied: {', '.join(corrections_made)}")
            return CorrectionResult(
                corrected_cypher=cypher_query,
                was_corrected=True,
                correction_type=CorrectionType.RULE_BASED,
                correction_description='; '.join(corrections_made)
            )
        
        logger.debug("No rule-based corrections needed")
        return CorrectionResult(
            corrected_cypher=cypher_query,
            was_corrected=False,
            correction_type=CorrectionType.RULE_BASED
        )
    
    def _llm_based_correction(
        self,
        cypher_query: str,
        user_query: str,
        error_message: Optional[str],
        schema: Optional[str]
    ) -> CorrectionResult:
        """Use LLM to correct the Cypher query based on error feedback.
        
        Args:
            cypher_query: The Cypher query to correct
            user_query: Original natural language query
            error_message: Error message from verification
            schema: Database schema for context
            
        Returns:
            CorrectionResult
        """
        logger.debug("Running LLM-based correction")
        
        # Handle None or empty query
        if cypher_query is None or not cypher_query:
            logger.warning("Cannot apply LLM-based correction to None or empty query")
            return CorrectionResult(
                corrected_cypher=cypher_query or "",
                was_corrected=False,
                correction_type=CorrectionType.LLM_BASED
            )
        
        if not self.llm_client:
            logger.warning("No LLM client available for LLM-based correction")
            return CorrectionResult(
                corrected_cypher=cypher_query,
                was_corrected=False,
                correction_type=CorrectionType.LLM_BASED
            )
        
        try:
            system_message = """You are an expert Cypher query corrector. Your job is to fix broken or invalid Cypher queries.

Given:
1. The original natural language question
2. A Cypher query that has errors
3. The error message describing what's wrong
4. Database schema (if available)

Generate a CORRECTED Cypher query that fixes the issues.

IMPORTANT:
- Return ONLY the corrected Cypher query
- Do NOT include explanations or markdown formatting
- Do NOT include code blocks (no ```)
- Ensure the query matches the user's intent"""

            schema_text = f"\n\nDatabase Schema:\n{schema}" if schema else ""
            error_text = f"\n\nError Message:\n{error_message}" if error_message else ""
            
            user_message = f"""Natural Language Question: {user_query}

Current (Invalid) Cypher Query:
{cypher_query}{error_text}{schema_text}

Please provide the CORRECTED Cypher query:"""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            corrected_cypher = self.llm_client.chat_completion(messages)
            
            # Clean up the response
            corrected_cypher = corrected_cypher.strip()
            
            # Remove markdown code blocks if present
            corrected_cypher = re.sub(r'```cypher\s*', '', corrected_cypher)
            corrected_cypher = re.sub(r'```\s*', '', corrected_cypher)
            corrected_cypher = corrected_cypher.strip()
            
            # Check if actually different from original
            if corrected_cypher != cypher_query and corrected_cypher:
                logger.info(f"LLM correction generated new query (length: {len(corrected_cypher)})")
                return CorrectionResult(
                    corrected_cypher=corrected_cypher,
                    was_corrected=True,
                    correction_type=CorrectionType.LLM_BASED,
                    correction_description="LLM-based correction applied"
                )
            else:
                logger.debug("LLM returned same or empty query")
                return CorrectionResult(
                    corrected_cypher=cypher_query,
                    was_corrected=False,
                    correction_type=CorrectionType.LLM_BASED
                )
                
        except Exception as e:
            logger.error(f"Error in LLM-based correction: {str(e)}")
            return CorrectionResult(
                corrected_cypher=cypher_query,
                was_corrected=False,
                correction_type=CorrectionType.LLM_BASED,
                correction_description=f"Error: {str(e)}"
            )

