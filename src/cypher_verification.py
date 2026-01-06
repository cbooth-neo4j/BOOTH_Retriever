"""Cypher verification techniques for iterative refinement."""

import re
from typing import Tuple, Optional, Dict, Any
from enum import Enum

from src.logger import setup_logger

logger = setup_logger("booth.verification")


class VerifierType(Enum):
    """Types of Cypher verifiers."""
    RULE_BASED = "rule_based"
    CYVER = "cyver"
    EXECUTION_BASED = "execution_based"
    LLM_BASED = "llm_based"


class VerificationResult:
    """Result from Cypher verification."""
    
    def __init__(
        self, 
        is_valid: bool, 
        error_message: Optional[str] = None,
        verifier_type: Optional[VerifierType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.is_valid = is_valid
        self.error_message = error_message
        self.verifier_type = verifier_type
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"VerificationResult(valid={self.is_valid}, type={self.verifier_type}, error={self.error_message[:50] if self.error_message else None})"


class CypherVerifier:
    """Main verifier class coordinating multiple verification techniques."""
    
    def __init__(self, neo4j_client=None, llm_client=None):
        """Initialize verifier with optional clients.
        
        Args:
            neo4j_client: Neo4j client for execution-based verification
            llm_client: LLM client for LLM-based verification
        """
        self.neo4j_client = neo4j_client
        self.llm_client = llm_client
    
    def verify(
        self, 
        cypher_query: str,
        user_query: str,
        verifier_types: list[VerifierType],
        use_metadata: bool = True
    ) -> VerificationResult:
        """Verify Cypher query using specified verification techniques.
        
        Executes verifiers in order: RULE_BASED -> CYVER -> EXECUTION_BASED -> LLM_BASED
        Stops at first failure or returns success if all pass.
        
        Args:
            cypher_query: The Cypher query to verify
            user_query: Original natural language query for context
            verifier_types: List of verifier types to use
            use_metadata: Whether to include detailed metadata in results
            
        Returns:
            VerificationResult with validation status and error details
        """
        logger.debug(f"Starting verification with {len(verifier_types)} verifiers: {[v.value for v in verifier_types]}")
        
        # Define execution order
        execution_order = [
            VerifierType.RULE_BASED,
            VerifierType.CYVER,
            VerifierType.EXECUTION_BASED,
            VerifierType.LLM_BASED
        ]
        
        # Filter to only requested verifiers in correct order
        ordered_verifiers = [v for v in execution_order if v in verifier_types]
        
        for verifier_type in ordered_verifiers:
            logger.debug(f"Running {verifier_type.value} verification")
            
            if verifier_type == VerifierType.RULE_BASED:
                result = self._rule_based_verification(cypher_query)
            elif verifier_type == VerifierType.CYVER:
                result = self._cyver_verification(cypher_query)
            elif verifier_type == VerifierType.EXECUTION_BASED:
                result = self._execution_based_verification(cypher_query)
            elif verifier_type == VerifierType.LLM_BASED:
                result = self._llm_based_verification(cypher_query, user_query)
            else:
                logger.warning(f"Unknown verifier type: {verifier_type}")
                continue
            
            if not result.is_valid:
                logger.info(f"Verification failed at {verifier_type.value}: {result.error_message[:100]}")
                return result
        
        logger.info("All verifications passed")
        return VerificationResult(is_valid=True)
    
    def _rule_based_verification(self, cypher_query: str) -> VerificationResult:
        """Fast rule-based verification checking relation direction syntax.
        
        Checks for common Cypher syntax errors:
        - Incorrect relationship direction syntax
        - Missing or malformed brackets
        - Basic syntax patterns
        
        Args:
            cypher_query: The Cypher query to verify
            
        Returns:
            VerificationResult
        """
        logger.debug("Running rule-based verification")
        
        # Check for empty query
        if not cypher_query or not cypher_query.strip():
            return VerificationResult(
                is_valid=False,
                error_message="Empty Cypher query",
                verifier_type=VerifierType.RULE_BASED
            )
        
        # Check for basic Cypher keywords
        cypher_upper = cypher_query.upper()
        has_cypher_keyword = any(keyword in cypher_upper for keyword in [
            'MATCH', 'CREATE', 'MERGE', 'DELETE', 'SET', 'RETURN', 'WITH', 'UNWIND'
        ])
        
        if not has_cypher_keyword:
            return VerificationResult(
                is_valid=False,
                error_message="No valid Cypher keywords found (MATCH, RETURN, etc.)",
                verifier_type=VerifierType.RULE_BASED
            )
        
        # Check for relationship direction errors
        # Valid: -[:REL]-, -[:REL]->, <-[:REL]-
        # Invalid: <-[:REL]->, >-[:REL]-<, etc.
        
        # Find all relationship patterns
        rel_pattern = r'<*-\[.*?\]-*>*'
        relationships = re.findall(rel_pattern, cypher_query)
        
        for rel in relationships:
            # Check for invalid bidirectional: <-[:REL]->
            if re.match(r'<-\[.*?\]->', rel):
                return VerificationResult(
                    is_valid=False,
                    error_message=f"Invalid bidirectional relationship pattern: {rel}",
                    verifier_type=VerifierType.RULE_BASED,
                    metadata={"invalid_pattern": rel}
                )
            
            # Check for invalid reverse: >-[:REL]-<
            if re.match(r'>-\[.*?\]-<', rel):
                return VerificationResult(
                    is_valid=False,
                    error_message=f"Invalid reverse relationship pattern: {rel}",
                    verifier_type=VerifierType.RULE_BASED,
                    metadata={"invalid_pattern": rel}
                )
        
        # Check for unmatched brackets
        if cypher_query.count('[') != cypher_query.count(']'):
            return VerificationResult(
                is_valid=False,
                error_message="Unmatched square brackets in query",
                verifier_type=VerifierType.RULE_BASED
            )
        
        if cypher_query.count('(') != cypher_query.count(')'):
            return VerificationResult(
                is_valid=False,
                error_message="Unmatched parentheses in query",
                verifier_type=VerifierType.RULE_BASED
            )
        
        logger.debug("Rule-based verification passed")
        return VerificationResult(is_valid=True, verifier_type=VerifierType.RULE_BASED)
    
    def _cyver_verification(self, cypher_query: str) -> VerificationResult:
        """Verify Cypher syntax using CyVer-like validation.
        
        This simulates CyVer verification by checking for common syntax errors.
        In production, you might use an actual CyVer library if available.
        
        Args:
            cypher_query: The Cypher query to verify
            
        Returns:
            VerificationResult
        """
        logger.debug("Running CyVer-style verification")
        
        try:
            # Basic Cypher syntax checks
            # Check for required RETURN clause in read queries
            cypher_upper = cypher_query.upper()
            
            # If it's a read query (MATCH/OPTIONAL MATCH), it should have RETURN
            if ('MATCH' in cypher_upper) and ('RETURN' not in cypher_upper) and ('DELETE' not in cypher_upper):
                return VerificationResult(
                    is_valid=False,
                    error_message="MATCH query missing RETURN clause",
                    verifier_type=VerifierType.CYVER
                )
            
            # Check for incomplete WHERE clauses
            if 'WHERE' in cypher_upper:
                # Simple check: WHERE should be followed by something meaningful
                where_parts = re.split(r'\bWHERE\b', cypher_query, flags=re.IGNORECASE)
                for part in where_parts[1:]:  # Skip first part (before WHERE)
                    condition = part.strip().split()[0] if part.strip() else ""
                    if not condition or condition.upper() in ['RETURN', 'WITH', 'ORDER', 'LIMIT']:
                        return VerificationResult(
                            is_valid=False,
                            error_message="Incomplete WHERE clause detected",
                            verifier_type=VerifierType.CYVER
                        )
            
            # Check for property access without dot notation
            # Pattern: word immediately followed by opening bracket without dot
            invalid_property = re.search(r'\b\w+\s*\[\s*["\']', cypher_query)
            if invalid_property:
                return VerificationResult(
                    is_valid=False,
                    error_message=f"Invalid property access syntax near: {invalid_property.group()}",
                    verifier_type=VerifierType.CYVER
                )
            
            logger.debug("CyVer-style verification passed")
            return VerificationResult(is_valid=True, verifier_type=VerifierType.CYVER)
            
        except Exception as e:
            logger.error(f"Error in CyVer verification: {str(e)}")
            return VerificationResult(
                is_valid=False,
                error_message=f"CyVer verification error: {str(e)}",
                verifier_type=VerifierType.CYVER
            )
    
    def _execution_based_verification(self, cypher_query: str) -> VerificationResult:
        """Verify by attempting to execute the query against the database.
        
        This is slower but most accurate - actually tries to run the query.
        
        Args:
            cypher_query: The Cypher query to verify
            
        Returns:
            VerificationResult
        """
        logger.debug("Running execution-based verification")
        
        if not self.neo4j_client:
            logger.warning("No Neo4j client available for execution-based verification")
            return VerificationResult(is_valid=True, verifier_type=VerifierType.EXECUTION_BASED)
        
        try:
            # Try to execute the query
            success, result_data, error_message = self.neo4j_client.execute_cypher(cypher_query)
            
            if success:
                logger.debug("Execution-based verification passed")
                return VerificationResult(
                    is_valid=True,
                    verifier_type=VerifierType.EXECUTION_BASED,
                    metadata={"execution_success": True, "result_count": len(result_data) if isinstance(result_data, list) else 1}
                )
            else:
                logger.debug(f"Execution failed: {error_message}")
                return VerificationResult(
                    is_valid=False,
                    error_message=f"Execution error: {error_message}",
                    verifier_type=VerifierType.EXECUTION_BASED,
                    metadata={"execution_success": False}
                )
                
        except Exception as e:
            logger.error(f"Error in execution-based verification: {str(e)}")
            return VerificationResult(
                is_valid=False,
                error_message=f"Execution verification error: {str(e)}",
                verifier_type=VerifierType.EXECUTION_BASED
            )
    
    def _llm_based_verification(self, cypher_query: str, user_query: str) -> VerificationResult:
        """Verify using LLM to check if Cypher matches user intent.
        
        This is the slowest but most comprehensive check.
        
        Args:
            cypher_query: The Cypher query to verify
            user_query: Original natural language query
            
        Returns:
            VerificationResult
        """
        logger.debug("Running LLM-based verification")
        
        if not self.llm_client:
            logger.warning("No LLM client available for LLM-based verification")
            return VerificationResult(is_valid=True, verifier_type=VerifierType.LLM_BASED)
        
        try:
            system_message = """You are a Cypher query validator. Your job is to verify if a generated Cypher query correctly represents the user's natural language question.

Respond with ONLY one of:
- VALID: if the Cypher correctly represents the intent
- INVALID: <brief reason>: if there are issues

Be strict but fair. Check for:
1. Does the query match the user's intent?
2. Are there syntax errors?
3. Are there logical errors?
4. Is it efficient and safe?"""

            user_message = f"""Natural Language Question: {user_query}

Generated Cypher Query:
{cypher_query}

Is this Cypher query valid and correct?"""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = self.llm_client.chat_completion(messages)
            response_upper = response.strip().upper()
            
            if response_upper.startswith('VALID'):
                logger.debug("LLM-based verification passed")
                return VerificationResult(
                    is_valid=True,
                    verifier_type=VerifierType.LLM_BASED,
                    metadata={"llm_response": response}
                )
            else:
                # Extract error message
                error_msg = response.replace('INVALID:', '').replace('INVALID', '').strip()
                logger.debug(f"LLM flagged as invalid: {error_msg}")
                return VerificationResult(
                    is_valid=False,
                    error_message=error_msg or "LLM validation failed",
                    verifier_type=VerifierType.LLM_BASED,
                    metadata={"llm_response": response}
                )
                
        except Exception as e:
            logger.error(f"Error in LLM-based verification: {str(e)}")
            # Don't fail the query due to verification error - be lenient
            return VerificationResult(
                is_valid=True,
                verifier_type=VerifierType.LLM_BASED,
                metadata={"error": str(e)}
            )

