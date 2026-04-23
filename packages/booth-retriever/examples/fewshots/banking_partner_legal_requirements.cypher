// Question: "What legal and compliance requirements must the potential banking partner meet?"
//
// Graph model (unstructured KG, chunk-based):
//   (:Chunk)-[:HAS_ENTITY]->(entity:__Entity__ :LEGAL_REQUIREMENT | :COMPLIANCE_FRAMEWORK | :CORPORATE_POLICY | :BANKING_PARTNER | ...)
// Entities don't connect to each other directly; they co-occur in shared chunks.
//
// Strategy: find entities typed as a legal/compliance/policy requirement that
// appear in at least one chunk alongside a BANKING_PARTNER entity.
MATCH (:BANKING_PARTNER)<-[:HAS_ENTITY]-(:Chunk)-[:HAS_ENTITY]->(req)
WHERE req.entity_type IN [
    'LEGAL_REQUIREMENT',
    'COMPLIANCE_FRAMEWORK',
    'CORPORATE_POLICY'
]
RETURN DISTINCT
    req.name AS requirement,
    req.description AS description,
    req.entity_type AS category
ORDER BY category, requirement
