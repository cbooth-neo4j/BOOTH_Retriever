# BOOTH Retriever — Demo Guide

## The demo question (high-risk)

> **Which RFPs require banking partners to compete on pricing and rebates, and
> what are the evaluation criteria?**

```cypher
// Which RFPs compete on pricing/rebates + their evaluation criteria.
// Returns PATHS (not just scalars) so the Ask-page popup can draw the
// actual Document -> Chunk -> EVALUATION_CRITERIA / BANKING_PARTNER subgraph
// that produced the answer.
MATCH path = (doc:Document)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(ec:EVALUATION_CRITERIA)
WHERE toLower(coalesce(ec.name, '')) CONTAINS 'pric'
   OR toLower(coalesce(ec.name, '')) CONTAINS 'rebate'
   OR toLower(coalesce(ec.name, '')) CONTAINS 'fee'
OPTIONAL MATCH partner = (doc)<-[:PART_OF]-(:Chunk)-[:HAS_ENTITY]->(:BANKING_PARTNER)
RETURN doc.name        AS rfp_document,
       ec.name         AS pricing_criterion,
       path            AS criteria_path,
       partner         AS banking_partner_path
ORDER BY rfp_document, pricing_criterion
```
