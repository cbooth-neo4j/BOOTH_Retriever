"""Seed a procedural-memory example into BOOTH's approved queries.

Procedural memory in BOOTH is modelled as a *command* request hung off an
approved ``Query`` whose steps form an ordered (and branching) chain. Each step
is performed by an agent, each agent calls one or more tools, and — this is the
part that makes the example realistic — each *data* tool is backed by a governed
data product sourced from one of the underlying systems:

    (Query)-[:HAS_STEP]->(Step)-[:NEXT]->(Step)-[:NEXT]-> ...
            (Step)-[:USES_AGENT]->(Agent)-[:USES_TOOL]->(Tool)
            (Tool)-[:BACKED_BY]->(DataProduct)-[:SOURCED_FROM]->(System)

This models a process-following agent with embedded AI judgement reconciling a
single **APAC custody / fund-accounting payment break**. The agent is given the
break's UUID and works it end to end:

    1. Retrieve the break from the Transaction Lifecycle Management (TLM)
       system by UUID.
    2. Map the break to its transaction flow — a deterministic lookup that
       recognises it as an APAC payment reconciliation and loads the playbook.
    3. Gather wider context from the two core banking systems (Citi, Vanguard).
    4. **Classify** the break — Payment in Live vs Return of Funds. This is the
       AI-judgement divergence; the label dynamically selects the substeps.
    5. (Payment in Live, expanded here) additional due diligence — pull deeper
       settlement evidence from TLM, Citi and Vanguard.
    6. **Decisioning** — synthesise the evidence into a disposition + rationale.
    7. Annotate the scenario and rationale back into OpsFlow *and* TLM.
    8. Apply the break age rule — age drives any additional follow-up steps.

    (Return of Funds is shown as a single stub branch; only Payment in Live is
    expanded in this example.)

What the graph captures (the build-time/run-time story made concrete):

    * **Steps** are tagged ``step_type`` = ``deterministic`` | ``ai_judgement``
      so the mix of rules-based work and AI judgement is visible.
    * **Agents** carry the same ``kind``; they are the unit that calls tools.
    * **Tools** are MCP-registry-style callable interfaces with intent-
      revealing names and rich descriptions (what they return and what they
      should NOT be used for) — exactly what an LLM reads when choosing a tool,
      and the investment that is reusable across processes. ``category`` is
      ``data`` (reads/writes a data product) or ``capability`` (pure judgement).
    * **DataProducts** are governed products on the data platform (DPM). Each
      carries a ``readiness_scenario`` (1-4, the four-scenario data-access
      problem), a pipeline ``status``, plus owner / freshness / entitlements /
      a semantic ``description``.
    * **Systems** are the four systems the data is sourced from: TLM, Citi,
      Vanguard and OpsFlow.

Usage:

    python packages/booth-retriever/examples/seed_procedural_memory.py

Requires the usual Neo4j env vars (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD /
NEO4J_DATABASE). Run ``booth init`` first so the Step / Agent / Tool / System /
DataProduct constraints exist. Idempotent: it MERGEs on stable ids/names and
cleans up prior versions of the example, so re-running does not create
duplicates.

If ``OPENAI_API_KEY`` is set, the command's embedding is computed and stored so
the example is retrievable on the Ask page (and its procedure graph shows in
the NVL popup). Without a key it is still created as an approved example for the
Curate page and the graph endpoint.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Stable ids so re-running this script is idempotent.
_QUERY_ID = "example-procedural-payments-recon"
_FEWSHOT_ID = "example-procedural-payments-recon-fewshot"

# A command request (imperative), keyed on a single break UUID. The UUID below
# is a representative sample; at run time the agent is given the real break id.
_BREAK_UUID = "7c2f9a84-3b6e-4f1a-9b2d-2f5e8c1a4d77"
_COMMAND = (
    "Reconcile the APAC custody payment break with UUID "
    f"{_BREAK_UUID} from the transaction lifecycle management system."
)

# Older examples / prior versions this seed supersedes; removed on run so the
# graph stays clean.
_OLD_QUERY_ID = "example-procedural-onboarding"
# Tool nodes from earlier versions of this example (system-as-tool era, then the
# WHT-recompute era). Deleted on run so they don't linger orphaned.
_LEGACY_TOOL_NAMES = [
    # original system-as-tool nodes
    "ReconBreakSystem", "CustodyLedger", "FundAccountingSystem", "PaymentsHub",
    "KeywordClassifier", "DataTransformer", "OpsCaseManager", "EscalationRouter",
    # WHT-recompute version tools
    "get_settled_payment", "get_reconciliation_breaks", "get_tax_treatment",
    "get_custody_entries", "classify_break_keywords", "recompute_net_amount",
    "annotate_ops_case", "route_escalation",
    # original onboarding-example tools
    "DocumentCollector", "WatchlistScreener", "AccountProvisioner",
]
# Agents from earlier versions that the new flow no longer uses.
_LEGACY_AGENT_NAMES = [
    # WHT-recompute version
    "FilterAgent", "CrossSystemLookupAgent", "LiveSettlementAgent",
    "TaxLookupAgent", "TransformAgent", "ReconcileAgent", "RootCauseAgent",
    "EscalationAgent",
    # original onboarding example
    "IntakeAgent", "ComplianceAgent", "ProvisioningAgent",
]
# Systems / data products from the WHT-recompute version, now replaced.
_LEGACY_SYSTEM_NAMES = [
    "Payments Hub", "Reconciliation Break System", "Fund Accounting System",
    "Custody Ledger",
]
_LEGACY_DATA_PRODUCT_IDS = [
    "dp_settled_payments", "dp_recon_breaks", "dp_fa_tax_treatments",
    "dp_custody_entries",
]

# The FewShot returns the whole procedure (both branches), ordered, with each
# step's type, agent, the tools it calls, and — for data tools — the systems and
# data-readiness scenarios behind them.
_FEWSHOT_CYPHER = (
    "MATCH (q:Query {id: 'example-procedural-payments-recon'})-[:HAS_STEP]->(head:Step) "
    "MATCH (head)-[:NEXT*0..]->(s:Step) "
    "OPTIONAL MATCH (s)-[:USES_AGENT]->(a:Agent) "
    "OPTIONAL MATCH (a)-[:USES_TOOL]->(t:Tool) "
    "OPTIONAL MATCH (t)-[:BACKED_BY]->(dp:DataProduct)-[:SOURCED_FROM]->(sys:System) "
    "RETURN s.scenario AS scenario, s.order AS step_order, s.name AS step, "
    "s.step_type AS step_type, a.name AS agent, a.kind AS agent_kind, "
    "collect(DISTINCT t.name) AS tools, "
    "collect(DISTINCT sys.name) AS systems, "
    "collect(DISTINCT dp.readiness_scenario) AS data_readiness_scenarios "
    "ORDER BY s.order"
)

# ---------------------------------------------------------------------------
# The data dimension: Systems -> DataProducts -> (BACKED_BY) Tools.
# ---------------------------------------------------------------------------

# Four systems the flow touches. ``kind`` distinguishes the bank's internal
# systems from external institutions (the harder data-access cases).
_SYSTEMS: dict[str, dict[str, str]] = {
    "Transaction Lifecycle Management": {
        "kind": "internal",
        "description": (
            "Source system for reconciliation breaks; holds each break by UUID "
            "and its end-to-end transaction lifecycle/event flow."
        ),
    },
    "OpsFlow": {
        "kind": "internal",
        "description": (
            "Operations case-management/workflow system; reconciliation cases "
            "are annotated here with the agent's decision and rationale."
        ),
    },
    "Vanguard": {
        "kind": "external",
        "description": (
            "External fund-administration banking system; payment context is "
            "portal-sourced, a consumption API is still in build."
        ),
    },
    "Citi": {
        "kind": "external",
        "description": (
            "External correspondent banking system; payment/settlement context "
            "has no direct API and must be fed via an Authoritative Data Source."
        ),
    },
}

# Data products on the platform (DPM). ``readiness_scenario`` is the four-
# scenario data-access problem; ``status`` is the pipeline view of where each
# product is on the road to being consumable.
#   1 = API exists, fields match            -> available
#   2 = API exists, fields differ (mapped)  -> available_with_mapping
#   3 = no consumption API (UI-only)        -> api_in_build
#   4 = vendor/external, needs ADS feed     -> ads_feed_required
_DATA_PRODUCTS: dict[str, dict] = {
    "dp_tlm_break_flow": {
        "name": "Transaction Lifecycle Break & Flow",
        "system": "Transaction Lifecycle Management",
        "readiness_scenario": 1,
        "status": "available",
        "owner": "TLM Platform",
        "freshness": "intraday (T+0)",
        "entitlements": "ops-apac-tlm",
        "description": (
            "The break record (by UUID) plus the transaction lifecycle/event "
            "flow — states, legs, value dates. Well-documented API; field names "
            "align with the process."
        ),
    },
    "dp_opsflow_cases": {
        "name": "OpsFlow Reconciliation Cases",
        "system": "OpsFlow",
        "readiness_scenario": 2,
        "status": "available_with_mapping",
        "owner": "Ops Platform",
        "freshness": "intraday (T+0)",
        "entitlements": "ops-apac-opsflow",
        "description": (
            "Reconciliation cases with disposition and rationale annotations. "
            "API exists; UI field names were mapped to API fields."
        ),
    },
    "dp_vanguard_context": {
        "name": "Vanguard Payment Context",
        "system": "Vanguard",
        "readiness_scenario": 3,
        "status": "api_in_build",
        "owner": "Fund Admin Data",
        "freshness": "daily batch (T+1)",
        "entitlements": "ops-apac-vanguard",
        "description": (
            "Fund-administration payment and settlement context. Portal-only "
            "today; a consumption API is being specified and built."
        ),
    },
    "dp_citi_context": {
        "name": "Citi Payment Context",
        "system": "Citi",
        "readiness_scenario": 4,
        "status": "ads_feed_required",
        "owner": "Correspondent Data",
        "freshness": "daily batch (T+1)",
        "entitlements": "ops-apac-citi",
        "description": (
            "Correspondent-banking payment and settlement context. External "
            "source with no direct API; must be fed into an Authoritative Data "
            "Source before exposure."
        ),
    },
}

# Tools = MCP-registry callable interfaces. ``data`` tools read or write a data
# product (``data_product`` key); ``capability`` tools are pure judgement.
# Rich, intent-revealing descriptions are what an LLM reads when deciding to call.
_TOOLS: dict[str, dict] = {
    "get_break_by_uuid": {
        "category": "data",
        "data_product": "dp_tlm_break_flow",
        "description": (
            "Return the reconciliation break record for a break UUID from the "
            "Transaction Lifecycle Management (TLM) system — status, amounts, "
            "references. Use to source the break under investigation. NOT for "
            "posting annotations."
        ),
    },
    "get_transaction_flow": {
        "category": "data",
        "data_product": "dp_tlm_break_flow",
        "description": (
            "Return the end-to-end transaction lifecycle/event flow (states, "
            "legs, timestamps) for a break from TLM, which identifies it as an "
            "APAC payment reconciliation. Use to load the applicable flow. "
            "Read-only."
        ),
    },
    "get_citi_payment_context": {
        "category": "data",
        "data_product": "dp_citi_context",
        "description": (
            "Return correspondent-banking payment/settlement context for a "
            "reference from Citi. External source fed via ADS. Use for wider "
            "cross-bank context. NOT a settlement instruction."
        ),
    },
    "get_vanguard_payment_context": {
        "category": "data",
        "data_product": "dp_vanguard_context",
        "description": (
            "Return fund-administration payment context for a reference from "
            "Vanguard. Portal-sourced, consumption API in build. Use for wider "
            "cross-bank context."
        ),
    },
    "get_payment_settlement_detail": {
        "category": "data",
        "data_product": "dp_tlm_break_flow",
        "description": (
            "Return detailed settlement evidence (value dates, settled legs, "
            "net/gross) for a payment from the TLM lifecycle. Use for Payment-"
            "in-Live due diligence. Read-only."
        ),
    },
    "get_citi_settlement_detail": {
        "category": "data",
        "data_product": "dp_citi_context",
        "description": (
            "Return deeper Citi settlement/value-date detail for a confirmed "
            "live payment. Use during Payment-in-Live due diligence. Read-only."
        ),
    },
    "get_vanguard_settlement_detail": {
        "category": "data",
        "data_product": "dp_vanguard_context",
        "description": (
            "Return deeper Vanguard payment/settlement detail for a live "
            "payment. Use during Payment-in-Live due diligence. Read-only."
        ),
    },
    "annotate_opsflow_case": {
        "category": "data",
        "data_product": "dp_opsflow_cases",
        "description": (
            "Write the disposition, scenario and supporting rationale onto the "
            "OpsFlow reconciliation case. Use to record the outcome in Ops. NOT "
            "for TLM updates."
        ),
    },
    "annotate_tlm_break": {
        "category": "data",
        "data_product": "dp_tlm_break_flow",
        "description": (
            "Annotate the TLM break with the agent's decision, scenario and "
            "backing evidence. Use to close the loop in the source system. NOT "
            "for OpsFlow."
        ),
    },
    "classify_break_type": {
        "category": "capability",
        "description": (
            "Classify a break as Payment-in-Live vs Return-of-Funds from the "
            "assembled cross-system context plus AI judgement; returns a label "
            "and confidence. Drives the branch and substep selection. NOT a "
            "settlement check."
        ),
    },
    "decide_break_disposition": {
        "category": "capability",
        "description": (
            "Synthesise the collected evidence into a reconciliation decision "
            "and rationale. AI-judgement decisioning step; performs no I/O."
        ),
    },
    "apply_age_rule": {
        "category": "capability",
        "description": (
            "Evaluate the break's age against SLA thresholds and determine any "
            "additional follow-up steps. Deterministic rule; age drives "
            "additional steps."
        ),
    },
}

# Agents: name -> kind ("deterministic" | "ai_judgement").
_AGENTS: dict[str, str] = {
    "RetrievalAgent": "deterministic",
    "FlowMappingAgent": "deterministic",
    "CrossBankContextAgent": "deterministic",
    "ClassificationAgent": "ai_judgement",
    "DueDiligenceAgent": "deterministic",
    "DecisioningAgent": "ai_judgement",
    "AnnotationAgent": "deterministic",
    "AgeingAgent": "deterministic",
    "ReturnsAgent": "deterministic",
}

# Steps: (key, order, scenario, name, step_type, agent, [tools]).
# Shared prefix uses orders 0-3; Payment-in-Live uses 100+, the Return-of-Funds
# stub uses 200, so a simple ``ORDER BY order`` reads shared -> PIL -> RoF.
_STEPS: list[tuple[str, int, str, str, str, str, list[str]]] = [
    # ---- Shared investigation (runs for every break) ----
    ("s1", 0, "shared", "Retrieve the break from TLM by UUID",
     "deterministic", "RetrievalAgent", ["get_break_by_uuid"]),
    ("s2", 1, "shared", "Map the break to its transaction flow (APAC payment reconciliation)",
     "deterministic", "FlowMappingAgent", ["get_transaction_flow"]),
    ("s3", 2, "shared", "Gather wider context from core banking systems (Citi, Vanguard)",
     "deterministic", "CrossBankContextAgent",
     ["get_citi_payment_context", "get_vanguard_payment_context"]),
    ("s4", 3, "shared", "Classify the break: Payment in Live vs Return of Funds",
     "ai_judgement", "ClassificationAgent", ["classify_break_type"]),
    # ---- Payment in Live (expanded path) ----
    ("pil1", 100, "payment_in_live",
     "Additional due diligence: collect detailed settlement evidence",
     "deterministic", "DueDiligenceAgent",
     ["get_payment_settlement_detail", "get_citi_settlement_detail",
      "get_vanguard_settlement_detail"]),
    ("pil2", 101, "payment_in_live", "Decisioning: determine the break disposition",
     "ai_judgement", "DecisioningAgent", ["decide_break_disposition"]),
    ("pil3", 102, "payment_in_live",
     "Annotate the scenario and rationale in OpsFlow and TLM",
     "deterministic", "AnnotationAgent",
     ["annotate_opsflow_case", "annotate_tlm_break"]),
    ("pil4", 103, "payment_in_live",
     "Apply the break age rule (age drives additional steps)",
     "deterministic", "AgeingAgent", ["apply_age_rule"]),
    # ---- Return of Funds (single stub branch; not expanded in this example) ----
    ("rof1", 200, "return_of_funds",
     "Follow the Return-of-Funds playbook (not expanded in this example)",
     "deterministic", "ReturnsAgent", []),
]

# NEXT edges. The classify step (s4) branches into both scenarios; the label
# from classification dynamically selects which branch is followed at run time.
_NEXT: list[tuple[str, str]] = [
    ("s1", "s2"), ("s2", "s3"), ("s3", "s4"),
    ("s4", "pil1"), ("pil1", "pil2"), ("pil2", "pil3"), ("pil3", "pil4"),
    ("s4", "rof1"),
]

_HEAD_KEY = "s1"


def _step_id(key: str) -> str:
    return f"{_QUERY_ID}-{key}"


def seed(driver, database: str | None, embedding: list[float] | None) -> None:
    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        session.execute_write(_seed_tx, embedding)


def _seed_tx(tx, embedding: list[float] | None) -> None:
    # Remove the older onboarding example so it doesn't linger alongside this one.
    tx.run(
        "MATCH (q:Query {id: $old_id}) "
        "OPTIONAL MATCH (q)-[:HAS_STEP]->(s:Step) "
        "OPTIONAL MATCH (q)-[:FEW_SHOT_EXAMPLE]->(fs:FewShot) "
        "DETACH DELETE q, s, fs",
        old_id=_OLD_QUERY_ID,
    )

    # Clean up the prior version of THIS example: its steps, plus the systems,
    # data products, tools and agents that the new flow supersedes.
    tx.run(
        "MATCH (q:Query {id: $query_id})-[:HAS_STEP]->(head:Step) "
        "MATCH (head)-[:NEXT*0..]->(s:Step) DETACH DELETE s",
        query_id=_QUERY_ID,
    )
    tx.run(
        "UNWIND $names AS name MATCH (t:Tool {name: name}) DETACH DELETE t",
        names=_LEGACY_TOOL_NAMES,
    )
    tx.run(
        "UNWIND $names AS name MATCH (a:Agent {name: name}) DETACH DELETE a",
        names=_LEGACY_AGENT_NAMES,
    )
    tx.run(
        "UNWIND $ids AS dp_id MATCH (dp:DataProduct {id: dp_id}) DETACH DELETE dp",
        ids=_LEGACY_DATA_PRODUCT_IDS,
    )
    tx.run(
        "UNWIND $names AS name MATCH (sys:System {name: name}) DETACH DELETE sys",
        names=_LEGACY_SYSTEM_NAMES,
    )

    # Approved Query (a command) + its FewShot.
    tx.run(
        "MERGE (q:Query {id: $query_id}) "
        "SET q.text = $command, q.status = 'approved', q.risk_level = 'low', "
        "q.kind = 'procedural_memory', q.request_type = 'command' "
        "MERGE (fs:FewShot {id: $fewshot_id}) "
        "SET fs.cypher_template = $cypher, fs.parameters = [], "
        "fs.category = 'PROCEDURE' "
        "MERGE (q)-[:FEW_SHOT_EXAMPLE]->(fs)",
        query_id=_QUERY_ID,
        command=_COMMAND,
        fewshot_id=_FEWSHOT_ID,
        cypher=_FEWSHOT_CYPHER,
    )

    if embedding is not None:
        tx.run(
            "MATCH (q:Query {id: $query_id}) SET q.embedding = $embedding",
            query_id=_QUERY_ID,
            embedding=embedding,
        )

    # Systems.
    for name, props in _SYSTEMS.items():
        tx.run(
            "MERGE (sys:System {name: $name}) "
            "SET sys.kind = $kind, sys.description = $description",
            name=name,
            kind=props["kind"],
            description=props["description"],
        )

    # Data products, each sourced from its system.
    for dp_id, props in _DATA_PRODUCTS.items():
        tx.run(
            "MERGE (dp:DataProduct {id: $id}) "
            "SET dp.name = $name, dp.readiness_scenario = $scenario, "
            "dp.status = $status, dp.owner = $owner, dp.freshness = $freshness, "
            "dp.entitlements = $entitlements, dp.description = $description "
            "WITH dp MATCH (sys:System {name: $system}) "
            "MERGE (dp)-[:SOURCED_FROM]->(sys)",
            id=dp_id,
            name=props["name"],
            scenario=props["readiness_scenario"],
            status=props["status"],
            owner=props["owner"],
            freshness=props["freshness"],
            entitlements=props["entitlements"],
            description=props["description"],
            system=props["system"],
        )

    # Tools; data tools are backed by their data product.
    for name, props in _TOOLS.items():
        tx.run(
            "MERGE (t:Tool {name: $name}) "
            "SET t.category = $category, t.description = $description",
            name=name,
            category=props["category"],
            description=props["description"],
        )
        if props.get("data_product"):
            tx.run(
                "MATCH (t:Tool {name: $name}), (dp:DataProduct {id: $dp}) "
                "MERGE (t)-[:BACKED_BY]->(dp)",
                name=name,
                dp=props["data_product"],
            )

    # Agents (deduped by name).
    for name, kind in _AGENTS.items():
        tx.run(
            "MERGE (a:Agent {name: $name}) SET a.kind = $kind",
            name=name,
            kind=kind,
        )

    # Steps, each linked to its agent, and the agent to its tools.
    for key, order, scenario, name, step_type, agent, tools in _STEPS:
        tx.run(
            "MERGE (s:Step {id: $step_id}) "
            "SET s.name = $name, s.order = $order, s.scenario = $scenario, "
            "s.step_type = $step_type "
            "WITH s "
            "MATCH (a:Agent {name: $agent}) "
            "MERGE (s)-[:USES_AGENT]->(a) "
            "WITH a "
            "UNWIND $tools AS tool_name "
            "MATCH (t:Tool {name: tool_name}) "
            "MERGE (a)-[:USES_TOOL]->(t)",
            step_id=_step_id(key),
            name=name,
            order=order,
            scenario=scenario,
            step_type=step_type,
            agent=agent,
            tools=tools,
        )

    # Attach the Query to the head step only; the chain unfolds via NEXT.
    tx.run(
        "MATCH (q:Query {id: $query_id}), (head:Step {id: $head_id}) "
        "MERGE (q)-[:HAS_STEP]->(head)",
        query_id=_QUERY_ID,
        head_id=_step_id(_HEAD_KEY),
    )

    # NEXT edges (including the branch out of the classify step).
    for from_key, to_key in _NEXT:
        tx.run(
            "MATCH (a:Step {id: $from_id}), (b:Step {id: $to_id}) "
            "MERGE (a)-[:NEXT]->(b)",
            from_id=_step_id(from_key),
            to_id=_step_id(to_key),
        )


def _maybe_embed(command: str) -> list[float] | None:
    """Return an embedding for the command, or None if unavailable.

    Best-effort: needs ``OPENAI_API_KEY`` and the ``neo4j-graphrag[openai]``
    extra. Any problem returns None — the example is still seeded, just not
    similarity-retrievable on the Ask page.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    except ImportError:
        return None
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    try:
        return OpenAIEmbeddings(model=model).embed_query(command)
    except Exception as exc:  # noqa: BLE001 - best-effort embedding
        print(f"Warning: could not embed command ({exc}); seeding without it.")
        return None


def main() -> int:
    load_dotenv()
    uri = os.environ.get("NEO4J_URI")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not password:
        print(
            "Error: NEO4J_URI and NEO4J_PASSWORD must be set (in .env or environment).",
            file=sys.stderr,
        )
        return 2
    user = os.environ.get("NEO4J_USER", "neo4j")
    database = os.environ.get("NEO4J_DATABASE")

    embedding = _maybe_embed(_COMMAND)

    try:
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            notifications_disabled_classifications=["UNRECOGNIZED"],
        )
    except TypeError:
        driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        seed(driver, database, embedding)
    finally:
        driver.close()

    print(f"Seeded procedural-memory example as Query id {_QUERY_ID!r}.")
    print(f"  command:   {_COMMAND}")
    print(
        f"  steps:     {len(_STEPS)} distinct "
        "(shared prefix + expanded 'Payment in Live' path + 'Return of Funds' stub)"
    )
    print(
        f"  agents:    {len(_AGENTS)}  tools: {len(_TOOLS)}  "
        f"data products: {len(_DATA_PRODUCTS)}  systems: {len(_SYSTEMS)}"
    )
    print(f"  retrievable on Ask page: {'yes' if embedding else 'no (set OPENAI_API_KEY)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
