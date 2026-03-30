"""BOOTH Retriever - Query Curation Interface

Updated for v2 data model with refinement workflow:
- When approving, triggers RefinementAgent to create parameterized templates
- Shows refinement results (category, template, cypher)
- New "Needs Human Support" tab for failed refinements
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from dotenv import load_dotenv

from src.booth_orchestrator import BOOTHOrchestrator
from src.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger("booth.curation")

# Page configuration
st.set_page_config(
    page_title="BOOTH Train",
    page_icon="✅",
    layout="wide"
)

# Initialize orchestrator
if 'orchestrator' not in st.session_state:
    try:
        logger.info("Initializing BOOTH Orchestrator for curation page")
        st.session_state.orchestrator = BOOTHOrchestrator()
        logger.info("BOOTH Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BOOTH: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize BOOTH: {str(e)}")
        st.stop()

# Header
st.title("✅ Train AI")
st.markdown("""
Review and approve queries to improve BOOTH's performance. Approved queries become few-shot examples for future similar questions.
""")

# Check if this is a new setup
try:
    with st.session_state.orchestrator.neo4j_client.driver.session() as session:
        # Check for entities
        result = session.run("MATCH (e:__Entity__) RETURN count(e) as count LIMIT 1")
        entity_count = result.single()["count"]
        
        if entity_count == 0:
            st.info("👋 No knowledge graph detected. Please complete setup first.")
            st.page_link("pages/0_Setup.py", label="→ Go to Setup", icon="⚙️")
            st.markdown("---")
except Exception as e:
    logger.warning(f"Could not check entity count: {e}")

# Refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

with col2:
    limit = st.number_input("Limit", min_value=10, max_value=100, value=50, step=10)

# Fetch pending queries
with st.spinner("Loading pending queries..."):
    try:
        logger.info(f"Fetching pending queries (limit={limit})")
        pending_queries = st.session_state.orchestrator.get_pending_queries_for_curation(limit=limit)
        logger.info(f"Loaded {len(pending_queries)} pending queries")
    except Exception as e:
        logger.error(f"Failed to load pending queries: {str(e)}", exc_info=True)
        st.error(f"Failed to load pending queries: {str(e)}")
        st.stop()

st.markdown("---")

# Display pending queries
if not pending_queries:
    st.info("🎉 No queries pending approval at the moment!")
    st.markdown("""
    Pending queries will appear here after users submit questions through the main interface.
    
    **What happens when you approve a query?**
    - The query becomes a "few-shot example" for the AI
    - Similar future queries will use this as a reference
    - This improves accuracy and consistency over time
    """)
else:
    st.header(f"Query store")
    
    # Group queries by status and success
    pending_successful = [q for q in pending_queries 
                         if q.get('status') == 'pending_approval' 
                         and q.get('summary')]
    pending_failed = [q for q in pending_queries 
                     if q.get('status') == 'pending_approval' 
                     and not q.get('summary')]
    # Declined includes ALL high-risk declined queries (with or without successful AI attempt)
    declined_queries = [q for q in pending_queries 
                       if q.get('status') == 'declined']
    rejected_queries = [q for q in pending_queries if q.get('status') == 'rejected']
    needs_review_queries = [q for q in pending_queries if q.get('status') == 'needs_review']
    needs_human_support_queries = [q for q in pending_queries if q.get('status') == 'needs_human_support']
    approved_queries = [q for q in pending_queries if q.get('status') == 'approved']
    
    # Tab definitions for tooltips
    tab_definitions = {
        "pending_approval": "Queries that executed successfully but haven't been validated by users yet. Review the Cypher and results before approving as few-shot examples.",
        "approved": "Queries that have been approved and are now being used as few-shot examples for similar future queries.",
        "failed": "Queries that failed to produce results. Review to understand why and reject or debug.",
        "declined": "High-risk queries that were declined to users. The AI ran in the background - some succeeded (🤖) and some failed (❌). Successful ones can be approved for training.",
        "rejected": "Previously rejected queries. Can be re-approved if needed.",
        "needs_review": "Queries marked as 'not helpful' by users. Review to improve system performance.",
        "needs_human_support": "🛠️ Refinement failed! The AI could not consolidate the multi-step query into a single parameterized template. Manual intervention needed."
    }
    
    # Tabs for different statuses
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"⏳ Pending Approval ({len(pending_successful)})",
        f"✅ Approved ({len(approved_queries)})", 
        f"❌ Failed ({len(pending_failed)})",
        f"⚠️ Declined ({len(declined_queries)})",
        f"🚫 Rejected ({len(rejected_queries)})",
        f"🔍 Needs Review ({len(needs_review_queries)})",
        f"🛠️ Needs Human Support ({len(needs_human_support_queries)})"
    ])
    
    with tab0:
        st.caption(f"ℹ️ {tab_definitions['pending_approval']}")
        st.markdown("---")
        
        if not pending_successful:
            st.info("No queries awaiting approval.")
        else:
            for idx, query in enumerate(pending_successful):
                with st.expander(
                    f"Query {idx + 1}: {query['query_text'][:80]}...",
                    expanded=(idx == 0)  # Expand first one
                ):
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Cypher query
                    st.markdown("### 🔧 Generated Cypher")
                    st.code(query['cypher_text'], language="cypher")
                    
                    # Results
                    st.markdown("### 📊 Results")
                    
                    st.markdown("**Summary:**")
                    st.success(query['summary'])
                    
                    # Show original cypher with multi-step queries and raw data
                    if query.get('result_data'):
                        with st.expander("🔧 View Original Cypher"):
                            st.markdown("**Last Query Executed:**")
                            st.code(query['cypher_text'], language="cypher")
                            
                            try:
                                data = json.loads(query['result_data'])
                                queries_executed = data.get('queries_executed', [])
                                
                                if queries_executed:
                                    st.markdown("**Multi-Step Queries Executed:**")
                                    for i, q in enumerate(queries_executed):
                                        st.markdown(f"**Query {i}:**")
                                        st.code(q, language="cypher")
                                
                                st.markdown("**Raw Data:**")
                                st.json(data)
                            except:
                                st.markdown("**Raw Data:**")
                                st.text(query['result_data'])
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button(
                            "✅ Approve",
                            key=f"approve_{idx}_{query['query_id']}",
                            type="primary",
                            use_container_width=True
                        ):
                            try:
                                logger.info(f"User approving query: {query['query_id']}")
                                st.session_state.orchestrator.approve_query(
                                    query['query_id'],
                                    query['cypher_id']
                                )
                                logger.info(f"Query {query['query_id']} approved successfully")
                                st.success("Query approved! It will now be used as a few-shot example.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to approve query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to approve: {str(e)}")
                    
                    with col2:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_{idx}_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                reason = "Rejected during curation"
                                logger.info(f"User rejecting query: {query['query_id']} (reason: {reason})")
                                st.session_state.orchestrator.reject_query(
                                    query['query_id'],
                                    reason
                                )
                                logger.info(f"Query {query['query_id']} rejected successfully")
                                st.warning("Query rejected.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to reject query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to reject: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab1:
        st.caption(f"ℹ️ {tab_definitions['approved']}")
        st.markdown("---")
        
        if not approved_queries:
            st.info("No approved queries found.")
            st.markdown("""
            Approved queries will appear here after you approve them in other tabs.
            
            These queries are now being used as few-shot examples for similar future queries.
            """)
        else:
            st.success(f"✅ {len(approved_queries)} approved query/queries are being used as few-shot examples.")
            
            for idx, query in enumerate(approved_queries):
                with st.expander(
                    f"✅ Query {idx + 1}: {query['query_text'][:80]}...",
                    expanded=(idx == 0)
                ):
                    
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Cypher query
                    if query.get('refined_cypher'):
                        st.markdown("### ✨ Refined Cypher Template")
                        st.code(query['refined_cypher'], language="cypher")
                        
                        # Show original attempt in expander with multi-step queries and raw data
                        if query.get('cypher_text') or query.get('result_data'):
                            with st.expander("🔧 View Original Cypher"):
                                if query.get('cypher_text'):
                                    st.markdown("**Last Query Executed:**")
                                    st.code(query['cypher_text'], language="cypher")
                                
                                if query.get('result_data'):
                                    try:
                                        data = json.loads(query['result_data'])
                                        queries_executed = data.get('queries_executed', [])
                                        
                                        if queries_executed:
                                            st.markdown("**Multi-Step Queries Executed:**")
                                            for i, q in enumerate(queries_executed):
                                                st.markdown(f"**Query {i}:**")
                                                st.code(q, language="cypher")
                                        
                                        st.markdown("**Raw Data:**")
                                        st.json(data)
                                    except:
                                        st.markdown("**Raw Data:**")
                                        st.text(query['result_data'])
                    elif query.get('cypher_text'):
                        st.markdown("### 🔧 Generated Cypher")
                        st.code(query['cypher_text'], language="cypher")
                    
                    # Results
                    if query.get('summary'):
                        st.markdown("### 📊 Results")
                        st.markdown("**Summary:**")
                        st.success(query['summary'])
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if st.button(
                            "🗑️ Delete",
                            key=f"delete_approved_{idx}_{query['query_id']}",
                            type="secondary",
                            use_container_width=True
                        ):
                            try:
                                logger.info(f"User deleting approved query: {query['query_id']}")
                                success = st.session_state.orchestrator.delete_query(query['query_id'])
                                if success:
                                    logger.info(f"Approved query {query['query_id']} deleted successfully")
                                    st.success("✅ Query deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error("Query not found or could not be deleted.")
                            except Exception as e:
                                logger.error(f"Failed to delete query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to delete: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab2:
        st.caption(f"ℹ️ {tab_definitions['failed']}")
        st.markdown("---")
        
        if not pending_failed:
            st.info("No failed queries pending review.")
        else:
            st.warning(f"These queries failed to execute successfully and may need manual review.")
            
            for idx, query in enumerate(pending_failed):
                with st.expander(f"Query {idx + 1}: {query['query_text'][:80]}..."):
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Cypher query
                    st.markdown("### 🔧 Generated Cypher")
                    st.code(query['cypher_text'], language="cypher")
                    
                    # No successful results
                    st.markdown("### ❌ Status")
                    st.error("This query did not produce successful results.")
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_failed_{idx}_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                reason = "Failed to execute successfully"
                                logger.info(f"User rejecting failed query: {query['query_id']} (reason: {reason})")
                                st.session_state.orchestrator.reject_query(
                                    query['query_id'],
                                    reason
                                )
                                logger.info(f"Failed query {query['query_id']} rejected successfully")
                                st.warning("Query rejected.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to reject query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to reject: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab3:
        st.caption(f"ℹ️ {tab_definitions['declined']}")
        st.markdown("---")
        
        if not declined_queries:
            st.info("No declined queries to review.")
        else:
            # Separate declined queries by refinement status
            declined_refined = [q for q in declined_queries if q.get('refinement_success')]
            declined_pending = [q for q in declined_queries if q.get('summary') and not q.get('refinement_success')]
            declined_failed = [q for q in declined_queries if not q.get('summary')]
            
            st.warning("These queries were marked as high-risk by users and automatically declined. The system ran the agent in the background and attempted refinement.")
            
            
            for idx, query in enumerate(declined_queries):
                # Determine if AI succeeded and if refinement was done
                ai_succeeded = bool(query.get('summary'))
                refinement_done = bool(query.get('refinement_success'))
                
                # Determine badge
                if refinement_done:
                    badge = "✨ Refined"
                elif ai_succeeded:
                    badge = "🤖 AI Succeeded"
                else:
                    badge = "❌ AI Failed"
                
                with st.expander(f"{badge} | Query {idx + 1}: {query['query_text'][:80]}..."):
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Status info
                    st.markdown("### ⚠️ Status")
                    risk_level = query.get('risk_level', 'unknown')
                    st.error(f"**Declined** - Risk Level: {risk_level.upper()}")
                    st.caption("Results were NOT shown to user, but agent ran in background for review.")
                    
                    if refinement_done:
                        st.success("✨ **Refinement complete** - Ready for curator approval")
                    elif ai_succeeded:
                        st.info("🤖 **AI generated a response** - Refinement may be in progress or failed")
                    else:
                        st.warning("❌ **AI failed to generate a response** - may need manual review")
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Show refined Cypher if available, otherwise show raw attempt
                    if query.get('refined_cypher'):
                        st.markdown("### ✨ Refined Cypher Template (Few-Shot Ready)")
                        st.code(query['refined_cypher'], language="cypher")
                        
                        # Show original attempt in expander with multi-step queries and raw data
                        with st.expander("🔧 View Original Cypher Attempt"):
                            if query.get('cypher_text'):
                                st.markdown("**Last Query Executed:**")
                                st.code(query['cypher_text'], language="cypher")
                            
                            if query.get('result_data'):
                                try:
                                    data = json.loads(query['result_data'])
                                    queries_executed = data.get('queries_executed', [])
                                    
                                    if queries_executed:
                                        st.markdown("**Multi-Step Queries Executed:**")
                                        for i, q in enumerate(queries_executed):
                                            st.markdown(f"**Query {i}:**")
                                            st.code(q, language="cypher")
                                    
                                    st.markdown("**Raw Data:**")
                                    st.json(data)
                                except:
                                    st.markdown("**Raw Data:**")
                                    st.text(query.get('result_data', ''))
                    elif query.get('cypher_text'):
                        st.markdown("### 🔧 Generated Cypher Attempt")
                        st.code(query['cypher_text'], language="cypher")
                    else:
                        st.info("No Cypher was generated - the agent may have failed early.")
                    
                    # Show response
                    if query.get('summary'):
                        st.markdown("### 📊 Results (Hidden from User)")
                        st.success(query['summary'])
                    elif query.get('cypher_text'):
                        st.markdown("### ⚠️ Execution Status")
                        st.warning("The agent did not produce successful results.")
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        # Determine button label based on refinement status
                        if refinement_done:
                            approve_label = "✅ Approve Template"
                        else:
                            approve_label = "✅ Approve"
                        
                        if st.button(
                            approve_label,
                            key=f"approve_declined_{idx}_{query['query_id']}",
                            type="primary",
                            use_container_width=True,
                            disabled=not (refinement_done or ai_succeeded)
                        ):
                            try:
                                logger.info(f"User approving declined query: {query['query_id']}")
                                
                                if refinement_done:
                                    # Refinement already done - just approve without re-refining
                                    result = st.session_state.orchestrator.approve_query_with_refinement(
                                        query['query_id'],
                                        query['cypher_id'],
                                        trigger_refinement=False  # Skip refinement
                                    )
                                    st.success("✅ Template approved!")
                                elif ai_succeeded:
                                    with st.spinner("🔄 Running refinement agent..."):
                                        result = st.session_state.orchestrator.approve_query_with_refinement(
                                            query['query_id'],
                                            query['cypher_id'],
                                            trigger_refinement=True
                                        )
                                    if result.get('success'):
                                        st.success("✅ Query approved and template created!")
                                    elif result.get('needs_human_support'):
                                        st.warning(f"⚠️ Refinement failed: {result.get('error')}")
                                    else:
                                        st.error(f"Approval failed: {result.get('error')}")
                                else:
                                    st.session_state.orchestrator.approve_query(
                                        query['query_id'],
                                        query['cypher_id']
                                    )
                                    st.success("Query approved!")
                                logger.info(f"Declined query {query['query_id']} approved successfully")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to approve query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to approve: {str(e)}")
                    
                    with col2:
                        if st.button(
                            "❌ Reject",
                            key=f"keep_rejected_declined_{idx}_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                reason = "Confirmed rejection - was correctly declined"
                                logger.info(f"User rejecting declined query: {query['query_id']}")
                                st.session_state.orchestrator.reject_query(
                                    query['query_id'],
                                    reason
                                )
                                logger.info(f"Declined query {query['query_id']} rejected")
                                st.warning("Query rejected.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to reject query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to reject: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab4:
        st.caption(f"ℹ️ {tab_definitions['rejected']}")
        st.markdown("---")
        
        if not rejected_queries:
            st.info("No rejected queries to review.")
        else:
            st.info("These queries were previously rejected during curation.")
            
            for idx, query in enumerate(rejected_queries):
                with st.expander(f"Query {idx + 1}: {query['query_text'][:80]}..."):
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Status info
                    st.markdown("### 🚫 Status")
                    st.warning("**Rejected during curation**")
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Cypher query if available
                    if query.get('cypher_text'):
                        st.markdown("### 🔧 Generated Cypher")
                        st.code(query['cypher_text'], language="cypher")
                        
                        if query.get('summary'):
                            st.markdown("### 📊 Results")
                            st.success(query['summary'])
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if st.button(
                            "✅ Re-approve",
                            key=f"reapprove_{idx}_{query['query_id']}",
                            type="primary",
                            use_container_width=True,
                            disabled=not query.get('cypher_text')
                        ):
                            try:
                                logger.info(f"User re-approving rejected query: {query['query_id']}")
                                st.session_state.orchestrator.approve_query(
                                    query['query_id'],
                                    query['cypher_id']
                                )
                                logger.info(f"Rejected query {query['query_id']} re-approved successfully")
                                st.success("Query re-approved! It will now be used as a few-shot example.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to re-approve query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to re-approve: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab5:
        st.caption(f"ℹ️ {tab_definitions['needs_review']}")
        st.markdown("---")
        
        if not needs_review_queries:
            st.info("No queries need review.")
            st.markdown("""
            Queries appear here when users click 👎 **Not Helpful** on the app page.
            
            Review these to understand why the response wasn't satisfactory and improve the system.
            """)
        else:
            st.warning("These queries were marked as 'not helpful' by users. Review to understand what went wrong.")
            
            for idx, query in enumerate(needs_review_queries):
                with st.expander(f"👎 Query {idx + 1}: {query['query_text'][:80]}..."):
                    # Status badge
                    st.error("👎 **User marked this response as not helpful**")
                    
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Cypher query if available
                    if query.get('cypher_text'):
                        st.markdown("### 🔧 Generated Cypher")
                        st.code(query['cypher_text'], language="cypher")
                        
                        if query.get('summary'):
                            st.markdown("### 📊 Results (User found unhelpful)")
                            st.warning(query['summary'])
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button(
                            "✅ Approve Anyway",
                            key=f"approve_review_{idx}_{query['query_id']}",
                            type="primary",
                            use_container_width=True,
                            disabled=not query.get('cypher_text')
                        ):
                            try:
                                logger.info(f"User approving needs-review query: {query['query_id']}")
                                st.session_state.orchestrator.approve_query(
                                    query['query_id'],
                                    query['cypher_id']
                                )
                                logger.info(f"Needs-review query {query['query_id']} approved")
                                st.success("Query approved! It will now be used as a few-shot example.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to approve query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to approve: {str(e)}")
                    
                    with col2:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_review_{idx}_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                reason = "User reported as not helpful"
                                logger.info(f"User rejecting needs-review query: {query['query_id']}")
                                st.session_state.orchestrator.reject_query(
                                    query['query_id'],
                                    reason
                                )
                                logger.info(f"Needs-review query {query['query_id']} rejected")
                                st.warning("Query rejected.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to reject query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to reject: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab6:
        st.caption(f"ℹ️ {tab_definitions['needs_human_support']}")
        st.markdown("---")
        
        if not needs_human_support_queries:
            st.info("🎉 No queries need human support at the moment!")
            st.markdown("""
            Queries appear here when the **RefinementAgent** fails to consolidate 
            multi-step agentic queries into a single parameterized template.
            
            When this happens, you can:
            1. **Manually create** a parameterized Cypher template
            2. **Reject** the query if it's not worth templating
            3. **Re-try** refinement with different settings
            """)
        else:
            st.error("🛠️ These queries require manual intervention to create templates.")
            st.markdown("""
            The RefinementAgent could not automatically consolidate these multi-step queries 
            into single parameterized Cypher templates. Review and manually create templates.
            """)
            
            for idx, query in enumerate(needs_human_support_queries):
                with st.expander(
                    f"🛠️ Query {idx + 1}: {query['query_text'][:80]}...",
                    expanded=(idx == 0)
                ):
                    # Status badge
                    st.error("🛠️ **Refinement Failed - Needs Human Support**")
                    
                    # Show refinement error if available
                    if query.get('refinement_error'):
                        st.warning(f"**Refinement Error:** {query['refinement_error']}")
                    
                    # Query details
                    st.markdown("### 📝 Original Question")
                    st.info(query['query_text'])
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Multi-step queries that were executed
                    if query.get('result_data'):
                        try:
                            data = json.loads(query['result_data'])
                            queries_executed = data.get('queries_executed', [])
                            if queries_executed:
                                st.markdown("### 🔧 Multi-Step Queries Executed")
                                for i, q in enumerate(queries_executed):
                                    st.code(q, language="cypher")
                        except:
                            pass
                    
                    # Results
                    if query.get('summary'):
                        st.markdown("### 📊 Suggested Answer")
                        st.success(query['summary'])
                    
                    # Manual template creation section
                    st.markdown("---")
                    st.markdown("### ✏️ Manual Template Creation")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        manual_template = st.text_area(
                            "Question Template",
                            placeholder="What {attribute} did {person} hold?",
                            key=f"manual_template_{idx}_{query['query_id']}",
                            help="Parameterize with {param_name} syntax"
                        )
                        
                        manual_category = st.selectbox(
                            "Category",
                            options=[
                                "PERSON_ATTRIBUTE", "PERSON_ROLE", "WORK_CREATOR", 
                                "WORK_PARTICIPANT", "LOCATION_COMPARISON", "LOCATION_ATTRIBUTE",
                                "TEMPORAL", "RELATIONSHIP", "FACTUAL", "MULTI_HOP", "OTHER"
                            ],
                            key=f"manual_category_{idx}_{query['query_id']}"
                        )
                    
                    with col_b:
                        manual_cypher = st.text_area(
                            "Parameterized Cypher",
                            placeholder="MATCH (p:PERSON) WHERE p.name CONTAINS $person_name RETURN p.name, p.description",
                            key=f"manual_cypher_{idx}_{query['query_id']}",
                            help="Use $param_name for parameters"
                        )
                        
                        manual_params = st.text_input(
                            "Parameters (comma-separated)",
                            placeholder="person_name, attribute",
                            key=f"manual_params_{idx}_{query['query_id']}"
                        )
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                    
                    with col1:
                        if st.button(
                            "✅ Approve Answer",
                            key=f"approve_answer_{idx}_{query['query_id']}",
                            type="primary",
                            use_container_width=True,
                            help="Approve this answer without creating a template. Use when multi-step inference is good enough."
                        ):
                            try:
                                logger.info(f"Approving answer for query {query['query_id']} without template creation")
                                # Get cypher_id if available
                                cypher_id = query.get('cypher_id')
                                if cypher_id:
                                    # Use approve_query which marks as approved and creates FEW_SHOT_PROMPT relationship
                                    st.session_state.orchestrator.approve_query(
                                        query['query_id'],
                                        cypher_id
                                    )
                                else:
                                    # Fallback: update Query node status directly (not UserQuestion)
                                    with st.session_state.orchestrator.neo4j_client.driver.session() as session:
                                        session.run("""
                                            MATCH (q:Query {id: $query_id})
                                            SET q.status = 'approved'
                                        """, query_id=query['query_id'])
                                logger.info(f"Answer approved for query {query['query_id']}")
                                st.success("✅ Answer approved! (No template created - multi-step inference is sufficient)")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to approve answer: {str(e)}", exc_info=True)
                                st.error(f"Failed to approve: {str(e)}")
                    
                    with col2:
                        if st.button(
                            "💾 Save Template",
                            key=f"save_template_{idx}_{query['query_id']}",
                            type="primary",
                            use_container_width=True,
                            disabled=not (manual_template and manual_cypher)
                        ):
                            try:
                                # Parse parameters
                                params = [p.strip() for p in manual_params.split(',') if p.strip()]
                                
                                # Create template manually
                                logger.info(f"Manually creating FewShot for query {query['query_id']}")
                                
                                # Store FewShot linked to the existing Query node
                                few_shot_id = st.session_state.orchestrator.neo4j_client.store_few_shot_for_query(
                                    query_id=query['query_id'],
                                    cypher_template=manual_cypher,
                                    parameters=params,
                                    example_values={"category": manual_category} if manual_category else None
                                )
                                
                                # Update Query node status to approved (not UserQuestion)
                                cypher_id = query.get('cypher_id')
                                if cypher_id:
                                    # Use approve_query to create FEW_SHOT_PROMPT relationship
                                    st.session_state.orchestrator.approve_query(
                                        query['query_id'],
                                        cypher_id
                                    )
                                else:
                                    # Fallback: just update Query status if no cypher_id
                                    with st.session_state.orchestrator.neo4j_client.driver.session() as session:
                                        session.run("""
                                            MATCH (q:Query {id: $query_id})
                                            SET q.status = 'approved'
                                        """, query_id=query['query_id'])
                                
                                logger.info(f"Manual template created for query {query['query_id']}")
                                st.success("✅ Template created successfully!")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to create manual template: {str(e)}", exc_info=True)
                                st.error(f"Failed to save template: {str(e)}")
                    
                    with col3:
                        if st.button(
                            "🔄 Retry Refinement",
                            key=f"retry_refine_{idx}_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                logger.info(f"Retrying refinement for query {query['query_id']}")
                                with st.spinner("🔄 Retrying refinement..."):
                                    result = st.session_state.orchestrator.approve_query_with_refinement(
                                        query['query_id'],
                                        query['cypher_id'],
                                        trigger_refinement=True
                                    )
                                
                                if result.get('success'):
                                    st.success("✅ Refinement succeeded on retry!")
                                    st.rerun()
                                else:
                                    st.error(f"Refinement failed again: {result.get('error')}")
                            except Exception as e:
                                logger.error(f"Retry refinement failed: {str(e)}", exc_info=True)
                                st.error(f"Retry failed: {str(e)}")
                    
                    with col4:
                        if st.button(
                            "❌ Reject",
                            key=f"reject_support_{idx}_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                reason = "Rejected - could not create template"
                                logger.info(f"Rejecting needs-support query: {query['query_id']}")
                                st.session_state.orchestrator.reject_query(
                                    query['query_id'],
                                    reason
                                )
                                logger.info(f"Needs-support query {query['query_id']} rejected")
                                st.warning("Query rejected.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to reject query: {str(e)}", exc_info=True)
                                st.error(f"Failed to reject: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")

# Sidebar with guidance
with st.sidebar:
    st.header("Quick Navigation")
    st.page_link("app.py", label="🔍 Query Interface")
    st.page_link("pages/0_Setup.py", label="⚙️ Setup")
    st.page_link("pages/2_Test_Set.py", label="📝 Test Set")
    
    st.markdown("---")
    st.header("Training Guidelines")
    st.markdown("""
    ### Tab Definitions
    
    **⏳ Pending Approval** - AI generated results but no user feedback yet. Ready for curator review.
    
    **✅ Approved** - Queries that have been approved and are now being used as few-shot examples.
    
    **❌ Failed** - Queries that didn't produce results.
    
    **⚠️ Declined** - High-risk queries where AI also failed.
    
    **🚫 Rejected** - Previously rejected, can be re-approved.
    
    **🔍 Needs Review** - Users marked as 'not helpful'.
    
    **🛠️ Needs Human Support** - Refinement failed, manual template creation needed.
    """)
    
    # Development: Dev Tools
    st.markdown("---")
    st.subheader("🧪 Dev Tools")
    st.caption("Development/Testing only")
    
    # Delete All Rejected
    if st.button("🗑️ Delete All Rejected", key="delete_all_rejected", use_container_width=True):
        try:
            deleted = st.session_state.orchestrator.neo4j_client.delete_all_rejected_queries()
            st.success(f"✅ Deleted {deleted} rejected queries")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {str(e)}")
    
    # Delete All BOOTH Data
    st.markdown("---")
    confirm = st.checkbox("Confirm reset all", key="confirm_delete_booth")
    if st.button("🗑️ Reset All BOOTH Data", key="delete_all_booth", use_container_width=True, disabled=not confirm, type="secondary"):
        try:
            deleted = st.session_state.orchestrator.neo4j_client.delete_all_booth_data()
            total = sum(deleted.values())
            st.success(f"✅ Deleted {total} nodes")
            st.json(deleted)
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>BOOTH Retriever v0.1.0 | Human-in-the-Loop Training</small>
</div>
""", unsafe_allow_html=True)

