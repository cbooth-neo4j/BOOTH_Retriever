"""BOOTH Retriever - Query Curation Interface"""

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

# Display statistics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    pending = sum(1 for q in pending_queries if q.get('status') == 'pending_approval')
    st.metric("Pending Approval", pending)
with col2:
    declined = sum(1 for q in pending_queries if q.get('status') == 'declined')
    st.metric("Declined (High-Risk)", declined)
with col3:
    rejected = sum(1 for q in pending_queries if q.get('status') == 'rejected')
    st.metric("Rejected", rejected)
with col4:
    successful = sum(1 for q in pending_queries if q.get('summary'))
    st.metric("Successful Queries", successful)

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
    st.header(f"Queries for Review ({len(pending_queries)})")
    
    # Group queries by status and success
    pending_successful = [q for q in pending_queries if q.get('status') == 'pending_approval' and q.get('summary')]
    pending_failed = [q for q in pending_queries if q.get('status') == 'pending_approval' and not q.get('summary')]
    declined_queries = [q for q in pending_queries if q.get('status') == 'declined']
    rejected_queries = [q for q in pending_queries if q.get('status') == 'rejected']
    
    # Tabs for different statuses
    tab1, tab2, tab3, tab4 = st.tabs([
        f"✅ Successful ({len(pending_successful)})", 
        f"❌ Failed ({len(pending_failed)})",
        f"⚠️ Declined/High-Risk ({len(declined_queries)})",
        f"🚫 Rejected ({len(rejected_queries)})"
    ])
    
    with tab1:
        if not pending_successful:
            st.info("No successful queries pending approval.")
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
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Summary:**")
                        st.success(query['summary'])
                    
                    with col2:
                        if query.get('result_data'):
                            with st.expander("View Raw Data"):
                                try:
                                    data = json.loads(query['result_data'])
                                    st.json(data)
                                except:
                                    st.text(query['result_data'])
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button(
                            "✅ Approve",
                            key=f"approve_{query['query_id']}",
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
                            key=f"reject_{query['query_id']}",
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
    
    with tab2:
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
                            key=f"reject_failed_{query['query_id']}",
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
        if not declined_queries:
            st.info("No declined queries to review.")
        else:
            st.warning("These queries were marked as high-risk by users and automatically declined. The system ran the full text2cypher pipeline in the background so you can see what would have happened.")
            
            for idx, query in enumerate(declined_queries):
                with st.expander(f"Query {idx + 1}: {query['query_text'][:80]}..."):
                    # Query details
                    st.markdown("### 📝 Query")
                    st.info(query['query_text'])
                    
                    # Status info
                    st.markdown("### ⚠️ Status")
                    risk_level = query.get('risk_level', 'unknown')
                    st.error(f"**Declined** - Risk Level: {risk_level.upper()}")
                    st.caption("Results were NOT shown to user, but pipeline ran in background for review.")
                    
                    # Timestamp
                    timestamp = query.get('timestamp', 'N/A')
                    st.caption(f"Submitted: {timestamp}")
                    
                    # Check if there's a Cypher attempt
                    if query.get('cypher_text'):
                        st.markdown("### 🔧 Generated Cypher Attempt")
                        st.code(query['cypher_text'], language="cypher")
                        
                        if query.get('summary'):
                            st.markdown("### 📊 Results (Hidden from User)")
                            st.success(query['summary'])
                            st.caption("✓ The query would have succeeded if executed")
                        else:
                            st.markdown("### ⚠️ Execution Status")
                            st.warning("The text2cypher pipeline did not produce successful results.")
                    else:
                        st.info("No Cypher was generated - the pipeline may have failed early.")
                    
                    # Action buttons
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button(
                            "✅ Approve",
                            key=f"approve_declined_{query['query_id']}",
                            type="primary",
                            use_container_width=True,
                            disabled=not query.get('cypher_text')
                        ):
                            try:
                                logger.info(f"User approving declined query: {query['query_id']}")
                                st.session_state.orchestrator.approve_query(
                                    query['query_id'],
                                    query['cypher_id']
                                )
                                logger.info(f"Declined query {query['query_id']} approved successfully")
                                st.success("Query approved! It will now be used as a few-shot example.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to approve query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to approve: {str(e)}")
                    
                    with col2:
                        if st.button(
                            "❌ Keep Rejected",
                            key=f"keep_rejected_declined_{query['query_id']}",
                            use_container_width=True
                        ):
                            try:
                                reason = "Confirmed rejection - was correctly declined"
                                logger.info(f"User keeping declined query rejected: {query['query_id']}")
                                st.session_state.orchestrator.reject_query(
                                    query['query_id'],
                                    reason
                                )
                                logger.info(f"Declined query {query['query_id']} kept as rejected")
                                st.warning("Query remains rejected.")
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Failed to reject query {query['query_id']}: {str(e)}", exc_info=True)
                                st.error(f"Failed to reject: {str(e)}")
                    
                    st.markdown(f"**Query ID:** `{query['query_id']}`")
    
    with tab4:
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
                            key=f"reapprove_{query['query_id']}",
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

# Sidebar with guidance
with st.sidebar:
    st.header("Training Guidelines")
    st.markdown("""
    ### When to Approve ✅
    - Query and Cypher are correct
    - Results are accurate
    - Good example for future use
    - Even declined queries can be approved if safe
    
    ### When to Reject ❌
    - Incorrect Cypher query
    - Wrong or misleading results
    - Security concerns
    - Not a good example
    
    ---
    
    ### About Declined Queries ⚠️
    
    High-risk queries are declined to users, but the system runs the full text2cypher pipeline in the background. Review them to see if they were actually safe.
    
    ---
    
    ### What Happens After Approval?
    
    1. Query becomes a **few-shot example**
    2. Added to vector database
    3. Similar future queries will reference it
    4. Improves system accuracy over time
    
    ---
    
    **Tip:** Approve high-quality queries to build a robust knowledge base!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>BOOTH Retriever v0.1.0 | Human-in-the-Loop Training</small>
</div>
""", unsafe_allow_html=True)

