"""BOOTH Retriever - Main Query Interface"""

import streamlit as st
import os
from dotenv import load_dotenv

from src.booth_orchestrator import BOOTHOrchestrator
from src.logger import setup_logger

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger("booth.app")

# Page configuration
st.set_page_config(
    page_title="BOOTH Retriever",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

if 'orchestrator' not in st.session_state:
    try:
        logger.info("Initializing BOOTH Orchestrator")
        st.session_state.orchestrator = BOOTHOrchestrator()
        logger.info("BOOTH Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BOOTH: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize BOOTH: {str(e)}")
        st.stop()

if 'query_history' not in st.session_state:
    st.session_state.query_history = []
    logger.debug("Initialized query history")

if 'pending_feedback' not in st.session_state:
    st.session_state.pending_feedback = {}  # {query_id: response}
    logger.debug("Initialized pending feedback tracker")


def submit_feedback(query_id: str, is_helpful: bool):
    """Submit user feedback for a query."""
    # Check if orchestrator has the new feedback method
    if not hasattr(st.session_state.orchestrator, 'submit_user_feedback'):
        logger.warning("Orchestrator doesn't support user feedback - please restart the app")
        return False
    
    success = st.session_state.orchestrator.submit_user_feedback(query_id, is_helpful)
    if success:
        # Remove from pending feedback
        if query_id in st.session_state.pending_feedback:
            del st.session_state.pending_feedback[query_id]
        return True
    return False

# Header
st.title("🔍 BOOTH Retriever")
st.markdown("""
**(Bounded Orchestration Of Text Handling)**

**Examples:**

- **Easy:** What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
- **Hard:** Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
- **Approved:** Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?
""")

# Check if graph is built - show welcome message for new users
try:
    with st.session_state.orchestrator.neo4j_client.driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
        node_count = result.single()["count"]
        
        if node_count == 0:
            st.info("👋 **Welcome to BOOTH Retriever!** It looks like you're new here. Please complete the setup to get started.")
            st.page_link("pages/0_Setup.py", label="→ Go to Setup", icon="⚙️")
            st.markdown("---")
            st.markdown("### What is BOOTH?")
            st.markdown("""
            BOOTH (Bounded Orchestration Of Text Handling) is an intelligent knowledge graph retrieval system that:
            - Learns from your approved queries to provide faster, more accurate responses
            - Uses agentic reasoning to explore complex questions
            - Improves over time through human feedback
            
            **Get Started:**
            1. **Setup** - Connect your data (files, folders, or Neo4j database)
            2. **Train AI** - Review and approve queries to build the knowledge base
            3. **Query** - Ask questions and get intelligent answers
            """)
            st.stop()
except Exception as e:
    logger.warning(f"Could not check graph status: {e}")
    # Continue anyway - let the user try to query

# Sidebar with info
with st.sidebar:
    
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))
    st.metric("Similarity Threshold", f"{threshold*100:.0f}%")
    
    model = os.getenv("AGENTIC_TEXT2CYPHER_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"))
    st.metric("Agent Model", model)
    
    st.markdown("---")

# Main query interface

# Query input
user_query = st.text_area(
    "Enter your question:",
    placeholder="e.g., How many users are in the system?",
    height=100
)

# Risk assessment checkbox
col1, col2 = st.columns([3, 1])

with col1:
    is_high_risk = st.checkbox(
        "⚠️ High-risk query",
        value=True,
        help="Check this if the query might contain sensitive operations or inappropriate content. High-risk queries will be declined to prevent execution, but the system will still attempt to generate and execute a Cypher query in the background for review purposes. Review results in the Train AI page."
    )

with col2:
    submit_button = st.button("Submit Query", type="primary", use_container_width=True)

# Process query
if submit_button and user_query.strip():
    logger.info(f"User submitted query: '{user_query.strip()[:100]}...' (high_risk={is_high_risk})")
    with st.spinner("Processing your query..."):
        try:
            # Process through BOOTH
            response = st.session_state.orchestrator.process_query(
                user_query=user_query.strip(),
                is_high_risk=is_high_risk
            )
            
            # Add to history
            st.session_state.query_history.append({
                'query': user_query.strip(),
                'response': response
            })
            logger.debug(f"Added query to history (total: {len(st.session_state.query_history)})")
            
            # Display response
            st.markdown("---")
            
            if response.declined:
                logger.info(f"Query declined (query_id: {response.query_id})")
                st.error("🚫 Query Declined")
                st.warning(response.answer)
                
                if response.query_id:
                    st.info(f"Query ID: `{response.query_id}` - Review in Train AI page to see what the system attempted")
            elif response.success:
                # Use getattr for backward compatibility with old response objects
                tool_used = getattr(response, 'tool_used', None)
                logger.info(f"Query successful (query_id: {response.query_id}, similar_match: {response.similar_match}, tool_used: {tool_used})")
                st.success("✅ Query Successful")
                
                # Show if similarity match was used
                if response.similar_match:
                    st.info("💡 Found similar query in database - using optimized approach")
                
                # Show which tool was used
                if tool_used:
                    tool_display = {
                        "hybrid_retriever": "🔍 Hybrid Retriever (vector + keyword search)",
                        "text2cypher": "⚡ Text2Cypher (natural language to Cypher)",
                        "agentic_text2cypher": "🤖 Agentic Text2Cypher (Deep Agent)"
                    }.get(tool_used, f"🔧 {tool_used}")
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(response.answer)
                
                # Show details in expander
                with st.expander("📊 View Details"):
                    if response.cypher_used:
                        st.markdown("**Generated Cypher Query:**")
                        st.code(response.cypher_used, language="cypher")
                    
                    if response.raw_data:
                        st.markdown("**Raw Data:**")
                        st.json(response.raw_data)
                    
                    st.markdown(f"**Query ID:** `{response.query_id}`")
                    st.info("This query is pending approval. Go to the Train AI page to review it.")
                
                # User feedback section (for low-risk queries)
                pending_feedback = getattr(response, 'pending_feedback', False)
                if pending_feedback and response.query_id:
                    st.markdown("---")
                    st.markdown("### 📝 Was this response accurate?")
                    st.caption("Your feedback helps improve BOOTH's responses and trains the system.")
                    
                    # Store response for feedback tracking
                    st.session_state.pending_feedback[response.query_id] = response
                    
                    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 2])
                    
                    with feedback_col1:
                        if st.button("👍 Helpful", key=f"helpful_{response.query_id}", type="primary"):
                            if submit_feedback(response.query_id, True):
                                st.success("Thank you! This query has been marked for final approval.")
                                st.rerun()
                            else:
                                st.error("Failed to record feedback. Please try again.")
                    
                    with feedback_col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{response.query_id}"):
                            if submit_feedback(response.query_id, False):
                                st.info("Thank you for your feedback. This will help us improve.")
                                st.rerun()
                            else:
                                st.error("Failed to record feedback. Please try again.")
            else:
                logger.warning(f"Query failed (query_id: {response.query_id}): {response.error_message}")
                st.error("❌ Query Failed")
                st.error(response.answer)
                
                if response.error_message:
                    with st.expander("View Error Details"):
                        st.code(response.error_message)
                
                if response.query_id:
                    st.info(f"Query ID: `{response.query_id}` - Logged for review")
        
        except Exception as e:
            logger.error(f"Unexpected error during query processing: {str(e)}", exc_info=True)
            st.error(f"An unexpected error occurred: {str(e)}")

elif submit_button:
    logger.debug("Submit button clicked but no query entered")
    st.warning("Please enter a question.")

# Query history
if st.session_state.query_history:
    st.markdown("---")
    st.header("Query History")
    
    for idx, item in enumerate(reversed(st.session_state.query_history[-5:])):
        with st.expander(f"Query {len(st.session_state.query_history) - idx}: {item['query'][:50]}..."):
            st.markdown(f"**Question:** {item['query']}")
            
            response = item['response']
            if response.success:
                st.success(response.answer)
            elif response.declined:
                st.warning(response.answer)
            else:
                st.error(response.answer)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>BOOTH Retriever v0.3.0 | Powered by OpenAI, Neo4j & Deep Agents</small>
</div>
""", unsafe_allow_html=True)

