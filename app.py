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

# Header
st.title("🔍 BOOTH Retriever")
st.markdown("""
**Bounded Orchestration Of Text Handling**

Ask questions about your Neo4j database in natural language.

**Example Questions:**

- **Easy:** What is NovaGrid Energy Corporation's annual revenue and how many employees does the company have?
- **Hard:** Across all three RFPs, list the top five digital capabilities the companies demand and explain why those capabilities matter to their industries.
""")

# Sidebar with info
with st.sidebar:
    st.header("About BOOTH")
    st.markdown("""
    BOOTH intelligently handles your queries:
    
    1. **Similarity Check**: Searches for similar past queries (>90% match)
    2. **Risk Assessment**: You mark queries as safe or high-risk
       - High-risk: Results hidden from user, but full pipeline runs for review
    3. **Text-to-Cypher**: Converts natural language to Cypher queries
    4. **Human Curation**: All queries (including declined) can be reviewed
    
    ---
    
    ### Configuration
    """)
    
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))
    st.metric("Similarity Threshold", f"{threshold*100:.0f}%")
    
    max_retries = int(os.getenv("MAX_CYPHER_RETRIES", "3"))
    st.metric("Max Cypher Retries", max_retries)
    
    st.markdown("---")
    st.markdown("Go to **Train AI** page to approve pending queries.")

# Main query interface
st.header("Ask a Question")

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
                logger.info(f"Query successful (query_id: {response.query_id}, similar_match: {response.similar_match})")
                st.success("✅ Query Successful")
                
                # Show if similarity match was used
                if response.similar_match:
                    st.info("💡 Found similar query in database - using optimized approach")
                
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
    <small>BOOTH Retriever v0.1.0 | Powered by OpenAI & Neo4j</small>
</div>
""", unsafe_allow_html=True)

