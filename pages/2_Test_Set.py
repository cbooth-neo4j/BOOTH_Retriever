"""BOOTH Retriever - Test Set Generation & Management

This page allows users to:
1. Upload existing test sets (CSV)
2. Generate test questions from the graph
3. Review and edit test questions
4. Run the agentic retriever to generate answers
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import csv
from io import StringIO

from src.logger import setup_logger
from src.neo4j_client import Neo4jClient
from src.llm_client import LLMClient
from src.booth_orchestrator import BOOTHOrchestrator
from src.onboarding_utils import (
    generate_test_questions,
    parse_test_set_csv,
    save_test_set_to_csv
)

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger("booth.testset")

# Page configuration
st.set_page_config(
    page_title="BOOTH Test Set",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'neo4j_client' not in st.session_state:
    st.session_state.neo4j_client = None
if 'test_questions' not in st.session_state:
    st.session_state.test_questions = []
if 'generated_answers' not in st.session_state:
    st.session_state.generated_answers = {}

# Initialize Neo4j client
try:
    if st.session_state.neo4j_client is None:
        st.session_state.neo4j_client = Neo4jClient()
except Exception as e:
    logger.error(f"Failed to initialize Neo4j client: {e}")
    st.error(f"Failed to connect to Neo4j: {str(e)}")
    st.stop()

# Header
st.title("📝 Test Set Management")
st.markdown("""
Create or upload test questions to evaluate BOOTH's performance.
""")

# Check if graph is built
try:
    with st.session_state.neo4j_client.driver.session() as session:
        result = session.run("MATCH (e:__Entity__) RETURN count(e) as count LIMIT 1")
        entity_count = result.single()["count"]
        
        if entity_count == 0:
            st.warning("⚠️ No entities found in your graph. Please complete setup first.")
            st.page_link("pages/0_Setup.py", label="Go to Setup →")
            st.stop()
except Exception as e:
    logger.warning(f"Could not check entity count: {e}")

st.markdown("---")

# Tabs for different operations
tab1, tab2, tab3 = st.tabs(["📤 Upload Test Set", "🤖 Generate Test Set", "📊 View Test Sets"])


# ============================
# TAB 1: UPLOAD TEST SET
# ============================
with tab1:
    st.header("Upload Existing Test Set")
    st.info("Upload a CSV file with columns: `question` and `expected_answer`")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="test_csv_upload"
    )
    
    if uploaded_file:
        questions, errors = parse_test_set_csv(uploaded_file)
        
        if errors:
            st.warning(f"⚠️ Found {len(errors)} error(s):")
            for error in errors[:10]:  # Show first 10 errors
                st.write(f"- {error}")
        
        if questions:
            st.success(f"✓ Parsed {len(questions)} questions")
            
            # Preview
            st.subheader("Preview")
            preview_df = pd.DataFrame(questions[:10])  # Show first 10
            st.dataframe(preview_df[['question', 'expected_answer']], use_container_width=True)
            
            if len(questions) > 10:
                st.caption(f"Showing first 10 of {len(questions)} questions")
            
            # Store button
            if st.button("💾 Store Test Set in Neo4j", type="primary", use_container_width=True):
                with st.spinner("Storing test questions..."):
                    try:
                        stored_count = 0
                        for q in questions:
                            st.session_state.neo4j_client.store_test_question(
                                question=q['question'],
                                expected_answer=q['expected_answer'],
                                metadata={
                                    'source': 'uploaded_csv',
                                    'uploaded_at': datetime.now().isoformat()
                                }
                            )
                            stored_count += 1
                        
                        st.success(f"✓ Stored {stored_count} test questions in Neo4j!")
                        logger.info(f"Stored {stored_count} test questions from CSV upload")
                        st.balloons()
                        
                    except Exception as e:
                        logger.error(f"Error storing test questions: {e}")
                        st.error(f"Error: {str(e)}")


# ============================
# TAB 2: GENERATE TEST SET
# ============================
with tab2:
    st.header("Generate Test Set from Graph")
    st.info("Automatically generate test questions based on entities in your knowledge graph")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        n_questions = st.number_input(
            "Number of questions to generate",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
    
    with col2:
        generate_answers = st.checkbox(
            "Generate answers using BOOTH Agent",
            value=True,
            help="Use the agentic retriever to generate first-draft answers"
        )
    
    if st.button("🤖 Generate Questions", type="primary", use_container_width=True):
        with st.spinner(f"Generating {n_questions} test questions..."):
            try:
                llm_client = LLMClient()
                
                # Generate questions
                questions = generate_test_questions(
                    neo4j_client=st.session_state.neo4j_client,
                    llm_client=llm_client,
                    n_questions=n_questions
                )
                
                if not questions:
                    st.error("Failed to generate questions. Please check that your graph has entities.")
                else:
                    st.session_state.test_questions = questions
                    st.success(f"✓ Generated {len(questions)} questions")
                    
                    # Generate answers if requested
                    if generate_answers:
                        st.info("Generating answers using BOOTH Agent...")
                        progress_bar = st.progress(0)
                        
                        try:
                            orchestrator = BOOTHOrchestrator()
                            
                            for idx, q in enumerate(questions):
                                try:
                                    # Run the agent
                                    response = orchestrator.process_query(
                                        user_query=q['question'],
                                        is_high_risk=False
                                    )
                                    
                                    if response.success:
                                        st.session_state.generated_answers[q['question']] = response.answer
                                    else:
                                        st.session_state.generated_answers[q['question']] = "[Failed to generate]"
                                    
                                    progress_bar.progress((idx + 1) / len(questions))
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to generate answer for '{q['question']}': {e}")
                                    st.session_state.generated_answers[q['question']] = "[Error]"
                            
                            st.success("✓ Answers generated!")
                            
                        except Exception as e:
                            logger.error(f"Error generating answers: {e}")
                            st.error(f"Error generating answers: {str(e)}")
                    
                    st.rerun()
                    
            except Exception as e:
                logger.error(f"Error generating questions: {e}")
                st.error(f"Error: {str(e)}")
    
    # Display generated questions
    if st.session_state.test_questions:
        st.markdown("---")
        st.subheader(f"Generated Questions ({len(st.session_state.test_questions)})")
        
        # Create editable dataframe
        questions_data = []
        for q in st.session_state.test_questions:
            questions_data.append({
                'Question': q['question'],
                'Expected Answer': q.get('answer', ''),
                'Generated Answer': st.session_state.generated_answers.get(q['question'], ''),
                'Entity': q.get('entity', ''),
                'Type': q.get('entity_type', '')
            })
        
        df = pd.DataFrame(questions_data)
        
        # Use data editor
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Expected Answer": st.column_config.TextColumn("Expected Answer", width="large"),
                "Generated Answer": st.column_config.TextColumn("Generated Answer (Read-only)", disabled=True, width="large"),
                "Entity": st.column_config.TextColumn("Entity", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small")
            }
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Store in Neo4j", type="primary", use_container_width=True):
                with st.spinner("Storing questions..."):
                    try:
                        stored_count = 0
                        for _, row in edited_df.iterrows():
                            st.session_state.neo4j_client.store_test_question(
                                question=row['Question'],
                                expected_answer=row['Expected Answer'],
                                metadata={
                                    'entity': row.get('Entity', ''),
                                    'entity_type': row.get('Type', ''),
                                    'generated_answer': row.get('Generated Answer', ''),
                                    'source': 'generated',
                                    'generated_at': datetime.now().isoformat()
                                }
                            )
                            stored_count += 1
                        
                        st.success(f"✓ Stored {stored_count} questions in Neo4j!")
                        logger.info(f"Stored {stored_count} generated test questions")
                        
                    except Exception as e:
                        logger.error(f"Error storing questions: {e}")
                        st.error(f"Error: {str(e)}")
        
        with col2:
            # Export to CSV
            csv_buffer = StringIO()
            export_df = edited_df[['Question', 'Expected Answer']].copy()
            export_df.columns = ['question', 'expected_answer']
            export_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"test_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.test_questions = []
                st.session_state.generated_answers = {}
                st.rerun()


# ============================
# TAB 3: VIEW TEST SETS
# ============================
with tab3:
    st.header("Stored Test Questions")
    
    # Refresh button
    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()
    
    try:
        # Retrieve stored questions
        stored_questions = st.session_state.neo4j_client.get_test_questions(limit=100)
        
        if not stored_questions:
            st.info("No test questions stored yet. Upload or generate questions in the other tabs.")
        else:
            st.success(f"✓ Found {len(stored_questions)} stored test questions")
            
            # Convert to dataframe
            questions_data = []
            for q in stored_questions:
                metadata = q.get('metadata', {})
                questions_data.append({
                    'Question': q['question'],
                    'Expected Answer': q['expected_answer'],
                    'Source': metadata.get('source', 'unknown'),
                    'Entity': metadata.get('entity', ''),
                    'Created': q.get('created_at', ''),
                    'ID': q['id']
                })
            
            df = pd.DataFrame(questions_data)
            
            # Display with filters
            col1, col2 = st.columns(2)
            with col1:
                source_filter = st.multiselect(
                    "Filter by source",
                    options=df['Source'].unique(),
                    default=list(df['Source'].unique())
                )
            
            # Apply filters
            filtered_df = df[df['Source'].isin(source_filter)]
            
            # Display
            st.dataframe(
                filtered_df[['Question', 'Expected Answer', 'Source', 'Entity', 'Created']],
                use_container_width=True,
                hide_index=True
            )
            
            # Export all
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                # Export to CSV
                csv_buffer = StringIO()
                export_df = filtered_df[['Question', 'Expected Answer']].copy()
                export_df.columns = ['question', 'expected_answer']
                export_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Export Filtered to CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"test_set_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🗑️ Delete All Test Questions", type="secondary", use_container_width=True):
                    if st.checkbox("Confirm deletion", key="confirm_delete_tests"):
                        try:
                            count = st.session_state.neo4j_client.delete_test_questions()
                            st.success(f"✓ Deleted {count} test questions")
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error deleting test questions: {e}")
                            st.error(f"Error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error retrieving test questions: {e}")
        st.error(f"Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>BOOTH Test Set Management | Use these questions to benchmark your system</small>
</div>
""", unsafe_allow_html=True)

