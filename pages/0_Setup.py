"""BOOTH Retriever - Setup & Onboarding Interface

This page guides users through:
1. Data source selection (upload files, folder, or Neo4j connection)
2. Entity type discovery and approval
3. Individual entity review
4. Graph building
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from src.logger import setup_logger
from src.neo4j_client import Neo4jClient
from src.llm_client import LLMClient
from src.onboarding_utils import (
    extract_sample_text_from_files,
    extract_entities_for_review,
    validate_neo4j_connection,
    update_env_file
)
from build_graph.main_processor import CustomGraphProcessor

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger("booth.setup")

# Page configuration
st.set_page_config(
    page_title="BOOTH Setup",
    page_icon="⚙️",
    layout="wide"
)

# Initialize session state
if 'setup_step' not in st.session_state:
    st.session_state.setup_step = 1
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'folder_path' not in st.session_state:
    st.session_state.folder_path = ""
if 'entity_types' not in st.session_state:
    st.session_state.entity_types = []
if 'approved_entities' not in st.session_state:
    st.session_state.approved_entities = {}
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False
if 'neo4j_uri' not in st.session_state:
    st.session_state.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
if 'neo4j_user' not in st.session_state:
    st.session_state.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
if 'neo4j_password' not in st.session_state:
    st.session_state.neo4j_password = os.getenv("NEO4J_PASSWORD", "")

# Header
st.title("⚙️ BOOTH Setup & Onboarding")
st.markdown("""
Welcome to BOOTH Retriever! Let's get your knowledge graph set up.
""")

# Progress indicator
progress_steps = ["Data Source", "Entity Types", "Review Entities", "Build Graph"]
current_step = st.session_state.setup_step
cols = st.columns(len(progress_steps))
for idx, (col, step_name) in enumerate(zip(cols, progress_steps), start=1):
    with col:
        if idx < current_step:
            st.success(f"✓ {step_name}")
        elif idx == current_step:
            st.info(f"→ {step_name}")
        else:
            st.text(f"  {step_name}")

st.markdown("---")


# ============================
# STEP 1: DATA SOURCE SELECTION
# ============================
if st.session_state.setup_step == 1:
    st.header("Step 1: Choose Your Data Source")
    
    data_source = st.radio(
        "How would you like to provide your data?",
        options=["upload", "folder", "neo4j"],
        format_func=lambda x: {
            "upload": "📁 Upload Files (PDF, TXT)",
            "folder": "📂 Select Local Folder",
            "neo4j": "🔗 Connect to Existing Neo4j Database"
        }[x],
        key="data_source_radio"
    )
    
    st.session_state.data_source = data_source
    
    # Option A: Upload Files
    if data_source == "upload":
        st.subheader("Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Drag and drop PDF or TXT files here",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"✓ {len(uploaded_files)} file(s) uploaded")
            
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("Continue with These Files →", type="primary", use_container_width=True):
                logger.info(f"User selected {len(uploaded_files)} files for upload")
                st.session_state.setup_step = 2
                st.rerun()
    
    # Option B: Local Folder
    elif data_source == "folder":
        st.subheader("Select Local Folder")
        folder_path = st.text_input(
            "Enter the full path to your folder containing PDF/TXT files:",
            value=st.session_state.folder_path,
            placeholder="C:\\Users\\YourName\\Documents\\data"
        )
        
        if folder_path:
            folder = Path(folder_path)
            if folder.exists() and folder.is_dir():
                # Count files
                pdf_files = list(folder.glob("*.pdf"))
                txt_files = list(folder.glob("*.txt"))
                total_files = len(pdf_files) + len(txt_files)
                
                if total_files > 0:
                    st.success(f"✓ Found {total_files} files ({len(pdf_files)} PDF, {len(txt_files)} TXT)")
                    st.session_state.folder_path = folder_path
                    
                    if st.button("Continue with This Folder →", type="primary", use_container_width=True):
                        logger.info(f"User selected folder: {folder_path} ({total_files} files)")
                        st.session_state.setup_step = 2
                        st.rerun()
                else:
                    st.warning("⚠️ No PDF or TXT files found in this folder")
            else:
                st.error("❌ Folder path does not exist or is not a directory")
    
    # Option C: Neo4j Connection
    elif data_source == "neo4j":
        st.subheader("Connect to Existing Neo4j Database")
        
        with st.form("neo4j_connection"):
            neo4j_uri = st.text_input(
                "Neo4j URI",
                value=st.session_state.neo4j_uri,
                placeholder="bolt://localhost:7687"
            )
            
            neo4j_user = st.text_input(
                "Username",
                value=st.session_state.neo4j_user,
                placeholder="neo4j"
            )
            
            neo4j_password = st.text_input(
                "Password",
                value=st.session_state.neo4j_password,
                type="password",
                placeholder="Enter password"
            )
            
            update_env = st.checkbox(
                "Save these credentials to .env file",
                value=False,
                help="This will update your .env file with the Neo4j connection details"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                test_button = st.form_submit_button("🔍 Test Connection", use_container_width=True)
            with col2:
                continue_button = st.form_submit_button("Continue →", type="primary", use_container_width=True)
        
        if test_button:
            with st.spinner("Testing connection..."):
                success, error = validate_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
                
                if success:
                    st.success("✓ Connection successful!")
                    st.session_state.neo4j_uri = neo4j_uri
                    st.session_state.neo4j_user = neo4j_user
                    st.session_state.neo4j_password = neo4j_password
                else:
                    st.error(f"❌ Connection failed: {error}")
        
        if continue_button:
            with st.spinner("Validating connection..."):
                success, error = validate_neo4j_connection(neo4j_uri, neo4j_user, neo4j_password)
                
                if success:
                    st.session_state.neo4j_uri = neo4j_uri
                    st.session_state.neo4j_user = neo4j_user
                    st.session_state.neo4j_password = neo4j_password
                    
                    # Update .env if requested
                    if update_env:
                        update_env_file("NEO4J_URI", neo4j_uri)
                        update_env_file("NEO4J_USER", neo4j_user)
                        update_env_file("NEO4J_PASSWORD", neo4j_password)
                        st.success("✓ Credentials saved to .env file")
                    
                    logger.info(f"User connected to Neo4j at {neo4j_uri}")
                    # Skip to step 4 (build graph) since we're using existing data
                    st.session_state.setup_step = 4
                    st.session_state.graph_built = True  # Assume graph already exists
                    st.rerun()
                else:
                    st.error(f"❌ Connection failed: {error}")


# ============================
# STEP 2: ENTITY TYPE DISCOVERY
# ============================
elif st.session_state.setup_step == 2:
    st.header("Step 2: Discover & Approve Entity Types")
    
    st.info("We'll analyze your documents to suggest entity types to extract (e.g., PERSON, ORGANIZATION, LOCATION)")
    
    if not st.session_state.entity_types:
        if st.button("🔍 Discover Entity Types", type="primary", use_container_width=True):
            with st.spinner("Analyzing documents..."):
                try:
                    # Extract sample text
                    if st.session_state.data_source == "upload":
                        sample_text = extract_sample_text_from_files(st.session_state.uploaded_files)
                    else:  # folder
                        folder = Path(st.session_state.folder_path)
                        # Read sample from first few files
                        sample_files = list(folder.glob("*.pdf"))[:3] + list(folder.glob("*.txt"))[:3]
                        sample_texts = []
                        for file_path in sample_files[:5]:
                            if file_path.suffix == '.txt':
                                sample_texts.append(file_path.read_text(encoding='utf-8', errors='ignore')[:3000])
                            elif file_path.suffix == '.pdf':
                                try:
                                    import PyPDF2
                                    with open(file_path, 'rb') as f:
                                        pdf = PyPDF2.PdfReader(f)
                                        text = ""
                                        for page_num in range(min(2, len(pdf.pages))):
                                            text += pdf.pages[page_num].extract_text()
                                        sample_texts.append(text[:3000])
                                except:
                                    pass
                        sample_text = "\n\n".join(sample_texts)
                    
                    # Use CustomGraphProcessor to discover labels
                    processor = CustomGraphProcessor()
                    discovered_labels = processor.discover_labels_for_text(sample_text)
                    processor.close()
                    
                    if discovered_labels:
                        st.session_state.entity_types = discovered_labels
                        logger.info(f"Discovered entity types: {discovered_labels}")
                        st.rerun()
                    else:
                        st.warning("No entity types discovered. Using defaults.")
                        st.session_state.entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"]
                        st.rerun()
                        
                except Exception as e:
                    logger.error(f"Error discovering entity types: {e}")
                    st.error(f"Error: {str(e)}")
    else:
        st.success("✓ Entity types discovered")
        
        # Display and allow editing
        st.subheader("Review Entity Types")
        st.markdown("Select the entity types you want to extract from your documents:")
        
        # Multiselect for existing types
        selected_types = st.multiselect(
            "Entity Types",
            options=st.session_state.entity_types,
            default=st.session_state.entity_types,
            help="Deselect any types you don't want to extract"
        )
        
        # Add custom types
        st.markdown("**Add Custom Types:**")
        custom_type = st.text_input(
            "Add a custom entity type",
            placeholder="e.g., PRODUCT, EVENT, TECHNOLOGY",
            key="custom_entity_type"
        )
        
        if st.button("➕ Add Custom Type"):
            if custom_type and custom_type.upper() not in selected_types:
                selected_types.append(custom_type.upper())
                st.session_state.entity_types = selected_types
                st.success(f"Added {custom_type.upper()}")
                st.rerun()
        
        # Update session state
        st.session_state.entity_types = selected_types
        
        # Show selected types
        st.write(f"**Selected Types ({len(selected_types)}):** {', '.join(selected_types)}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.setup_step = 1
                st.rerun()
        with col2:
            if st.button("Continue →", type="primary", use_container_width=True, disabled=len(selected_types) == 0):
                logger.info(f"User approved entity types: {selected_types}")
                st.session_state.setup_step = 3
                st.rerun()


# ============================
# STEP 3: REVIEW INDIVIDUAL ENTITIES
# ============================
elif st.session_state.setup_step == 3:
    st.header("Step 3: Review Individual Entities (Optional)")
    
    st.info("Review and edit specific entities that were detected. This step is optional but helps improve accuracy.")
    
    if not st.session_state.approved_entities:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔍 Extract & Review Entities", type="primary", use_container_width=True):
                with st.spinner("Extracting entities for review..."):
                    try:
                        # Extract sample text
                        if st.session_state.data_source == "upload":
                            sample_text = extract_sample_text_from_files(st.session_state.uploaded_files, max_chars=5000)
                        else:  # folder
                            folder = Path(st.session_state.folder_path)
                            sample_files = list(folder.glob("*.pdf"))[:2] + list(folder.glob("*.txt"))[:2]
                            sample_texts = []
                            for file_path in sample_files:
                                if file_path.suffix == '.txt':
                                    sample_texts.append(file_path.read_text(encoding='utf-8', errors='ignore')[:2500])
                                elif file_path.suffix == '.pdf':
                                    try:
                                        import PyPDF2
                                        with open(file_path, 'rb') as f:
                                            pdf = PyPDF2.PdfReader(f)
                                            text = pdf.pages[0].extract_text()
                                            sample_texts.append(text[:2500])
                                    except:
                                        pass
                            sample_text = "\n\n".join(sample_texts)
                        
                        # Extract entities
                        llm_client = LLMClient()
                        entities = extract_entities_for_review(
                            sample_text,
                            st.session_state.entity_types,
                            llm_client,
                            max_entities_per_type=20
                        )
                        
                        st.session_state.approved_entities = entities
                        logger.info(f"Extracted entities for review: {sum(len(v) for v in entities.values())} total")
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Error extracting entities: {e}")
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Skip This Step →", use_container_width=True):
                logger.info("User skipped entity review")
                st.session_state.setup_step = 4
                st.rerun()
    
    else:
        # Display entities by type
        st.success(f"✓ Extracted {sum(len(v) for v in st.session_state.approved_entities.values())} entities")
        
        for entity_type in st.session_state.entity_types:
            entities = st.session_state.approved_entities.get(entity_type, [])
            
            if entities:
                with st.expander(f"**{entity_type}** ({len(entities)} entities)", expanded=False):
                    for idx, entity in enumerate(entities):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{entity['name']}**")
                            st.caption(entity.get('context', ''))
                        with col2:
                            if st.button("🗑️ Remove", key=f"remove_{entity_type}_{idx}"):
                                st.session_state.approved_entities[entity_type].pop(idx)
                                st.rerun()
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.setup_step = 2
                st.rerun()
        with col2:
            if st.button("Continue to Build Graph →", type="primary", use_container_width=True):
                logger.info("User approved entities, moving to graph building")
                st.session_state.setup_step = 4
                st.rerun()


# ============================
# STEP 4: BUILD GRAPH
# ============================
elif st.session_state.setup_step == 4:
    st.header("Step 4: Build Knowledge Graph")
    
    if not st.session_state.graph_built:
        st.info("Ready to build your knowledge graph. This may take several minutes depending on the size of your data.")
        
        # Show summary
        st.subheader("Configuration Summary")
        st.write(f"**Data Source:** {st.session_state.data_source}")
        
        if st.session_state.data_source == "upload":
            st.write(f"**Files:** {len(st.session_state.uploaded_files)}")
        elif st.session_state.data_source == "folder":
            st.write(f"**Folder:** {st.session_state.folder_path}")
        elif st.session_state.data_source == "neo4j":
            st.write(f"**Neo4j URI:** {st.session_state.neo4j_uri}")
        
        if st.session_state.entity_types:
            st.write(f"**Entity Types:** {', '.join(st.session_state.entity_types)}")
        
        if st.button("🚀 Build Graph", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if st.session_state.data_source == "neo4j":
                    # Just validate the connection
                    status_text.text("Validating Neo4j connection...")
                    progress_bar.progress(50)
                    
                    neo4j_client = Neo4jClient(
                        uri=st.session_state.neo4j_uri,
                        user=st.session_state.neo4j_user,
                        password=st.session_state.neo4j_password
                    )
                    
                    # Get schema to verify
                    schema = neo4j_client.get_database_schema()
                    neo4j_client.close()
                    
                    status_text.text("Connection validated!")
                    progress_bar.progress(100)
                    
                    st.session_state.graph_built = True
                    st.success("✓ Successfully connected to Neo4j database!")
                    st.info("Your existing graph is ready to use.")
                    
                else:
                    # Build graph from files/folder
                    status_text.text("Initializing graph processor...")
                    progress_bar.progress(10)
                    
                    processor = CustomGraphProcessor()
                    processor.discovered_labels = st.session_state.entity_types
                    
                    if st.session_state.data_source == "upload":
                        # Save uploaded files to temp directory
                        status_text.text("Processing uploaded files...")
                        progress_bar.progress(20)
                        
                        with tempfile.TemporaryDirectory() as tmpdir:
                            temp_path = Path(tmpdir)
                            
                            # Save files
                            for file in st.session_state.uploaded_files:
                                file_path = temp_path / file.name
                                with open(file_path, 'wb') as f:
                                    f.write(file.getbuffer())
                            
                            # Process directory
                            status_text.text("Building knowledge graph...")
                            progress_bar.progress(50)
                            
                            result = processor.process_directory(
                                str(temp_path),
                                perform_resolution=True,
                                prompt_for_advanced=False,
                                auto_advanced=True,  # Run advanced processing
                                mode='fresh'
                            )
                    
                    else:  # folder
                        status_text.text("Building knowledge graph...")
                        progress_bar.progress(50)
                        
                        result = processor.process_directory(
                            st.session_state.folder_path,
                            perform_resolution=True,
                            prompt_for_advanced=False,
                            auto_advanced=True,  # Run advanced processing
                            mode='fresh'
                        )
                    
                    processor.close()
                    
                    status_text.text("Graph building complete!")
                    progress_bar.progress(100)
                    
                    st.session_state.graph_built = True
                    
                    # Display results
                    st.success("✓ Knowledge graph built successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Documents", result.get('total_documents', 0))
                    with col2:
                        st.metric("Chunks", f"{result.get('total_chunks_created', 0):,}")
                    with col3:
                        st.metric("Entities", f"{result.get('total_entities_created', 0):,}")
                
                logger.info("Graph building completed successfully")
                st.balloons()
                
            except Exception as e:
                logger.error(f"Error building graph: {e}", exc_info=True)
                st.error(f"❌ Error building graph: {str(e)}")
                status_text.text("Build failed")
                progress_bar.progress(0)
    
    else:
        st.success("✓ Graph already built!")
        st.info("Your knowledge graph is ready. You can now:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("app.py", label="🔍 Start Querying", use_container_width=True)
        with col2:
            st.page_link("pages/2_Test_Set.py", label="📝 Create Test Set", use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>BOOTH Retriever Setup | Need help? Check the documentation</small>
</div>
""", unsafe_allow_html=True)

