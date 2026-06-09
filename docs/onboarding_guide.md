# BOOTH Onboarding System Guide

## Overview

The BOOTH Onboarding System provides a comprehensive, user-friendly interface for setting up your knowledge graph retrieval system. It guides users through data ingestion, entity discovery, graph building, and test set creation.

## Architecture

The onboarding system consists of three main pages:

1. **Setup (pages/0_Setup.py)** - Data source selection and graph building
2. **Train AI (pages/1_Train_AI.py)** - Query curation and approval (existing, enhanced)
3. **Test Set (pages/2_Test_Set.py)** - Test question generation and management

## Components

### 1. Onboarding Utilities (`src/onboarding_utils.py`)

Helper functions for the onboarding workflow:

- `extract_sample_text_from_files()` - Extract text samples from uploaded PDF/TXT files
- `extract_entities_for_review()` - Extract entities with context for user review
- `generate_test_questions()` - Generate test questions from the knowledge graph
- `validate_neo4j_connection()` - Test Neo4j connection with credentials
- `update_env_file()` - Update .env file with new configuration
- `parse_test_set_csv()` - Parse uploaded CSV test sets
- `save_test_set_to_csv()` - Export test sets to CSV

### 2. Neo4j Client Extensions (`src/neo4j_client.py`)

New methods for test set management:

- `store_test_question()` - Store a test question in Neo4j
- `get_test_questions()` - Retrieve stored test questions
- `delete_test_questions()` - Delete all test questions

### 3. Setup Page (`pages/0_Setup.py`)

Main onboarding interface with 4 steps:

#### Step 1: Data Source Selection

Three options:

**a) Upload Files**
- Drag and drop PDF or TXT files
- Files are validated and stored temporarily
- Preview of uploaded files with size information

**b) Select Local Folder**
- Enter full path to folder containing documents
- Automatic detection of PDF and TXT files
- File count preview before processing

**c) Connect to Neo4j**
- Enter connection credentials (URI, username, password)
- Test connection before proceeding
- Optional: Save credentials to .env file
- Skips to Step 4 (existing database assumed)

#### Step 2: Entity Type Discovery

- Analyzes sample text from documents
- Uses `CustomGraphProcessor.discover_labels_for_text()`
- Displays discovered entity types (e.g., PERSON, ORGANIZATION, LOCATION)
- Users can:
  - Deselect unwanted types
  - Add custom entity types
  - Approve final list

#### Step 3: Individual Entity Review (Optional)

- Extracts sample entities using approved types
- Displays entities grouped by type
- Shows context snippet for each entity
- Users can:
  - Remove unwanted entities
  - Skip this step entirely

#### Step 4: Build Knowledge Graph

- Progress bar with status updates
- Uses `CustomGraphProcessor` to build graph
- For uploaded files: Saves to temp directory, then processes
- For folders: Processes directly
- For Neo4j: Validates connection and schema
- Displays final statistics:
  - Number of documents processed
  - Chunks created
  - Entities extracted

### 4. Test Set Page (`pages/2_Test_Set.py`)

Three tabs for test set management:

#### Tab 1: Upload Test Set

- CSV upload with validation
- Expected columns: `question`, `expected_answer`
- Preview before storing
- Stores in Neo4j as `TestQuestion` nodes

#### Tab 2: Generate Test Set

- Generate questions from graph entities
- Configurable number of questions (5-100)
- Optional: Generate answers using BOOTH Agent
- Editable dataframe for review
- Actions:
  - Store in Neo4j
  - Download as CSV
  - Clear generated questions

#### Tab 3: View Test Sets

- Display all stored test questions
- Filter by source (uploaded/generated)
- Export filtered results to CSV
- Delete all test questions

### 5. Navigation Updates

#### Main App (`app.py`)

- Checks if graph is built on startup
- Shows welcome message for new users
- Redirects to Setup page if no data detected

#### Train AI Page (`pages/1_Train_AI.py`)

- Added navigation links in sidebar
- Check for graph existence
- Redirect to Setup if needed

## User Flow

```
1. User visits app → No graph detected → Redirected to Setup

2. Setup Page
   ├─ Choose data source (upload/folder/neo4j)
   ├─ [If files/folder] Discover entity types
   ├─ [Optional] Review individual entities
   └─ Build knowledge graph

3. [Optional] Test Set Page
   ├─ Upload existing test set (CSV)
   ├─ OR Generate test questions from graph
   └─ Review and store questions

4. Main Query Interface
   └─ Start querying the knowledge graph

5. Train AI Page
   └─ Review and approve queries for continuous improvement
```

## Technical Implementation

### Session State Variables

```python
st.session_state.graph_built        # True if graph exists
st.session_state.data_source        # "upload" | "folder" | "neo4j"
st.session_state.uploaded_files     # List of UploadedFile objects
st.session_state.folder_path        # String path to folder
st.session_state.entity_types       # List of approved entity types
st.session_state.approved_entities  # Dict of entities by type
st.session_state.neo4j_uri          # Neo4j connection URI
st.session_state.neo4j_user         # Neo4j username
st.session_state.neo4j_password     # Neo4j password
st.session_state.test_questions     # Generated test questions
st.session_state.generated_answers  # Generated answers dict
```

### Neo4j Schema

**TestQuestion Node:**
```cypher
CREATE (tq:TestQuestion {
    id: $test_id,
    question: $question,
    expected_answer: $expected_answer,
    created_at: datetime(),
    metadata: $metadata  // JSON string
})
```

### Integration with build_graph Module

The onboarding system integrates with the existing `build_graph` module:

- Uses `CustomGraphProcessor` for graph building
- Respects discovered entity types
- Supports both lean and advanced processing modes
- Handles entity resolution and community detection

## File Structure

```
BOOTH_Retriever/
├── src/
│   ├── onboarding_utils.py        # NEW: Helper utilities
│   └── neo4j_client.py            # MODIFIED: Added test set methods
├── pages/
│   ├── 0_Setup.py                 # NEW: Main onboarding page
│   ├── 1_Train_AI.py              # MODIFIED: Added navigation
│   └── 2_Test_Set.py              # NEW: Test set management
├── app.py                         # MODIFIED: Added setup check
├── test_onboarding_integration.py # NEW: Integration test
└── docs/
    └── onboarding_guide.md        # NEW: This file
```

## Testing

Run the integration test:

```bash
python test_onboarding_integration.py
```

This validates:
- All modules can be imported
- Neo4j client has new methods
- All required files exist
- Neo4j connection (optional)

## Usage Examples

### Example 1: Upload Files

1. Navigate to Setup page
2. Select "Upload Files"
3. Drag and drop PDFs
4. Click "Continue"
5. Review and approve entity types
6. (Optional) Review individual entities
7. Click "Build Graph"
8. Wait for completion

### Example 2: Local Folder

1. Navigate to Setup page
2. Select "Select Local Folder"
3. Enter folder path: `C:\Users\YourName\Documents\data`
4. Click "Continue"
5. Follow entity discovery steps
6. Click "Build Graph"

### Example 3: Existing Neo4j

1. Navigate to Setup page
2. Select "Connect to Existing Neo4j Database"
3. Enter credentials:
   - URI: `bolt://localhost:7687`
   - Username: `neo4j`
   - Password: `your_password`
4. Check "Save to .env" (optional)
5. Click "Test Connection"
6. Click "Continue"
7. Ready to query!

### Example 4: Generate Test Set

1. Complete setup first
2. Navigate to Test Set page
3. Go to "Generate Test Set" tab
4. Set number of questions (e.g., 20)
5. Check "Generate answers using BOOTH Agent"
6. Click "Generate Questions"
7. Wait for generation (with progress bar)
8. Review questions in editable table
9. Click "Store in Neo4j"

## Best Practices

1. **Start Small**: Upload a few documents first to test the system
2. **Review Entity Types**: Remove unwanted types to improve accuracy
3. **Skip Individual Review**: Unless you need fine-grained control
4. **Generate Test Sets**: Create benchmarks to measure improvements
5. **Use Existing Neo4j**: If you already have a knowledge graph

## Troubleshooting

### "No entities found in graph"

- Ensure you completed the Setup page
- Check that documents were actually processed
- Verify Neo4j connection in environment variables

### "Failed to connect to Neo4j"

- Check Neo4j is running
- Verify credentials in .env file
- Test connection using Setup page

### "Import errors" in test script

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment if using one
- Some import warnings are expected for build_graph dependencies

### "Entity discovery takes too long"

- This is normal for large document sets
- The system samples text intelligently
- Consider processing fewer documents initially

## Advanced Configuration

### Custom Entity Types

Add domain-specific entity types:
- Financial: `FINANCIAL_INSTRUMENT`, `CURRENCY`, `MARKET`
- Medical: `DISEASE`, `TREATMENT`, `MEDICATION`
- Legal: `LAW`, `CASE`, `COURT`

### Graph Processing Options

In `CustomGraphProcessor`:
- `lean_mode=True` - Skip advanced processing
- `relationship_strategy="smart"` - Intelligent relationship discovery
- `auto_advanced=True` - Run community detection

### Environment Variables

Key variables:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Future Enhancements

Potential improvements:
- Support for more file formats (DOCX, CSV, JSON)
- Batch file upload with progress tracking
- Entity type suggestions based on domain
- Test set validation and quality metrics
- Graph statistics and visualization
- Export/import configurations

## Support

For issues or questions:
1. Check this guide first
2. Review existing documentation in `docs/`
3. Check the logs in `logs/` folder
4. Run integration test for diagnostics

---

**Version:** 1.0  
**Last Updated:** January 2026  

