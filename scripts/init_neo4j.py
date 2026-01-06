"""Initialize Neo4j database schema, indexes, and constraints for BOOTH."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def init_neo4j_schema():
    """Initialize Neo4j database with required schema, indexes, and constraints."""
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable is required")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            print("Initializing Neo4j schema for BOOTH...")
            
            # Create constraints for unique IDs
            print("Creating constraints...")
            
            # Query node constraint
            try:
                session.run("""
                    CREATE CONSTRAINT query_id_unique IF NOT EXISTS
                    FOR (q:Query) REQUIRE q.id IS UNIQUE
                """)
                print("✓ Query ID constraint created")
            except Exception as e:
                print(f"  Query constraint may already exist: {e}")
            
            # CypherAttempt node constraint
            try:
                session.run("""
                    CREATE CONSTRAINT cypher_attempt_id_unique IF NOT EXISTS
                    FOR (c:CypherAttempt) REQUIRE c.id IS UNIQUE
                """)
                print("✓ CypherAttempt ID constraint created")
            except Exception as e:
                print(f"  CypherAttempt constraint may already exist: {e}")
            
            # Response node constraint
            try:
                session.run("""
                    CREATE CONSTRAINT response_id_unique IF NOT EXISTS
                    FOR (r:Response) REQUIRE r.id IS UNIQUE
                """)
                print("✓ Response ID constraint created")
            except Exception as e:
                print(f"  Response constraint may already exist: {e}")
            
            # Create vector index for Query embeddings
            print("\nCreating vector index...")
            
            try:
                # Check if index exists
                result = session.run("""
                    SHOW INDEXES
                    YIELD name, type
                    WHERE name = 'query_embeddings'
                    RETURN count(*) as count
                """)
                index_exists = result.single()["count"] > 0
                
                if not index_exists:
                    # Create vector index
                    # Using 1536 dimensions for text-embedding-3-small
                    session.run("""
                        CREATE VECTOR INDEX query_embeddings IF NOT EXISTS
                        FOR (q:Query)
                        ON q.embedding
                        OPTIONS {
                            indexConfig: {
                                `vector.dimensions`: 1536,
                                `vector.similarity_function`: 'cosine'
                            }
                        }
                    """)
                    print("✓ Vector index 'query_embeddings' created (1536 dimensions, cosine similarity)")
                else:
                    print("✓ Vector index 'query_embeddings' already exists")
            except Exception as e:
                print(f"  Error creating vector index: {e}")
                print("  Note: Vector indexes require Neo4j 5.11+ with vector search support")
            
            # Create regular indexes for faster lookups
            print("\nCreating property indexes...")
            
            try:
                session.run("""
                    CREATE INDEX query_status_idx IF NOT EXISTS
                    FOR (q:Query) ON (q.status)
                """)
                print("✓ Query status index created")
            except Exception as e:
                print(f"  Status index may already exist: {e}")
            
            try:
                session.run("""
                    CREATE INDEX query_timestamp_idx IF NOT EXISTS
                    FOR (q:Query) ON (q.timestamp)
                """)
                print("✓ Query timestamp index created")
            except Exception as e:
                print(f"  Timestamp index may already exist: {e}")
            
            print("\n✅ Neo4j schema initialization complete!")
            print("\nNode Types:")
            print("  - Query: Stores user queries with embeddings and metadata")
            print("  - CypherAttempt: Stores generated cypher queries")
            print("  - Response: Stores results returned to user")
            print("\nRelationships:")
            print("  - (Query)-[:GENERATED]->(CypherAttempt) - for unapproved queries")
            print("  - (Query)-[:FEW_SHOT_PROMPT]->(CypherAttempt) - for approved examples")
            print("  - (CypherAttempt)-[:PRODUCED]->(Response) - links attempts to results")
            print("  - (Query)-[:SIMILAR_TO]->(Query) - tracks similarity matches")
            
    except Exception as e:
        print(f"❌ Error initializing schema: {e}")
        raise
    finally:
        driver.close()


if __name__ == "__main__":
    init_neo4j_schema()

