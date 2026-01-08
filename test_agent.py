"""Test script for BOOTH Agent integration."""

import os
import sys
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.agents.state import BOOTHAgentState, MatchedQueryContext
        print("  [OK] state.py imports successful")
    except Exception as e:
        print(f"  [FAIL] state.py import failed: {e}")
        return False
    
    try:
        from src.agents.tools import BOOTHTools, create_tools
        print("  [OK] tools.py imports successful")
    except Exception as e:
        print(f"  [FAIL] tools.py import failed: {e}")
        return False
    
    try:
        from src.agents.booth_agent import BOOTHAgent, create_booth_agent
        print("  [OK] booth_agent.py imports successful")
    except Exception as e:
        print(f"  [FAIL] booth_agent.py import failed: {e}")
        return False
    
    try:
        from src.booth_orchestrator import BOOTHOrchestrator, BOOTHResponse
        print("  [OK] booth_orchestrator.py imports successful")
    except Exception as e:
        print(f"  [FAIL] booth_orchestrator.py import failed: {e}")
        return False
    
    print("\nAll imports successful!")
    return True


def test_orchestrator_init():
    """Test that the orchestrator initializes correctly."""
    print("\nTesting orchestrator initialization...")
    
    # Check required env vars
    if not os.getenv("OPENAI_API_KEY"):
        print("  [SKIP] OPENAI_API_KEY not set - skipping live test")
        return None
    
    if not os.getenv("NEO4J_PASSWORD"):
        print("  [SKIP] NEO4J_PASSWORD not set - skipping live test")
        return None
    
    try:
        from src.booth_orchestrator import BOOTHOrchestrator
        
        # Try with agent enabled
        orchestrator = BOOTHOrchestrator(use_agent=True)
        
        if orchestrator.use_agent and orchestrator.agent:
            print("  [OK] Orchestrator initialized with BOOTH Agent")
            print(f"    - Agent tools: {[t.name for t in orchestrator.agent.tools]}")
        else:
            print("  [WARN] Agent initialization failed, using legacy mode")
        
        # Clean up
        orchestrator.close()
        return True
        
    except Exception as e:
        print(f"  [FAIL] Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_tools():
    """Test that the agent tools work correctly."""
    print("\nTesting agent tools...")
    
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("NEO4J_PASSWORD"):
        print("  [SKIP] Missing API keys - skipping tool test")
        return None
    
    try:
        from src.llm_client import LLMClient
        from src.neo4j_client import Neo4jClient
        from src.agents.tools import BOOTHTools
        
        llm_client = LLMClient()
        neo4j_client = Neo4jClient()
        
        tool_container = BOOTHTools(neo4j_client, llm_client)
        tools = tool_container.get_tools()
        
        print(f"  [OK] Created {len(tools)} tools:")
        for tool in tools:
            print(f"    - {tool.name}: {tool.description[:50]}...")
        
        neo4j_client.close()
        return True
        
    except Exception as e:
        print(f"  [FAIL] Tool creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("BOOTH Agent Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Orchestrator initialization
    results.append(("Orchestrator Init", test_orchestrator_init()))
    
    # Test 3: Agent tools
    results.append(("Agent Tools", test_agent_tools()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        if result is True:
            status = "[PASSED]"
        elif result is False:
            status = "[FAILED]"
        else:
            status = "[SKIPPED]"
        print(f"  {name}: {status}")
    
    print("\nDone!")

