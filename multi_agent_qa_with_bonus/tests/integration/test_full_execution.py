#!/usr/bin/env python3

import json
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app import load_api_keys
from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig

def test_full_execution():
    """Test the full multi-agent execution with real API."""
    
    print("🚀 Starting Multi-Agent QA System Test")
    print("=" * 50)
    
    # Load API keys
    if not load_api_keys():
        print("❌ Failed to load API keys")
        return False
    
    # Create orchestrator
    engine_params = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    config = QASystemConfig(engine_params=engine_params)
    orchestrator = MultiAgentQAOrchestrator(config)
    
    # Test question
    test_question = "What is the capital of France and why is it important?"
    
    print(f"🤔 Question: {test_question}")
    print("\n📋 Starting execution...")
    
    try:
        # Execute the multi-agent workflow
        result = orchestrator.run_qa_test(
            task_description=test_question,
            app_context={"type": "knowledge_test"},
            test_config={"timeout": 60}
        )
        
        print("\n✅ Execution completed!")
        print("=" * 50)
        print("📊 RESULTS:")
        print("=" * 50)
        
        if hasattr(result, 'final_answer'):
            print(f"Final Answer: {result.final_answer}")
        else:
            print(f"Result: {result}")
            
        print("\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_execution()
    sys.exit(0 if success else 1)
