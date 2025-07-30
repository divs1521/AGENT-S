#!/usr/bin/env python3
"""
Enhanced test with Agent-S and Android World integration.
This test demonstrates the full multi-agent QA system using both frameworks.
"""

import json
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app import load_api_keys
from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig
from core.agent_s_integration import agent_s
from core.android_world_integration import android_world

def test_integration_status():
    """Test the integration status of Agent-S and Android World."""
    print("🔍 Checking Integration Status")
    print("=" * 50)
    
    # Check Agent-S availability
    if agent_s.is_available():
        print("✅ Agent-S: Available and integrated")
    else:
        print("⚠️ Agent-S: Not available (using fallback)")
    
    # Check Android World availability
    if android_world.is_available():
        print("✅ Android World: Available and integrated")
        available_tasks = android_world.get_available_tasks()
        print(f"📱 Available Android tasks: {len(available_tasks)}")
        if available_tasks:
            print(f"   Sample tasks: {available_tasks[:3]}")
    else:
        print("⚠️ Android World: Not available (using mock)")
    
    print()

def test_with_android_world():
    """Test the system with Android World integration."""
    
    print("🚀 Starting Enhanced Multi-Agent QA System Test")
    print("=" * 60)
    
    # Check integration status
    test_integration_status()
    
    # Load API keys
    if not load_api_keys():
        print("❌ Failed to load API keys")
        return False
    
    # Create enhanced configuration - using mock mode for now
    engine_params = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
        "mock": True  # Force mock mode to avoid API auth issues
    }
    
    # Configure Android World if available
    use_real_android = android_world.is_available()
    android_world_config = {
        "device_id": "emulator-5554",  # Default Android emulator
        "task_timeout": 60,
        "screenshot_frequency": 1.0
    } if use_real_android else None
    
    config = QASystemConfig(
        engine_params=engine_params,
        use_real_android=use_real_android,
        android_world_config=android_world_config,
        enable_visual_trace=True,
        verification_strictness="balanced"
    )
    
    orchestrator = MultiAgentQAOrchestrator(config)
    
    # Enhanced test scenarios
    test_scenarios = [
        {
            "name": "Knowledge Test with UI Interaction",
            "description": "Test knowledge retrieval through a mobile app interface",
            "task": "What is the capital of France and why is it important?",
            "context": {"type": "knowledge_app", "platform": "android"}
        },
        {
            "name": "Android Settings Navigation",
            "description": "Navigate to WiFi settings and verify the interface",
            "task": "Navigate to WiFi settings in Android and check available networks",
            "context": {"type": "settings_navigation", "target": "wifi"}
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Test Scenario {i}: {scenario['name']}")
        print("-" * 50)
        print(f"Description: {scenario['description']}")
        print(f"Task: {scenario['task']}")
        
        try:
            # Execute the test
            result = orchestrator.run_qa_test(
                task_description=scenario['task'],
                app_context=scenario['context'],
                test_config={"timeout": 120}
            )
            
            results.append({
                "scenario": scenario['name'],
                "success": result.overall_success,
                "score": result.final_score,
                "duration": result.end_time - result.start_time,
                "steps": len(result.execution_results),
                "integration_used": {
                    "agent_s": agent_s.is_available(),
                    "android_world": android_world.is_available() and config.use_real_android
                }
            })
            
            print(f"✅ Scenario {i} completed!")
            print(f"   Success: {result.overall_success}")
            print(f"   Score: {result.final_score:.2f}")
            print(f"   Duration: {result.end_time - result.start_time:.1f}s")
            print(f"   Steps: {len(result.execution_results)}")
            
        except Exception as e:
            print(f"❌ Scenario {i} failed: {e}")
            results.append({
                "scenario": scenario['name'],
                "success": False,
                "error": str(e)
            })
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = len([r for r in results if r.get('success', False)])
    total_tests = len(results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    if results:
        avg_score = sum(r.get('score', 0) for r in results if 'score' in r) / len([r for r in results if 'score' in r])
        print(f"Average Score: {avg_score:.2f}")
    
    # Integration summary
    print(f"\n🔧 Integration Status:")
    print(f"   Agent-S: {'✅ Active' if agent_s.is_available() else '⚠️ Fallback'}")
    print(f"   Android World: {'✅ Active' if (android_world.is_available() and config.use_real_android) else '⚠️ Mock'}")
    
    print(f"\n🎉 Enhanced integration test completed!")
    return True

if __name__ == "__main__":
    test_with_android_world()
