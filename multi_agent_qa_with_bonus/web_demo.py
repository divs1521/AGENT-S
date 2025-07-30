"""
Quick Demo of Multi-Agent QA System with Web Interface
Shows both programmatic usage and web interface capabilities.
"""

import time
import requests
import json
from datetime import datetime

def test_web_interface():
    """Test the web interface programmatically."""
    base_url = "http://localhost:5000"
    
    print("🌐 Testing Multi-Agent QA System Web Interface")
    print("=" * 50)
    
    try:
        # Test if server is running
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Web interface is accessible")
        else:
            print(f"❌ Web interface returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Web interface is not running: {e}")
        print("💡 Start the web interface with: python app.py")
        return False
    
    # Test API endpoints
    endpoints = [
        ("/api/agent-status", "Agent Status API"),
        ("/api/logs", "Logs API"),
        ("/history", "History Page")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} working")
            else:
                print(f"⚠️  {name} returned status {response.status_code}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    return True

def demo_api_interaction():
    """Demonstrate API interaction with the web interface."""
    base_url = "http://localhost:5000"
    
    print("\n🔧 Testing API Key Setup")
    print("-" * 30)
    
    # Test API key endpoint
    try:
        # Simulate API key testing (mock keys)
        test_data = {
            "openai_key": "sk-test-key-for-demo",
            "anthropic_key": "",
            "google_key": ""
        }
        
        response = requests.post(f"{base_url}/api/test-keys", 
                               json=test_data, timeout=10)
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ API key testing endpoint working")
            print(f"   Results: {results}")
        else:
            print(f"⚠️  API testing returned status {response.status_code}")
    except Exception as e:
        print(f"❌ API key testing failed: {e}")

def demo_test_execution():
    """Demonstrate test execution via API."""
    base_url = "http://localhost:5000"
    
    print("\n🚀 Testing QA Execution API")
    print("-" * 30)
    
    test_config = {
        "task": "Demo test - Check WiFi settings functionality",
        "android_task": "settings_wifi",
        "use_real_env": False,  # Use mock environment
        "enable_screenshots": True,
        "enable_recovery": True,
        "verbose_logging": True
    }
    
    try:
        print(f"📝 Starting test: {test_config['task']}")
        response = requests.post(f"{base_url}/api/start-test", 
                               json=test_config, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                episode_id = result.get('episode_id')
                print(f"✅ Test started successfully")
                print(f"   Episode ID: {episode_id}")
                
                # Monitor test progress
                print("📊 Monitoring test progress...")
                for i in range(10):  # Monitor for up to 10 seconds
                    time.sleep(1)
                    
                    # Get agent status
                    status_response = requests.get(f"{base_url}/api/agent-status")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        print(f"   Agents: {', '.join([f'{k}:{v.get('status', 'unknown')}' for k, v in status.items()])}")
                    
                    # Check if test is complete (simplified check)
                    # In a real scenario, you'd use WebSocket for real-time updates
                    
                print("✅ Test monitoring completed")
            else:
                print(f"❌ Failed to start test: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Test start request failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")

def show_usage_examples():
    """Show various usage examples."""
    print("\n📚 Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "title": "🌐 Access Web Interface",
            "description": "Open your browser and go to:",
            "code": "http://localhost:5000"
        },
        {
            "title": "🔧 Setup API Keys",
            "description": "Configure your LLM provider credentials:",
            "code": "http://localhost:5000/setup"
        },
        {
            "title": "▶️ Execute Tests",
            "description": "Run QA tests with real-time monitoring:",
            "code": "http://localhost:5000/execute"
        },
        {
            "title": "📊 View History",
            "description": "Browse execution history and reports:",
            "code": "http://localhost:5000/history"
        },
        {
            "title": "🖥️ Command Line",
            "description": "Run tests via command line:",
            "code": "python main.py --mode single --task 'Test WiFi settings'"
        },
        {
            "title": "📁 History Folders",
            "description": "Each execution creates a timestamped folder:",
            "code": "history/20250730_143022_a1b2c3d4/"
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}")
        print(f"   {example['description']}")
        print(f"   {example['code']}")

def main():
    """Main demo function."""
    print(f"🤖 Multi-Agent QA System Demo")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test web interface
    web_working = test_web_interface()
    
    if web_working:
        # Test API interactions
        demo_api_interaction()
        demo_test_execution()
    
    # Show usage examples regardless
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\n💡 Next Steps:")
    print("   1. Visit http://localhost:5000 in your browser")
    print("   2. Set up your API keys in the Setup page")
    print("   3. Run your first QA test in the Execute page")
    print("   4. Monitor results in real-time")
    print("   5. Review history and generate reports")
    
if __name__ == "__main__":
    main()
