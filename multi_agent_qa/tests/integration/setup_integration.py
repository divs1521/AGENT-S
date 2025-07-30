#!/usr/bin/env python3
"""
Setup script for Agent-S and Android World integration.
This script helps configure the multi-agent QA system with both frameworks.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_dependencies():
    """Check if required Python packages are installed."""
    print("🔍 Checking Python dependencies...")
    
    required_packages = [
        'openai',
        'anthropic', 
        'google-generativeai',
        'flask',
        'flask-socketio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_agent_s_setup():
    """Check Agent-S setup and dependencies."""
    print("\n🤖 Checking Agent-S setup...")
    
    agent_s_path = Path(__file__).parent.parent / "Agent-S-main"
    
    if not agent_s_path.exists():
        print("   ❌ Agent-S not found at expected location")
        return False
    
    # Check key directories
    key_dirs = [
        "gui_agents/s2/core",
        "gui_agents/s1"
    ]
    
    for dir_path in key_dirs:
        full_path = agent_s_path / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
    
    # Check if Agent-S requirements are installed
    agent_s_requirements = agent_s_path / "requirements.txt"
    if agent_s_requirements.exists():
        print("   📋 Agent-S requirements.txt found")
        # Could add logic to check if requirements are installed
    
    return True

def check_android_world_setup():
    """Check Android World setup and dependencies."""
    print("\n📱 Checking Android World setup...")
    
    android_world_path = Path(__file__).parent.parent / "android_world-main"
    
    if not android_world_path.exists():
        print("   ❌ Android World not found at expected location")
        return False
    
    # Check key directories
    key_dirs = [
        "android_world/env",
        "android_world/agents"
    ]
    
    for dir_path in key_dirs:
        full_path = android_world_path / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
    
    # Check if Android World requirements are installed
    android_world_requirements = android_world_path / "requirements.txt"
    if android_world_requirements.exists():
        print("   📋 Android World requirements.txt found")
    
    return True

def check_api_keys():
    """Check if API keys are properly configured."""
    print("\n🔑 Checking API keys configuration...")
    
    config_file = Path(__file__).parent / "config" / "api_keys.json"
    
    if not config_file.exists():
        print("   ❌ API keys file not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            keys = json.load(f)
        
        for provider in ['openai', 'anthropic', 'google']:
            if keys.get(provider):
                print(f"   ✅ {provider}: Configured")
            else:
                print(f"   ⚠️ {provider}: Not configured")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading API keys: {e}")
        return False

def create_integrated_config():
    """Create a configuration file for the integrated system."""
    print("\n⚙️ Creating integrated system configuration...")
    
    config = {
        "system": {
            "name": "Multi-Agent QA System with Agent-S and Android World",
            "version": "1.0.0",
            "integration_mode": "full"
        },
        "agent_s": {
            "enabled": True,
            "fallback_to_mock": True,
            "engine_config": {
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        },
        "android_world": {
            "enabled": True,
            "fallback_to_mock": True,
            "environment_config": {
                "device_id": "emulator-5554",
                "task_timeout": 60,
                "screenshot_frequency": 1.0
            }
        },
        "qa_system": {
            "max_execution_time": 300,
            "max_retries": 3,
            "enable_visual_trace": True,
            "verification_strictness": "balanced"
        }
    }
    
    config_file = Path(__file__).parent / "config" / "integrated_config.json"
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✅ Configuration saved to: {config_file}")
    return True

def main():
    """Main setup function."""
    print("🚀 Multi-Agent QA System Integration Setup")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Run all checks
    all_checks_passed &= check_python_dependencies()
    all_checks_passed &= check_agent_s_setup()
    all_checks_passed &= check_android_world_setup()
    all_checks_passed &= check_api_keys()
    
    # Create configuration
    create_integrated_config()
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("   1. Run: python test_integration.py")
        print("   2. Or start the web interface: python app.py")
    else:
        print("⚠️ Setup completed with warnings.")
        print("   Some components may not be fully functional.")
        print("   Check the messages above and install missing dependencies.")
    
    print("\n💡 System Features:")
    print("   • Agent-S integration for advanced LLM agents")
    print("   • Android World integration for realistic Android testing")
    print("   • Fallback to mock implementations when needed")
    print("   • Web interface for monitoring and control")

if __name__ == "__main__":
    main()
