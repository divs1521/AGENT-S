"""Example script demonstrating the multi-agent QA system in action."""

import os
import json
import time
from main import run_single_test, run_multiple_tests, create_sample_test_scenarios
from utils.config_utils import create_sample_config_files, ConfigManager
from utils.logging_utils import QALogger

def setup_example_environment():
    """Setup the example environment with sample configurations."""
    print("Setting up example environment...")
    
    # Create necessary directories
    directories = ["config", "logs", "output", "screenshots"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create sample configuration files
    create_sample_config_files("config")
    
    print("✓ Example environment setup complete")


def demo_single_test():
    """Demonstrate a single QA test execution."""
    print("\n" + "="*60)
    print("DEMO: Single QA Test Execution")
    print("="*60)
    
    # Setup logger
    logger = QALogger("logs")
    
    task_description = "Test WiFi settings - navigate to settings, find WiFi option, and toggle WiFi on/off"
    
    print(f"Task: {task_description}")
    print("Running test with mock Android environment...")
    
    # Run the test
    results = run_single_test(
        task_description=task_description,
        android_task="settings_wifi",
        use_mock=True,
        output_dir="output/demo_single"
    )
    
    # Display results
    episode = results["episode"]
    analysis = results["analysis"]
    
    print(f"\n📊 Test Results:")
    print(f"   Episode ID: {episode['episode_id']}")
    print(f"   Success: {episode['overall_success']}")
    print(f"   Final Score: {episode['final_score']:.2f}")
    print(f"   Duration: {episode['end_time'] - episode['start_time']:.2f} seconds")
    print(f"   Plan Steps: {len(episode['plan']['subgoals'])}")
    print(f"   Executions: {len(episode['execution_results'])}")
    print(f"   Verifications: {len(episode['verification_results'])}")
    
    print(f"\n🔍 Supervisor Analysis:")
    print(f"   Bug Detection Accuracy: {analysis['bug_detection_accuracy']:.2f}")
    print(f"   Recovery Ability: {analysis['agent_recovery_ability']:.2f}")
    print(f"   Coverage Score: {analysis['test_coverage_score']:.2f}")
    
    if analysis['critical_issues']:
        print(f"   Critical Issues Found: {len(analysis['critical_issues'])}")
        for issue in analysis['critical_issues'][:3]:  # Show first 3
            print(f"     - {issue}")
    
    print(f"\n💡 Improvement Suggestions:")
    for suggestion in analysis['prompt_improvement_suggestions'][:3]:  # Show first 3
        print(f"   - {suggestion}")
    
    return results


def demo_multiple_tests():
    """Demonstrate multiple test scenarios execution."""
    print("\n" + "="*60)
    print("DEMO: Multiple QA Test Scenarios")
    print("="*60)
    
    # Create diverse test scenarios
    test_scenarios = [
        {
            "name": "WiFi Settings Test",
            "task_description": "Navigate to WiFi settings and toggle WiFi on/off",
            "app_context": {
                "android_task": "settings_wifi",
                "expected_elements": ["wifi_toggle", "wifi_settings"]
            }
        },
        {
            "name": "Alarm Creation Test", 
            "task_description": "Create a new alarm for 7:00 AM with custom label",
            "app_context": {
                "android_task": "clock_alarm",
                "expected_elements": ["add_alarm", "time_picker", "alarm_label"]
            }
        },
        {
            "name": "Email Search Test",
            "task_description": "Search for emails containing 'meeting' keyword",
            "app_context": {
                "android_task": "email_search",
                "expected_elements": ["search_bar", "email_list", "search_results"]
            }
        }
    ]
    
    print(f"Running {len(test_scenarios)} test scenarios...")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"  {i}. {scenario['name']}")
    
    # Run the tests
    results = run_multiple_tests(
        test_scenarios=test_scenarios,
        android_task="settings_wifi",  # Base task
        use_mock=True,
        output_dir="output/demo_multiple"
    )
    
    # Display results
    summary = results["summary"]
    
    print(f"\n📊 Overall Results:")
    print(f"   Total Scenarios: {results['total_scenarios']}")
    print(f"   Success Rate: {summary['success_rate']:.2%}")
    print(f"   Average Score: {summary['average_score']:.2f}")
    print(f"   Total Execution Time: {summary['total_execution_time']:.2f} seconds")
    
    print(f"\n📈 Individual Test Results:")
    for i, episode in enumerate(results["episodes"]):
        scenario_name = test_scenarios[i]["name"]
        print(f"   {scenario_name}: {'✅ PASS' if episode['overall_success'] else '❌ FAIL'} (Score: {episode['final_score']:.2f})")
    
    # Show comprehensive report highlights
    report = results.get("comprehensive_report", {})
    if report:
        exec_summary = report.get("executive_summary", {})
        if exec_summary:
            print(f"\n🎯 Executive Summary:")
            print(f"   Overall Grade: {exec_summary.get('overall_grade', 'N/A')}")
            print(f"   System Maturity: {exec_summary.get('system_maturity', 'N/A')}")
    
    return results


def demo_agent_interaction():
    """Demonstrate detailed agent interaction flow."""
    print("\n" + "="*60)
    print("DEMO: Detailed Agent Interaction Flow")
    print("="*60)
    
    # This is a simulation of what happens inside the orchestrator
    print("🤖 Agent Workflow Simulation:")
    print()
    
    # Planner Agent
    print("1️⃣ PLANNER AGENT")
    print("   📝 Task: Create an alarm for 7:00 AM")
    print("   🧠 Processing: Breaking down into subgoals...")
    time.sleep(1)
    
    subgoals = [
        "Open clock/alarm application",
        "Navigate to alarm creation screen", 
        "Set alarm time to 7:00 AM",
        "Configure alarm settings (repeat, sound, etc.)",
        "Save the new alarm",
        "Verify alarm appears in alarm list"
    ]
    
    print(f"   ✅ Generated {len(subgoals)} subgoals:")
    for i, goal in enumerate(subgoals, 1):
        print(f"      {i}. {goal}")
    print()
    
    # Executor Agent  
    print("2️⃣ EXECUTOR AGENT")
    print("   🎯 Executing subgoal 1: Open clock/alarm application")
    print("   👀 Analyzing UI state...")
    print("   📱 Action: click(x=540, y=960) on 'Clock' app icon")
    time.sleep(0.5)
    print("   ✅ Execution completed successfully")
    print()
    
    # Verifier Agent
    print("3️⃣ VERIFIER AGENT") 
    print("   🔍 Verifying execution result...")
    print("   📊 Expected: Clock app should be open")
    print("   📊 Actual: Clock app interface visible with alarm tab")
    print("   🎯 Confidence: 0.92")
    print("   ✅ Verification PASSED")
    print()
    
    # Continue with next step
    print("   📈 Proceeding to next subgoal...")
    print("   🔄 Repeating Executor → Verifier cycle...")
    print()
    
    # Supervisor Agent
    print("4️⃣ SUPERVISOR AGENT")
    print("   📋 Analyzing complete test episode...")
    print("   📊 Bug Detection Accuracy: 0.88")
    print("   🔧 Recovery Ability: 0.75") 
    print("   📐 Coverage Score: 0.82")
    print("   💡 Generating improvement suggestions...")
    
    suggestions = [
        "Enhance time picker interaction logic",
        "Add validation for alarm sound selection",
        "Improve error handling for scheduling conflicts"
    ]
    
    print("   📝 Improvement Suggestions:")
    for suggestion in suggestions:
        print(f"      • {suggestion}")
    
    print("   ✅ Analysis complete")
    print()
    
    print("🏁 Multi-Agent QA Test Complete!")
    return True


def demo_visual_trace():
    """Demonstrate visual trace capabilities."""
    print("\n" + "="*60)
    print("DEMO: Visual Trace and Screenshot Analysis")
    print("="*60)
    
    print("📸 Visual Trace Simulation:")
    print()
    
    # Simulate screenshot capture sequence
    steps = [
        ("step_1_before", "Home screen with app icons visible"),
        ("step_1_after", "Clock app opening with loading animation"),
        ("step_2_before", "Clock app main screen with alarm tab"),
        ("step_2_after", "Alarm creation screen with time picker"),
        ("step_3_before", "Time picker showing default time"),
        ("step_3_after", "Time picker set to 7:00 AM"),
        ("step_4_before", "Alarm settings configuration screen"),
        ("step_4_after", "New alarm saved and visible in list")
    ]
    
    screenshot_dir = "screenshots/demo"
    os.makedirs(screenshot_dir, exist_ok=True)
    
    for step_name, description in steps:
        screenshot_path = f"{screenshot_dir}/{step_name}.png"
        
        # Create placeholder screenshot file
        with open(screenshot_path, 'w') as f:
            f.write(f"Screenshot placeholder: {description}")
        
        print(f"   📷 {step_name}: {description}")
        print(f"      Saved to: {screenshot_path}")
        time.sleep(0.3)
    
    print(f"\n🔍 Visual Analysis Results:")
    print("   ✅ UI flow follows expected pattern")
    print("   ✅ No visual inconsistencies detected")
    print("   ✅ All interactive elements properly highlighted")
    print("   ⚠️  Minor: Loading animation could be faster")
    
    return len(steps)


def run_complete_demo():
    """Run the complete demonstration of the multi-agent QA system."""
    print("🚀 Multi-Agent QA System Demonstration")
    print("=" * 80)
    
    # Setup
    setup_example_environment()
    
    # Demo 1: Single test
    single_results = demo_single_test()
    
    # Demo 2: Multiple tests  
    multiple_results = demo_multiple_tests()
    
    # Demo 3: Agent interaction
    demo_agent_interaction()
    
    # Demo 4: Visual trace
    screenshot_count = demo_visual_trace()
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETE")
    print("="*80)
    
    print("📊 Demo Summary:")
    print(f"   ✅ Single test executed successfully")
    print(f"   ✅ Multiple tests completed ({multiple_results['total_scenarios']} scenarios)")
    print(f"   ✅ Agent workflow demonstrated")
    print(f"   ✅ Visual trace captured ({screenshot_count} screenshots)")
    
    print(f"\n📁 Generated Files:")
    print(f"   📂 output/demo_single/ - Single test results")
    print(f"   📂 output/demo_multiple/ - Multiple test results")  
    print(f"   📂 logs/ - System and agent logs")
    print(f"   📂 screenshots/demo/ - Screenshot placeholders")
    print(f"   📂 config/ - Sample configuration files")
    
    print(f"\n🔗 Next Steps:")
    print(f"   1. Review generated reports in output directories")
    print(f"   2. Customize configuration files in config/")
    print(f"   3. Add your own test scenarios")
    print(f"   4. Integrate with real Android devices") 
    print(f"   5. Set up API keys for LLM providers")
    
    print(f"\n💡 Tips:")
    print(f"   • Use --verbose flag for detailed logging")
    print(f"   • Start with mock environment for development")
    print(f"   • Review supervisor suggestions for improvements")
    print(f"   • Customize agent prompts for your specific needs")
    
    return {
        "single_results": single_results,
        "multiple_results": multiple_results,
        "demo_completed": True
    }


if __name__ == "__main__":
    # Check if this is being run directly
    print("🎯 Running Multi-Agent QA System Demo...")
    print("This demo showcases the capabilities of the system using mock environments.")
    print("No real Android devices or API keys required for this demonstration.\n")
    
    try:
        results = run_complete_demo()
        print("\n✨ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check the logs for more details.")
