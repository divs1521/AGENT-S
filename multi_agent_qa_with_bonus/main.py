"""Main application script for the multi-agent QA system."""

import logging
import json
import argparse
import os
from typing import Dict, List, Optional
from dataclasses import asdict

from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig
from utils.android_integration import create_android_environment, get_supported_actions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config(android_task: str = "settings_wifi", use_mock: bool = True) -> QASystemConfig:
    """Create default configuration for the QA system.
    
    Args:
        android_task: Android World task to use
        use_mock: Whether to use mock Android environment
        
    Returns:
        QA system configuration
    """
    # Default engine parameters (would need real API keys for production use)
    engine_params = {
        "engine_type": "openai",  # or "anthropic", "gemini", etc.
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    # Create Android environment
    android_env = create_android_environment(android_task, use_mock)
    
    return QASystemConfig(
        engine_params=engine_params,
        android_env=android_env,
        max_execution_time=300.0,
        max_retries=3,
        enable_visual_trace=True,
        log_directory="logs",
        screenshot_directory="screenshots",
        enable_recovery=True,
        verification_strictness="balanced"
    )


def run_single_test(
    task_description: str,
    android_task: str = "settings_wifi",
    use_mock: bool = True,
    output_dir: str = "output"
) -> Dict:
    """Run a single QA test.
    
    Args:
        task_description: Description of what to test
        android_task: Android World task to use
        use_mock: Whether to use mock environment
        output_dir: Directory to save results
        
    Returns:
        Test results dictionary
    """
    logger.info(f"Running single test: {task_description}")
    
    # Create configuration
    config = create_default_config(android_task, use_mock)
    config.log_directory = os.path.join(output_dir, "logs")
    config.screenshot_directory = os.path.join(output_dir, "screenshots")
    
    # Create orchestrator
    orchestrator = MultiAgentQAOrchestrator(config)
    
    # Run test
    episode = orchestrator.run_qa_test(
        task_description=task_description,
        app_context={
            "android_task": android_task,
            "environment_type": "mock" if use_mock else "real"
        }
    )
    
    # Analyze episode
    analysis = orchestrator.analyze_episode(episode)
    
    # Save results
    results = {
        "episode": asdict(episode),
        "analysis": asdict(analysis),
        "system_status": orchestrator.get_system_status()
    }
    
    results_path = os.path.join(output_dir, f"test_results_{episode.episode_id}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Test completed. Results saved to {results_path}")
    logger.info(f"Success: {episode.overall_success}, Score: {episode.final_score:.2f}")
    
    return results


def run_multiple_tests(
    test_scenarios: List[Dict],
    android_task: str = "settings_wifi", 
    use_mock: bool = True,
    output_dir: str = "output"
) -> Dict:
    """Run multiple QA tests.
    
    Args:
        test_scenarios: List of test scenario configurations
        android_task: Android World task to use
        use_mock: Whether to use mock environment
        output_dir: Directory to save results
        
    Returns:
        Combined results from all tests
    """
    logger.info(f"Running {len(test_scenarios)} test scenarios")
    
    # Create configuration
    config = create_default_config(android_task, use_mock)
    config.log_directory = os.path.join(output_dir, "logs")
    config.screenshot_directory = os.path.join(output_dir, "screenshots")
    
    # Create orchestrator
    orchestrator = MultiAgentQAOrchestrator(config)
    
    all_episodes = []
    all_analyses = []
    
    for i, scenario in enumerate(test_scenarios):
        logger.info(f"Running scenario {i + 1}/{len(test_scenarios)}: {scenario.get('name', 'Unnamed')}")
        
        # Run test
        episode = orchestrator.run_qa_test(
            task_description=scenario["task_description"],
            app_context=scenario.get("app_context", {}),
            test_config=scenario.get("test_config", {})
        )
        
        # Analyze episode
        analysis = orchestrator.analyze_episode(episode)
        
        all_episodes.append(episode)
        all_analyses.append(analysis)
    
    # Generate comprehensive report
    report_path = os.path.join(output_dir, "comprehensive_report.json")
    report = orchestrator.generate_comprehensive_report(all_episodes, report_path)
    
    # Compile results
    results = {
        "total_scenarios": len(test_scenarios),
        "episodes": [asdict(ep) for ep in all_episodes],
        "analyses": [asdict(analysis) for analysis in all_analyses],
        "comprehensive_report": report,
        "system_status": orchestrator.get_system_status(),
        "summary": {
            "success_rate": len([ep for ep in all_episodes if ep.overall_success]) / len(all_episodes),
            "average_score": sum(ep.final_score for ep in all_episodes) / len(all_episodes),
            "total_execution_time": sum(ep.end_time - ep.start_time for ep in all_episodes)
        }
    }
    
    # Save combined results
    results_path = os.path.join(output_dir, "multiple_tests_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"All tests completed. Results saved to {results_path}")
    logger.info(f"Overall success rate: {results['summary']['success_rate']:.2f}")
    
    return results


def run_continuous_testing(
    test_scenarios: List[Dict],
    max_iterations: int = 5,
    android_task: str = "settings_wifi",
    use_mock: bool = True,
    output_dir: str = "output"
) -> Dict:
    """Run continuous testing with iterative improvement.
    
    Args:
        test_scenarios: List of test scenarios
        max_iterations: Maximum improvement iterations
        android_task: Android World task to use
        use_mock: Whether to use mock environment
        output_dir: Directory to save results
        
    Returns:
        Continuous testing results
    """
    logger.info(f"Running continuous testing with {max_iterations} iterations")
    
    # Create configuration
    config = create_default_config(android_task, use_mock)
    config.log_directory = os.path.join(output_dir, "logs")
    config.screenshot_directory = os.path.join(output_dir, "screenshots")
    
    # Create orchestrator
    orchestrator = MultiAgentQAOrchestrator(config)
    
    # Run continuous testing
    results = orchestrator.run_continuous_testing(
        test_scenarios=test_scenarios,
        max_iterations=max_iterations,
        improvement_threshold=0.05
    )
    
    # Save results
    results_path = os.path.join(output_dir, "continuous_testing_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Continuous testing completed. Results saved to {results_path}")
    logger.info(f"Total iterations: {results['total_iterations']}")
    
    return results


def create_sample_test_scenarios() -> List[Dict]:
    """Create sample test scenarios for demonstration.
    
    Returns:
        List of sample test scenarios
    """
    return [
        {
            "name": "WiFi Toggle Test",
            "task_description": "Test turning WiFi on and off in Android settings",
            "app_context": {
                "android_task": "settings_wifi",
                "expected_elements": ["wifi_toggle", "wifi_settings"]
            },
            "test_config": {
                "max_steps": 10,
                "timeout": 120
            }
        },
        {
            "name": "Alarm Creation Test", 
            "task_description": "Test creating a new alarm in the clock app",
            "app_context": {
                "android_task": "clock_alarm",
                "expected_elements": ["add_alarm", "time_picker", "save_button"]
            },
            "test_config": {
                "max_steps": 15,
                "timeout": 180
            }
        },
        {
            "name": "Email Search Test",
            "task_description": "Test searching for emails in the email app",
            "app_context": {
                "android_task": "email_search", 
                "expected_elements": ["search_bar", "email_list", "search_results"]
            },
            "test_config": {
                "max_steps": 12,
                "timeout": 150
            }
        }
    ]


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Multi-Agent QA System for Android Apps")
    
    parser.add_argument(
        "--mode",
        choices=["single", "multiple", "continuous"],
        default="single", 
        help="Testing mode to run"
    )
    
    parser.add_argument(
        "--task",
        default="Test WiFi toggle functionality",
        help="Task description for single test mode"
    )
    
    parser.add_argument(
        "--android-task",
        default="settings_wifi",
        help="Android World task to use"
    )
    
    parser.add_argument(
        "--use-real-env",
        action="store_true",
        help="Use real Android environment instead of mock"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum iterations for continuous testing"
    )
    
    parser.add_argument(
        "--scenarios-file",
        help="JSON file containing test scenarios"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load test scenarios
    if args.scenarios_file and os.path.exists(args.scenarios_file):
        with open(args.scenarios_file, 'r') as f:
            test_scenarios = json.load(f)
    else:
        test_scenarios = create_sample_test_scenarios()
    
    use_mock = not args.use_real_env
    
    try:
        if args.mode == "single":
            results = run_single_test(
                task_description=args.task,
                android_task=args.android_task,
                use_mock=use_mock,
                output_dir=args.output_dir
            )
            print(f"Single test completed. Success: {results['episode']['overall_success']}")
            
        elif args.mode == "multiple":
            results = run_multiple_tests(
                test_scenarios=test_scenarios,
                android_task=args.android_task,
                use_mock=use_mock,
                output_dir=args.output_dir
            )
            print(f"Multiple tests completed. Success rate: {results['summary']['success_rate']:.2f}")
            
        elif args.mode == "continuous":
            results = run_continuous_testing(
                test_scenarios=test_scenarios,
                max_iterations=args.max_iterations,
                android_task=args.android_task,
                use_mock=use_mock,
                output_dir=args.output_dir
            )
            print(f"Continuous testing completed. Iterations: {results['total_iterations']}")
        
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
