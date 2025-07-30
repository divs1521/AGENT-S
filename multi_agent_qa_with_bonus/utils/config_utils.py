"""Configuration utilities for the multi-agent QA system."""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    engine_params: Dict = field(default_factory=dict)
    max_retries: int = 3
    timeout: float = 30.0
    enable_logging: bool = True
    custom_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestConfig:
    """Configuration for test execution."""
    max_execution_time: float = 300.0
    max_steps_per_test: int = 50
    screenshot_enabled: bool = True
    recovery_enabled: bool = True
    verification_strictness: str = "balanced"  # strict, balanced, lenient
    parallel_execution: bool = False
    retry_failed_tests: bool = True


@dataclass
class SystemConfig:
    """Overall system configuration."""
    log_directory: str = "logs"
    output_directory: str = "output"
    screenshot_directory: str = "screenshots"
    config_directory: str = "config"
    
    # Agent configurations
    planner_config: AgentConfig = field(default_factory=AgentConfig)
    executor_config: AgentConfig = field(default_factory=AgentConfig)
    verifier_config: AgentConfig = field(default_factory=AgentConfig)
    supervisor_config: AgentConfig = field(default_factory=AgentConfig)
    
    # Test configuration
    test_config: TestConfig = field(default_factory=TestConfig)
    
    # Android World integration
    android_task: str = "settings_wifi"
    use_mock_environment: bool = True
    android_world_config: Dict = field(default_factory=dict)


class ConfigManager:
    """Manage configuration loading and saving."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_system_config(self, config_file: str = "system_config.json") -> SystemConfig:
        """Load system configuration from file.
        
        Args:
            config_file: Configuration file name
            
        Returns:
            System configuration object
        """
        config_path = self.config_dir / config_file
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                return self._dict_to_system_config(config_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading config from {config_path}: {e}")
                print("Using default configuration")
        
        # Return default configuration
        return SystemConfig()
    
    def save_system_config(self, config: SystemConfig, config_file: str = "system_config.json"):
        """Save system configuration to file.
        
        Args:
            config: System configuration to save
            config_file: Configuration file name
        """
        config_path = self.config_dir / config_file
        
        config_dict = self._system_config_to_dict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    
    def load_test_scenarios(self, scenarios_file: str = "test_scenarios.json") -> List[Dict]:
        """Load test scenarios from file.
        
        Args:
            scenarios_file: Test scenarios file name
            
        Returns:
            List of test scenario configurations
        """
        scenarios_path = self.config_dir / scenarios_file
        
        if scenarios_path.exists():
            try:
                with open(scenarios_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading scenarios from {scenarios_path}: {e}")
        
        # Return default scenarios
        return self._get_default_scenarios()
    
    def save_test_scenarios(self, scenarios: List[Dict], scenarios_file: str = "test_scenarios.json"):
        """Save test scenarios to file.
        
        Args:
            scenarios: Test scenarios to save
            scenarios_file: Test scenarios file name
        """
        scenarios_path = self.config_dir / scenarios_file
        
        with open(scenarios_path, 'w') as f:
            json.dump(scenarios, f, indent=2)
        
        print(f"Test scenarios saved to {scenarios_path}")
    
    def load_agent_prompts(self, prompts_file: str = "agent_prompts.json") -> Dict[str, str]:
        """Load custom agent prompts from file.
        
        Args:
            prompts_file: Prompts file name
            
        Returns:
            Dictionary of agent prompts
        """
        prompts_path = self.config_dir / prompts_file
        
        if prompts_path.exists():
            try:
                with open(prompts_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading prompts from {prompts_path}: {e}")
        
        return {}
    
    def save_agent_prompts(self, prompts: Dict[str, str], prompts_file: str = "agent_prompts.json"):
        """Save custom agent prompts to file.
        
        Args:
            prompts: Agent prompts to save
            prompts_file: Prompts file name
        """
        prompts_path = self.config_dir / prompts_file
        
        with open(prompts_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"Agent prompts saved to {prompts_path}")
    
    def _dict_to_system_config(self, config_data: Dict) -> SystemConfig:
        """Convert dictionary to SystemConfig object.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            SystemConfig object
        """
        # Extract agent configurations
        agent_configs = {}
        for agent_name in ["planner", "executor", "verifier", "supervisor"]:
            agent_data = config_data.get(f"{agent_name}_config", {})
            agent_configs[f"{agent_name}_config"] = AgentConfig(**agent_data)
        
        # Extract test configuration
        test_data = config_data.get("test_config", {})
        test_config = TestConfig(**test_data)
        
        # Create system config
        system_config = SystemConfig(
            log_directory=config_data.get("log_directory", "logs"),
            output_directory=config_data.get("output_directory", "output"),
            screenshot_directory=config_data.get("screenshot_directory", "screenshots"),
            config_directory=config_data.get("config_directory", "config"),
            test_config=test_config,
            android_task=config_data.get("android_task", "settings_wifi"),
            use_mock_environment=config_data.get("use_mock_environment", True),
            android_world_config=config_data.get("android_world_config", {}),
            **agent_configs
        )
        
        return system_config
    
    def _system_config_to_dict(self, config: SystemConfig) -> Dict:
        """Convert SystemConfig object to dictionary.
        
        Args:
            config: SystemConfig object
            
        Returns:
            Configuration dictionary
        """
        return {
            "log_directory": config.log_directory,
            "output_directory": config.output_directory,
            "screenshot_directory": config.screenshot_directory,
            "config_directory": config.config_directory,
            "android_task": config.android_task,
            "use_mock_environment": config.use_mock_environment,
            "android_world_config": config.android_world_config,
            "planner_config": {
                "engine_params": config.planner_config.engine_params,
                "max_retries": config.planner_config.max_retries,
                "timeout": config.planner_config.timeout,
                "enable_logging": config.planner_config.enable_logging,
                "custom_prompts": config.planner_config.custom_prompts
            },
            "executor_config": {
                "engine_params": config.executor_config.engine_params,
                "max_retries": config.executor_config.max_retries,
                "timeout": config.executor_config.timeout,
                "enable_logging": config.executor_config.enable_logging,
                "custom_prompts": config.executor_config.custom_prompts
            },
            "verifier_config": {
                "engine_params": config.verifier_config.engine_params,
                "max_retries": config.verifier_config.max_retries,
                "timeout": config.verifier_config.timeout,
                "enable_logging": config.verifier_config.enable_logging,
                "custom_prompts": config.verifier_config.custom_prompts
            },
            "supervisor_config": {
                "engine_params": config.supervisor_config.engine_params,
                "max_retries": config.supervisor_config.max_retries,
                "timeout": config.supervisor_config.timeout,
                "enable_logging": config.supervisor_config.enable_logging,
                "custom_prompts": config.supervisor_config.custom_prompts
            },
            "test_config": {
                "max_execution_time": config.test_config.max_execution_time,
                "max_steps_per_test": config.test_config.max_steps_per_test,
                "screenshot_enabled": config.test_config.screenshot_enabled,
                "recovery_enabled": config.test_config.recovery_enabled,
                "verification_strictness": config.test_config.verification_strictness,
                "parallel_execution": config.test_config.parallel_execution,
                "retry_failed_tests": config.test_config.retry_failed_tests
            }
        }
    
    def _get_default_scenarios(self) -> List[Dict]:
        """Get default test scenarios.
        
        Returns:
            List of default test scenarios
        """
        return [
            {
                "name": "WiFi Settings Test",
                "task_description": "Test navigation to WiFi settings and toggle WiFi on/off",
                "app_context": {
                    "android_task": "settings_wifi",
                    "expected_elements": ["wifi_toggle", "wifi_settings", "network_list"]
                },
                "test_config": {
                    "max_steps": 15,
                    "timeout": 180
                }
            },
            {
                "name": "Clock Alarm Test",
                "task_description": "Test creating and configuring a new alarm",
                "app_context": {
                    "android_task": "clock_alarm", 
                    "expected_elements": ["add_alarm", "time_picker", "alarm_list"]
                },
                "test_config": {
                    "max_steps": 20,
                    "timeout": 240
                }
            },
            {
                "name": "Email Search Test",
                "task_description": "Test searching for specific emails in the email app",
                "app_context": {
                    "android_task": "email_search",
                    "expected_elements": ["search_bar", "email_list", "compose_button"]
                },
                "test_config": {
                    "max_steps": 12,
                    "timeout": 150
                }
            }
        ]
    
    def create_environment_config(self) -> Dict:
        """Create environment-specific configuration.
        
        Returns:
            Environment configuration dictionary
        """
        return {
            "api_keys": {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "google_api_key": os.getenv("GOOGLE_API_KEY", "")
            },
            "model_configs": {
                "openai": {
                    "model": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "anthropic": {
                    "model": "claude-3-5-sonnet-20240620",
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                "gemini": {
                    "model": "gemini-1.5-pro",
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            },
            "android_world": {
                "available_tasks": [
                    "settings_wifi",
                    "clock_alarm",
                    "email_search",
                    "contacts_add",
                    "calendar_event",
                    "notes_create",
                    "camera_photo",
                    "browser_search"
                ],
                "default_task": "settings_wifi",
                "mock_mode": True
            }
        }
    
    def validate_config(self, config: SystemConfig) -> List[str]:
        """Validate system configuration.
        
        Args:
            config: System configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate directories
        required_dirs = [
            config.log_directory,
            config.output_directory,
            config.screenshot_directory
        ]
        
        for directory in required_dirs:
            if not directory or not isinstance(directory, str):
                errors.append(f"Invalid directory path: {directory}")
        
        # Validate test configuration
        if config.test_config.max_execution_time <= 0:
            errors.append("max_execution_time must be positive")
        
        if config.test_config.max_steps_per_test <= 0:
            errors.append("max_steps_per_test must be positive")
        
        if config.test_config.verification_strictness not in ["strict", "balanced", "lenient"]:
            errors.append("verification_strictness must be 'strict', 'balanced', or 'lenient'")
        
        # Validate agent configurations
        agents = [
            ("planner", config.planner_config),
            ("executor", config.executor_config),
            ("verifier", config.verifier_config),
            ("supervisor", config.supervisor_config)
        ]
        
        for agent_name, agent_config in agents:
            if agent_config.max_retries < 0:
                errors.append(f"{agent_name} max_retries must be non-negative")
            
            if agent_config.timeout <= 0:
                errors.append(f"{agent_name} timeout must be positive")
        
        return errors
    
    def setup_directories(self, config: SystemConfig):
        """Setup required directories based on configuration.
        
        Args:
            config: System configuration
        """
        directories = [
            config.log_directory,
            config.output_directory,
            config.screenshot_directory,
            config.config_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Directory ensured: {directory}")


def load_default_config() -> SystemConfig:
    """Load default system configuration.
    
    Returns:
        Default system configuration
    """
    config_manager = ConfigManager()
    return config_manager.load_system_config()


def create_sample_config_files(config_dir: str = "config"):
    """Create sample configuration files for reference.
    
    Args:
        config_dir: Directory to create sample files in
    """
    config_manager = ConfigManager(config_dir)
    
    # Create sample system config
    sample_config = SystemConfig()
    sample_config.planner_config.engine_params = {
        "engine_type": "openai",
        "model": "gpt-4o",
        "temperature": 0.7
    }
    
    config_manager.save_system_config(sample_config, "sample_system_config.json")
    
    # Create sample test scenarios
    sample_scenarios = config_manager._get_default_scenarios()
    config_manager.save_test_scenarios(sample_scenarios, "sample_test_scenarios.json")
    
    # Create sample agent prompts
    sample_prompts = {
        "planner_system_prompt": "You are an expert QA planning agent...",
        "executor_system_prompt": "You are an expert executor agent...",
        "verifier_system_prompt": "You are an expert verification agent...",
        "supervisor_system_prompt": "You are an expert supervisor agent..."
    }
    config_manager.save_agent_prompts(sample_prompts, "sample_agent_prompts.json")
    
    print(f"Sample configuration files created in {config_dir}")
