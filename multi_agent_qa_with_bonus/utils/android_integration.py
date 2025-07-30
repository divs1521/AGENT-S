"""Android World integration utilities for the multi-agent QA system."""

import logging
import sys
import os
from typing import Dict, List, Optional, Any, Union
import time

# Add Android World path
current_dir = os.path.dirname(__file__)
android_world_path = os.path.join(current_dir, "..", "..", "android_world-main")
sys.path.insert(0, android_world_path)

logger = logging.getLogger(__name__)

# Mock classes for development/testing
class MockAndroidEnv:
    """Mock Android environment for development and testing."""
    
    def __init__(self, task_name: str = "mock_task"):
        self.task_name = task_name
        self.state_history = []
        self.action_history = []
        self.current_step = 0
        
    def reset(self, go_home: bool = False):
        """Reset the environment."""
        self.state_history = []
        self.action_history = []
        self.current_step = 0
        logger.info(f"Mock Android environment reset for task: {self.task_name}")
        return self._create_mock_state()
    
    def step(self, action: Dict):
        """Execute an action in the environment."""
        self.action_history.append(action)
        self.current_step += 1
        logger.debug(f"Mock action executed: {action}")
        time.sleep(0.1)  # Simulate action execution time
        return self._create_mock_state()
    
    def get_state(self, wait_to_stabilize: bool = False):
        """Get current state of the environment."""
        if wait_to_stabilize:
            time.sleep(0.5)  # Simulate waiting for UI to stabilize
        
        state = self._create_mock_state()
        self.state_history.append(state)
        return state
    
    def _create_mock_state(self):
        """Create a mock state object."""
        return MockState(
            pixels=f"mock_pixels_{self.current_step}",
            forest=f"mock_forest_{self.current_step}",
            ui_elements=[
                MockUIElement(f"button_{i}", "button", f"Button {i}", (100 + i*50, 200 + i*30))
                for i in range(5)
            ]
        )


class MockState:
    """Mock state object."""
    
    def __init__(self, pixels, forest, ui_elements):
        self.pixels = pixels
        self.forest = forest
        self.ui_elements = ui_elements


class MockUIElement:
    """Mock UI element."""
    
    def __init__(self, element_id: str, element_type: str, text: str, bbox: tuple):
        self.element_id = element_id
        self.element_type = element_type
        self.text = text
        self.bbox = bbox  # (x, y, width, height)
        
    def __dict__(self):
        return {
            "element_id": self.element_id,
            "element_type": self.element_type,
            "text": self.text,
            "bbox": self.bbox
        }


# Try to import real Android World classes
try:
    from android_world.env import interface
    from android_world.env import android_world_controller
    from android_world.agents import base_agent
    from android_world.env import json_action
    
    ANDROID_WORLD_AVAILABLE = True
    logger.info("Android World modules imported successfully")
    
except ImportError as e:
    logger.warning(f"Could not import Android World modules: {e}")
    logger.warning("Using mock implementations for development")
    ANDROID_WORLD_AVAILABLE = False
    
    # Create mock classes
    interface = type('MockInterface', (), {})
    android_world_controller = type('MockController', (), {})
    base_agent = type('MockBaseAgent', (), {})
    json_action = type('MockJsonAction', (), {})


class AndroidWorldAdapter:
    """Adapter class to integrate with Android World environment."""
    
    def __init__(self, task_name: str = "settings_wifi", use_mock: bool = False):
        """Initialize Android World adapter.
        
        Args:
            task_name: Name of the Android World task to run
            use_mock: Whether to use mock environment (for testing)
        """
        self.task_name = task_name
        self.use_mock = use_mock or not ANDROID_WORLD_AVAILABLE
        
        if self.use_mock:
            logger.info("Using mock Android environment")
            self.env = MockAndroidEnv(task_name)
        else:
            logger.info(f"Initializing real Android World environment for task: {task_name}")
            self.env = self._create_real_environment(task_name)
        
        self.action_history = []
        self.state_history = []
    
    def _create_real_environment(self, task_name: str):
        """Create a real Android World environment.
        
        Args:
            task_name: Task name for the environment
            
        Returns:
            Android World environment instance
        """
        try:
            # This would need to be implemented based on actual Android World setup
            # For now, return mock
            logger.warning("Real Android World environment creation not implemented, using mock")
            return MockAndroidEnv(task_name)
        except Exception as e:
            logger.error(f"Failed to create real Android environment: {e}")
            return MockAndroidEnv(task_name)
    
    def reset(self, go_home: bool = False):
        """Reset the Android environment.
        
        Args:
            go_home: Whether to navigate to home screen on reset
            
        Returns:
            Initial state after reset
        """
        logger.info(f"Resetting Android environment (go_home={go_home})")
        return self.env.reset(go_home)
    
    def execute_action(self, action: Dict) -> Dict:
        """Execute an action in the Android environment.
        
        Args:
            action: Action to execute (e.g., {"action_type": "click", "x": 100, "y": 200})
            
        Returns:
            Result of action execution
        """
        logger.debug(f"Executing action: {action}")
        
        try:
            # Validate action format
            validated_action = self._validate_action(action)
            
            # Execute in environment
            result = self.env.step(validated_action)
            
            # Record action
            self.action_history.append({
                "action": validated_action,
                "timestamp": time.time(),
                "result": "success"
            })
            
            return {
                "success": True,
                "result": result,
                "action": validated_action
            }
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            
            self.action_history.append({
                "action": action,
                "timestamp": time.time(),
                "result": "error",
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    def get_current_state(self, wait_to_stabilize: bool = True) -> Dict:
        """Get current state of the Android environment.
        
        Args:
            wait_to_stabilize: Whether to wait for UI to stabilize
            
        Returns:
            Current state information
        """
        try:
            state = self.env.get_state(wait_to_stabilize)
            
            # Convert state to dictionary format
            state_dict = {
                "ui_elements": self._extract_ui_elements(state),
                "screen_size": self._get_screen_size(state),
                "timestamp": time.time(),
                "raw_state": state
            }
            
            self.state_history.append(state_dict)
            return state_dict
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {
                "error": str(e),
                "ui_elements": [],
                "screen_size": [1080, 1920],
                "timestamp": time.time()
            }
    
    def _validate_action(self, action: Dict) -> Dict:
        """Validate and normalize action format.
        
        Args:
            action: Raw action dictionary
            
        Returns:
            Validated action dictionary
        """
        action_type = action.get("action_type", "unknown")
        
        if action_type == "click":
            if "x" not in action or "y" not in action:
                raise ValueError("Click action requires 'x' and 'y' coordinates")
            return {
                "action_type": "click",
                "x": int(action["x"]),
                "y": int(action["y"])
            }
        
        elif action_type == "input_text":
            if "text" not in action:
                raise ValueError("Input text action requires 'text' field")
            return {
                "action_type": "input_text",
                "text": str(action["text"]),
                "x": int(action.get("x", 540)),  # Default to center
                "y": int(action.get("y", 960))
            }
        
        elif action_type == "swipe":
            return {
                "action_type": "swipe",
                "direction": action.get("direction", "up"),
                "x": int(action.get("x", 540)),
                "y": int(action.get("y", 960))
            }
        
        elif action_type == "scroll":
            return {
                "action_type": "scroll",
                "direction": action.get("direction", "down")
            }
        
        elif action_type in ["navigate_back", "navigate_home", "keyboard_enter"]:
            return {"action_type": action_type}
        
        elif action_type == "long_press":
            if "x" not in action or "y" not in action:
                raise ValueError("Long press action requires 'x' and 'y' coordinates")
            return {
                "action_type": "long_press",
                "x": int(action["x"]),
                "y": int(action["y"])
            }
        
        elif action_type == "wait":
            return {
                "action_type": "wait",
                "duration": float(action.get("duration", 2.0))
            }
        
        else:
            logger.warning(f"Unknown action type: {action_type}")
            return action
    
    def _extract_ui_elements(self, state) -> List[Dict]:
        """Extract UI elements from state object.
        
        Args:
            state: State object from Android World
            
        Returns:
            List of UI element dictionaries
        """
        try:
            if hasattr(state, 'ui_elements'):
                elements = []
                for element in state.ui_elements:
                    if hasattr(element, '__dict__'):
                        elements.append(element.__dict__)
                    elif hasattr(element, 'element_id'):
                        elements.append({
                            "element_id": getattr(element, 'element_id', 'unknown'),
                            "element_type": getattr(element, 'element_type', 'unknown'),
                            "text": getattr(element, 'text', ''),
                            "bbox": getattr(element, 'bbox', (0, 0, 0, 0))
                        })
                    else:
                        elements.append({"raw": str(element)})
                return elements
            else:
                return []
        except Exception as e:
            logger.error(f"Error extracting UI elements: {e}")
            return []
    
    def _get_screen_size(self, state) -> List[int]:
        """Get screen size from state object.
        
        Args:
            state: State object from Android World
            
        Returns:
            List of [width, height]
        """
        try:
            if hasattr(state, 'pixels'):
                if hasattr(state.pixels, 'shape'):
                    return list(state.pixels.shape[:2])
                elif isinstance(state.pixels, str) and 'mock' in state.pixels:
                    return [1080, 1920]  # Default mock size
            return [1080, 1920]  # Default size
        except Exception as e:
            logger.error(f"Error getting screen size: {e}")
            return [1080, 1920]
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available Android World tasks.
        
        Returns:
            List of available task names
        """
        # Common Android World tasks
        tasks = [
            "settings_wifi",
            "clock_alarm", 
            "email_search",
            "contacts_add",
            "calendar_event",
            "notes_create",
            "camera_photo",
            "browser_search",
            "maps_navigation",
            "music_play"
        ]
        
        return tasks
    
    def switch_task(self, new_task_name: str):
        """Switch to a different Android World task.
        
        Args:
            new_task_name: Name of the new task to switch to
        """
        logger.info(f"Switching from task '{self.task_name}' to '{new_task_name}'")
        self.task_name = new_task_name
        
        if self.use_mock:
            self.env = MockAndroidEnv(new_task_name)
        else:
            self.env = self._create_real_environment(new_task_name)
        
        # Reset histories
        self.action_history = []
        self.state_history = []
    
    def get_task_info(self) -> Dict:
        """Get information about the current task.
        
        Returns:
            Dictionary with task information
        """
        return {
            "task_name": self.task_name,
            "environment_type": "mock" if self.use_mock else "real",
            "actions_executed": len(self.action_history),
            "states_captured": len(self.state_history),
            "android_world_available": ANDROID_WORLD_AVAILABLE
        }
    
    def create_test_scenario(self, task_description: str) -> Dict:
        """Create a test scenario configuration for a given task.
        
        Args:
            task_description: Description of what the test should accomplish
            
        Returns:
            Test scenario configuration
        """
        return {
            "name": f"Test: {task_description}",
            "task_description": task_description,
            "app_context": {
                "android_task": self.task_name,
                "environment": "mock" if self.use_mock else "real",
                "expected_elements": self._get_expected_ui_elements()
            },
            "test_config": {
                "max_steps": 20,
                "timeout": 300,
                "screenshot_enabled": True,
                "recovery_enabled": True
            }
        }
    
    def _get_expected_ui_elements(self) -> List[str]:
        """Get expected UI elements for the current task.
        
        Returns:
            List of expected UI element types/IDs
        """
        task_elements = {
            "settings_wifi": ["wifi_toggle", "wifi_settings", "network_list"],
            "clock_alarm": ["alarm_list", "add_alarm", "time_picker"],
            "email_search": ["search_bar", "email_list", "compose_button"],
            "contacts_add": ["add_contact", "name_field", "phone_field"],
            "calendar_event": ["calendar_view", "add_event", "date_picker"]
        }
        
        return task_elements.get(self.task_name, ["generic_ui_elements"])
    
    def cleanup(self):
        """Clean up resources and connections."""
        logger.info("Cleaning up Android World adapter")
        
        # Save session data if needed
        session_data = {
            "task_name": self.task_name,
            "total_actions": len(self.action_history),
            "total_states": len(self.state_history),
            "session_duration": time.time() - (self.action_history[0]["timestamp"] if self.action_history else time.time())
        }
        
        logger.info(f"Session completed: {session_data}")


def create_android_environment(task_name: str = "settings_wifi", use_mock: bool = False) -> AndroidWorldAdapter:
    """Factory function to create an Android World environment adapter.
    
    Args:
        task_name: Android World task to initialize
        use_mock: Whether to use mock environment
        
    Returns:
        AndroidWorldAdapter instance
    """
    logger.info(f"Creating Android environment for task: {task_name}")
    return AndroidWorldAdapter(task_name, use_mock)


def get_supported_actions() -> List[Dict]:
    """Get list of supported Android actions.
    
    Returns:
        List of action specifications
    """
    return [
        {
            "action_type": "click",
            "description": "Tap on a UI element",
            "required_params": ["x", "y"],
            "optional_params": []
        },
        {
            "action_type": "input_text", 
            "description": "Type text into a field",
            "required_params": ["text"],
            "optional_params": ["x", "y"]
        },
        {
            "action_type": "swipe",
            "description": "Swipe gesture in a direction",
            "required_params": ["direction"],
            "optional_params": ["x", "y"]
        },
        {
            "action_type": "scroll",
            "description": "Scroll content in a direction",
            "required_params": ["direction"],
            "optional_params": []
        },
        {
            "action_type": "long_press",
            "description": "Long press on a UI element",
            "required_params": ["x", "y"],
            "optional_params": []
        },
        {
            "action_type": "navigate_back",
            "description": "Press the back button",
            "required_params": [],
            "optional_params": []
        },
        {
            "action_type": "navigate_home",
            "description": "Go to home screen",
            "required_params": [],
            "optional_params": []
        },
        {
            "action_type": "keyboard_enter",
            "description": "Press enter key",
            "required_params": [],
            "optional_params": []
        },
        {
            "action_type": "wait",
            "description": "Wait for specified duration",
            "required_params": [],
            "optional_params": ["duration"]
        }
    ]
