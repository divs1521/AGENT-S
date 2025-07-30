"""Executor Agent for multi-agent QA system."""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from core.base_module import BaseQAModule
from .planner_agent import SubgoalStep

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a subgoal."""
    success: bool
    step_id: int
    action_taken: str
    ui_state_before: Dict
    ui_state_after: Dict
    error_message: Optional[str] = None
    execution_time: float = 0.0
    screenshot_path: Optional[str] = None
    subgoal_description: Optional[str] = None


class ExecutorAgent(BaseQAModule):
    """Agent responsible for executing subgoals in the Android UI environment."""
    
    EXECUTOR_SYSTEM_PROMPT = """
    You are an expert executor agent for Android UI automation and testing. Your job is to:
    
    1. Analyze the current UI state and accessibility tree
    2. Execute specific subgoals with precise mobile gestures
    3. Handle UI elements like buttons, text fields, lists, etc.
    4. Navigate through Android apps efficiently
    5. Deal with dynamic UI states and modal dialogs
    6. Provide detailed execution feedback
    
    For each action you take:
    - Analyze the current UI hierarchy and elements
    - Select the most appropriate UI element for the action
    - Choose the correct gesture type (tap, long_press, swipe, input_text, etc.)
    - Handle edge cases like loading states, popups, and permissions
    - Provide clear feedback about what was accomplished
    
    Available Android actions:
    - click: Tap on UI elements
    - double_tap: Double tap on elements
    - long_press: Long press for context menus
    - input_text: Type text into fields
    - swipe: Swipe in directions (up, down, left, right)
    - scroll: Scroll through lists and content
    - navigate_back: Use back button
    - navigate_home: Go to home screen
    - keyboard_enter: Press enter key
    - open_app: Launch applications
    - wait: Wait for UI to stabilize
    
    Always ground your actions in the current UI state and provide specific element coordinates or identifiers.
    """
    
    def __init__(self, engine_params: Dict, android_env=None, platform: str = "android"):
        """Initialize the Executor Agent.
        
        Args:
            engine_params: Configuration parameters for the LLM engine
            android_env: Android environment for action execution
            platform: Operating system platform
        """
        super().__init__(engine_params, platform)
        self.executor_agent = self._create_agent(self.EXECUTOR_SYSTEM_PROMPT)
        self.android_env = android_env
        self.execution_history: List[ExecutionResult] = []
        self.current_ui_state: Optional[Dict] = None
        
    def set_android_env(self, android_env):
        """Set the Android environment for execution.
        
        Args:
            android_env: Android environment instance
        """
        self.android_env = android_env
        
    def execute_subgoal(self, subgoal: SubgoalStep, current_ui_state: Dict) -> ExecutionResult:
        """Execute a specific subgoal in the Android environment.
        
        Args:
            subgoal: Subgoal to execute
            current_ui_state: Current state of the Android UI
            
        Returns:
            Execution result with success status and details
        """
        logger.info(f"Executing subgoal {subgoal.id}: {subgoal.description}")
        
        start_time = time.time()
        ui_state_before = current_ui_state.copy()
        
        try:
            # Analyze current UI and plan action
            action_plan = self._analyze_and_plan_action(subgoal, current_ui_state)
            
            # Execute the planned action
            action_result = self._execute_action(action_plan)
            
            # Get updated UI state
            if self.android_env:
                time.sleep(1)  # Wait for UI to stabilize
                updated_state = self._get_current_ui_state()
            else:
                # Generate realistic mock UI state based on action
                updated_state = self._get_current_ui_state()
                
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=action_result["success"],
                step_id=subgoal.id,
                action_taken=action_result["action"],
                ui_state_before=ui_state_before,
                ui_state_after=updated_state,
                error_message=action_result.get("error"),
                execution_time=execution_time,
                screenshot_path=action_result.get("screenshot_path"),
                subgoal_description=subgoal.description
            )
            
            self.execution_history.append(result)
            logger.info(f"Execution completed in {execution_time:.2f}s, success: {result.success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing subgoal {subgoal.id}: {str(e)}")
            
            execution_time = time.time() - start_time
            result = ExecutionResult(
                success=False,
                step_id=subgoal.id,
                action_taken="error",
                ui_state_before=ui_state_before,
                ui_state_after=current_ui_state,
                error_message=str(e),
                execution_time=execution_time,
                subgoal_description=subgoal.description
            )
            
            self.execution_history.append(result)
            return result
    
    def _analyze_and_plan_action(self, subgoal: SubgoalStep, ui_state: Dict) -> Dict:
        """Analyze UI state and plan the specific action to take.
        
        Args:
            subgoal: Subgoal to execute
            ui_state: Current UI state
            
        Returns:
            Action plan with specific details
        """
        analysis_prompt = f"""
        Analyze the current Android UI state and plan the specific action needed to accomplish this subgoal:
        
        Subgoal: {subgoal.description}
        Action Type: {subgoal.action_type}
        Target Element: {subgoal.target_element}
        Expected Outcome: {subgoal.expected_outcome}
        
        Current UI State:
        UI Elements: {json.dumps(ui_state.get('ui_elements', []), indent=2)}
        Screen Size: {ui_state.get('screen_size', [1080, 1920])}
        
        Please analyze the UI and provide EXACTLY ONE action plan in JSON format (no arrays, no multiple objects):
        {{
            "action_type": "click|input_text|swipe|scroll|long_press|etc",
            "target_element_id": "specific element ID or index",
            "coordinates": {{"x": x_position, "y": y_position}},
            "text": "text to input (if applicable)",
            "direction": "swipe/scroll direction (if applicable)",
            "reasoning": "explanation of why this action was chosen",
            "confidence": 0.0-1.0
        }}
        
        IMPORTANT: Return ONLY the JSON object, no explanations, no multiple actions, no arrays.
        Base your decision on the actual UI elements present and the subgoal requirements.
        If the target element is not visible, suggest scrolling or navigation actions.
        """
        
        self.executor_agent.add_message(analysis_prompt)
        response = self.executor_agent.generate_response()
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = self._clean_json_response(response)
            action_plan = json.loads(cleaned_response)
            logger.info(f"Action planned: {action_plan['action_type']} with confidence {action_plan.get('confidence', 0.5)}")
            return action_plan
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse action plan: {e}")
            logger.error(f"Response was: {response}")
            
            # Return fallback action plan
            return self._create_fallback_action(subgoal, ui_state)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract pure JSON.
        
        Removes markdown code blocks, extra whitespace, and other formatting.
        Handles cases where multiple JSON objects are returned.
        """
        import json
        
        # Remove markdown code blocks
        if "```json" in response:
            # Extract content between ```json and ```
            start_marker = "```json"
            end_marker = "```"
            
            start_idx = response.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = response.find(end_marker, start_idx)
                if end_idx != -1:
                    response = response[start_idx:end_idx]
        
        # Remove any remaining backticks
        response = response.replace("```", "")
        
        # Remove JSON comments (// style)
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove inline comments
            if '//' in line:
                line = line.split('//')[0]
            cleaned_lines.append(line)
        response = '\n'.join(cleaned_lines)
        
        # Strip whitespace
        response = response.strip()
        
        # Handle multiple JSON objects - take the first valid one
        try:
            # Try to parse as-is first
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # If that fails, look for multiple JSON objects separated by commas
            # and try to extract just the first one
            try:
                # Find the first complete JSON object
                brace_count = 0
                first_json_end = -1
                in_string = False
                escape_next = False
                
                for i, char in enumerate(response):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                first_json_end = i + 1
                                break
                
                if first_json_end > 0:
                    first_json = response[:first_json_end]
                    # Validate it's proper JSON
                    json.loads(first_json)
                    return first_json
                
            except (json.JSONDecodeError, IndexError):
                pass
        
        return response

    def _execute_action(self, action_plan: Dict) -> Dict:
        """Execute the planned action in the Android environment.
        
        Args:
            action_plan: Planned action details
            
        Returns:
            Action execution result
        """
        if not self.android_env:
            # Enhanced mock execution for testing
            return self._execute_mock_action(action_plan)
        
        try:
            action_type = action_plan["action_type"]
            
            if action_type == "click":
                return self._execute_click(action_plan)
            elif action_type == "input_text":
                return self._execute_input_text(action_plan)
            elif action_type == "swipe":
                return self._execute_swipe(action_plan)
            elif action_type == "scroll":
                return self._execute_scroll(action_plan)
            elif action_type == "long_press":
                return self._execute_long_press(action_plan)
            elif action_type == "navigate_back":
                return self._execute_navigate_back()
            elif action_type == "navigate_home":
                return self._execute_navigate_home()
            elif action_type == "keyboard_enter":
                return self._execute_keyboard_enter()
            elif action_type == "wait":
                return self._execute_wait(action_plan)
            else:
                return {
                    "success": False,
                    "action": action_type,
                    "error": f"Unsupported action type: {action_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "action": action_plan.get("action_type", "unknown"),
                "error": str(e)
            }
    
    def _execute_mock_action(self, action_plan: Dict) -> Dict:
        """Enhanced mock execution that simulates realistic app behavior.
        
        Args:
            action_plan: Planned action details
            
        Returns:
            Mock action execution result
        """
        action_type = action_plan.get("action_type", "unknown")
        
        # Simulate different execution times for realism
        import random
        execution_delay = random.uniform(0.5, 2.0)
        time.sleep(execution_delay)
        
        # Map action types to more realistic mock actions
        action_mapping = {
            "click": "mock_click",
            "input_text": "mock_input_text", 
            "swipe": "mock_swipe",
            "scroll": "mock_scroll",
            "long_press": "mock_long_press",
            "navigate_back": "mock_navigate_back",
            "navigate_home": "mock_navigate_home", 
            "keyboard_enter": "mock_keyboard_enter",
            "wait": "mock_wait",
            "navigate": "mock_navigate",
            "tap": "mock_click",
            "input": "mock_input_text",
            "verify": "mock_wait"
        }
        
        mock_action = action_mapping.get(action_type, "mock_action")
        
        return {
            "success": True,
            "action": mock_action,
            "details": f"Mock execution of {action_type} completed successfully",
            "mock_response": True
        }
    
    def _execute_click(self, action_plan: Dict) -> Dict:
        """Execute a click action."""
        try:
            if "coordinates" in action_plan:
                x, y = action_plan["coordinates"]["x"], action_plan["coordinates"]["y"]
            else:
                # Use element index if available
                element_id = action_plan.get("target_element_id")
                if element_id:
                    # In a real implementation, we would get coordinates from element_id
                    x, y = self._get_element_coordinates(element_id)
                else:
                    raise ValueError("No coordinates or element ID provided for click")
            
            # Execute click in Android environment
            action = {
                "action_type": "click",
                "x": x,
                "y": y
            }
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": f"click at ({x}, {y})",
                "details": f"Clicked at coordinates ({x}, {y})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "click",
                "error": str(e)
            }
    
    def _execute_input_text(self, action_plan: Dict) -> Dict:
        """Execute a text input action."""
        try:
            text = action_plan.get("text", "")
            if not text:
                return {
                    "success": False,
                    "action": "input_text",
                    "error": "No text provided for input"
                }
            
            # First click on the text field if coordinates provided
            if "coordinates" in action_plan:
                click_action = {
                    "action_type": "click",
                    "x": action_plan["coordinates"]["x"],
                    "y": action_plan["coordinates"]["y"]
                }
                if hasattr(self.android_env, 'step'):
                    self.android_env.step(click_action)
                time.sleep(0.5)  # Wait for keyboard to appear
            
            # Input text
            text_action = {
                "action_type": "input_text",
                "text": text
            }
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(text_action)
            
            return {
                "success": True,
                "action": f"input_text: '{text}'",
                "details": f"Input text: {text}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "input_text",
                "error": str(e)
            }
    
    def _execute_swipe(self, action_plan: Dict) -> Dict:
        """Execute a swipe action."""
        try:
            direction = action_plan.get("direction", "up")
            start_coords = action_plan.get("start_coordinates", {"x": 540, "y": 960})
            
            action = {
                "action_type": "swipe",
                "direction": direction,
                "x": start_coords["x"],
                "y": start_coords["y"]
            }
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": f"swipe {direction}",
                "details": f"Swiped {direction} from ({start_coords['x']}, {start_coords['y']})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "swipe",
                "error": str(e)
            }
    
    def _execute_scroll(self, action_plan: Dict) -> Dict:
        """Execute a scroll action."""
        try:
            direction = action_plan.get("direction", "down")
            
            action = {
                "action_type": "scroll",
                "direction": direction
            }
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": f"scroll {direction}",
                "details": f"Scrolled {direction}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "scroll",
                "error": str(e)
            }
    
    def _execute_long_press(self, action_plan: Dict) -> Dict:
        """Execute a long press action."""
        try:
            if "coordinates" in action_plan:
                x, y = action_plan["coordinates"]["x"], action_plan["coordinates"]["y"]
            else:
                raise ValueError("No coordinates provided for long press")
            
            action = {
                "action_type": "long_press",
                "x": x,
                "y": y
            }
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": f"long_press at ({x}, {y})",
                "details": f"Long pressed at coordinates ({x}, {y})"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "long_press",
                "error": str(e)
            }
    
    def _execute_navigate_back(self) -> Dict:
        """Execute navigate back action."""
        try:
            action = {"action_type": "navigate_back"}
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": "navigate_back",
                "details": "Navigated back"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "navigate_back",
                "error": str(e)
            }
    
    def _execute_navigate_home(self) -> Dict:
        """Execute navigate home action."""
        try:
            action = {"action_type": "navigate_home"}
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": "navigate_home",
                "details": "Navigated to home"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "navigate_home",
                "error": str(e)
            }
    
    def _execute_keyboard_enter(self) -> Dict:
        """Execute keyboard enter action."""
        try:
            action = {"action_type": "keyboard_enter"}
            
            if hasattr(self.android_env, 'step'):
                self.android_env.step(action)
            
            return {
                "success": True,
                "action": "keyboard_enter",
                "details": "Pressed enter key"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "keyboard_enter",
                "error": str(e)
            }
    
    def _execute_wait(self, action_plan: Dict) -> Dict:
        """Execute wait action."""
        try:
            duration = action_plan.get("duration", 2.0)
            time.sleep(duration)
            
            return {
                "success": True,
                "action": f"wait {duration}s",
                "details": f"Waited for {duration} seconds"
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": "wait",
                "error": str(e)
            }
    
    def _get_current_ui_state(self) -> Dict:
        """Get current UI state from Android environment.
        
        Returns:
            Dictionary containing current UI state
        """
        if not self.android_env:
            # Enhanced mock UI state for better testing
            return self._generate_mock_ui_state()
        
        try:
            if hasattr(self.android_env, 'get_state'):
                state = self.android_env.get_state()
                return {
                    "ui_elements": [elem.__dict__ if hasattr(elem, '__dict__') else str(elem) 
                                   for elem in state.ui_elements],
                    "screen_size": list(state.pixels.shape[:2]) if hasattr(state, 'pixels') else [1080, 1920],
                    "timestamp": time.time()
                }
            else:
                return {"error": "Environment does not support get_state"}
                
        except Exception as e:
            logger.error(f"Error getting UI state: {e}")
            return {"error": str(e)}
    
    def _generate_mock_ui_state(self) -> Dict:
        """Generate realistic mock UI state for testing.
        
        Returns:
            Mock UI state with simulated elements
        """
        import random
        
        # Simulate some UI elements based on execution history
        mock_elements = []
        
        # Add some basic elements that would be present in a knowledge test app
        if len(self.execution_history) > 0:
            last_action = self.execution_history[-1].action_taken
            
            if "navigate" in last_action or "click" in last_action:
                mock_elements.extend([
                    {"type": "button", "text": "Home", "bounds": [50, 100, 150, 150]},
                    {"type": "textview", "text": "Knowledge Test App", "bounds": [200, 50, 600, 100]},
                    {"type": "edittext", "hint": "Enter your question", "bounds": [100, 300, 900, 400]},
                    {"type": "button", "text": "Submit", "bounds": [400, 450, 600, 500]}
                ])
            
            if "input" in last_action:
                mock_elements.extend([
                    {"type": "textview", "text": "Processing...", "bounds": [100, 550, 400, 600]},
                    {"type": "progressbar", "bounds": [100, 610, 900, 650]}
                ])
            
            if len(self.execution_history) > 3:  # Simulate results appearing
                mock_elements.extend([
                    {"type": "textview", "text": "Answer: Paris", "bounds": [100, 700, 800, 750]},
                    {"type": "textview", "text": "Paris is important because it is the capital and largest city of France...", "bounds": [100, 760, 900, 850]}
                ])
        
        return {
            "mock": True,
            "ui_elements": mock_elements,
            "screen_size": [1080, 1920],
            "timestamp": time.time(),
            "execution_step": len(self.execution_history)
        }
    
    def _get_element_coordinates(self, element_id: str) -> Tuple[int, int]:
        """Get coordinates for a UI element by ID.
        
        Args:
            element_id: Element identifier
            
        Returns:
            Tuple of (x, y) coordinates
        """
        # This would need to be implemented based on the actual UI element structure
        # For now, return center of screen as fallback
        logger.warning(f"Using fallback coordinates for element: {element_id}")
        return (540, 960)  # Center of typical phone screen
    
    def _create_fallback_action(self, subgoal: SubgoalStep, ui_state: Dict) -> Dict:
        """Create a fallback action when planning fails.
        
        Args:
            subgoal: Subgoal to execute
            ui_state: Current UI state
            
        Returns:
            Fallback action plan
        """
        logger.warning(f"Creating fallback action for subgoal: {subgoal.description}")
        
        # Simple mapping of action types to default actions
        action_mapping = {
            "navigate": {"action_type": "click", "coordinates": {"x": 100, "y": 100}},
            "tap": {"action_type": "click", "coordinates": {"x": 540, "y": 960}},
            "input": {"action_type": "input_text", "text": "test text", "coordinates": {"x": 540, "y": 960}},
            "verify": {"action_type": "wait", "duration": 1.0},
            "scroll": {"action_type": "scroll", "direction": "down"},
            "wait": {"action_type": "wait", "duration": 2.0}
        }
        
        fallback = action_mapping.get(subgoal.action_type, action_mapping["tap"])
        fallback["reasoning"] = "Fallback action due to planning failure"
        fallback["confidence"] = 0.1
        
        return fallback
    
    def get_execution_summary(self) -> Dict:
        """Get summary of all execution results.
        
        Returns:
            Summary of execution history
        """
        if not self.execution_history:
            return {"total_executions": 0, "success_rate": 0.0}
        
        successful = len([r for r in self.execution_history if r.success])
        total = len(self.execution_history)
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": total - successful,
            "success_rate": successful / total,
            "average_execution_time": sum(r.execution_time for r in self.execution_history) / total,
            "last_execution": self.execution_history[-1].__dict__ if self.execution_history else None
        }
