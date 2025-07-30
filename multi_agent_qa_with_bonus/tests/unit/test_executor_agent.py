"""Unit tests for ExecutorAgent."""

import unittest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.planner_agent import SubgoalStep


class TestExecutorAgent(unittest.TestCase):
    """Test cases for ExecutorAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine_params = {'model': 'gpt-4o-mini', 'temperature': 0.7}
        self.executor = ExecutorAgent(self.engine_params)
        
        # Sample subgoal for testing
        self.sample_subgoal = SubgoalStep(
            id=1,
            description="Open the Settings application",
            action_type="tap",
            target_element="Settings app icon",
            expected_outcome="Settings app opens",
            verification_criteria="Settings screen is displayed",
            dependencies=[]
        )
        
        # Sample UI state
        self.sample_ui_state = {
            "ui_elements": [
                {"type": "button", "text": "Settings", "bounds": [100, 200, 200, 250]},
                {"type": "textview", "text": "Home Screen", "bounds": [0, 0, 1080, 100]}
            ],
            "screen_size": [1080, 1920]
        }
    
    def test_init(self):
        """Test ExecutorAgent initialization."""
        self.assertIsNotNone(self.executor)
        self.assertEqual(self.executor.execution_history, [])
        self.assertIsNone(self.executor.current_ui_state)
    
    def test_clean_json_response_single_object(self):
        """Test cleaning a single JSON object response."""
        response = '''```json
        {
            "action_type": "click",
            "coordinates": {"x": 100, "y": 200},
            "reasoning": "Click on button"
        }
        ```'''
        
        cleaned = self.executor._clean_json_response(response)
        parsed = json.loads(cleaned)
        
        self.assertEqual(parsed["action_type"], "click")
        self.assertEqual(parsed["coordinates"]["x"], 100)
    
    def test_clean_json_response_multiple_objects(self):
        """Test cleaning response with multiple JSON objects."""
        response = '''```json
        {
            "action_type": "wait",
            "duration": 2.0
        },
        {
            "action_type": "click",
            "coordinates": {"x": 100, "y": 200}
        }
        ```'''
        
        cleaned = self.executor._clean_json_response(response)
        parsed = json.loads(cleaned)
        
        # Should return only the first object
        self.assertEqual(parsed["action_type"], "wait")
        self.assertEqual(parsed["duration"], 2.0)
        self.assertNotIn("coordinates", parsed)
    
    def test_execute_mock_action(self):
        """Test mock action execution."""
        action_plan = {
            "action_type": "click",
            "coordinates": {"x": 100, "y": 200},
            "reasoning": "Test click"
        }
        
        result = self.executor._execute_mock_action(action_plan)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "mock_click")
        self.assertTrue(result["mock_response"])
    
    def test_create_fallback_action(self):
        """Test fallback action creation."""
        fallback = self.executor._create_fallback_action(self.sample_subgoal, self.sample_ui_state)
        
        self.assertIn("action_type", fallback)
        self.assertEqual(fallback["reasoning"], "Fallback action due to planning failure")
        self.assertEqual(fallback["confidence"], 0.1)
    
    def test_execute_subgoal_mock(self):
        """Test subgoal execution with mock environment."""
        # Execute without real Android environment (should use mock)
        result = self.executor.execute_subgoal(self.sample_subgoal, self.sample_ui_state)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.step_id, 1)
        self.assertIsNotNone(result.action_taken)
        self.assertGreater(result.execution_time, 0)
    
    def test_get_execution_summary_empty(self):
        """Test execution summary when no executions have occurred."""
        summary = self.executor.get_execution_summary()
        
        self.assertEqual(summary["total_executions"], 0)
        self.assertEqual(summary["success_rate"], 0.0)
    
    def test_get_execution_summary_with_history(self):
        """Test execution summary with execution history."""
        # Add some mock execution results
        result1 = ExecutionResult(
            success=True, step_id=1, action_taken="mock_click",
            ui_state_before={}, ui_state_after={}, execution_time=1.0
        )
        result2 = ExecutionResult(
            success=False, step_id=2, action_taken="mock_input",
            ui_state_before={}, ui_state_after={}, execution_time=2.0,
            error_message="Mock error"
        )
        
        self.executor.execution_history = [result1, result2]
        summary = self.executor.get_execution_summary()
        
        self.assertEqual(summary["total_executions"], 2)
        self.assertEqual(summary["successful_executions"], 1)
        self.assertEqual(summary["failed_executions"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["average_execution_time"], 1.5)


if __name__ == '__main__':
    unittest.main()
