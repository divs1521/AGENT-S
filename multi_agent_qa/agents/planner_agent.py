"""Planner Agent for multi-agent QA system."""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from core.base_module import BaseQAModule

logger = logging.getLogger(__name__)


@dataclass
class SubgoalStep:
    """Represents a single subgoal step in the QA plan."""
    id: int
    description: str
    action_type: str  # e.g., "navigate", "tap", "verify", "input"
    target_element: Optional[str] = None
    expected_outcome: Optional[str] = None
    verification_criteria: Optional[str] = None
    dependencies: Optional[List[int]] = None
    

@dataclass
class QAPlan:
    """Represents the complete QA test plan."""
    task_id: str
    description: str
    subgoals: List[SubgoalStep]
    success_criteria: str
    estimated_steps: int
    

class PlannerAgent(BaseQAModule):
    """Agent responsible for parsing high-level QA goals and decomposing them into subgoals."""
    
    PLANNER_SYSTEM_PROMPT = """
    You are an expert QA planning agent for mobile applications. Your job is to analyze high-level test requirements 
    and break them down into detailed, actionable subgoals for testing Android applications.
    
    Your responsibilities:
    1. Parse the high-level QA goal into specific, testable subgoals
    2. Consider modal states, navigation flows, and potential edge cases
    3. Plan for verification steps after each action
    4. Adapt plans dynamically when unexpected states occur
    5. Ensure comprehensive test coverage
    
    For each subgoal, specify:
    - Clear description of what needs to be done
    - Action type (navigate, tap, input, verify, wait, etc.)
    - Target UI element (if applicable)
    - Expected outcome
    - Verification criteria
    - Dependencies on previous steps
    
    Output your plan as a structured JSON with clear subgoals that can be executed sequentially.
    Be thorough but efficient - aim for complete coverage without unnecessary redundancy.
    
    Consider Android-specific UI patterns like:
    - Navigation drawer, bottom navigation, tabs
    - Modal dialogs and popups
    - Permission requests
    - Keyboard interactions
    - Scroll behaviors
    - App state transitions
    """
    
    def __init__(self, engine_params: Dict, platform: str = "android"):
        """Initialize the Planner Agent.
        
        Args:
            engine_params: Configuration parameters for the LLM engine
            platform: Operating system platform
        """
        super().__init__(engine_params, platform)
        self.planner_agent = self._create_agent(self.PLANNER_SYSTEM_PROMPT)
        self.current_plan: Optional[QAPlan] = None
        self.execution_history: List[Dict] = []
        
    def create_plan(self, task_description: str, app_context: Optional[str] = None) -> QAPlan:
        """Create a QA test plan from high-level task description.
        
        Args:
            task_description: High-level description of what to test
            app_context: Additional context about the app being tested
            
        Returns:
            Complete QA plan with subgoals
        """
        logger.info(f"Creating plan for task: {task_description}")
        
        planning_prompt = f"""
        Create a detailed QA test plan for the following task:
        
        Task: {task_description}
        {f"App Context: {app_context}" if app_context else ""}
        
        Please provide a JSON response with the following structure:
        {{
            "task_id": "unique_identifier",
            "description": "clear description of the test",
            "subgoals": [
                {{
                    "id": 1,
                    "description": "specific action description",
                    "action_type": "navigate|tap|input|verify|wait|scroll",
                    "target_element": "UI element identifier or description",
                    "expected_outcome": "what should happen after this step",
                    "verification_criteria": "how to verify success",
                    "dependencies": [list of step IDs this depends on]
                }}
            ],
            "success_criteria": "overall success criteria for the test",
            "estimated_steps": number_of_steps
        }}
        
        Ensure each subgoal is specific, actionable, and includes verification steps.
        """
        
        self.planner_agent.add_message(planning_prompt)
        response = self.planner_agent.generate_response()
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = self._clean_json_response(response)
            plan_data = json.loads(cleaned_response)
            subgoals = [SubgoalStep(**step) for step in plan_data["subgoals"]]
            
            plan = QAPlan(
                task_id=plan_data["task_id"],
                description=plan_data["description"],
                subgoals=subgoals,
                success_criteria=plan_data["success_criteria"],
                estimated_steps=plan_data["estimated_steps"]
            )
            
            self.current_plan = plan
            logger.info(f"Created plan with {len(subgoals)} subgoals")
            return plan
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse plan response: {e}")
            logger.error(f"Response was: {response}")
            
            # Create a fallback simple plan
            return self._create_fallback_plan(task_description)
        
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract pure JSON.
        
        Removes markdown code blocks, extra whitespace, and other formatting.
        """
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
        
        return response

    def adapt_plan(self, current_step: int, issue_description: str, current_ui_state: Dict) -> QAPlan:
        """Adapt the current plan when issues are encountered.
        
        Args:
            current_step: Current step being executed
            issue_description: Description of the issue encountered
            current_ui_state: Current state of the UI
            
        Returns:
            Updated QA plan
        """
        if not self.current_plan:
            raise ValueError("No current plan to adapt")
            
        logger.info(f"Adapting plan at step {current_step} due to: {issue_description}")
        
        adaptation_prompt = f"""
        The current test execution encountered an issue and needs plan adaptation.
        
        Current Plan: {json.dumps(self._plan_to_dict(self.current_plan), indent=2)}
        Current Step: {current_step}
        Issue: {issue_description}
        Current UI State: {json.dumps(current_ui_state, indent=2)}
        
        Please provide an updated plan that:
        1. Handles the current issue appropriately
        2. Maintains the original test objectives
        3. Adds recovery steps if needed
        4. Updates subsequent steps as necessary
        
        Provide the response in the same JSON format as the original plan.
        """
        
        self.planner_agent.add_message(adaptation_prompt)
        response = self.planner_agent.generate_response()
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = self._clean_json_response(response)
            plan_data = json.loads(cleaned_response)
            subgoals = [SubgoalStep(**step) for step in plan_data["subgoals"]]
            
            adapted_plan = QAPlan(
                task_id=plan_data["task_id"],
                description=plan_data["description"],
                subgoals=subgoals,
                success_criteria=plan_data["success_criteria"],
                estimated_steps=plan_data["estimated_steps"]
            )
            
            self.current_plan = adapted_plan
            logger.info(f"Adapted plan with {len(subgoals)} subgoals")
            return adapted_plan
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse adapted plan: {e}")
            return self.current_plan  # Return original plan if adaptation fails
    
    def get_next_subgoal(self, completed_steps: List[int]) -> Optional[SubgoalStep]:
        """Get the next subgoal to execute.
        
        Args:
            completed_steps: List of completed step IDs
            
        Returns:
            Next subgoal to execute, or None if plan is complete
        """
        if not self.current_plan:
            return None
            
        for subgoal in self.current_plan.subgoals:
            if subgoal.id not in completed_steps:
                # Check if dependencies are satisfied
                if not subgoal.dependencies or all(dep in completed_steps for dep in subgoal.dependencies):
                    return subgoal
                    
        return None  # All steps completed or no executable steps
    
    def _create_fallback_plan(self, task_description: str) -> QAPlan:
        """Create a simple fallback plan when parsing fails.
        
        Args:
            task_description: Original task description
            
        Returns:
            Simple fallback QA plan
        """
        logger.warning("Creating fallback plan due to parsing failure")
        
        fallback_subgoals = [
            SubgoalStep(
                id=1,
                description=f"Navigate to app and prepare for: {task_description}",
                action_type="navigate",
                target_element="app_home",
                expected_outcome="App is ready for testing",
                verification_criteria="App UI is loaded and responsive"
            ),
            SubgoalStep(
                id=2,
                description=f"Execute main task: {task_description}",
                action_type="tap",
                target_element="main_action_element",
                expected_outcome="Task action is performed",
                verification_criteria="Expected UI change occurs",
                dependencies=[1]
            ),
            SubgoalStep(
                id=3,
                description="Verify task completion",
                action_type="verify",
                expected_outcome="Task completed successfully",
                verification_criteria="Success indicators are present",
                dependencies=[2]
            )
        ]
        
        return QAPlan(
            task_id="fallback_plan",
            description=task_description,
            subgoals=fallback_subgoals,
            success_criteria="Task completed without errors",
            estimated_steps=3
        )
    
    def _plan_to_dict(self, plan: QAPlan) -> Dict:
        """Convert QAPlan to dictionary for JSON serialization.
        
        Args:
            plan: QA plan to convert
            
        Returns:
            Dictionary representation of the plan
        """
        return {
            "task_id": plan.task_id,
            "description": plan.description,
            "subgoals": [
                {
                    "id": step.id,
                    "description": step.description,
                    "action_type": step.action_type,
                    "target_element": step.target_element,
                    "expected_outcome": step.expected_outcome,
                    "verification_criteria": step.verification_criteria,
                    "dependencies": step.dependencies
                }
                for step in plan.subgoals
            ],
            "success_criteria": plan.success_criteria,
            "estimated_steps": plan.estimated_steps
        }
    
    def log_execution_step(self, step_id: int, result: Dict):
        """Log the execution result of a step.
        
        Args:
            step_id: ID of the executed step
            result: Execution result data
        """
        self.execution_history.append({
            "step_id": step_id,
            "timestamp": result.get("timestamp"),
            "success": result.get("success", False),
            "details": result.get("details", "")
        })
        
    def get_plan_status(self) -> Dict:
        """Get current plan execution status.
        
        Returns:
            Dictionary with plan status information
        """
        if not self.current_plan:
            return {"status": "no_plan", "total_steps": 0, "completed_steps": 0}
            
        completed_steps = [entry["step_id"] for entry in self.execution_history if entry["success"]]
        
        return {
            "status": "in_progress" if len(completed_steps) < len(self.current_plan.subgoals) else "completed",
            "total_steps": len(self.current_plan.subgoals),
            "completed_steps": len(completed_steps),
            "current_step": len(completed_steps) + 1 if len(completed_steps) < len(self.current_plan.subgoals) else None,
            "success_rate": len([e for e in self.execution_history if e["success"]]) / max(len(self.execution_history), 1)
        }
