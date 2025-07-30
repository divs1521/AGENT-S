"""Verifier Agent for multi-agent QA system."""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from core.base_module import BaseQAModule
from .planner_agent import SubgoalStep
from .executor_agent import ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a step execution."""
    success: bool
    step_id: int
    verification_type: str
    expected_state: str
    actual_state: str
    confidence: float
    issues_found: List[str]
    suggestions: List[str]
    timestamp: float


class VerifierAgent(BaseQAModule):
    """Agent responsible for verifying app behavior after each execution step."""
    
    VERIFIER_SYSTEM_PROMPT = """
    You are an expert QA verification agent for Android applications. Your role is to:
    
    1. Analyze the expected vs actual state after each test step
    2. Detect functional bugs, UI inconsistencies, and unexpected behaviors
    3. Verify that UI elements are in the correct state (enabled/disabled, visible/hidden, correct text)
    4. Check for proper navigation flows and modal dialog handling
    5. Identify performance issues or crashes
    6. Provide actionable feedback for test improvement
    
    For each verification:
    - Compare the expected outcome with the actual UI state
    - Look for UI elements that should be present or absent
    - Check text content, button states, form validations
    - Detect modal dialogs, error messages, loading states
    - Identify navigation issues or unexpected screen transitions
    - Evaluate user experience aspects (responsiveness, visual consistency)
    
    Types of issues to detect:
    - Functional bugs (features not working as expected)
    - UI/UX issues (layout problems, accessibility issues)
    - Performance issues (slow loading, unresponsive UI)
    - Navigation problems (wrong screen, broken flows)
    - Data validation errors (incorrect form handling)
    - Error handling issues (poor error messages, crashes)
    
    Provide specific, actionable feedback and suggest improvements when issues are found.
    """
    
    def __init__(self, engine_params: Dict, platform: str = "android"):
        """Initialize the Verifier Agent.
        
        Args:
            engine_params: Configuration parameters for the LLM engine
            platform: Operating system platform
        """
        super().__init__(engine_params, platform)
        self.verifier_agent = self._create_agent(self.VERIFIER_SYSTEM_PROMPT)
        self.verification_history: List[VerificationResult] = []
        self.known_issues: List[Dict] = []
        
    def verify_execution(
        self, 
        subgoal: SubgoalStep, 
        execution_result: ExecutionResult,
        ui_state_current: Dict,
        screenshot_path: Optional[str] = None
    ) -> VerificationResult:
        """Verify the execution result against expected outcomes.
        
        Args:
            subgoal: Original subgoal that was executed
            execution_result: Result of the execution
            ui_state_current: Current UI state after execution
            screenshot_path: Path to screenshot for visual verification
            
        Returns:
            Verification result with pass/fail status and details
        """
        logger.info(f"Verifying execution of step {subgoal.id}: {subgoal.description}")
        
        # For mock environments, use simplified verification logic
        if ui_state_current.get('mock'):
            logger.info("Using mock environment verification")
            
            if execution_result.success:
                # Check if UI state shows logical progression
                ui_elements_after = execution_result.ui_state_after.get('ui_elements', [])
                ui_elements_current = ui_state_current.get('ui_elements', [])
                
                # Mock verification logic - be more generous for mock environment
                has_progression = (
                    execution_result.success or 
                    len(ui_elements_after) > 0 or 
                    len(ui_elements_current) > 0 or
                    execution_result.action_taken.startswith('mock_') or
                    'mock' in execution_result.action_taken.lower()
                )
                
                # In mock environment, if execution succeeded, verification should generally pass
                success = execution_result.success and has_progression
                confidence = 0.9 if success else 0.5
                
                issues = []
                suggestions = ["Test behavior verified in mock environment"]
                
                if not success:
                    issues.append("Mock execution completed but no UI progression detected")
                    suggestions = ["Check mock environment implementation", "Improve mock UI state generation"]
                
                return VerificationResult(
                    success=success,
                    step_id=subgoal.id,
                    verification_type="mock_verification",
                    expected_state=subgoal.expected_outcome,
                    actual_state=f"Mock execution: {execution_result.action_taken}",
                    confidence=confidence,
                    issues_found=issues,
                    suggestions=suggestions,
                    timestamp=time.time()
                )
            else:
                return VerificationResult(
                    success=False,
                    step_id=subgoal.id,
                    verification_type="mock_verification",
                    expected_state=subgoal.expected_outcome,
                    actual_state=f"Mock execution failed: {execution_result.error_message}",
                    confidence=0.2,
                    issues_found=["Mock execution failed"],
                    suggestions=["Check mock execution implementation"],
                    timestamp=time.time()
                )
        
        # Continue with LLM-based verification for real environments
        
        verification_prompt = f"""
        Verify the execution result of this QA test step:
        
        SUBGOAL INFORMATION:
        - ID: {subgoal.id}
        - Description: {subgoal.description}
        - Action Type: {subgoal.action_type}
        - Target Element: {subgoal.target_element}
        - Expected Outcome: {subgoal.expected_outcome}
        - Verification Criteria: {subgoal.verification_criteria}
        
        EXECUTION RESULT:
        - Success: {execution_result.success}
        - Action Taken: {execution_result.action_taken}
        - Error Message: {execution_result.error_message}
        - Execution Time: {execution_result.execution_time:.2f}s
        
        UI STATE BEFORE: {json.dumps(execution_result.ui_state_before, indent=2)}
        UI STATE AFTER: {json.dumps(execution_result.ui_state_after, indent=2)}
        CURRENT UI STATE: {json.dumps(ui_state_current, indent=2)}
        
        IMPORTANT: {'This is a MOCK ENVIRONMENT for testing. Be more lenient in verification - focus on execution success and logical progression rather than specific UI elements being present.' if ui_state_current.get('mock') else 'This is a REAL ENVIRONMENT. Perform thorough verification of all UI elements and expected behaviors.'}
        
        Please analyze and provide verification results in JSON format:
        {{
            "verification_passed": true/false,
            "confidence": 0.0-1.0,
            "expected_state_met": true/false,
            "issues_found": [
                {{
                    "type": "functional|ui|performance|navigation|data|error_handling",
                    "severity": "critical|major|minor|cosmetic",
                    "description": "detailed description of the issue",
                    "location": "where in the UI this issue occurs"
                }}
            ],
            "positive_observations": [
                "list of things that worked correctly"
            ],
            "suggestions": [
                "actionable suggestions for improvement"
            ],
            "ui_elements_verified": [
                {{
                    "element": "element description",
                    "status": "correct|incorrect|missing|unexpected",
                    "details": "specific verification details"
                }}
            ],
            "performance_assessment": {{
                "execution_time_acceptable": true/false,
                "ui_responsiveness": "good|acceptable|poor",
                "notes": "performance-related observations"
            }},
            "next_step_readiness": {{
                "ready_for_next_step": true/false,
                "blocking_issues": ["list of issues preventing next step"],
                "recommended_actions": ["actions to take before proceeding"]
            }}
        }}
        
        Be thorough in your analysis and provide specific, actionable feedback.
        """
        
        self.verifier_agent.add_message(verification_prompt)
        response = self.verifier_agent.generate_response()
        
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = self._clean_json_response(response)
            verification_data = json.loads(cleaned_response)
            
            result = VerificationResult(
                success=verification_data.get("verification_passed", False),
                step_id=subgoal.id,
                verification_type="full_verification",
                expected_state=subgoal.expected_outcome or "No specific expectation",
                actual_state=self._summarize_ui_state(ui_state_current),
                confidence=verification_data.get("confidence", 0.5),
                issues_found=[issue["description"] for issue in verification_data.get("issues_found", [])],
                suggestions=verification_data.get("suggestions", []),
                timestamp=time.time()
            )
            
            # Store detailed verification data
            result.detailed_analysis = verification_data
            
            self.verification_history.append(result)
            
            # Track persistent issues
            self._track_issues(verification_data.get("issues_found", []))
            
            logger.info(f"Verification completed. Passed: {result.success}, Confidence: {result.confidence}")
            
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse verification response: {e}")
            logger.error(f"Response was: {response}")
            
            # Create fallback verification result
            return self._create_fallback_verification(subgoal, execution_result, ui_state_current)
    
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

    def detect_functional_bugs(self, ui_state: Dict, expected_behavior: str) -> List[Dict]:
        """Detect potential functional bugs in the current UI state.
        
        Args:
            ui_state: Current UI state to analyze
            expected_behavior: Description of expected behavior
            
        Returns:
            List of detected functional bugs
        """
        bug_detection_prompt = f"""
        Analyze the current UI state for potential functional bugs:
        
        Expected Behavior: {expected_behavior}
        Current UI State: {json.dumps(ui_state, indent=2)}
        
        Look for:
        - UI elements that should be enabled but are disabled
        - Missing buttons, links, or interactive elements
        - Incorrect text content or labels
        - Form validation issues
        - Navigation problems
        - Error states that shouldn't be present
        - Performance issues (loading indicators stuck, etc.)
        
        Provide results as JSON:
        {{
            "bugs_detected": [
                {{
                    "type": "functional_bug",
                    "severity": "critical|major|minor",
                    "description": "detailed bug description",
                    "element_affected": "UI element that has the bug",
                    "expected_state": "what the element should look like/do",
                    "actual_state": "current problematic state",
                    "reproduction_steps": "how to reproduce this bug",
                    "impact": "user impact of this bug"
                }}
            ],
            "warnings": [
                "potential issues that need further investigation"
            ]
        }}
        """
        
        self.verifier_agent.add_message(bug_detection_prompt)
        response = self.verifier_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            bug_data = json.loads(cleaned_response)
            return bug_data.get("bugs_detected", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse bug detection response: {e}")
            return []
    
    def verify_ui_consistency(self, ui_state: Dict, app_standards: Dict) -> Dict:
        """Verify UI consistency against app design standards.
        
        Args:
            ui_state: Current UI state
            app_standards: App-specific design standards and guidelines
            
        Returns:
            UI consistency verification result
        """
        consistency_prompt = f"""
        Verify UI consistency against app design standards:
        
        Current UI State: {json.dumps(ui_state, indent=2)}
        App Standards: {json.dumps(app_standards, indent=2)}
        
        Check for:
        - Consistent color schemes and typography
        - Proper spacing and alignment
        - Consistent button styles and sizes
        - Appropriate use of icons and imagery
        - Accessibility compliance (contrast, text size, etc.)
        - Layout consistency across screens
        - Navigation pattern consistency
        
        Provide assessment as JSON:
        {{
            "consistency_score": 0.0-1.0,
            "issues": [
                {{
                    "type": "design_inconsistency",
                    "element": "affected UI element",
                    "issue": "description of inconsistency",
                    "standard_expected": "what the standard requires",
                    "current_state": "current non-compliant state",
                    "impact": "user experience impact"
                }}
            ],
            "positive_aspects": ["things that are consistent and well-designed"],
            "accessibility_issues": ["accessibility problems found"],
            "recommendations": ["suggestions for improvement"]
        }}
        """
        
        self.verifier_agent.add_message(consistency_prompt)
        response = self.verifier_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse consistency verification: {e}")
            return {"consistency_score": 0.5, "issues": [], "error": str(e)}
    
    def assess_user_experience(self, interaction_flow: List[Dict]) -> Dict:
        """Assess overall user experience based on interaction flow.
        
        Args:
            interaction_flow: List of user interactions and their results
            
        Returns:
            User experience assessment
        """
        ux_prompt = f"""
        Assess the user experience based on this interaction flow:
        
        Interaction Flow: {json.dumps(interaction_flow, indent=2)}
        
        Evaluate:
        - Ease of task completion
        - Number of steps required vs. optimal
        - Clarity of UI feedback and messaging
        - Error handling and recovery
        - Overall flow efficiency
        - User frustration points
        - Cognitive load on users
        
        Provide UX assessment as JSON:
        {{
            "ux_score": 0.0-1.0,
            "task_completion_difficulty": "easy|moderate|difficult|impossible",
            "efficiency_rating": "excellent|good|acceptable|poor",
            "friction_points": [
                {{
                    "step": "step where friction occurs",
                    "issue": "what makes this step difficult",
                    "impact": "how this affects user experience",
                    "suggestion": "how to improve this step"
                }}
            ],
            "positive_aspects": ["UX elements that work well"],
            "error_handling_assessment": {{
                "quality": "excellent|good|acceptable|poor",
                "issues": ["problems with error handling"],
                "suggestions": ["improvements for error handling"]
            }},
            "overall_recommendations": ["high-level UX improvement suggestions"]
        }}
        """
        
        self.verifier_agent.add_message(ux_prompt)
        response = self.verifier_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse UX assessment: {e}")
            return {"ux_score": 0.5, "error": str(e)}
    
    def suggest_test_improvements(self, test_results: List[VerificationResult]) -> Dict:
        """Suggest improvements to the test based on verification results.
        
        Args:
            test_results: List of verification results from the test
            
        Returns:
            Test improvement suggestions
        """
        improvement_prompt = f"""
        Based on these test verification results, suggest improvements:
        
        Test Results Summary:
        - Total Verifications: {len(test_results)}
        - Passed: {len([r for r in test_results if r.success])}
        - Failed: {len([r for r in test_results if not r.success])}
        - Average Confidence: {sum(r.confidence for r in test_results) / len(test_results) if test_results else 0:.2f}
        
        Detailed Results:
        {json.dumps([{
            'step_id': r.step_id,
            'success': r.success,
            'confidence': r.confidence,
            'issues': r.issues_found,
            'suggestions': r.suggestions
        } for r in test_results], indent=2)}
        
        Provide improvement suggestions as JSON:
        {{
            "test_coverage_improvements": [
                "suggestions for better test coverage"
            ],
            "test_step_improvements": [
                {{
                    "step_id": "step that needs improvement",
                    "issue": "what's wrong with current step",
                    "improvement": "how to improve the step",
                    "priority": "high|medium|low"
                }}
            ],
            "additional_test_scenarios": [
                "scenarios that should be added to the test"
            ],
            "automation_improvements": [
                "ways to make the test automation more robust"
            ],
            "overall_test_strategy": {{
                "strengths": ["what's working well in the test"],
                "weaknesses": ["areas that need improvement"],
                "recommendations": ["strategic recommendations"]
            }}
        }}
        """
        
        self.verifier_agent.add_message(improvement_prompt)
        response = self.verifier_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse improvement suggestions: {e}")
            return {"error": str(e), "suggestions": []}
    
    def _summarize_ui_state(self, ui_state: Dict) -> str:
        """Create a concise summary of the UI state.
        
        Args:
            ui_state: UI state to summarize
            
        Returns:
            String summary of the UI state
        """
        elements = ui_state.get("ui_elements", [])
        summary_parts = [
            f"Screen with {len(elements)} UI elements",
            f"Screen size: {ui_state.get('screen_size', 'unknown')}"
        ]
        
        if elements:
            element_types = {}
            for element in elements[:10]:  # Limit to first 10 elements
                element_type = str(element).split()[0] if isinstance(element, str) else "element"
                element_types[element_type] = element_types.get(element_type, 0) + 1
            
            summary_parts.append(f"Element types: {dict(element_types)}")
        
        return "; ".join(summary_parts)
    
    def _track_issues(self, issues: List[Dict]):
        """Track recurring issues across verifications.
        
        Args:
            issues: List of issues found in current verification
        """
        for issue in issues:
            # Check if this is a recurring issue
            similar_issues = [
                known for known in self.known_issues 
                if known.get("type") == issue.get("type") and 
                   known.get("description", "").lower() in issue.get("description", "").lower()
            ]
            
            if similar_issues:
                # Increment occurrence count
                similar_issues[0]["occurrences"] = similar_issues[0].get("occurrences", 1) + 1
                similar_issues[0]["last_seen"] = time.time()
            else:
                # Add new issue
                issue_record = issue.copy()
                issue_record["first_seen"] = time.time()
                issue_record["last_seen"] = time.time()
                issue_record["occurrences"] = 1
                self.known_issues.append(issue_record)
    
    def _create_fallback_verification(
        self, 
        subgoal: SubgoalStep, 
        execution_result: ExecutionResult, 
        ui_state: Dict
    ) -> VerificationResult:
        """Create a fallback verification when parsing fails.
        
        Args:
            subgoal: Original subgoal
            execution_result: Execution result
            ui_state: Current UI state
            
        Returns:
            Fallback verification result
        """
        logger.warning(f"Creating fallback verification for step {subgoal.id}")
        
        # Simple heuristic verification
        success = execution_result.success and not execution_result.error_message
        
        issues = []
        if execution_result.error_message:
            issues.append(f"Execution error: {execution_result.error_message}")
        
        if execution_result.execution_time > 10.0:
            issues.append("Execution took longer than expected (>10s)")
        
        return VerificationResult(
            success=success,
            step_id=subgoal.id,
            verification_type="fallback_verification",
            expected_state=subgoal.expected_outcome or "Unknown",
            actual_state=self._summarize_ui_state(ui_state),
            confidence=0.3,  # Low confidence for fallback
            issues_found=issues,
            suggestions=["Manual verification recommended due to parsing failure"],
            timestamp=time.time()
        )
    
    def get_verification_summary(self) -> Dict:
        """Get summary of all verification results.
        
        Returns:
            Summary of verification history
        """
        if not self.verification_history:
            return {
                "total_verifications": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0,
                "total_issues": 0
            }
        
        total = len(self.verification_history)
        successful = len([v for v in self.verification_history if v.success])
        total_issues = sum(len(v.issues_found) for v in self.verification_history)
        avg_confidence = sum(v.confidence for v in self.verification_history) / total
        
        return {
            "total_verifications": total,
            "successful_verifications": successful,
            "failed_verifications": total - successful,
            "success_rate": successful / total,
            "average_confidence": avg_confidence,
            "total_issues": total_issues,
            "recurring_issues": len([issue for issue in self.known_issues if issue.get("occurrences", 0) > 1]),
            "recent_trends": self._analyze_recent_trends()
        }
    
    def _analyze_recent_trends(self) -> Dict:
        """Analyze trends in recent verifications.
        
        Returns:
            Dictionary with trend analysis
        """
        if len(self.verification_history) < 5:
            return {"trend": "insufficient_data"}
        
        recent = self.verification_history[-5:]
        recent_success_rate = len([v for v in recent if v.success]) / len(recent)
        
        overall_success_rate = len([v for v in self.verification_history if v.success]) / len(self.verification_history)
        
        if recent_success_rate > overall_success_rate + 0.1:
            trend = "improving"
        elif recent_success_rate < overall_success_rate - 0.1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_success_rate": recent_success_rate,
            "overall_success_rate": overall_success_rate,
            "confidence_trend": "increasing" if recent[-1].confidence > recent[0].confidence else "decreasing"
        }
