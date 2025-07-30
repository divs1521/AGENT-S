"""Core orchestrator for the multi-agent QA system with Agent-S and Android World integration."""

import logging
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import uuid

from agents.planner_agent import PlannerAgent, QAPlan, SubgoalStep
from agents.executor_agent import ExecutorAgent, ExecutionResult
from agents.verifier_agent import VerifierAgent, VerificationResult
from agents.supervisor_agent import SupervisorAgent, TestEpisode, SupervisorAnalysis
from .android_world_integration import android_world

logger = logging.getLogger(__name__)


@dataclass
class QASystemConfig:
    """Configuration for the QA system with Android World integration."""
    engine_params: Dict
    android_env: Optional[Any] = None
    android_world_config: Optional[Dict] = None  # Android World specific config
    use_real_android: bool = False  # Whether to use real Android environment
    max_execution_time: float = 300.0  # 5 minutes max per test
    max_retries: int = 3
    enable_visual_trace: bool = True
    log_directory: str = "logs"
    screenshot_directory: str = "screenshots" 
    enable_recovery: bool = True
    verification_strictness: str = "balanced"  # strict, balanced, lenient


class MultiAgentQAOrchestrator:
    """Main orchestrator that coordinates all QA agents."""
    
    def __init__(self, config: QASystemConfig, status_callback=None):
        """Initialize the multi-agent QA orchestrator with Android World support.
        
        Args:
            config: Configuration for the QA system
            status_callback: Optional callback function for status updates
        """
        self.config = config
        self.status_callback = status_callback  # For real-time status updates
        
        # Initialize Android World environment if requested
        if config.use_real_android and android_world.is_available():
            logger.info("Initializing Android World environment...")
            android_config = config.android_world_config or {}
            self.android_env = android_world.create_environment(android_config)
            if self.android_env:
                self.android_interface = android_world.create_interface(self.android_env)
                logger.info("✅ Android World environment initialized")
            else:
                logger.warning("⚠️ Failed to initialize Android World, falling back to mock")
                self.android_env = None
                self.android_interface = None
        else:
            logger.info("Using mock Android environment")
            self.android_env = None
            self.android_interface = None
        
        # Initialize agents
        logger.info("Initializing QA agents...")
        self.planner = PlannerAgent(config.engine_params)
        self.executor = ExecutorAgent(config.engine_params, self.android_env)
        self.verifier = VerifierAgent(config.engine_params)
        self.supervisor = SupervisorAgent(config.engine_params)
        
        # System state
        self.current_episode: Optional[TestEpisode] = None
        self.episode_history: List[TestEpisode] = []
        
        # Setup logging and directories
        self._setup_directories()
        
        logger.info("Multi-agent QA orchestrator initialized successfully")
        
    def _update_agent_status(self, agent_name: str, status: str, progress: int = 0, message: str = "", decision: str = ""):
        """Update agent status via callback if available."""
        if self.status_callback:
            self.status_callback(agent_name, status, progress, message, decision)
        
        # Log with decision if provided
        log_message = f"[{agent_name}] Status: {status} ({progress}%) - {message}"
        if decision:
            log_message += f" | Decision: {decision}"
        logger.info(log_message)
    
    def run_qa_test(
        self, 
        task_description: str, 
        app_context: Optional[Dict] = None,
        test_config: Optional[Dict] = None
    ) -> TestEpisode:
        """Run a complete QA test using all agents.
        
        Args:
            task_description: High-level description of what to test
            app_context: Additional context about the app being tested
            test_config: Optional test-specific configuration
            
        Returns:
            Complete test episode with results
        """
        episode_id = str(uuid.uuid4())
        logger.info(f"Starting QA test episode {episode_id}: {task_description}")
        
        start_time = time.time()
        
        try:
            # Phase 1: Planning
            self._update_agent_status("planner", "planning", 10, "Analyzing task requirements", "")
            logger.info("Phase 1: Creating test plan")
            plan = self.planner.create_plan(task_description, app_context)
            
            plan_decision = f"Created {len(plan.subgoals)} step plan: {plan.description}"
            self._update_agent_status("planner", "completed", 100, "Test plan created successfully", plan_decision)
            
            # Phase 2: Execution and Verification
            self._update_agent_status("executor", "ready", 0, "Preparing to execute test plan")
            logger.info("Phase 2: Executing test plan with verification")
            execution_results, verification_results, visual_trace = self._execute_plan_with_verification(plan)
            
            # Phase 3: Episode completion
            end_time = time.time()
            
            # Calculate overall success and score
            overall_success = self._calculate_overall_success(execution_results, verification_results)
            final_score = self._calculate_final_score(execution_results, verification_results)
            
            # Create episode
            episode = TestEpisode(
                episode_id=episode_id,
                plan=plan,
                execution_results=execution_results,
                verification_results=verification_results,
                visual_trace=visual_trace,
                start_time=start_time,
                end_time=end_time,
                overall_success=overall_success,
                final_score=final_score
            )
            
            # Phase 4: Supervisor Analysis
            self._update_agent_status("supervisor", "analyzing", 90, "Analyzing test episode", "")
            logger.info("Phase 4: Analyzing episode with supervisor")
            try:
                analysis = self.supervisor.analyze_test_episode(episode, app_context)
                episode.supervisor_analysis = analysis
                
                supervisor_decision = f"Coverage: {analysis.test_coverage_score:.2f} | Accuracy: {analysis.bug_detection_accuracy:.2f} | Recovery: {analysis.agent_recovery_ability:.2f}"
                self._update_agent_status("supervisor", "completed", 100, "Episode analysis completed", supervisor_decision)
            except Exception as e:
                logger.error(f"Supervisor analysis failed: {e}")
                self._update_agent_status("supervisor", "error", 90, "Analysis failed", f"Error: {str(e)[:50]}")
            
            self.current_episode = episode
            self.episode_history.append(episode)
            
            # Log episode summary
            self._log_episode_summary(episode)
            
            logger.info(f"QA test episode completed. Success: {overall_success}, Score: {final_score:.2f}")
            return episode
            
        except Exception as e:
            logger.error(f"Error during QA test execution: {e}")
            
            # Create failed episode
            end_time = time.time()
            episode = TestEpisode(
                episode_id=episode_id,
                plan=QAPlan("failed", task_description, [], "Test failed", 0),
                execution_results=[],
                verification_results=[],
                visual_trace=[],
                start_time=start_time,
                end_time=end_time,
                overall_success=False,
                final_score=0.0
            )
            
            self.episode_history.append(episode)
            return episode
    
    def analyze_episode(self, episode: TestEpisode, app_context: Optional[Dict] = None) -> SupervisorAnalysis:
        """Analyze a test episode using the supervisor agent.
        
        Args:
            episode: Test episode to analyze
            app_context: Additional app context for analysis
            
        Returns:
            Comprehensive analysis from supervisor
        """
        logger.info(f"Analyzing episode {episode.episode_id}")
        return self.supervisor.analyze_test_episode(episode, app_context)
    
    def generate_comprehensive_report(
        self, 
        episodes: Optional[List[TestEpisode]] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """Generate a comprehensive evaluation report.
        
        Args:
            episodes: Episodes to include (defaults to all episodes)
            output_path: Path to save the report
            
        Returns:
            Comprehensive evaluation report
        """
        episodes = episodes or self.episode_history
        
        if not output_path:
            output_path = os.path.join(
                self.config.log_directory, 
                f"qa_report_{int(time.time())}.json"
            )
        
        return self.supervisor.generate_evaluation_report(episodes, output_path)
    
    def run_continuous_testing(
        self, 
        test_scenarios: List[Dict],
        max_iterations: int = 10,
        improvement_threshold: float = 0.1
    ) -> Dict:
        """Run continuous testing with iterative improvement.
        
        Args:
            test_scenarios: List of test scenarios to run
            max_iterations: Maximum number of improvement iterations
            improvement_threshold: Minimum improvement required to continue
            
        Returns:
            Summary of continuous testing results
        """
        logger.info(f"Starting continuous testing with {len(test_scenarios)} scenarios")
        
        iteration_results = []
        
        for iteration in range(max_iterations):
            logger.info(f"Continuous testing iteration {iteration + 1}/{max_iterations}")
            
            iteration_episodes = []
            
            # Run all test scenarios
            for i, scenario in enumerate(test_scenarios):
                logger.info(f"Running scenario {i + 1}/{len(test_scenarios)}: {scenario.get('name', 'Unnamed')}")
                
                episode = self.run_qa_test(
                    scenario["task_description"],
                    scenario.get("app_context"),
                    scenario.get("test_config")
                )
                
                iteration_episodes.append(episode)
            
            # Analyze all episodes
            analyses = []
            for episode in iteration_episodes:
                analysis = self.analyze_episode(episode, episode.plan.__dict__.get("app_context"))
                analyses.append(analysis)
            
            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(iteration_episodes, analyses)
            iteration_results.append({
                "iteration": iteration + 1,
                "episodes": len(iteration_episodes),
                "metrics": iteration_metrics,
                "timestamp": time.time()
            })
            
            # Check for improvement
            if iteration > 0:
                improvement = self._calculate_improvement(iteration_results[-2], iteration_results[-1])
                logger.info(f"Iteration {iteration + 1} improvement: {improvement:.3f}")
                
                if improvement < improvement_threshold:
                    logger.info("Improvement threshold not met, stopping continuous testing")
                    break
            
            # Apply improvements based on supervisor feedback
            if iteration < max_iterations - 1:  # Don't improve on last iteration
                self._apply_iterative_improvements(analyses)
        
        return {
            "total_iterations": len(iteration_results),
            "total_episodes": sum(r["episodes"] for r in iteration_results),
            "iteration_results": iteration_results,
            "final_metrics": iteration_results[-1]["metrics"] if iteration_results else {},
            "improvement_summary": self._summarize_improvements(iteration_results)
        }
    
    def _execute_plan_with_verification(self, plan: QAPlan) -> Tuple[List[ExecutionResult], List[VerificationResult], List[str]]:
        """Execute a plan with step-by-step verification.
        
        Args:
            plan: QA plan to execute
            
        Returns:
            Tuple of (execution_results, verification_results, visual_trace)
        """
        execution_results = []
        verification_results = []
        visual_trace = []
        completed_steps = []
        
        max_attempts = len(plan.subgoals) * self.config.max_retries
        attempts = 0
        
        total_steps = len(plan.subgoals)
        
        while len(completed_steps) < len(plan.subgoals) and attempts < max_attempts:
            attempts += 1
            
            # Find next step to execute
            current_step = self._get_next_executable_step(plan.subgoals, completed_steps)
            if not current_step:
                logger.warning("No executable steps found, breaking execution loop")
                break
            
            step_progress = int((len(completed_steps) / total_steps) * 100)
            
            # Execute step
            self._update_agent_status("executor", "executing", step_progress, 
                                    f"Executing step {current_step.id}: {current_step.description}", "")
            
            current_ui_state = self._get_current_ui_state()
            execution_result = self.executor.execute_subgoal(current_step, current_ui_state)
            execution_results.append(execution_result)
            
            execution_decision = f"Action: {execution_result.action_taken} | Success: {execution_result.success}"
            if execution_result.error_message:
                execution_decision += f" | Error: {execution_result.error_message[:30]}"
            
            # Verify step
            verify_progress = int(((len(completed_steps) + 0.5) / total_steps) * 100)
            self._update_agent_status("verifier", "verifying", verify_progress,
                                    f"Verifying step {current_step.id}", "")
            
            verification_result = self.verifier.verify_execution(
                current_step, execution_result, self._get_current_ui_state(), None
            )
            verification_results.append(verification_result)
            
            verification_decision = f"Result: {'Pass' if verification_result.success else 'Fail'} | Confidence: {verification_result.confidence:.2f}"
            if verification_result.issues_found:
                verification_decision += f" | Issues: {len(verification_result.issues_found)}"
            
            # Handle verification result
            if verification_result.success:
                completed_steps.append(current_step.id)
                self._update_agent_status("verifier", "ready", verify_progress + 10, 
                                        f"Step {current_step.id} verified successfully", verification_decision)
            else:
                self._update_agent_status("verifier", "error", verify_progress, 
                                        f"Step {current_step.id} verification failed", verification_decision)
                
                if self.config.enable_recovery and verification_result.suggestions:
                    logger.info(f"Attempting recovery for step {current_step.id}")
                    # In a full implementation, we would retry with modified approach
                else:
                    logger.error(f"Step {current_step.id} failed without recovery")
            
            # Capture visual trace
            if self.config.enable_visual_trace:
                screenshot_path = self._capture_screenshot(f"step_{current_step.id}")
                if screenshot_path:
                    visual_trace.append(screenshot_path)
            
            # Small delay between steps
            time.sleep(0.5)
        
        # Mark execution as complete
        final_progress = 100 if len(completed_steps) == total_steps else 80
        self._update_agent_status("executor", "completed", final_progress, 
                                f"Execution completed: {len(completed_steps)}/{total_steps} steps")
        self._update_agent_status("verifier", "completed", 100, 
                                f"Verification completed: {len(completed_steps)}/{total_steps} steps")
        
        return execution_results, verification_results, visual_trace
    
    def _get_next_executable_step(self, subgoals: List[SubgoalStep], completed_steps: List[int]) -> Optional[SubgoalStep]:
        """Get the next executable step based on dependencies."""
        for subgoal in subgoals:
            if subgoal.id not in completed_steps:
                # Check if all dependencies are completed
                if all(dep in completed_steps for dep in subgoal.dependencies):
                    return subgoal
        return None
    
    def _get_current_ui_state(self) -> Dict:
        """Get current UI state from the Android environment.
        
        Returns:
            Dictionary containing current UI state
        """
        if self.config.android_env:
            try:
                if hasattr(self.config.android_env, 'get_state'):
                    state = self.config.android_env.get_state()
                    return {
                        "ui_elements": [elem.__dict__ if hasattr(elem, '__dict__') else str(elem) 
                                       for elem in state.ui_elements],
                        "screen_size": list(state.pixels.shape[:2]) if hasattr(state, 'pixels') else [1080, 1920],
                        "timestamp": time.time()
                    }
            except Exception as e:
                logger.error(f"Error getting UI state: {e}")
        
        # Return mock state if no environment or error
        return {
            "mock": True,
            "ui_elements": [],
            "screen_size": [1080, 1920],
            "timestamp": time.time()
        }
    
    def _capture_screenshot(self, step_name: str) -> str:
        """Capture a screenshot for the visual trace.
        
        Args:
            step_name: Name/identifier for this step
            
        Returns:
            Path to the saved screenshot
        """
        timestamp = int(time.time() * 1000)
        filename = f"{step_name}_{timestamp}.png"
        filepath = os.path.join(self.config.screenshot_directory, filename)
        
        try:
            if self.config.android_env and hasattr(self.config.android_env, 'get_state'):
                state = self.config.android_env.get_state()
                if hasattr(state, 'pixels'):
                    # Save screenshot (would need proper image saving implementation)
                    # For now, just create a placeholder file
                    with open(filepath, 'w') as f:
                        f.write(f"Screenshot placeholder: {step_name} at {timestamp}")
                    logger.debug(f"Screenshot saved: {filepath}")
                    return filepath
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
        
        # Return placeholder path if capture fails
        return f"screenshot_placeholder_{step_name}_{timestamp}"
    
    def _attempt_recovery(
        self, 
        subgoal: SubgoalStep, 
        execution_result: ExecutionResult, 
        verification_result: VerificationResult
    ) -> bool:
        """Attempt to recover from a failed step.
        
        Args:
            subgoal: Failed subgoal
            execution_result: Execution result
            verification_result: Verification result
            
        Returns:
            True if recovery was successful
        """
        logger.info(f"Attempting recovery for step {subgoal.id}")
        
        # Simple recovery strategies
        recovery_strategies = [
            "wait_and_retry",
            "navigate_back_and_retry",
            "refresh_and_retry"
        ]
        
        for strategy in recovery_strategies:
            try:
                if strategy == "wait_and_retry":
                    time.sleep(2)
                    # Retry execution
                    current_ui_state = self._get_current_ui_state()
                    retry_result = self.executor.execute_subgoal(subgoal, current_ui_state)
                    if retry_result.success:
                        return True
                
                elif strategy == "navigate_back_and_retry":
                    # Try going back and retrying
                    back_action = {"action_type": "navigate_back"}
                    if self.config.android_env and hasattr(self.config.android_env, 'step'):
                        self.config.android_env.step(back_action)
                        time.sleep(1)
                        
                        current_ui_state = self._get_current_ui_state()
                        retry_result = self.executor.execute_subgoal(subgoal, current_ui_state)
                        if retry_result.success:
                            return True
                
                # Add more recovery strategies as needed
                
            except Exception as e:
                logger.error(f"Recovery strategy {strategy} failed: {e}")
                continue
        
        logger.warning(f"All recovery strategies failed for step {subgoal.id}")
        return False
    
    def _calculate_overall_success(
        self, 
        execution_results: List[ExecutionResult], 
        verification_results: List[VerificationResult]
    ) -> bool:
        """Calculate overall success of the test episode.
        
        Args:
            execution_results: List of execution results
            verification_results: List of verification results
            
        Returns:
            True if overall test was successful
        """
        if not execution_results or not verification_results:
            return False
        
        execution_success_rate = len([r for r in execution_results if r.success]) / len(execution_results)
        verification_success_rate = len([v for v in verification_results if v.success]) / len(verification_results)
        
        # Consider test successful if both rates are above threshold
        return execution_success_rate >= 0.8 and verification_success_rate >= 0.8
    
    def _calculate_final_score(
        self, 
        execution_results: List[ExecutionResult], 
        verification_results: List[VerificationResult]
    ) -> float:
        """Calculate final score for the test episode.
        
        Args:
            execution_results: List of execution results
            verification_results: List of verification results
            
        Returns:
            Final score between 0.0 and 1.0
        """
        if not execution_results or not verification_results:
            return 0.0
        
        execution_score = len([r for r in execution_results if r.success]) / len(execution_results)
        verification_score = len([v for v in verification_results if v.success]) / len(verification_results)
        
        # Average confidence from verifications
        avg_confidence = sum(v.confidence for v in verification_results) / len(verification_results)
        
        # Weighted final score
        final_score = (execution_score * 0.4 + verification_score * 0.4 + avg_confidence * 0.2)
        
        return min(1.0, max(0.0, final_score))
    
    def _setup_directories(self):
        """Setup required directories for logging and screenshots."""
        directories = [
            self.config.log_directory,
            self.config.screenshot_directory
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    def _log_episode_summary(self, episode: TestEpisode):
        """Log a summary of the test episode.
        
        Args:
            episode: Test episode to log
        """
        summary = {
            "episode_id": episode.episode_id,
            "duration": episode.end_time - episode.start_time,
            "success": episode.overall_success,
            "score": episode.final_score,
            "plan_steps": len(episode.plan.subgoals),
            "executions": len(episode.execution_results),
            "verifications": len(episode.verification_results),
            "screenshots": len(episode.visual_trace)
        }
        
        log_path = os.path.join(self.config.log_directory, f"episode_{episode.episode_id}.json")
        try:
            with open(log_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Episode summary logged to {log_path}")
        except Exception as e:
            logger.error(f"Failed to log episode summary: {e}")
    
    def _calculate_iteration_metrics(
        self, 
        episodes: List[TestEpisode], 
        analyses: List[SupervisorAnalysis]
    ) -> Dict:
        """Calculate metrics for a continuous testing iteration.
        
        Args:
            episodes: Episodes from this iteration
            analyses: Supervisor analyses from this iteration
            
        Returns:
            Dictionary of iteration metrics
        """
        if not episodes:
            return {}
        
        return {
            "success_rate": len([e for e in episodes if e.overall_success]) / len(episodes),
            "average_score": sum(e.final_score for e in episodes) / len(episodes),
            "average_duration": sum(e.end_time - e.start_time for e in episodes) / len(episodes),
            "bug_detection_accuracy": sum(a.bug_detection_accuracy for a in analyses) / len(analyses) if analyses else 0,
            "recovery_ability": sum(a.agent_recovery_ability for a in analyses) / len(analyses) if analyses else 0,
            "coverage_score": sum(a.test_coverage_score for a in analyses) / len(analyses) if analyses else 0
        }
    
    def _calculate_improvement(self, prev_iteration: Dict, current_iteration: Dict) -> float:
        """Calculate improvement between iterations.
        
        Args:
            prev_iteration: Previous iteration results
            current_iteration: Current iteration results
            
        Returns:
            Improvement score (positive means improvement)
        """
        prev_metrics = prev_iteration["metrics"]
        current_metrics = current_iteration["metrics"]
        
        improvement_weights = {
            "success_rate": 0.3,
            "average_score": 0.3,
            "bug_detection_accuracy": 0.2,
            "recovery_ability": 0.1,
            "coverage_score": 0.1
        }
        
        total_improvement = 0
        for metric, weight in improvement_weights.items():
            if metric in prev_metrics and metric in current_metrics:
                improvement = current_metrics[metric] - prev_metrics[metric]
                total_improvement += improvement * weight
        
        return total_improvement
    
    def _apply_iterative_improvements(self, analyses: List[SupervisorAnalysis]):
        """Apply improvements based on supervisor analyses.
        
        Args:
            analyses: List of supervisor analyses to learn from
        """
        logger.info("Applying iterative improvements based on supervisor feedback")
        
        # Collect all improvement suggestions
        all_suggestions = []
        for analysis in analyses:
            all_suggestions.extend(analysis.prompt_improvement_suggestions)
            all_suggestions.extend(analysis.plan_quality_feedback)
            all_suggestions.extend(analysis.execution_quality_feedback)
            all_suggestions.extend(analysis.verification_quality_feedback)
        
        # For now, just log the suggestions
        # In a full implementation, this would update agent prompts and configurations
        logger.info(f"Collected {len(all_suggestions)} improvement suggestions")
        for i, suggestion in enumerate(all_suggestions[:5]):  # Log first 5
            logger.info(f"Improvement {i + 1}: {suggestion}")
    
    def _summarize_improvements(self, iteration_results: List[Dict]) -> Dict:
        """Summarize improvements across all iterations.
        
        Args:
            iteration_results: Results from all iterations
            
        Returns:
            Summary of improvements made
        """
        if len(iteration_results) < 2:
            return {"improvements": "insufficient_data"}
        
        first = iteration_results[0]["metrics"]
        last = iteration_results[-1]["metrics"]
        
        improvements = {}
        for metric in first.keys():
            if metric in last:
                improvement = last[metric] - first[metric]
                improvements[metric] = {
                    "absolute_change": improvement,
                    "relative_change": improvement / first[metric] if first[metric] != 0 else 0,
                    "trend": "improved" if improvement > 0 else "declined" if improvement < 0 else "stable"
                }
        
        return {
            "total_iterations": len(iteration_results),
            "metric_improvements": improvements,
            "overall_trend": "improving" if sum(imp["absolute_change"] for imp in improvements.values()) > 0 else "declining"
        }
    
    def get_system_status(self) -> Dict:
        """Get current status of the QA system.
        
        Returns:
            Dictionary with system status information
        """
        return {
            "total_episodes": len(self.episode_history),
            "current_episode": self.current_episode.episode_id if self.current_episode else None,
            "agent_status": {
                "planner": self.planner.get_plan_status(),
                "executor": self.executor.get_execution_summary(),
                "verifier": self.verifier.get_verification_summary(),
                "supervisor": self.supervisor.get_supervisor_summary()
            },
            "recent_performance": self._get_recent_performance(),
            "system_health": self._assess_system_health()
        }
    
    def _get_recent_performance(self) -> Dict:
        """Get performance metrics from recent episodes.
        
        Returns:
            Recent performance metrics
        """
        if not self.episode_history:
            return {"performance": "no_data"}
        
        recent_episodes = self.episode_history[-5:]  # Last 5 episodes
        
        return {
            "recent_success_rate": len([e for e in recent_episodes if e.overall_success]) / len(recent_episodes),
            "average_score": sum(e.final_score for e in recent_episodes) / len(recent_episodes),
            "average_duration": sum(e.end_time - e.start_time for e in recent_episodes) / len(recent_episodes),
            "episodes_analyzed": len(recent_episodes)
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health.
        
        Returns:
            String describing system health
        """
        if not self.episode_history:
            return "unknown"
        
        recent_performance = self._get_recent_performance()
        
        if recent_performance["recent_success_rate"] >= 0.8:
            return "healthy"
        elif recent_performance["recent_success_rate"] >= 0.6:
            return "acceptable"
        elif recent_performance["recent_success_rate"] >= 0.4:
            return "concerning"
        else:
            return "poor"
