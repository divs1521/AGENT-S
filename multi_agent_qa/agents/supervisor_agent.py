"""Supervisor Agent for multi-agent QA system."""

import logging
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from core.base_module import BaseQAModule
from .planner_agent import QAPlan, SubgoalStep
from .executor_agent import ExecutionResult
from .verifier_agent import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class TestEpisode:
    """Represents a complete test episode."""
    episode_id: str
    plan: QAPlan
    execution_results: List[ExecutionResult]
    verification_results: List[VerificationResult]
    visual_trace: List[str]  # Paths to screenshots
    start_time: float
    end_time: float
    overall_success: bool
    final_score: float


@dataclass
class SupervisorAnalysis:
    """Analysis result from supervisor agent."""
    episode_id: str
    overall_assessment: str
    bug_detection_accuracy: float
    agent_recovery_ability: float
    test_coverage_score: float
    prompt_improvement_suggestions: List[str]
    plan_quality_feedback: List[str]
    execution_quality_feedback: List[str]
    verification_quality_feedback: List[str]
    recommended_test_expansions: List[str]
    critical_issues: List[str]
    strengths: List[str]
    timestamp: float


class SupervisorAgent(BaseQAModule):
    """Agent responsible for reviewing test episodes and proposing improvements."""
    
    SUPERVISOR_SYSTEM_PROMPT = """
    You are an expert QA supervisor and test architect with deep knowledge of mobile app testing, 
    agent-based testing systems, and quality assurance best practices. Your role is to:
    
    1. Review complete test episodes and provide comprehensive analysis
    2. Evaluate the quality of planning, execution, and verification
    3. Assess bug detection accuracy and agent recovery capabilities
    4. Identify gaps in test coverage and suggest improvements
    5. Propose better prompts and strategies for each agent
    6. Recommend test scenario expansions
    7. Analyze visual traces and interaction patterns
    
    For each test episode review:
    - Analyze the planning quality and completeness
    - Evaluate execution accuracy and error handling
    - Assess verification thoroughness and accuracy
    - Review visual traces for missed opportunities
    - Identify patterns that could improve future tests
    - Suggest specific improvements for each agent
    - Recommend additional test scenarios
    
    Focus on:
    - Test coverage gaps (edge cases, error conditions, accessibility)
    - Agent coordination and communication effectiveness
    - Robustness to UI changes and unexpected states
    - Efficiency of test execution (time, steps, resource usage)
    - Quality of bug detection and reporting
    - User experience considerations
    - Maintainability and scalability of test approaches
    
    Provide actionable, specific recommendations that can directly improve the multi-agent system.
    """
    
    def __init__(self, engine_params: Dict, platform: str = "android"):
        """Initialize the Supervisor Agent.
        
        Args:
            engine_params: Configuration parameters for the LLM engine
            platform: Operating system platform
        """
        super().__init__(engine_params, platform)
        self.supervisor_agent = self._create_agent(self.SUPERVISOR_SYSTEM_PROMPT)
        self.episode_history: List[TestEpisode] = []
        self.analysis_history: List[SupervisorAnalysis] = []
        
    def analyze_test_episode(
        self, 
        episode: TestEpisode,
        app_context: Optional[Dict] = None
    ) -> SupervisorAnalysis:
        """Analyze a complete test episode and provide comprehensive feedback.
        
        Args:
            episode: Complete test episode to analyze
            app_context: Additional context about the app being tested
            
        Returns:
            Comprehensive analysis with improvement suggestions
        """
        logger.info(f"Analyzing test episode: {episode.episode_id}")
        
        # Prepare episode summary for analysis
        episode_summary = self._create_episode_summary(episode)
        
        analysis_prompt = f"""
        Analyze this complete QA test episode and provide comprehensive feedback:
        
        EPISODE SUMMARY:
        {episode_summary}
        
        APP CONTEXT: {json.dumps(app_context or {}, indent=2)}
        
        Please provide a detailed analysis in JSON format:
        {{
            "overall_assessment": {{
                "grade": "A|B|C|D|F",
                "summary": "high-level assessment of the test episode",
                "key_findings": ["most important findings from the analysis"]
            }},
            "planning_analysis": {{
                "quality_score": 0.0-1.0,
                "strengths": ["what the planner did well"],
                "weaknesses": ["areas where planning could improve"],
                "missing_scenarios": ["test scenarios that should have been included"],
                "improved_prompts": ["suggestions for better planner prompts"]
            }},
            "execution_analysis": {{
                "accuracy_score": 0.0-1.0,
                "efficiency_score": 0.0-1.0,
                "error_handling_score": 0.0-1.0,
                "strengths": ["what the executor did well"],
                "issues": ["execution problems found"],
                "recovery_instances": ["times the executor recovered from errors"],
                "improved_strategies": ["suggestions for better execution"]
            }},
            "verification_analysis": {{
                "thoroughness_score": 0.0-1.0,
                "accuracy_score": 0.0-1.0,
                "bug_detection_effectiveness": 0.0-1.0,
                "false_positives": ["issues incorrectly flagged as problems"],
                "false_negatives": ["real issues that were missed"],
                "verification_gaps": ["things that should have been verified"],
                "improved_criteria": ["suggestions for better verification"]
            }},
            "agent_coordination": {{
                "effectiveness_score": 0.0-1.0,
                "communication_quality": ["how well agents shared information"],
                "coordination_issues": ["problems with agent interaction"],
                "improvement_suggestions": ["how to improve coordination"]
            }},
            "test_coverage_assessment": {{
                "coverage_score": 0.0-1.0,
                "covered_scenarios": ["scenarios that were well tested"],
                "coverage_gaps": ["important scenarios not covered"],
                "edge_cases_missed": ["edge cases that should be tested"],
                "accessibility_coverage": ["accessibility testing gaps"],
                "performance_coverage": ["performance testing gaps"]
            }},
            "bug_detection_analysis": {{
                "detection_accuracy": 0.0-1.0,
                "critical_bugs_found": ["serious bugs discovered"],
                "minor_issues_found": ["minor issues discovered"],
                "missed_opportunities": ["bugs that likely exist but weren't found"],
                "false_alarms": ["non-issues flagged as bugs"]
            }},
            "visual_trace_analysis": {{
                "ui_flow_assessment": "smooth|acceptable|problematic",
                "visual_inconsistencies": ["UI problems visible in screenshots"],
                "missed_visual_cues": ["visual elements that should have been noticed"],
                "ux_observations": ["user experience insights from visual trace"]
            }},
            "recommendations": {{
                "immediate_improvements": ["changes to implement right away"],
                "strategic_improvements": ["longer-term improvements to consider"],
                "additional_test_scenarios": ["new test cases to add"],
                "agent_prompt_improvements": {{
                    "planner": ["improved prompts for planner agent"],
                    "executor": ["improved prompts for executor agent"],
                    "verifier": ["improved prompts for verifier agent"]
                }},
                "system_architecture_suggestions": ["improvements to the multi-agent system"]
            }},
            "metrics_summary": {{
                "overall_score": 0.0-1.0,
                "planning_score": 0.0-1.0,
                "execution_score": 0.0-1.0,
                "verification_score": 0.0-1.0,
                "coverage_score": 0.0-1.0,
                "efficiency_score": 0.0-1.0
            }}
        }}
        
        Be specific, actionable, and focus on improvements that will make the system more effective.
        """
        
        self.supervisor_agent.add_message(analysis_prompt)
        response = self.supervisor_agent.generate_response()
        
        try:
            # Clean the response of markdown code blocks
            cleaned_response = self._clean_json_response(response)
            analysis_data = json.loads(cleaned_response)
            
            analysis = SupervisorAnalysis(
                episode_id=episode.episode_id,
                overall_assessment=analysis_data.get("overall_assessment", {}).get("summary", "Analysis completed"),
                bug_detection_accuracy=analysis_data.get("bug_detection_analysis", {}).get("detection_accuracy", 0.5),
                agent_recovery_ability=analysis_data.get("execution_analysis", {}).get("error_handling_score", 0.5),
                test_coverage_score=analysis_data.get("test_coverage_assessment", {}).get("coverage_score", 0.5),
                prompt_improvement_suggestions=self._extract_prompt_suggestions(analysis_data),
                plan_quality_feedback=analysis_data.get("planning_analysis", {}).get("improved_prompts", []),
                execution_quality_feedback=analysis_data.get("execution_analysis", {}).get("improved_strategies", []),
                verification_quality_feedback=analysis_data.get("verification_analysis", {}).get("improved_criteria", []),
                recommended_test_expansions=analysis_data.get("recommendations", {}).get("additional_test_scenarios", []),
                critical_issues=analysis_data.get("bug_detection_analysis", {}).get("critical_bugs_found", []),
                strengths=self._extract_strengths(analysis_data),
                timestamp=time.time()
            )
            
            # Store detailed analysis
            analysis.detailed_analysis = analysis_data
            
            self.analysis_history.append(analysis)
            logger.info(f"Analysis completed. Overall score: {analysis_data.get('metrics_summary', {}).get('overall_score', 'N/A')}")
            
            return analysis
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse supervisor analysis: {e}")
            
            # Log a cleaner version of the response for debugging
            cleaned_response = self._clean_json_response(response)
            logger.error("=== SUPERVISOR ANALYSIS PARSING ERROR ===")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Response length: {len(response)} characters")
            logger.error(f"Cleaned response length: {len(cleaned_response)} characters")
            logger.error("=== RAW RESPONSE (first 500 chars) ===")
            logger.error(response[:500] + "..." if len(response) > 500 else response)
            logger.error("=== END OF RESPONSE ===")
            
            return self._create_fallback_analysis(episode)
    
    def generate_evaluation_report(
        self, 
        episodes: List[TestEpisode],
        output_path: Optional[str] = None
    ) -> Dict:
        """Generate a comprehensive evaluation report for multiple episodes.
        
        Args:
            episodes: List of test episodes to include in report
            output_path: Optional path to save the report
            
        Returns:
            Comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {len(episodes)} episodes")
        
        report_prompt = f"""
        Generate a comprehensive evaluation report for this multi-agent QA system based on multiple test episodes:
        
        EPISODES SUMMARY:
        Total Episodes: {len(episodes)}
        Overall Success Rate: {len([e for e in episodes if e.overall_success]) / len(episodes) if episodes else 0:.2f}
        Average Episode Duration: {sum((e.end_time - e.start_time) for e in episodes) / len(episodes) if episodes else 0:.2f} seconds
        
        DETAILED EPISODE DATA:
        {json.dumps([self._create_episode_summary(episode) for episode in episodes], indent=2)}
        
        HISTORICAL ANALYSIS:
        {json.dumps([{
            'episode_id': a.episode_id,
            'bug_detection_accuracy': a.bug_detection_accuracy,
            'recovery_ability': a.agent_recovery_ability,
            'coverage_score': a.test_coverage_score,
            'critical_issues': a.critical_issues
        } for a in self.analysis_history], indent=2)}
        
        Please provide a comprehensive report in JSON format:
        {{
            "executive_summary": {{
                "overall_grade": "A|B|C|D|F",
                "system_maturity": "early|developing|mature|production_ready",
                "key_achievements": ["major accomplishments of the system"],
                "critical_gaps": ["most important areas needing improvement"],
                "readiness_assessment": "assessment of production readiness"
            }},
            "performance_metrics": {{
                "bug_detection": {{
                    "average_accuracy": 0.0-1.0,
                    "improvement_trend": "improving|stable|declining",
                    "false_positive_rate": 0.0-1.0,
                    "false_negative_rate": 0.0-1.0
                }},
                "agent_recovery": {{
                    "average_ability": 0.0-1.0,
                    "improvement_trend": "improving|stable|declining",
                    "recovery_success_rate": 0.0-1.0
                }},
                "test_coverage": {{
                    "average_score": 0.0-1.0,
                    "coverage_consistency": "consistent|variable|inconsistent",
                    "gap_categories": ["types of coverage gaps found"]
                }},
                "efficiency": {{
                    "average_execution_time": "time in seconds",
                    "steps_per_test": "average number of steps",
                    "resource_utilization": "assessment of resource usage"
                }}
            }},
            "system_strengths": {{
                "planning": ["strengths of the planning agent"],
                "execution": ["strengths of the execution agent"],
                "verification": ["strengths of the verification agent"],
                "coordination": ["strengths of agent coordination"],
                "overall": ["overall system strengths"]
            }},
            "improvement_priorities": {{
                "high_priority": ["critical improvements needed immediately"],
                "medium_priority": ["important improvements for next iteration"],
                "low_priority": ["nice-to-have improvements"],
                "research_needed": ["areas requiring further research"]
            }},
            "comparative_analysis": {{
                "vs_manual_testing": {{
                    "advantages": ["where the system beats manual testing"],
                    "disadvantages": ["where manual testing is still better"],
                    "recommendations": ["how to best combine automated and manual"]
                }},
                "vs_traditional_automation": {{
                    "advantages": ["benefits over traditional test automation"],
                    "disadvantages": ["where traditional automation is better"],
                    "hybrid_approach": ["how to combine both approaches"]
                }}
            }},
            "future_development": {{
                "next_milestones": ["key development milestones"],
                "feature_additions": ["new features to add"],
                "architecture_evolution": ["how the system should evolve"],
                "scaling_considerations": ["how to scale the system"]
            }},
            "deployment_recommendations": {{
                "current_suitability": ["what this system is ready for now"],
                "prerequisites": ["what needs to be done before deployment"],
                "risk_mitigation": ["how to mitigate deployment risks"],
                "success_metrics": ["how to measure deployment success"]
            }}
        }}
        
        Be comprehensive, honest about limitations, and provide actionable recommendations.
        """
        
        self.supervisor_agent.add_message(report_prompt)
        response = self.supervisor_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            report_data = json.loads(cleaned_response)
            
            # Add metadata
            report_data["metadata"] = {
                "report_generated": time.time(),
                "episodes_analyzed": len(episodes),
                "total_analysis_history": len(self.analysis_history),
                "report_version": "1.0"
            }
            
            # Save report if path provided
            if output_path:
                self._save_report(report_data, output_path)
            
            return report_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse evaluation report: {e}")
            return {"error": str(e), "raw_response": response}
    
    def analyze_visual_trace(self, screenshot_paths: List[str], expected_flow: str) -> Dict:
        """Analyze visual traces from screenshots to identify UI flow issues.
        
        Args:
            screenshot_paths: List of paths to screenshot files
            expected_flow: Description of the expected UI flow
            
        Returns:
            Analysis of the visual trace
        """
        # Note: This would need to be enhanced with actual image processing
        # For now, we'll analyze based on file metadata and naming patterns
        
        visual_analysis_prompt = f"""
        Analyze this visual trace of UI interactions:
        
        Expected Flow: {expected_flow}
        Screenshots: {len(screenshot_paths)} images captured
        Screenshot Paths: {screenshot_paths}
        
        Based on the screenshot sequence, provide analysis in JSON format:
        {{
            "flow_assessment": {{
                "follows_expected_flow": true/false,
                "flow_efficiency": 0.0-1.0,
                "user_confusion_points": ["points where users might get confused"],
                "smooth_transitions": ["transitions that work well"],
                "jarring_transitions": ["transitions that are abrupt or confusing"]
            }},
            "ui_consistency": {{
                "consistent_design": true/false,
                "style_inconsistencies": ["design inconsistencies noticed"],
                "layout_issues": ["layout problems in the flow"],
                "accessibility_observations": ["accessibility issues visible"]
            }},
            "interaction_patterns": {{
                "effective_patterns": ["UI patterns that work well"],
                "problematic_patterns": ["UI patterns that cause issues"],
                "missing_affordances": ["UI cues that should be present"],
                "cognitive_load": "low|medium|high"
            }},
            "improvement_suggestions": {{
                "immediate_fixes": ["UI fixes that should be implemented"],
                "design_improvements": ["longer-term design improvements"],
                "interaction_improvements": ["better interaction patterns"],
                "accessibility_improvements": ["accessibility enhancements"]
            }}
        }}
        
        Note: This analysis is based on screenshot metadata. In a full implementation, 
        actual image analysis would provide more detailed insights.
        """
        
        self.supervisor_agent.add_message(visual_analysis_prompt)
        response = self.supervisor_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse visual analysis: {e}")
            return {"error": str(e), "analysis": "Visual analysis failed"}
    
    def suggest_prompt_improvements(self, agent_type: str, current_performance: Dict) -> List[str]:
        """Suggest specific improvements to agent prompts based on performance.
        
        Args:
            agent_type: Type of agent (planner, executor, verifier)
            current_performance: Current performance metrics for the agent
            
        Returns:
            List of specific prompt improvement suggestions
        """
        improvement_prompt = f"""
        Suggest specific improvements to the {agent_type} agent prompt based on performance data:
        
        Agent Type: {agent_type}
        Current Performance: {json.dumps(current_performance, indent=2)}
        
        Provide specific prompt improvements as JSON:
        {{
            "prompt_improvements": [
                {{
                    "issue": "specific performance issue to address",
                    "current_approach": "how the prompt currently handles this",
                    "improved_approach": "better way to handle this in the prompt",
                    "example_prompt_text": "example of improved prompt text",
                    "expected_improvement": "what performance improvement to expect"
                }}
            ],
            "additional_instructions": [
                "new instructions to add to the prompt"
            ],
            "removal_suggestions": [
                "parts of current prompt that should be removed"
            ],
            "context_improvements": [
                "better ways to provide context to the agent"
            ]
        }}
        
        Be specific and actionable in your suggestions.
        """
        
        self.supervisor_agent.add_message(improvement_prompt)
        response = self.supervisor_agent.generate_response()
        
        try:
            cleaned_response = self._clean_json_response(response)
            improvement_data = json.loads(cleaned_response)
            return improvement_data.get("prompt_improvements", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse prompt improvements: {e}")
            return []
    
    def _create_episode_summary(self, episode: TestEpisode) -> str:
        """Create a concise summary of a test episode.
        
        Args:
            episode: Test episode to summarize
            
        Returns:
            String summary of the episode
        """
        duration = episode.end_time - episode.start_time
        
        summary_parts = [
            f"Episode ID: {episode.episode_id}",
            f"Duration: {duration:.2f} seconds",
            f"Success: {episode.overall_success}",
            f"Final Score: {episode.final_score:.2f}",
            f"Plan Steps: {len(episode.plan.subgoals)}",
            f"Execution Results: {len(episode.execution_results)}",
            f"Verification Results: {len(episode.verification_results)}",
            f"Screenshots: {len(episode.visual_trace)}"
        ]
        
        if episode.execution_results:
            successful_executions = len([r for r in episode.execution_results if r.success])
            summary_parts.append(f"Execution Success Rate: {successful_executions / len(episode.execution_results):.2f}")
        
        if episode.verification_results:
            successful_verifications = len([v for v in episode.verification_results if v.success])
            summary_parts.append(f"Verification Success Rate: {successful_verifications / len(episode.verification_results):.2f}")
        
        return "; ".join(summary_parts)
    
    def _extract_prompt_suggestions(self, analysis_data: Dict) -> List[str]:
        """Extract prompt improvement suggestions from analysis data.
        
        Args:
            analysis_data: Full analysis data
            
        Returns:
            List of prompt improvement suggestions
        """
        suggestions = []
        
        # Extract from different sections
        for agent_type in ["planner", "executor", "verifier"]:
            agent_suggestions = analysis_data.get("recommendations", {}).get("agent_prompt_improvements", {}).get(agent_type, [])
            suggestions.extend([f"{agent_type}: {s}" for s in agent_suggestions])
        
        return suggestions
    
    def _extract_strengths(self, analysis_data: Dict) -> List[str]:
        """Extract system strengths from analysis data.
        
        Args:
            analysis_data: Full analysis data
            
        Returns:
            List of system strengths
        """
        strengths = []
        
        # Extract from different sections
        sections = ["planning_analysis", "execution_analysis", "verification_analysis"]
        for section in sections:
            section_strengths = analysis_data.get(section, {}).get("strengths", [])
            strengths.extend(section_strengths)
        
        return strengths
    
    def _create_fallback_analysis(self, episode: TestEpisode) -> SupervisorAnalysis:
        """Create a fallback analysis when parsing fails.
        
        Args:
            episode: Test episode to analyze
            
        Returns:
            Fallback analysis result
        """
        logger.warning(f"Creating fallback analysis for episode {episode.episode_id}")
        
        # Simple heuristic analysis
        success_rate = len([r for r in episode.execution_results if r.success]) / max(len(episode.execution_results), 1)
        
        return SupervisorAnalysis(
            episode_id=episode.episode_id,
            overall_assessment="Fallback analysis due to parsing failure",
            bug_detection_accuracy=0.5,
            agent_recovery_ability=success_rate,
            test_coverage_score=0.5,
            prompt_improvement_suggestions=["Manual review recommended due to analysis failure"],
            plan_quality_feedback=["Review planning approach"],
            execution_quality_feedback=["Review execution accuracy"],
            verification_quality_feedback=["Review verification criteria"],
            recommended_test_expansions=["Consider additional test scenarios"],
            critical_issues=["Analysis parsing failed - manual review needed"],
            strengths=["System completed test execution"],
            timestamp=time.time()
        )
    
    def _save_report(self, report_data: Dict, output_path: str):
        """Save evaluation report to file.
        
        Args:
            report_data: Report data to save
            output_path: Path to save the report
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def get_supervisor_summary(self) -> Dict:
        """Get summary of supervisor analysis history.
        
        Returns:
            Summary of supervisor activities
        """
        if not self.analysis_history:
            return {
                "total_analyses": 0,
                "average_scores": {},
                "trends": "insufficient_data"
            }
        
        return {
            "total_analyses": len(self.analysis_history),
            "episodes_analyzed": len(set(a.episode_id for a in self.analysis_history)),
            "average_scores": {
                "bug_detection": sum(a.bug_detection_accuracy for a in self.analysis_history) / len(self.analysis_history),
                "recovery_ability": sum(a.agent_recovery_ability for a in self.analysis_history) / len(self.analysis_history),
                "coverage": sum(a.test_coverage_score for a in self.analysis_history) / len(self.analysis_history)
            },
            "common_issues": self._identify_common_issues(),
            "improvement_trends": self._analyze_improvement_trends()
        }
    
    def _identify_common_issues(self) -> List[str]:
        """Identify common issues across analyses.
        
        Returns:
            List of common issues found
        """
        all_issues = []
        for analysis in self.analysis_history:
            all_issues.extend(analysis.critical_issues)
        
        # Simple frequency analysis
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return issues that appear more than once
        return [issue for issue, count in issue_counts.items() if count > 1]
    
    def _analyze_improvement_trends(self) -> Dict:
        """Analyze improvement trends over time.
        
        Returns:
            Trend analysis results
        """
        if len(self.analysis_history) < 3:
            return {"trend": "insufficient_data"}
        
        recent = self.analysis_history[-3:]
        early = self.analysis_history[:3]
        
        metrics = ["bug_detection_accuracy", "agent_recovery_ability", "test_coverage_score"]
        trends = {}
        
        for metric in metrics:
            recent_avg = sum(getattr(a, metric) for a in recent) / len(recent)
            early_avg = sum(getattr(a, metric) for a in early) / len(early)
            
            if recent_avg > early_avg + 0.1:
                trends[metric] = "improving"
            elif recent_avg < early_avg - 0.1:
                trends[metric] = "declining"
            else:
                trends[metric] = "stable"
        
        return trends

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing markdown code blocks and extra formatting."""
        # Remove markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1]
        if "```" in response:
            response = response.split("```")[0]
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Handle incomplete JSON by attempting to fix common issues
        if response:
            # Find the last complete structure
            # Look for incomplete strings or objects at the end
            lines = response.split('\n')
            
            # Find the last line with content
            last_meaningful_line = -1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() and not lines[i].strip().endswith(','):
                    last_meaningful_line = i
                    break
            
            # If we found an incomplete line, try to fix it
            if last_meaningful_line >= 0:
                last_line = lines[last_meaningful_line].strip()
                
                # If line ends with incomplete string or incomplete object
                if last_line.endswith('"') and last_line.count('"') % 2 == 1:
                    # Complete the incomplete string
                    lines[last_meaningful_line] = last_line + '"'
                    logger.warning("Fixed incomplete string at end of JSON")
                elif ':' in last_line and not last_line.endswith('}') and not last_line.endswith(']'):
                    # Incomplete value, add a placeholder
                    if last_line.rstrip().endswith(':'):
                        lines[last_meaningful_line] = last_line + ' "incomplete"'
                        logger.warning("Added placeholder value for incomplete JSON property")
                
                # Remove any lines after the fixed line that might be incomplete
                lines = lines[:last_meaningful_line + 1]
                response = '\n'.join(lines)
            
            # Count opening and closing braces
            open_braces = response.count('{')
            close_braces = response.count('}')
            
            # If we have unmatched opening braces, try to close them
            if open_braces > close_braces:
                missing_braces = open_braces - close_braces
                response += '}' * missing_braces
                logger.warning(f"Added {missing_braces} closing braces to incomplete JSON")
            
            # Handle unterminated strings by looking for quotes
            quote_count = response.count('"')
            if quote_count % 2 != 0:
                # Odd number of quotes means unterminated string
                response += '"'
                logger.warning("Added closing quote to incomplete JSON string")
        
        return response
