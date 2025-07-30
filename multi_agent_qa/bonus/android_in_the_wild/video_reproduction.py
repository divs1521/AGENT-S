"""
Video Analysis and Reproduction Engine
Reproduces android_in_the_wild video flows using the multi-agent system.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import core system components (without disrupting them)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig
from bonus.android_in_the_wild.dataset_processor import VideoTrace, ComparisonResult

logger = logging.getLogger(__name__)

@dataclass
class ReproductionResult:
    """Results from reproducing a video trace with the multi-agent system."""
    original_video: str
    generated_task: str
    agent_actions: List[Dict]
    execution_success: bool
    comparison_result: ComparisonResult
    performance_metrics: Dict
    
class VideoReproductionEngine:
    """Reproduces android_in_the_wild video flows using multi-agent QA system."""
    
    def __init__(self, orchestrator: MultiAgentQAOrchestrator):
        self.orchestrator = orchestrator
        self.reproductions_completed = 0
        self.total_accuracy = 0.0
        
    def reproduce_video_flow(self, video_trace: VideoTrace) -> ReproductionResult:
        """Reproduce a video flow using the multi-agent system."""
        
        logger.info(f"Starting reproduction of video: {video_trace.video_path}")
        logger.info(f"Generated task: {video_trace.task_prompt}")
        
        try:
            # Execute the generated task using the multi-agent system
            results = self.orchestrator.run_qa_test(video_trace.task_prompt)
            
            # Extract agent actions from results
            agent_actions = self._extract_agent_actions(results)
            
            # Compare with ground truth
            comparison_result = self._compare_with_ground_truth(
                agent_actions, 
                video_trace.ground_truth_actions,
                video_trace.ui_traces
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                results, 
                comparison_result
            )
            
            reproduction_result = ReproductionResult(
                original_video=video_trace.video_path,
                generated_task=video_trace.task_prompt,
                agent_actions=agent_actions,
                execution_success=getattr(results, 'overall_success', False),
                comparison_result=comparison_result,
                performance_metrics=performance_metrics
            )
            
            self.reproductions_completed += 1
            self.total_accuracy += comparison_result.accuracy_score
            
            logger.info(f"Reproduction completed with accuracy: {comparison_result.accuracy_score:.2f}")
            return reproduction_result
            
        except Exception as e:
            logger.error(f"Failed to reproduce video flow: {e}")
            
            # Return failed reproduction result
            return ReproductionResult(
                original_video=video_trace.video_path,
                generated_task=video_trace.task_prompt,
                agent_actions=[],
                execution_success=False,
                comparison_result=ComparisonResult(0.0, 0.0, 0.0, {}, [], ["Execution failed"]),
                performance_metrics={"error": str(e)}
            )

    def reproduce_video_sequence(self, video_data: Dict, use_enhanced_training: bool = False, real_time_updates: bool = False) -> ReproductionResult:
        """
        Reproduce a video sequence from video data dictionary.
        This method provides compatibility with the Flask web interface.
        """
        try:
            # Convert video data to VideoTrace if needed
            if isinstance(video_data, dict):
                # Create a mock VideoTrace from the video data
                from bonus.android_in_the_wild.dataset_processor import VideoTrace, ComparisonResult
                
                video_trace = VideoTrace(
                    video_path=video_data.get('path', video_data.get('id', 'unknown')),
                    episode_id=video_data.get('id', 'unknown_episode'),
                    metadata=video_data.get('metadata', {}),
                    ui_traces=video_data.get('ui_states', []),
                    task_prompt=video_data.get('task_description', 'Navigate through Android interface'),
                    ground_truth_actions=video_data.get('actions', []),
                    app_package=video_data.get('metadata', {}).get('app_package', 'unknown'),
                    device_info=video_data.get('metadata', {})
                )
            else:
                video_trace = video_data
            
            # Use the main reproduction method
            return self.reproduce_video_flow(video_trace)
            
        except Exception as e:
            logger.error(f"Failed to reproduce video sequence: {e}")
            
            # Return failed result
            from bonus.android_in_the_wild.dataset_processor import ComparisonResult
            return ReproductionResult(
                original_video=str(video_data),
                generated_task="Failed to generate task",
                agent_actions=[],
                execution_success=False,
                comparison_result=ComparisonResult(0.0, 0.0, 0.0, {}, [], [f"Reproduction failed: {str(e)}"]),
                performance_metrics={"error": str(e)}
            )
    
    def _extract_agent_actions(self, results) -> List[Dict]:
        """Extract action sequence from orchestrator results."""
        actions = []
        
        # Extract from execution results
        if hasattr(results, 'execution_results'):
            for execution in results.execution_results:
                if hasattr(execution, 'action'):
                    action_data = {
                        'type': getattr(execution.action, 'action_type', 'unknown'),
                        'element_id': getattr(execution.action, 'element_id', ''),
                        'coordinates': getattr(execution.action, 'coordinates', [0, 0]),
                        'timestamp': getattr(execution, 'timestamp', 0.0),
                        'success': getattr(execution, 'success', False)
                    }
                    actions.append(action_data)
        
        return actions
    
    def _compare_with_ground_truth(self, 
                                  agent_actions: List[Dict], 
                                  ground_truth: List[Dict],
                                  ui_traces: List[Dict]) -> ComparisonResult:
        """Compare agent actions with ground truth video trace."""
        
        # Calculate accuracy metrics
        accuracy = self._calculate_action_accuracy(agent_actions, ground_truth)
        robustness = self._calculate_robustness_score(agent_actions, ui_traces)
        generalization = self._calculate_generalization_score(agent_actions, ground_truth)
        
        # Identify deviations
        failed_steps = self._identify_deviations(agent_actions, ground_truth)
        
        # Generate recommendations
        recommendations = self._generate_improvement_recommendations(failed_steps)
        
        detailed_metrics = {
            'action_sequence_similarity': accuracy,
            'timing_accuracy': self._calculate_timing_accuracy(agent_actions, ground_truth),
            'gesture_precision': self._calculate_gesture_precision(agent_actions, ground_truth),
            'ui_state_coverage': robustness,
            'error_handling': self._calculate_error_handling_score(agent_actions)
        }
        
        return ComparisonResult(
            accuracy_score=accuracy,
            robustness_score=robustness,
            generalization_score=generalization,
            detailed_metrics=detailed_metrics,
            failed_steps=failed_steps,
            recommendations=recommendations
        )
    
    def _calculate_action_accuracy(self, agent_actions: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate accuracy of action sequence."""
        if not ground_truth:
            return 1.0 if not agent_actions else 0.0
        
        if not agent_actions:
            return 0.0
        
        # Calculate similarity based on action types and targets
        matches = 0
        for i, gt_action in enumerate(ground_truth):
            if i < len(agent_actions):
                agent_action = agent_actions[i]
                if (agent_action.get('type') == gt_action.get('type') and
                    agent_action.get('element_id') == gt_action.get('element_id')):
                    matches += 1
        
        return matches / len(ground_truth)
    
    def _calculate_robustness_score(self, agent_actions: List[Dict], ui_traces: List[Dict]) -> float:
        """Calculate robustness based on handling of UI variations."""
        # Mock calculation - in reality would analyze adaptation to UI changes
        return 0.8 if agent_actions else 0.0
    
    def _calculate_generalization_score(self, agent_actions: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate generalization capability."""
        # Mock calculation - would analyze transferability across different contexts
        return 0.75 if agent_actions else 0.0
    
    def _calculate_timing_accuracy(self, agent_actions: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate timing accuracy between actions."""
        return 0.85  # Mock value
    
    def _calculate_gesture_precision(self, agent_actions: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate precision of gesture coordinates."""
        if not agent_actions or not ground_truth:
            return 0.0
        
        # Calculate coordinate accuracy
        total_precision = 0.0
        valid_comparisons = 0
        
        for i, gt_action in enumerate(ground_truth):
            if (i < len(agent_actions) and 
                'coordinates' in gt_action and 
                'coordinates' in agent_actions[i]):
                
                gt_coords = gt_action['coordinates']
                agent_coords = agent_actions[i]['coordinates']
                
                # Calculate distance accuracy (closer = better)
                distance = ((gt_coords[0] - agent_coords[0]) ** 2 + 
                           (gt_coords[1] - agent_coords[1]) ** 2) ** 0.5
                
                # Normalize to 0-1 scale (assume max acceptable distance is 100 pixels)
                precision = max(0.0, 1.0 - distance / 100.0)
                total_precision += precision
                valid_comparisons += 1
        
        return total_precision / valid_comparisons if valid_comparisons > 0 else 0.0
    
    def _calculate_error_handling_score(self, agent_actions: List[Dict]) -> float:
        """Calculate error handling capability."""
        # Mock calculation based on successful action completion
        if not agent_actions:
            return 0.0
        
        successful_actions = sum(1 for action in agent_actions if action.get('success', False))
        return successful_actions / len(agent_actions)
    
    def _identify_deviations(self, agent_actions: List[Dict], ground_truth: List[Dict]) -> List[Dict]:
        """Identify steps where agent deviated from ground truth."""
        deviations = []
        
        # Check sequence length
        if len(agent_actions) != len(ground_truth):
            deviations.append({
                'type': 'sequence_length_mismatch',
                'expected_steps': len(ground_truth),
                'actual_steps': len(agent_actions),
                'impact': 'high'
            })
        
        # Check individual action deviations
        min_length = min(len(agent_actions), len(ground_truth))
        for i in range(min_length):
            agent_action = agent_actions[i]
            gt_action = ground_truth[i]
            
            if agent_action.get('type') != gt_action.get('type'):
                deviations.append({
                    'type': 'action_type_mismatch',
                    'step': i,
                    'expected': gt_action.get('type'),
                    'actual': agent_action.get('type'),
                    'impact': 'medium'
                })
            
            if agent_action.get('element_id') != gt_action.get('element_id'):
                deviations.append({
                    'type': 'target_element_mismatch',
                    'step': i,
                    'expected': gt_action.get('element_id'),
                    'actual': agent_action.get('element_id'),
                    'impact': 'high'
                })
        
        return deviations
    
    def _generate_improvement_recommendations(self, failed_steps: List[Dict]) -> List[str]:
        """Generate recommendations based on failed steps."""
        recommendations = []
        
        if not failed_steps:
            recommendations.append("Agent performance meets expectations")
            return recommendations
        
        # Analyze failure patterns
        failure_types = [step.get('type') for step in failed_steps]
        
        if 'sequence_length_mismatch' in failure_types:
            recommendations.append("Improve planning accuracy to match expected sequence length")
        
        if 'action_type_mismatch' in failure_types:
            recommendations.append("Enhance action selection logic for better type accuracy")
        
        if 'target_element_mismatch' in failure_types:
            recommendations.append("Improve UI element detection and targeting precision")
        
        # Check impact levels
        high_impact_failures = [step for step in failed_steps if step.get('impact') == 'high']
        if high_impact_failures:
            recommendations.append("Focus on critical failure points that significantly impact task completion")
        
        return recommendations
    
    def _calculate_performance_metrics(self, results, comparison_result: ComparisonResult) -> Dict:
        """Calculate overall performance metrics."""
        return {
            'execution_time': getattr(results, 'duration', 0.0),
            'steps_completed': len(getattr(results, 'execution_results', [])),
            'overall_accuracy': comparison_result.accuracy_score,
            'robustness': comparison_result.robustness_score,
            'generalization': comparison_result.generalization_score,
            'success_rate': 1.0 if getattr(results, 'overall_success', False) else 0.0
        }
    
    def generate_batch_report(self, reproduction_results: List[ReproductionResult]) -> Dict:
        """Generate comprehensive report for batch of reproductions."""
        if not reproduction_results:
            return {"error": "No reproduction results to analyze"}
        
        # Calculate aggregate metrics
        total_accuracy = sum(r.comparison_result.accuracy_score for r in reproduction_results)
        total_robustness = sum(r.comparison_result.robustness_score for r in reproduction_results)
        total_generalization = sum(r.comparison_result.generalization_score for r in reproduction_results)
        
        successful_reproductions = sum(1 for r in reproduction_results if r.execution_success)
        
        # Identify common failure patterns
        all_failed_steps = []
        for result in reproduction_results:
            all_failed_steps.extend(result.comparison_result.failed_steps)
        
        failure_types = {}
        for step in all_failed_steps:
            step_type = step.get('type', 'unknown')
            failure_types[step_type] = failure_types.get(step_type, 0) + 1
        
        # Generate comprehensive recommendations
        batch_recommendations = self._generate_batch_recommendations(reproduction_results)
        
        report = {
            'summary': {
                'total_videos_processed': len(reproduction_results),
                'successful_reproductions': successful_reproductions,
                'success_rate': successful_reproductions / len(reproduction_results),
                'average_accuracy': total_accuracy / len(reproduction_results),
                'average_robustness': total_robustness / len(reproduction_results),
                'average_generalization': total_generalization / len(reproduction_results)
            },
            'failure_analysis': {
                'common_failure_types': failure_types,
                'most_challenging_videos': self._identify_challenging_videos(reproduction_results)
            },
            'performance_trends': {
                'accuracy_distribution': self._calculate_accuracy_distribution(reproduction_results),
                'execution_time_stats': self._calculate_time_stats(reproduction_results)
            },
            'recommendations': batch_recommendations,
            'detailed_results': [
                {
                    'video': r.original_video,
                    'task': r.generated_task,
                    'accuracy': r.comparison_result.accuracy_score,
                    'success': r.execution_success
                } for r in reproduction_results
            ]
        }
        
        return report
    
    def _generate_batch_recommendations(self, results: List[ReproductionResult]) -> List[str]:
        """Generate system-wide recommendations based on batch results."""
        recommendations = []
        
        # Analyze overall performance
        avg_accuracy = sum(r.comparison_result.accuracy_score for r in results) / len(results)
        
        if avg_accuracy < 0.7:
            recommendations.append("Overall accuracy below threshold - consider retraining Planner Agent")
        
        if avg_accuracy < 0.5:
            recommendations.append("Critical accuracy issues - review entire agent coordination pipeline")
        
        # Analyze success rate
        success_rate = sum(1 for r in results if r.execution_success) / len(results)
        
        if success_rate < 0.8:
            recommendations.append("Low success rate indicates need for improved error handling")
        
        return recommendations
    
    def _identify_challenging_videos(self, results: List[ReproductionResult]) -> List[Dict]:
        """Identify videos that were most challenging for the agent system."""
        challenging = []
        
        for result in results:
            if (result.comparison_result.accuracy_score < 0.5 or 
                not result.execution_success):
                challenging.append({
                    'video': result.original_video,
                    'accuracy': result.comparison_result.accuracy_score,
                    'issues': len(result.comparison_result.failed_steps)
                })
        
        # Sort by difficulty (lowest accuracy first)
        challenging.sort(key=lambda x: x['accuracy'])
        return challenging[:5]  # Return top 5 most challenging
    
    def _calculate_accuracy_distribution(self, results: List[ReproductionResult]) -> Dict:
        """Calculate distribution of accuracy scores."""
        accuracies = [r.comparison_result.accuracy_score for r in results]
        
        return {
            'excellent': sum(1 for a in accuracies if a >= 0.9) / len(accuracies),
            'good': sum(1 for a in accuracies if 0.7 <= a < 0.9) / len(accuracies),
            'fair': sum(1 for a in accuracies if 0.5 <= a < 0.7) / len(accuracies),
            'poor': sum(1 for a in accuracies if a < 0.5) / len(accuracies)
        }
    
    def _calculate_time_stats(self, results: List[ReproductionResult]) -> Dict:
        """Calculate execution time statistics."""
        times = [r.performance_metrics.get('execution_time', 0) for r in results]
        
        if not times:
            return {}
        
        return {
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'median': sorted(times)[len(times) // 2]
        }

if __name__ == "__main__":
    # Example usage with real android_in_the_wild dataset
    from bonus.android_in_the_wild.dataset_processor import AndroidInTheWildProcessor
    
    # Initialize components with GitHub dataset
    processor = AndroidInTheWildProcessor(
        dataset_url="https://github.com/google-research/google-research/tree/master/android_in_the_wild",
        local_cache_dir="datasets/android_in_the_wild"
    )
    
    # Create orchestrator for reproduction
    config = QASystemConfig(
        engine_params={'mock': True},  # Use mock for demo
        android_env=None,
        max_execution_time=300.0,
        enable_visual_trace=True
    )
    orchestrator = MultiAgentQAOrchestrator(config)
    
    # Initialize reproduction engine
    reproduction_engine = VideoReproductionEngine(orchestrator)
    
    # Process sample videos
    sample_videos = processor.download_sample_videos(3)
    video_traces = [processor.process_video(v['path']) for v in sample_videos]
    
    # Reproduce each video flow
    reproduction_results = []
    for trace in video_traces:
        result = reproduction_engine.reproduce_video_flow(trace)
        reproduction_results.append(result)
    
    # Generate batch report
    batch_report = reproduction_engine.generate_batch_report(reproduction_results)
    
    print("Batch Reproduction Report:")
    print(f"Success Rate: {batch_report['summary']['success_rate']:.2f}")
    print(f"Average Accuracy: {batch_report['summary']['average_accuracy']:.2f}")
    print(f"Recommendations: {len(batch_report['recommendations'])}")
    
    for recommendation in batch_report['recommendations']:
        print(f"- {recommendation}")
