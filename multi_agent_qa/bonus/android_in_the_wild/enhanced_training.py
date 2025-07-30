"""
Enhanced Agent Training with android_in_the_wild Dataset
Implements bonus features for improving agent training using real user data.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("Warning: numpy not available. Enhanced training features will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    print("Warning: opencv-python not available. Video processing features will be limited.")

# Import core system components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bonus.android_in_the_wild.dataset_processor import AndroidInTheWildProcessor, VideoTrace

# Helper functions for handling missing dependencies
def safe_numpy_operation(operation, *args, **kwargs):
    """Safely perform numpy operations with fallback."""
    if not NUMPY_AVAILABLE:
        # Return mock data for demonstration
        if operation == "random":
            return [0.5] * kwargs.get('size', 1)
        elif operation == "mean":
            return sum(args[0]) / len(args[0]) if args[0] else 0.0
        elif operation == "std":
            return 0.1  # Mock standard deviation
        elif operation == "array":
            return list(args[0]) if args else []
        else:
            return 0.0
    else:
        if operation == "random":
            return np.random.random(**kwargs).tolist()
        elif operation == "mean":
            return np.mean(args[0])
        elif operation == "std":
            return np.std(args[0])
        elif operation == "array":
            return np.array(args[0])
        else:
            return getattr(np, operation)(*args, **kwargs)

def safe_cv2_operation(operation, *args, **kwargs):
    """Safely perform cv2 operations with fallback."""
    if not CV2_AVAILABLE:
        # Return mock data for demonstration
        if operation == "resize":
            return [[0, 0, 0] for _ in range(kwargs.get('height', 100))]
        elif operation == "imread":
            return [[0, 0, 0] for _ in range(100)]
        else:
            return None
    else:
        return getattr(cv2, operation)(*args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class TrainingDataPoint:
    """Single training data point for agent enhancement."""
    episode_id: str
    input_data: Dict
    target_output: Dict
    agent_type: str  # planner, executor, verifier, supervisor
    difficulty: str
    metadata: Dict

class EnhancedPlannerTraining:
    """Enhanced training for Planner Agent using real user session traces."""
    
    def __init__(self, dataset_processor: AndroidInTheWildProcessor):
        self.dataset_processor = dataset_processor
        self.training_data = []
        self.session_patterns = {}
    
    def extract_planning_patterns(self, video_traces: List[VideoTrace]) -> List[TrainingDataPoint]:
        """Extract planning patterns from real user sessions for pretraining."""
        logger.info("Extracting planning patterns from user session traces...")
        
        training_points = []
        
        for trace in video_traces:
            # Analyze how humans sequence app tasks
            planning_data = self._analyze_human_task_sequencing(trace)
            
            # Create training data for different planning scenarios
            for scenario in planning_data:
                training_point = TrainingDataPoint(
                    episode_id=trace.episode_id,
                    input_data={
                        'task_description': trace.task_prompt,
                        'app_context': trace.app_package,
                        'device_info': trace.device_info,
                        'initial_ui_state': trace.ui_traces[0] if trace.ui_traces else {}
                    },
                    target_output={
                        'action_sequence': scenario['sequence'],
                        'decision_points': scenario['decisions'],
                        'contingency_plans': scenario['alternatives']
                    },
                    agent_type='planner',
                    difficulty=self._assess_planning_difficulty(scenario),
                    metadata=trace.metadata
                )
                training_points.append(training_point)
        
        logger.info(f"Generated {len(training_points)} planning training points")
        return training_points
    
    def _analyze_human_task_sequencing(self, trace: VideoTrace) -> List[Dict]:
        """Analyze how humans sequence tasks in the given trace."""
        sequences = []
        
        # Extract action patterns
        actions = trace.ground_truth_actions
        if not actions:
            return sequences
        
        # Group actions into logical sequences
        current_sequence = []
        decision_points = []
        
        for i, action in enumerate(actions):
            current_sequence.append(action)
            
            # Identify decision points (pauses, UI changes, etc.)
            if self._is_decision_point(action, actions[i+1:]):
                decision_points.append({
                    'position': len(current_sequence) - 1,
                    'context': action,
                    'alternatives': self._identify_alternatives(action, trace.ui_traces)
                })
        
        sequences.append({
            'sequence': current_sequence,
            'decisions': decision_points,
            'alternatives': self._generate_alternative_sequences(current_sequence)
        })
        
        return sequences
    
    def _is_decision_point(self, current_action: Dict, remaining_actions: List[Dict]) -> bool:
        """Determine if current action represents a decision point."""
        # Check for pauses (timing gaps)
        if remaining_actions and remaining_actions[0].get('timestamp', 0) - current_action.get('timestamp', 0) > 2.0:
            return True
        
        # Check for navigation actions
        if current_action.get('type') in ['navigate_back', 'navigate_home', 'app_switch']:
            return True
        
        return False
    
    def _identify_alternatives(self, action: Dict, ui_traces: List[Dict]) -> List[Dict]:
        """Identify alternative actions that could have been taken."""
        alternatives = []
        
        # Find UI state at time of action
        action_time = action.get('timestamp', 0)
        ui_state = None
        for trace in ui_traces:
            if abs(trace.get('timestamp', 0) - action_time) < 0.5:
                ui_state = trace
                break
        
        if ui_state and 'ui_elements' in ui_state:
            # Find other interactive elements
            for element in ui_state['ui_elements']:
                if (element.get('type') in ['button', 'menu_item', 'clickable'] and 
                    element.get('id') != action.get('element_id')):
                    alternatives.append({
                        'type': 'click',
                        'element_id': element.get('id'),
                        'coordinates': element.get('center', [0, 0]),
                        'reasoning': f"Alternative clickable element: {element.get('text', 'unnamed')}"
                    })
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    def _generate_alternative_sequences(self, sequence: List[Dict]) -> List[List[Dict]]:
        """Generate alternative action sequences."""
        alternatives = []
        
        # Generate shorter sequence (more direct)
        if len(sequence) > 3:
            direct_sequence = [sequence[0], sequence[-1]]
            alternatives.append(direct_sequence)
        
        # Generate sequence with different navigation
        if any(action.get('type') == 'navigate_back' for action in sequence):
            no_back_sequence = [action for action in sequence if action.get('type') != 'navigate_back']
            if no_back_sequence:
                alternatives.append(no_back_sequence)
        
        return alternatives
    
    def _assess_planning_difficulty(self, scenario: Dict) -> str:
        """Assess the difficulty level of a planning scenario."""
        sequence_length = len(scenario['sequence'])
        decision_count = len(scenario['decisions'])
        alternative_count = sum(len(d['alternatives']) for d in scenario['decisions'])
        
        if sequence_length <= 3 and decision_count <= 1:
            return 'easy'
        elif sequence_length <= 6 and decision_count <= 3:
            return 'medium'
        else:
            return 'hard'

class EnhancedExecutorTraining:
    """Enhanced training for Executor Agent with gesture control and visual grounding."""
    
    def __init__(self, dataset_processor: AndroidInTheWildProcessor):
        self.dataset_processor = dataset_processor
        self.gesture_patterns = {}
        self.visual_grounding_data = []
    
    def extract_gesture_training_data(self, video_traces: List[VideoTrace]) -> List[TrainingDataPoint]:
        """Extract gesture control training data from touchpoints and motion paths."""
        logger.info("Extracting gesture training data from user interactions...")
        
        training_points = []
        
        for trace in video_traces:
            # Analyze touchpoint locations and motion paths
            gesture_data = self._analyze_gesture_patterns(trace)
            
            for gesture in gesture_data:
                training_point = TrainingDataPoint(
                    episode_id=trace.episode_id,
                    input_data={
                        'ui_screenshot': gesture['screenshot'],
                        'target_element': gesture['target_element'],
                        'task_context': trace.task_prompt,
                        'device_resolution': trace.device_info.get('screen_resolution', [1080, 1920])
                    },
                    target_output={
                        'gesture_type': gesture['type'],
                        'start_coordinates': gesture['start_coords'],
                        'end_coordinates': gesture.get('end_coords'),
                        'duration': gesture.get('duration', 0),
                        'pressure': gesture.get('pressure', 1.0)
                    },
                    agent_type='executor',
                    difficulty=self._assess_gesture_difficulty(gesture),
                    metadata={
                        'device_type': trace.device_info.get('model', 'unknown'),
                        'screen_density': trace.device_info.get('density', 'hdpi'),
                        'layout_variation': gesture.get('layout_id', 'default')
                    }
                )
                training_points.append(training_point)
        
        logger.info(f"Generated {len(training_points)} gesture training points")
        return training_points
    
    def _analyze_gesture_patterns(self, trace: VideoTrace) -> List[Dict]:
        """Analyze gesture patterns from user interactions."""
        gestures = []
        
        for action in trace.ground_truth_actions:
            gesture_data = {
                'type': action.get('type', 'click'),
                'start_coords': action.get('coordinates', [0, 0]),
                'timestamp': action.get('timestamp', 0),
                'target_element': action.get('element_id', ''),
                'screenshot': self._get_screenshot_at_time(trace, action.get('timestamp', 0))
            }
            
            # Add motion data for swipe/scroll gestures
            if action.get('type') in ['swipe', 'scroll']:
                gesture_data.update({
                    'end_coords': action.get('end_coordinates', gesture_data['start_coords']),
                    'duration': action.get('duration', 0.5),
                    'velocity': action.get('velocity', 1.0)
                })
            
            # Add pressure data for different touch types
            if action.get('type') == 'long_click':
                gesture_data['pressure'] = 1.5
                gesture_data['duration'] = action.get('duration', 1.0)
            
            gestures.append(gesture_data)
        
        return gestures
    
    def _get_screenshot_at_time(self, trace: VideoTrace, timestamp: float) -> str:
        """Get screenshot filename closest to the given timestamp."""
        closest_trace = None
        min_diff = float('inf')
        
        for ui_trace in trace.ui_traces:
            time_diff = abs(ui_trace.get('timestamp', 0) - timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_trace = ui_trace
        
        return closest_trace.get('screenshot', 'default.png') if closest_trace else 'default.png'
    
    def _assess_gesture_difficulty(self, gesture: Dict) -> str:
        """Assess the difficulty of a gesture based on precision requirements."""
        gesture_type = gesture.get('type', 'click')
        
        # Handle target_element - it could be a string (element_id) or dict
        target_element = gesture.get('target_element', {})
        if isinstance(target_element, str):
            # If it's a string (element_id), use default size based on element type
            if 'button' in target_element.lower():
                target_size = [120, 48]  # Standard button size
            elif 'input' in target_element.lower() or 'text' in target_element.lower():
                target_size = [200, 48]  # Input field size
            else:
                target_size = [100, 100]  # Default size
        else:
            target_size = target_element.get('size', [100, 100])
        
        # Calculate target area
        target_area = target_size[0] * target_size[1] if len(target_size) >= 2 else 10000
        
        if gesture_type == 'click':
            if target_area > 5000:  # Large targets
                return 'easy'
            elif target_area > 1000:  # Medium targets
                return 'medium'
            else:  # Small targets
                return 'hard'
        elif gesture_type in ['swipe', 'scroll']:
            return 'medium'  # Motion gestures are generally medium difficulty
        elif gesture_type == 'long_click':
            return 'medium'  # Timing-sensitive
        else:
            return 'medium'

class EnhancedVerifierTraining:
    """Enhanced training for Verifier Agent using contrastive learning."""
    
    def __init__(self, dataset_processor: AndroidInTheWildProcessor):
        self.dataset_processor = dataset_processor
        self.anomaly_patterns = {}
        self.expected_flows = {}
    
    def create_contrastive_training_data(self, video_traces: List[VideoTrace]) -> List[TrainingDataPoint]:
        """Create contrastive training data for anomaly detection."""
        logger.info("Creating contrastive training data for verifier agent...")
        
        training_points = []
        
        for trace in video_traces:
            # Create positive examples (expected flows)
            positive_examples = self._create_positive_examples(trace)
            
            # Create negative examples (anomalous flows)
            negative_examples = self._create_negative_examples(trace)
            
            # Combine for contrastive learning
            for pos_example in positive_examples:
                training_point = TrainingDataPoint(
                    episode_id=trace.episode_id,
                    input_data=pos_example['input'],
                    target_output={'label': 'expected', 'confidence': 1.0, 'reasoning': pos_example['reasoning']},
                    agent_type='verifier',
                    difficulty='medium',
                    metadata={'example_type': 'positive', 'flow_type': pos_example['flow_type']}
                )
                training_points.append(training_point)
            
            for neg_example in negative_examples:
                training_point = TrainingDataPoint(
                    episode_id=trace.episode_id,
                    input_data=neg_example['input'],
                    target_output={'label': 'anomalous', 'confidence': 1.0, 'reasoning': neg_example['reasoning']},
                    agent_type='verifier',
                    difficulty='hard',
                    metadata={'example_type': 'negative', 'anomaly_type': neg_example['anomaly_type']}
                )
                training_points.append(training_point)
        
        logger.info(f"Generated {len(training_points)} contrastive training points")
        return training_points
    
    def _create_positive_examples(self, trace: VideoTrace) -> List[Dict]:
        """Create positive examples from expected user flows."""
        positive_examples = []
        
        # Normal action sequences
        for i, action in enumerate(trace.ground_truth_actions):
            if i < len(trace.ui_traces):
                example = {
                    'input': {
                        'previous_action': trace.ground_truth_actions[i-1] if i > 0 else None,
                        'current_action': action,
                        'ui_state': trace.ui_traces[i],
                        'expected_outcome': trace.ui_traces[i+1] if i+1 < len(trace.ui_traces) else None
                    },
                    'reasoning': 'Normal user action with expected UI transition',
                    'flow_type': 'sequential'
                }
                positive_examples.append(example)
        
        return positive_examples
    
    def _create_negative_examples(self, trace: VideoTrace) -> List[Dict]:
        """Create negative examples by introducing anomalies."""
        negative_examples = []
        
        for i, action in enumerate(trace.ground_truth_actions):
            # Create various types of anomalies
            
            # Wrong element clicked
            wrong_element_action = action.copy()
            wrong_element_action['element_id'] = 'wrong_element_id'
            negative_examples.append({
                'input': {
                    'previous_action': trace.ground_truth_actions[i-1] if i > 0 else None,
                    'current_action': wrong_element_action,
                    'ui_state': trace.ui_traces[i] if i < len(trace.ui_traces) else {},
                    'expected_outcome': None
                },
                'reasoning': 'Action targets incorrect UI element',
                'anomaly_type': 'wrong_target'
            })
            
            # Action at wrong time
            if i > 0:
                out_of_order_action = trace.ground_truth_actions[i-1].copy()
                negative_examples.append({
                    'input': {
                        'previous_action': action,
                        'current_action': out_of_order_action,
                        'ui_state': trace.ui_traces[i] if i < len(trace.ui_traces) else {},
                        'expected_outcome': None
                    },
                    'reasoning': 'Action performed out of sequence',
                    'anomaly_type': 'wrong_timing'
                })
            
            # Missing required action
            if action.get('type') == 'click' and i < len(trace.ground_truth_actions) - 1:
                negative_examples.append({
                    'input': {
                        'previous_action': action,
                        'current_action': None,  # Missing action
                        'ui_state': trace.ui_traces[i] if i < len(trace.ui_traces) else {},
                        'expected_outcome': trace.ui_traces[i+1] if i+1 < len(trace.ui_traces) else None
                    },
                    'reasoning': 'Required action was not performed',
                    'anomaly_type': 'missing_action'
                })
        
        return negative_examples[:len(trace.ground_truth_actions)]  # Limit negative examples

class EnhancedSupervisorTraining:
    """Enhanced training for Supervisor Agent using video input processing."""
    
    def __init__(self, dataset_processor: AndroidInTheWildProcessor):
        self.dataset_processor = dataset_processor
        self.video_analysis_patterns = {}
    
    def create_video_analysis_training_data(self, video_traces: List[VideoTrace]) -> List[TrainingDataPoint]:
        """Create training data for video input processing and analysis."""
        logger.info("Creating video analysis training data for supervisor agent...")
        
        training_points = []
        
        for trace in video_traces:
            # Generate task prompts from video analysis
            task_generation_data = self._create_task_generation_examples(trace)
            
            # Create improvement suggestion examples
            improvement_data = self._create_improvement_examples(trace)
            
            # Combine all supervisor training data
            for example in task_generation_data + improvement_data:
                training_point = TrainingDataPoint(
                    episode_id=trace.episode_id,
                    input_data=example['input'],
                    target_output=example['output'],
                    agent_type='supervisor',
                    difficulty=example.get('difficulty', 'medium'),
                    metadata=example.get('metadata', {})
                )
                training_points.append(training_point)
        
        logger.info(f"Generated {len(training_points)} supervisor training points")
        return training_points
    
    def _create_task_generation_examples(self, trace: VideoTrace) -> List[Dict]:
        """Create examples for generating task prompts from video analysis."""
        examples = []
        
        # Full video to task prompt
        examples.append({
            'input': {
                'video_path': trace.video_path,
                'ui_sequence': trace.ui_traces,
                'action_sequence': trace.ground_truth_actions,
                'app_package': trace.app_package,
                'device_info': trace.device_info
            },
            'output': {
                'generated_task': trace.task_prompt,
                'confidence': 0.9,
                'task_category': self._categorize_task(trace.task_prompt),
                'estimated_difficulty': self._estimate_task_difficulty(trace),
                'key_interactions': self._identify_key_interactions(trace)
            },
            'difficulty': 'hard',
            'metadata': {
                'training_type': 'task_generation',
                'video_duration': trace.metadata.get('duration', 0)
            }
        })
        
        return examples
    
    def _create_improvement_examples(self, trace: VideoTrace) -> List[Dict]:
        """Create examples for suggesting agent improvements."""
        examples = []
        
        # Identify potential improvements based on user behavior
        inefficiencies = self._identify_inefficiencies(trace)
        
        for inefficiency in inefficiencies:
            examples.append({
                'input': {
                    'agent_actions': trace.ground_truth_actions,
                    'execution_time': trace.metadata.get('duration', 0),
                    'success_rate': 1.0,  # Assume human actions are successful
                    'ui_coverage': len(set(action.get('element_id', '') for action in trace.ground_truth_actions))
                },
                'output': {
                    'improvement_type': inefficiency['type'],
                    'suggestion': inefficiency['suggestion'],
                    'priority': inefficiency['priority'],
                    'expected_benefit': inefficiency['benefit']
                },
                'difficulty': 'medium',
                'metadata': {
                    'training_type': 'improvement_suggestion',
                    'inefficiency_category': inefficiency['category']
                }
            })
        
        return examples
    
    def _categorize_task(self, task_prompt: str) -> str:
        """Categorize the task based on the prompt."""
        task_lower = task_prompt.lower()
        
        if any(word in task_lower for word in ['wifi', 'settings', 'network']):
            return 'settings'
        elif any(word in task_lower for word in ['alarm', 'clock', 'time']):
            return 'time_management'
        elif any(word in task_lower for word in ['email', 'gmail', 'message']):
            return 'communication'
        elif any(word in task_lower for word in ['contact', 'phone', 'call']):
            return 'contacts'
        elif any(word in task_lower for word in ['calendar', 'event', 'appointment']):
            return 'calendar'
        else:
            return 'general'
    
    def _estimate_task_difficulty(self, trace: VideoTrace) -> str:
        """Estimate task difficulty based on trace characteristics."""
        action_count = len(trace.ground_truth_actions)
        duration = trace.metadata.get('duration', 0)
        ui_changes = len(trace.ui_traces)
        
        if action_count <= 3 and duration <= 10:
            return 'easy'
        elif action_count <= 6 and duration <= 20:
            return 'medium'
        else:
            return 'hard'
    
    def _identify_key_interactions(self, trace: VideoTrace) -> List[str]:
        """Identify key interactions in the trace."""
        key_interactions = []
        
        for action in trace.ground_truth_actions:
            action_type = action.get('type', 'unknown')
            element_id = action.get('element_id', '')
            
            if action_type in ['click', 'long_click'] and element_id:
                key_interactions.append(f"{action_type}:{element_id}")
            elif action_type in ['swipe', 'scroll']:
                direction = action.get('direction', 'unknown')
                key_interactions.append(f"{action_type}:{direction}")
        
        return key_interactions[:5]  # Limit to top 5
    
    def _identify_inefficiencies(self, trace: VideoTrace) -> List[Dict]:
        """Identify potential inefficiencies in the user's actions."""
        inefficiencies = []
        
        actions = trace.ground_truth_actions
        
        # Check for redundant actions
        for i in range(len(actions) - 1):
            if (actions[i].get('type') == actions[i+1].get('type') and
                actions[i].get('element_id') == actions[i+1].get('element_id')):
                inefficiencies.append({
                    'type': 'redundant_action',
                    'category': 'execution',
                    'suggestion': 'Avoid duplicate actions on the same element',
                    'priority': 'medium',
                    'benefit': 'Reduce execution time by 10-20%'
                })
        
        # Check for excessive navigation
        nav_actions = [a for a in actions if a.get('type') in ['navigate_back', 'navigate_home']]
        if len(nav_actions) > len(actions) * 0.3:
            inefficiencies.append({
                'type': 'excessive_navigation',
                'category': 'planning',
                'suggestion': 'Plan more direct paths to reduce navigation overhead',
                'priority': 'high',
                'benefit': 'Improve task completion efficiency by 25-40%'
            })
        
        # Check for long pauses (potential confusion)
        for i in range(len(actions) - 1):
            time_gap = actions[i+1].get('timestamp', 0) - actions[i].get('timestamp', 0)
            if time_gap > 5.0:  # 5 second pause
                inefficiencies.append({
                    'type': 'decision_delay',
                    'category': 'planning',
                    'suggestion': 'Improve UI element recognition to reduce hesitation',
                    'priority': 'medium',
                    'benefit': 'Reduce average execution time by 15-30%'
                })
                break  # Only report one instance
        
        return inefficiencies

def create_comprehensive_training_dataset(dataset_processor: AndroidInTheWildProcessor, 
                                        num_episodes: int = 10) -> Dict[str, List[TrainingDataPoint]]:
    """Create comprehensive training dataset for all agents."""
    logger.info(f"Creating comprehensive training dataset with {num_episodes} episodes")
    
    # Download sample episodes
    sample_videos = dataset_processor.download_sample_videos(num_episodes)
    video_traces = [dataset_processor.process_video(video['path']) for video in sample_videos]
    
    # Initialize training modules
    planner_trainer = EnhancedPlannerTraining(dataset_processor)
    executor_trainer = EnhancedExecutorTraining(dataset_processor)
    verifier_trainer = EnhancedVerifierTraining(dataset_processor)
    supervisor_trainer = EnhancedSupervisorTraining(dataset_processor)
    
    # Generate training data for each agent
    training_dataset = {
        'planner': planner_trainer.extract_planning_patterns(video_traces),
        'executor': executor_trainer.extract_gesture_training_data(video_traces),
        'verifier': verifier_trainer.create_contrastive_training_data(video_traces),
        'supervisor': supervisor_trainer.create_video_analysis_training_data(video_traces)
    }
    
    # Log statistics
    for agent_type, data_points in training_dataset.items():
        logger.info(f"{agent_type.capitalize()} Agent: {len(data_points)} training points")
        
        # Difficulty distribution
        difficulty_counts = {}
        for point in data_points:
            diff = point.difficulty
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        logger.info(f"  Difficulty distribution: {difficulty_counts}")
    
    return training_dataset

if __name__ == "__main__":
    # Test enhanced training data generation
    processor = AndroidInTheWildProcessor(
        dataset_url="https://github.com/google-research/google-research/tree/master/android_in_the_wild",
        local_cache_dir="datasets/android_in_the_wild"
    )
    
    # Create comprehensive training dataset
    training_data = create_comprehensive_training_dataset(processor, num_episodes=5)
    
    print("Enhanced Training Dataset Summary:")
    print("=" * 50)
    
    total_points = 0
    for agent_type, data_points in training_data.items():
        total_points += len(data_points)
        print(f"{agent_type.capitalize()} Agent: {len(data_points)} training points")
        
        if data_points:
            example = data_points[0]
            print(f"  Example input keys: {list(example.input_data.keys())}")
            print(f"  Example output keys: {list(example.target_output.keys())}")
            print()
    
    print(f"Total training points generated: {total_points}")
    print("\nThis dataset can be used to enhance agent performance using:")
    print("- Real user interaction patterns")
    print("- Visual grounding on actual UI elements")
    print("- Contrastive learning for anomaly detection")
    print("- Video analysis for task understanding")
