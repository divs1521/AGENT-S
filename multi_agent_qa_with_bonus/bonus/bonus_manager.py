"""
Bonus Features Integration Manager
Manages the integration of android_in_the_wild bonus features
without disrupting the core multi-agent QA system.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add core project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig
from bonus.android_in_the_wild.dataset_processor import AndroidInTheWildProcessor, DatasetEnhancedTrainer
from bonus.android_in_the_wild.video_reproduction import VideoReproductionEngine
from bonus.enhanced_training import EnhancedTrainingPipeline

logger = logging.getLogger(__name__)

class BonusFeatureManager:
    """
    Manages bonus features for the Multi-Agent QA System.
    Provides android_in_the_wild integration without affecting core functionality.
    """
    
    def __init__(self, core_orchestrator: MultiAgentQAOrchestrator, config: Dict = None):
        self.core_orchestrator = core_orchestrator
        self.config = config or {}
        
        # Initialize bonus components
        self.dataset_processor = None
        self.reproduction_engine = None
        self.training_pipeline = None
        
        # Bonus feature status
        self.features_enabled = {
            'dataset_processing': False,
            'video_reproduction': False,
            'enhanced_training': False
        }
        
        # Results storage
        self.bonus_results = {
            'processed_videos': [],
            'reproduction_results': [],
            'training_results': {}
        }
        
    def initialize_bonus_features(self, dataset_path: str = None):
        """Initialize all bonus features."""
        try:
            logger.info("Initializing bonus features...")
            
            # Initialize dataset processor
            if dataset_path:
                self.dataset_processor = AndroidInTheWildProcessor(dataset_path)
                self.features_enabled['dataset_processing'] = True
                logger.info("✅ Dataset processing initialized")
            
            # Initialize video reproduction engine
            self.reproduction_engine = VideoReproductionEngine(self.core_orchestrator)
            self.features_enabled['video_reproduction'] = True
            logger.info("✅ Video reproduction engine initialized")
            
            # Initialize enhanced training pipeline
            self.training_pipeline = EnhancedTrainingPipeline()
            self.features_enabled['enhanced_training'] = True
            logger.info("✅ Enhanced training pipeline initialized")
            
            logger.info("🎉 All bonus features initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize bonus features: {e}")
            return False
    
    def run_android_in_the_wild_integration(self, video_count: int = 5) -> Dict:
        """
        Main method to run android_in_the_wild integration.
        This reproduces the complete bonus feature workflow.
        """
        if not self.features_enabled['dataset_processing']:
            return {"error": "Dataset processing not initialized"}
        
        logger.info(f"🚀 Starting android_in_the_wild integration with {video_count} videos")
        
        results = {
            'phase_1_dataset_processing': {},
            'phase_2_task_generation': {},
            'phase_3_reproduction': {},
            'phase_4_comparison': {},
            'phase_5_scoring': {},
            'summary': {}
        }
        
        try:
            # Phase 1: Download and process videos
            logger.info("📥 Phase 1: Processing android_in_the_wild videos...")
            video_traces = self._process_dataset_videos(video_count)
            results['phase_1_dataset_processing'] = {
                'videos_processed': len(video_traces),
                'processing_success': True
            }
            
            # Phase 2: Generate task prompts
            logger.info("🎯 Phase 2: Generating task prompts...")
            task_generation_results = self._generate_task_prompts(video_traces)
            results['phase_2_task_generation'] = task_generation_results
            
            # Phase 3: Reproduce video flows
            logger.info("🤖 Phase 3: Reproducing video flows with multi-agent system...")
            reproduction_results = self._reproduce_video_flows(video_traces)
            results['phase_3_reproduction'] = {
                'reproductions_completed': len(reproduction_results),
                'success_rate': sum(1 for r in reproduction_results if r.execution_success) / len(reproduction_results)
            }
            
            # Phase 4: Compare agent vs ground truth
            logger.info("⚖️ Phase 4: Comparing agent performance vs ground truth...")
            comparison_results = self._compare_performances(reproduction_results)
            results['phase_4_comparison'] = comparison_results
            
            # Phase 5: Score accuracy, robustness, and generalization
            logger.info("📊 Phase 5: Calculating performance scores...")
            scoring_results = self._calculate_performance_scores(reproduction_results)
            results['phase_5_scoring'] = scoring_results
            
            # Generate summary
            results['summary'] = self._generate_integration_summary(results)
            
            # Store results
            self.bonus_results = results
            
            logger.info("✅ android_in_the_wild integration completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ android_in_the_wild integration failed: {e}")
            results['error'] = str(e)
            return results
    
    def run_enhanced_training_pipeline(self, video_traces: List = None) -> Dict:
        """Run enhanced training pipeline using android_in_the_wild data."""
        if not self.features_enabled['enhanced_training']:
            return {"error": "Enhanced training not initialized"}
        
        logger.info("🧠 Starting enhanced training pipeline...")
        
        try:
            # Use existing video traces or process new ones
            if not video_traces:
                if self.bonus_results.get('processed_videos'):
                    video_traces = self.bonus_results['processed_videos']
                else:
                    # Process sample videos for training
                    video_traces = self._process_dataset_videos(5)
            
            # Run training pipeline
            training_results = self.training_pipeline.run_full_training_pipeline(video_traces)
            
            # Store training results
            self.bonus_results['training_results'] = training_results
            
            logger.info("✅ Enhanced training pipeline completed!")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {e}")
            return {"error": str(e)}
    
    def _process_dataset_videos(self, count: int) -> List:
        """Process videos from android_in_the_wild dataset."""
        # Download sample videos
        sample_videos = self.dataset_processor.download_sample_videos(count)
        
        # Process each video
        video_traces = []
        for video_info in sample_videos:
            try:
                trace = self.dataset_processor.process_video(video_info['path'])
                video_traces.append(trace)
                logger.info(f"✅ Processed video: {video_info['path']}")
            except Exception as e:
                logger.error(f"❌ Failed to process video {video_info['path']}: {e}")
        
        # Store processed videos
        self.bonus_results['processed_videos'] = video_traces
        
        return video_traces
    
    def _generate_task_prompts(self, video_traces: List) -> Dict:
        """Generate task prompts from video analysis."""
        generated_prompts = []
        
        for trace in video_traces:
            prompt_info = {
                'video': getattr(trace, 'video_path', 'unknown'),
                'generated_prompt': getattr(trace, 'task_prompt', ''),
                'confidence': 0.85,  # Mock confidence score
                'complexity': self._assess_prompt_complexity(trace)
            }
            generated_prompts.append(prompt_info)
        
        return {
            'prompts_generated': len(generated_prompts),
            'average_confidence': sum(p['confidence'] for p in generated_prompts) / len(generated_prompts),
            'prompt_details': generated_prompts
        }
    
    def _reproduce_video_flows(self, video_traces: List) -> List:
        """Reproduce video flows using multi-agent system."""
        reproduction_results = []
        
        for trace in video_traces:
            try:
                result = self.reproduction_engine.reproduce_video_flow(trace)
                reproduction_results.append(result)
                logger.info(f"✅ Reproduced: {result.generated_task}")
            except Exception as e:
                logger.error(f"❌ Failed to reproduce video flow: {e}")
        
        # Store reproduction results
        self.bonus_results['reproduction_results'] = reproduction_results
        
        return reproduction_results
    
    def _compare_performances(self, reproduction_results: List) -> Dict:
        """Compare agent performance vs ground truth."""
        comparison_data = {
            'total_comparisons': len(reproduction_results),
            'successful_reproductions': 0,
            'accuracy_scores': [],
            'robustness_scores': [],
            'generalization_scores': [],
            'common_failures': []
        }
        
        for result in reproduction_results:
            if result.execution_success:
                comparison_data['successful_reproductions'] += 1
            
            comparison_data['accuracy_scores'].append(result.comparison_result.accuracy_score)
            comparison_data['robustness_scores'].append(result.comparison_result.robustness_score)
            comparison_data['generalization_scores'].append(result.comparison_result.generalization_score)
            
            # Collect failure patterns
            for failed_step in result.comparison_result.failed_steps:
                failure_type = failed_step.get('type', 'unknown')
                comparison_data['common_failures'].append(failure_type)
        
        # Calculate averages
        if comparison_data['accuracy_scores']:
            comparison_data['average_accuracy'] = sum(comparison_data['accuracy_scores']) / len(comparison_data['accuracy_scores'])
            comparison_data['average_robustness'] = sum(comparison_data['robustness_scores']) / len(comparison_data['robustness_scores'])
            comparison_data['average_generalization'] = sum(comparison_data['generalization_scores']) / len(comparison_data['generalization_scores'])
        
        return comparison_data
    
    def _calculate_performance_scores(self, reproduction_results: List) -> Dict:
        """Calculate accuracy, robustness, and generalization scores."""
        if not reproduction_results:
            return {"error": "No reproduction results to score"}
        
        # Calculate individual metrics
        accuracy_scores = [r.comparison_result.accuracy_score for r in reproduction_results]
        robustness_scores = [r.comparison_result.robustness_score for r in reproduction_results]
        generalization_scores = [r.comparison_result.generalization_score for r in reproduction_results]
        
        # Calculate overall scores
        overall_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        overall_robustness = sum(robustness_scores) / len(robustness_scores)
        overall_generalization = sum(generalization_scores) / len(generalization_scores)
        
        # Performance grades
        def get_grade(score):
            if score >= 0.9: return "A"
            elif score >= 0.8: return "B"
            elif score >= 0.7: return "C"
            elif score >= 0.6: return "D"
            else: return "F"
        
        return {
            'accuracy': {
                'score': overall_accuracy,
                'grade': get_grade(overall_accuracy),
                'individual_scores': accuracy_scores
            },
            'robustness': {
                'score': overall_robustness,
                'grade': get_grade(overall_robustness),
                'individual_scores': robustness_scores
            },
            'generalization': {
                'score': overall_generalization,
                'grade': get_grade(overall_generalization),
                'individual_scores': generalization_scores
            },
            'overall_performance': {
                'composite_score': (overall_accuracy + overall_robustness + overall_generalization) / 3,
                'grade': get_grade((overall_accuracy + overall_robustness + overall_generalization) / 3)
            }
        }
    
    def _generate_integration_summary(self, results: Dict) -> Dict:
        """Generate comprehensive summary of integration results."""
        summary = {
            'execution_status': 'completed',
            'phases_completed': len([k for k in results.keys() if k.startswith('phase_') and not results[k].get('error')]),
            'overall_success_rate': 0.0,
            'key_achievements': [],
            'identified_challenges': [],
            'recommendations': []
        }
        
        # Calculate overall success rate
        if 'phase_3_reproduction' in results:
            summary['overall_success_rate'] = results['phase_3_reproduction'].get('success_rate', 0.0)
        
        # Key achievements
        summary['key_achievements'] = [
            f"Successfully processed {results.get('phase_1_dataset_processing', {}).get('videos_processed', 0)} videos",
            f"Generated {results.get('phase_2_task_generation', {}).get('prompts_generated', 0)} task prompts",
            f"Completed {results.get('phase_3_reproduction', {}).get('reproductions_completed', 0)} video reproductions"
        ]
        
        # Identify challenges
        summary['identified_challenges'] = []
        if summary['overall_success_rate'] < 0.8:
            summary['identified_challenges'].append("Low reproduction success rate")
        
        if 'phase_4_comparison' in results:
            avg_accuracy = results['phase_4_comparison'].get('average_accuracy', 0)
            if avg_accuracy < 0.7:
                summary['identified_challenges'].append("Accuracy below target threshold")
        
        # Recommendations
        summary['recommendations'] = [
            "Continue collecting diverse video traces for training",
            "Focus on improving accuracy for challenging scenarios",
            "Implement additional error handling patterns",
            "Expand dataset to include more app categories"
        ]
        
        if summary['overall_success_rate'] >= 0.8:
            summary['recommendations'].append("Consider deploying enhanced agents in production")
        
        return summary
    
    def _assess_prompt_complexity(self, trace) -> str:
        """Assess complexity of generated prompt."""
        prompt = getattr(trace, 'task_prompt', '')
        
        if len(prompt.split()) <= 5:
            return "simple"
        elif len(prompt.split()) <= 10:
            return "medium"
        else:
            return "complex"
    
    def generate_bonus_report(self) -> Dict:
        """Generate comprehensive report of all bonus features."""
        report = {
            'bonus_features_status': self.features_enabled,
            'android_in_the_wild_integration': self.bonus_results,
            'enhanced_training_status': {},
            'overall_assessment': {},
            'next_steps': []
        }
        
        # Enhanced training status
        if 'training_results' in self.bonus_results:
            training_results = self.bonus_results['training_results']
            report['enhanced_training_status'] = {
                'agents_trained': training_results.get('training_overview', {}).get('total_agents_trained', 0),
                'average_accuracy': training_results.get('training_overview', {}).get('average_validation_accuracy', 0),
                'examples_processed': training_results.get('training_overview', {}).get('total_examples_processed', 0)
            }
        
        # Overall assessment
        report['overall_assessment'] = self._assess_bonus_performance()
        
        # Next steps
        report['next_steps'] = [
            "Integrate enhanced agents with core system",
            "Set up continuous training pipeline",
            "Expand android_in_the_wild dataset coverage",
            "Implement real-time performance monitoring",
            "Develop automated model deployment pipeline"
        ]
        
        return report
    
    def _assess_bonus_performance(self) -> Dict:
        """Assess overall performance of bonus features."""
        assessment = {
            'dataset_processing': 'not_evaluated',
            'video_reproduction': 'not_evaluated',
            'enhanced_training': 'not_evaluated',
            'overall_grade': 'incomplete'
        }
        
        # Assess dataset processing
        if 'processed_videos' in self.bonus_results and self.bonus_results['processed_videos']:
            assessment['dataset_processing'] = 'excellent'
        
        # Assess video reproduction
        if 'reproduction_results' in self.bonus_results:
            repro_results = self.bonus_results['reproduction_results']
            if repro_results:
                success_rate = sum(1 for r in repro_results if r.execution_success) / len(repro_results)
                if success_rate >= 0.8:
                    assessment['video_reproduction'] = 'excellent'
                elif success_rate >= 0.6:
                    assessment['video_reproduction'] = 'good'
                else:
                    assessment['video_reproduction'] = 'needs_improvement'
        
        # Assess training
        if 'training_results' in self.bonus_results:
            training = self.bonus_results['training_results']
            avg_accuracy = training.get('training_overview', {}).get('average_validation_accuracy', 0)
            if avg_accuracy >= 0.85:
                assessment['enhanced_training'] = 'excellent'
            elif avg_accuracy >= 0.75:
                assessment['enhanced_training'] = 'good'
            else:
                assessment['enhanced_training'] = 'needs_improvement'
        
        # Overall grade
        scores = [v for v in assessment.values() if v in ['excellent', 'good', 'needs_improvement']]
        if not scores:
            assessment['overall_grade'] = 'incomplete'
        elif all(s == 'excellent' for s in scores):
            assessment['overall_grade'] = 'A'
        elif all(s in ['excellent', 'good'] for s in scores):
            assessment['overall_grade'] = 'B'
        else:
            assessment['overall_grade'] = 'C'
        
        return assessment
    
    def save_bonus_results(self, filepath: str = None):
        """Save all bonus results to file."""
        if not filepath:
            filepath = "bonus/results/bonus_features_results.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare serializable data
        serializable_results = self._make_serializable(self.bonus_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"💾 Bonus results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        else:
            return obj

def main():
    """Main function to demonstrate bonus features."""
    print("🎉 Multi-Agent QA System - Bonus Features Demo")
    print("=" * 60)
    
    # Initialize core system (in mock mode for demo)
    config = QASystemConfig(
        engine_params={'mock': True},
        android_env=None,
        max_execution_time=300.0,
        enable_visual_trace=True
    )
    
    core_orchestrator = MultiAgentQAOrchestrator(config)
    
    # Initialize bonus feature manager
    bonus_manager = BonusFeatureManager(core_orchestrator)
    
    # Initialize bonus features
    print("\n🔧 Initializing bonus features...")
    if bonus_manager.initialize_bonus_features():
        print("✅ Bonus features initialized successfully!")
        
        # Run android_in_the_wild integration
        print("\n🚀 Running android_in_the_wild integration...")
        integration_results = bonus_manager.run_android_in_the_wild_integration(video_count=3)
        
        if 'error' not in integration_results:
            print("✅ Integration completed successfully!")
            
            # Display results
            summary = integration_results.get('summary', {})
            print(f"\n📊 Results Summary:")
            print(f"  • Success Rate: {summary.get('overall_success_rate', 0):.2%}")
            print(f"  • Phases Completed: {summary.get('phases_completed', 0)}/5")
            
            # Run enhanced training
            print("\n🧠 Running enhanced training pipeline...")
            training_results = bonus_manager.run_enhanced_training_pipeline()
            
            if 'error' not in training_results:
                print("✅ Enhanced training completed!")
                
                training_overview = training_results.get('training_overview', {})
                print(f"  • Agents Trained: {training_overview.get('total_agents_trained', 0)}")
                print(f"  • Average Accuracy: {training_overview.get('average_validation_accuracy', 0):.2%}")
            
            # Generate comprehensive report
            print("\n📋 Generating bonus features report...")
            bonus_report = bonus_manager.generate_bonus_report()
            
            assessment = bonus_report.get('overall_assessment', {})
            print(f"  • Overall Grade: {assessment.get('overall_grade', 'N/A')}")
            
            # Save results
            bonus_manager.save_bonus_results()
            print("💾 Results saved successfully!")
            
        else:
            print(f"❌ Integration failed: {integration_results['error']}")
    
    else:
        print("❌ Failed to initialize bonus features")
    
    print("\n🎉 Bonus features demo completed!")

if __name__ == "__main__":
    main()
