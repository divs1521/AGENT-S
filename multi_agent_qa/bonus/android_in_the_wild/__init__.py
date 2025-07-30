"""
Android in the Wild Dataset Integration (Bonus Features)

This package integrates the real android_in_the_wild dataset from Google Research
to enhance the multi-agent QA system with actual user interaction data.

Key Components:
- dataset_processor: Real GitHub dataset integration and processing
- video_reproduction: Video flow reproduction with multi-agent system
- enhanced_training: Enhanced training modules for all four agents
- main_demo: Complete integration demonstration

Usage:
    from bonus.android_in_the_wild.main_demo import demonstrate_android_in_the_wild_integration
    
    # Run complete integration
    results = demonstrate_android_in_the_wild_integration()
    print(f"Success Rate: {results['batch_report']['summary']['success_rate']:.2%}")

Features:
- Real dataset from https://github.com/google-research/google-research/tree/master/android_in_the_wild
- Video processing and task generation
- Multi-agent reproduction system with accuracy scoring
- Enhanced training for Planner, Executor, Verifier, and Supervisor agents
- Comprehensive performance metrics and analysis
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent QA Team"

# Import main components for easy access
try:
    from .dataset_processor import AndroidInTheWildProcessor
    from .video_reproduction import VideoReproductionEngine
    from .main_demo import demonstrate_android_in_the_wild_integration, run_specific_bonus_feature_demo
    
    # Try to import enhanced training (may fail due to missing dependencies)
    try:
        from .enhanced_training import create_comprehensive_training_dataset
        ENHANCED_TRAINING_AVAILABLE = True
    except ImportError:
        ENHANCED_TRAINING_AVAILABLE = False
        create_comprehensive_training_dataset = None
    
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Some android_in_the_wild components unavailable: {e}")
    AndroidInTheWildProcessor = None
    VideoReproductionEngine = None
    demonstrate_android_in_the_wild_integration = None
    run_specific_bonus_feature_demo = None
    create_comprehensive_training_dataset = None
    ENHANCED_TRAINING_AVAILABLE = False

# Export public interface
__all__ = [
    'AndroidInTheWildProcessor',
    'VideoReproductionEngine', 
    'demonstrate_android_in_the_wild_integration',
    'run_specific_bonus_feature_demo',
    'create_comprehensive_training_dataset',
    'ENHANCED_TRAINING_AVAILABLE'
]

def get_feature_status():
    """Return status of available features."""
    return {
        'dataset_processor': AndroidInTheWildProcessor is not None,
        'video_reproduction': VideoReproductionEngine is not None,
        'main_demo': demonstrate_android_in_the_wild_integration is not None,
        'enhanced_training': ENHANCED_TRAINING_AVAILABLE,
        'version': __version__
    }

def print_feature_status():
    """Print status of available features."""
    status = get_feature_status()
    print("Android in the Wild Integration Status:")
    print(f"  Version: {status['version']}")
    print(f"  Dataset Processor: {'✅' if status['dataset_processor'] else '❌'}")
    print(f"  Video Reproduction: {'✅' if status['video_reproduction'] else '❌'}")
    print(f"  Main Demo: {'✅' if status['main_demo'] else '❌'}")
    print(f"  Enhanced Training: {'✅' if status['enhanced_training'] else '❌'}")
    
    if not status['enhanced_training']:
        print("\n💡 To enable enhanced training features, install:")
        print("  pip install opencv-python numpy scikit-learn matplotlib seaborn")

# Quick access functions
def quick_demo():
    """Run a quick demonstration of the android_in_the_wild integration."""
    if demonstrate_android_in_the_wild_integration is None:
        print("❌ Demo not available. Check dependencies.")
        return None
    
    print("🚀 Running quick android_in_the_wild demo...")
    return demonstrate_android_in_the_wild_integration()

def check_dataset_access():
    """Check if we can access the real android_in_the_wild dataset."""
    if AndroidInTheWildProcessor is None:
        return False, "Dataset processor not available"
    
    try:
        processor = AndroidInTheWildProcessor(
            dataset_url="https://github.com/google-research/google-research/tree/master/android_in_the_wild",
            local_cache_dir="datasets/android_in_the_wild"
        )
        stats = processor.get_dataset_stats()
        return True, f"Dataset accessible: {stats.get('total_episodes', 'Unknown')} episodes"
    except Exception as e:
        return False, f"Dataset access failed: {e}"

if __name__ == "__main__":
    print_feature_status()
    
    # Check dataset access
    accessible, message = check_dataset_access()
    print(f"\nDataset Access: {'✅' if accessible else '❌'} {message}")
