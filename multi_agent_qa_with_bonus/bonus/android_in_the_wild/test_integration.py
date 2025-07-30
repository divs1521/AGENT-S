"""
Quick test script for android_in_the_wild integration
Verifies that the bonus features are working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_basic_imports():
    """Test basic imports work."""
    print("🧪 Testing basic imports...")
    
    try:
        from bonus.android_in_the_wild import AndroidInTheWildProcessor
        print("✅ AndroidInTheWildProcessor imported")
    except ImportError as e:
        print(f"❌ AndroidInTheWildProcessor import failed: {e}")
        return False
    
    try:
        from bonus.android_in_the_wild import VideoReproductionEngine
        print("✅ VideoReproductionEngine imported")
    except ImportError as e:
        print(f"❌ VideoReproductionEngine import failed: {e}")
        return False
    
    try:
        from bonus.android_in_the_wild import demonstrate_android_in_the_wild_integration
        print("✅ Main demo function imported")
    except ImportError as e:
        print(f"❌ Main demo function import failed: {e}")
        return False
    
    return True

def test_dataset_processor():
    """Test dataset processor functionality."""
    print("\n🔍 Testing dataset processor...")
    
    try:
        from bonus.android_in_the_wild.dataset_processor import AndroidInTheWildProcessor
        
        processor = AndroidInTheWildProcessor(
            dataset_url="https://github.com/google-research/google-research/tree/master/android_in_the_wild",
            local_cache_dir="datasets/android_in_the_wild"
        )
        
        # Test getting dataset stats
        stats = processor.get_dataset_stats()
        print(f"✅ Dataset stats retrieved: {stats}")
        
        # Test mock video download (doesn't actually download)
        print("✅ Dataset processor initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dataset processor test failed: {e}")
        return False

def test_video_reproduction():
    """Test video reproduction functionality."""
    print("\n🎬 Testing video reproduction...")
    
    try:
        from bonus.android_in_the_wild.video_reproduction import VideoReproductionEngine
        from core.orchestrator import MultiAgentQAOrchestrator, QASystemConfig
        
        # Create mock orchestrator
        config = QASystemConfig(
            engine_params={'mock': True},
            android_env=None,
            max_execution_time=30.0,
            enable_visual_trace=True,
            enable_recovery=True,
            verification_strictness='balanced'
        )
        orchestrator = MultiAgentQAOrchestrator(config)
        
        # Create reproduction engine
        reproduction_engine = VideoReproductionEngine(orchestrator)
        print("✅ Video reproduction engine created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Video reproduction test failed: {e}")
        return False

def test_enhanced_training():
    """Test enhanced training functionality."""
    print("\n🎓 Testing enhanced training...")
    
    try:
        from bonus.android_in_the_wild.enhanced_training import (
            EnhancedPlannerTraining, 
            EnhancedExecutorTraining,
            EnhancedVerifierTraining,
            EnhancedSupervisorTraining
        )
        from bonus.android_in_the_wild.dataset_processor import AndroidInTheWildProcessor
        
        # Create a dataset processor for training
        processor = AndroidInTheWildProcessor(
            dataset_url="https://github.com/google-research/google-research/tree/master/android_in_the_wild",
            local_cache_dir="datasets/android_in_the_wild"
        )
        
        # Test creating training instances
        planner_training = EnhancedPlannerTraining(processor)
        executor_training = EnhancedExecutorTraining(processor)
        verifier_training = EnhancedVerifierTraining(processor)
        supervisor_training = EnhancedSupervisorTraining(processor)
        
        print("✅ All enhanced training modules created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced training test failed: {e}")
        return False

def test_feature_availability():
    """Test feature availability check."""
    print("\n📊 Testing feature availability...")
    
    try:
        from bonus.android_in_the_wild import get_feature_status, ENHANCED_TRAINING_AVAILABLE
        
        status = get_feature_status()
        print(f"Feature status: {status}")
        print(f"Enhanced training available: {ENHANCED_TRAINING_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature availability test failed: {e}")
        return False

def run_quick_demo():
    """Run a very quick demo of the main functionality."""
    print("\n🚀 Running quick demo...")
    
    try:
        from bonus.android_in_the_wild.main_demo import run_specific_bonus_feature_demo
        
        print("Testing task generation demo...")
        run_specific_bonus_feature_demo("task_generation")
        
        print("\nTesting accuracy scoring demo...")
        run_specific_bonus_feature_demo("accuracy_scoring")
        
        print("✅ Quick demo completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Android in the Wild Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Dataset Processor", test_dataset_processor),
        ("Video Reproduction", test_video_reproduction),
        ("Enhanced Training", test_enhanced_training),
        ("Feature Availability", test_feature_availability),
        ("Quick Demo", run_quick_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! android_in_the_wild integration is working correctly.")
        print("\nYou can now run the full demo:")
        print("  python bonus/android_in_the_wild/main_demo.py --full")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("\nFor missing dependencies, try:")
        print("  python bonus/android_in_the_wild/install_dependencies.py --install")
    
    return passed == total

if __name__ == "__main__":
    main()
