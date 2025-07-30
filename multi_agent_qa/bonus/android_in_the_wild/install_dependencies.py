"""
Installation script for android_in_the_wild bonus features
Installs optional dependencies for enhanced functionality.
"""

import subprocess
import sys
import os

def install_package(package_name, import_name=None):
    """Install a package and test if it can be imported."""
    import_name = import_name or package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            __import__(import_name)
            print(f"✅ {package_name} installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def install_bonus_dependencies():
    """Install all optional dependencies for bonus features."""
    print("🚀 Installing android_in_the_wild bonus feature dependencies...")
    print("=" * 60)
    
    dependencies = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for package, import_name in dependencies:
        if install_package(package, import_name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Summary: {success_count}/{total_count} packages installed")
    
    if success_count == total_count:
        print("🎉 All bonus features are now available!")
        print("\nYou can now run:")
        print("  python main_demo.py --full")
    else:
        print("⚠️  Some packages failed to install. Basic features will still work.")
        print("\nCore features available:")
        print("  python main_demo.py --feature task_generation")
        print("  python main_demo.py --feature comparison")
    
    return success_count == total_count

def test_installation():
    """Test if all installations are working correctly."""
    print("\n🧪 Testing installation...")
    
    tests = [
        ("numpy", "import numpy; print('numpy version:', numpy.__version__)"),
        ("opencv", "import cv2; print('opencv version:', cv2.__version__)"),
        ("sklearn", "import sklearn; print('sklearn version:', sklearn.__version__)"),
        ("matplotlib", "import matplotlib; print('matplotlib version:', matplotlib.__version__)"),
        ("seaborn", "import seaborn; print('seaborn version:', seaborn.__version__)")
    ]
    
    for name, test_code in tests:
        try:
            exec(test_code)
            print(f"✅ {name} test passed")
        except Exception as e:
            print(f"❌ {name} test failed: {e}")

def check_bonus_features_available():
    """Check which bonus features are available."""
    print("\n📋 Checking bonus feature availability...")
    
    try:
        # Test basic imports
        from bonus.android_in_the_wild import get_feature_status
        status = get_feature_status()
        
        print(f"Dataset Processor: {'✅' if status['dataset_processor'] else '❌'}")
        print(f"Video Reproduction: {'✅' if status['video_reproduction'] else '❌'}")
        print(f"Main Demo: {'✅' if status['main_demo'] else '❌'}")
        print(f"Enhanced Training: {'✅' if status['enhanced_training'] else '❌'}")
        
        if all(status.values()):
            print("\n🎯 All bonus features are ready to use!")
        else:
            print("\n⚠️  Some features may have limited functionality.")
            
    except ImportError as e:
        print(f"❌ Cannot import bonus features: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Install android_in_the_wild bonus dependencies")
    parser.add_argument("--install", action="store_true", help="Install optional dependencies")
    parser.add_argument("--test", action="store_true", help="Test installation")
    parser.add_argument("--check", action="store_true", help="Check feature availability")
    
    args = parser.parse_args()
    
    if args.install:
        install_bonus_dependencies()
    
    if args.test:
        test_installation()
    
    if args.check:
        check_bonus_features_available()
    
    if not any([args.install, args.test, args.check]):
        # Default: install and test
        print("Installing and testing android_in_the_wild bonus features...")
        success = install_bonus_dependencies()
        if success:
            test_installation()
        check_bonus_features_available()
