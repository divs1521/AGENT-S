#!/usr/bin/env python3
"""Test runner for Multi-Agent QA System."""

import sys
import os
import unittest
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all tests in the test directory."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_unit_tests():
    """Run only unit tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / 'unit'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run only integration tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / 'integration'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_api_tests():
    """Run only API tests."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / 'api'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for Multi-Agent QA System')
    parser.add_argument('--type', choices=['all', 'unit', 'integration', 'api'], 
                       default='all', help='Type of tests to run')
    
    args = parser.parse_args()
    
    print(f"Running {args.type} tests...")
    print("=" * 50)
    
    if args.type == 'all':
        success = run_all_tests()
    elif args.type == 'unit':
        success = run_unit_tests()
    elif args.type == 'integration':
        success = run_integration_tests()
    elif args.type == 'api':
        success = run_api_tests()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
