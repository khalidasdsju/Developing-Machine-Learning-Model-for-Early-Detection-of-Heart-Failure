"""
Script to run all tests for the Heart Failure Detection project.
"""

import unittest
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)
