#!/usr/bin/env python3
"""
Test status runner - executes the test check
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, '/workspace')

# Import and run the test check
try:
    from run_quick_test_check import main
    success = main()
    print(f"\nTest check result: {'SUCCESS' if success else 'NEEDS WORK'}")
except Exception as e:
    print(f"Test runner error: {e}")
    import traceback
    traceback.print_exc()
    success = False

sys.exit(0 if success else 1)