#!/usr/bin/env python3
"""Run basic test"""

import subprocess
import sys

def run_test():
    try:
        # Run the basic functionality test
        result = subprocess.run([sys.executable, "test_basic_functionality.py"], 
                              capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"Failed to run test: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("✓ Basic tests passed")
    else:
        print("✗ Basic tests failed")