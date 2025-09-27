#!/usr/bin/env python3
"""Check if test_adversarial.py can be imported successfully"""

import sys
import os

# Add workspace to Python path
sys.path.insert(0, '/workspace')

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'app_adversarial',
        'app_config', 
        'app_telemetry',
        'numpy',
        'pytest'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError:
            print(f"❌ {module} missing")
            missing.append(module)
    
    return len(missing) == 0

def check_test_file():
    """Check if test_adversarial.py can be imported"""
    try:
        import test_adversarial
        print("✓ test_adversarial.py imported successfully")
        
        # Test basic functionality
        record = test_adversarial.make_record()
        print(f"✓ make_record() works: {record.event_id}")
        
        return True
    except Exception as e:
        print(f"❌ Error importing test_adversarial.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Checking dependencies and imports...\n")
    
    deps_ok = check_dependencies()
    print()
    
    if deps_ok:
        test_ok = check_test_file()
        if test_ok:
            print("\n✅ SUCCESS: test_adversarial.py works correctly!")
            return True
        else:
            print("\n❌ FAILURE: test_adversarial.py has issues")
            return False
    else:
        print("\n❌ FAILURE: Missing dependencies")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"Exit code: {exit_code}")