#!/usr/bin/env python3
"""Simple test runner to verify test_adversarial.py works"""

import sys
import os
sys.path.insert(0, '/workspace')

def test_imports():
    """Test that all imports work correctly"""
    try:
        print("Testing imports...")
        
        # Test main module import
        import test_adversarial
        print("✓ test_adversarial module imported")
        
        # Test key functions exist
        assert hasattr(test_adversarial, 'make_record'), "make_record function missing"
        assert hasattr(test_adversarial, 'TestFixtures'), "TestFixtures class missing"
        print("✓ Key functions and classes exist")
        
        # Test basic functionality
        record = test_adversarial.make_record()
        assert record.event_id == "evt", f"Expected 'evt', got {record.event_id}"
        print("✓ make_record() works correctly")
        
        # Test TestFixtures
        sample = test_adversarial.TestFixtures.create_sample_telemetry()
        assert sample.event_id == "test_event_001", f"Expected 'test_event_001', got {sample.event_id}"
        print("✓ TestFixtures.create_sample_telemetry() works")
        
        config = test_adversarial.TestFixtures.create_test_config()
        print(f"✓ TestFixtures.create_test_config() works - mode: {config.adversarial_mode}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_functions():
    """Test individual test functions"""
    try:
        import test_adversarial
        
        print("\nTesting individual functions...")
        
        # List of test functions to validate
        test_functions = [
            'test_noise_injection_does_not_mutate_original',
            'test_gradient_masking_adds_time_jitter',
            'test_input_transformation_logarithmic',
            'test_adaptive_perturbation_with_epsilon',
            'test_attack_vector_defaults',
            'test_adversarial_config_validation',
            'test_engine_generates_attack'
        ]
        
        for func_name in test_functions:
            if hasattr(test_adversarial, func_name):
                try:
                    func = getattr(test_adversarial, func_name)
                    func()
                    print(f"✓ {func_name}() passed")
                except Exception as e:
                    print(f"❌ {func_name}() failed: {e}")
                    return False
            else:
                print(f"⚠ {func_name}() not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("=== Testing test_adversarial.py ===")
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import tests failed")
        return False
    
    # Test 2: Individual functions  
    if not test_individual_functions():
        print("❌ Function tests failed")
        return False
    
    print("\n✅ ALL TESTS PASSED - test_adversarial.py works correctly!")
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\nExit code: {exit_code}")