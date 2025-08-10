#!/usr/bin/env python3
"""
Quick verification that test_adversarial.py works after fixing the enum issue
"""

def test_basic_import():
    """Test that test_adversarial.py can be imported"""
    try:
        print("Testing import of test_adversarial...")
        import test_adversarial
        print("✅ test_adversarial imported successfully!")
        
        # Test make_record function
        record = test_adversarial.make_record()
        print(f"✅ make_record() works: {record.event_id}")
        
        # Test TestFixtures
        sample = test_adversarial.TestFixtures.create_sample_telemetry()
        print(f"✅ TestFixtures.create_sample_telemetry() works: {sample.event_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_functions():
    """Test individual functions"""
    try:
        import test_adversarial
        
        print("\nTesting individual test functions...")
        
        # Test each function
        functions_to_test = [
            'test_noise_injection_does_not_mutate_original',
            'test_gradient_masking_adds_time_jitter', 
            'test_input_transformation_logarithmic',
            'test_adaptive_perturbation_with_epsilon',
            'test_attack_vector_defaults',
            'test_adversarial_config_validation',
            'test_engine_generates_attack'
        ]
        
        passed = 0
        for func_name in functions_to_test:
            try:
                func = getattr(test_adversarial, func_name)
                func()
                print(f"✅ {func_name}() passed")
                passed += 1
            except Exception as e:
                print(f"❌ {func_name}() failed: {e}")
        
        print(f"\n📊 Results: {passed}/{len(functions_to_test)} functions passed")
        return passed == len(functions_to_test)
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING test_adversarial.py AFTER ENUM FIX")
    print("=" * 60)
    
    # Test 1: Basic import
    import_ok = test_basic_import()
    
    if import_ok:
        # Test 2: Individual functions
        functions_ok = test_individual_functions()
        
        if functions_ok:
            print("\n🎉 ALL TESTS PASSED! test_adversarial.py works correctly!")
            return True
        else:
            print("\n⚠️  Import worked but some functions failed")
            return False
    else:
        print("\n❌ Import failed")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    print(f"\nFinal Result: {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1)