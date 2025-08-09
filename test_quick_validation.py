#!/usr/bin/env python3
"""
Quick validation of the fixes applied to test_adversarial.py
"""

def test_imports():
    """Test that all imports work after fixes"""
    try:
        import test_adversarial
        print("✅ test_adversarial imported successfully")
        
        # Test basic functionality
        record = test_adversarial.make_record()
        print(f"✅ make_record() works: {record.event_id}")
        
        # Test fixtures
        sample = test_adversarial.TestFixtures.create_sample_telemetry() 
        print(f"✅ TestFixtures work: {sample.event_id}")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functions():
    """Test the standalone functions work"""
    try:
        import test_adversarial
        
        print("Testing standalone functions...")
        
        # Test each function
        test_adversarial.test_noise_injection_does_not_mutate_original()
        print("✅ test_noise_injection_does_not_mutate_original")
        
        test_adversarial.test_gradient_masking_adds_time_jitter()
        print("✅ test_gradient_masking_adds_time_jitter")
        
        test_adversarial.test_input_transformation_logarithmic()
        print("✅ test_input_transformation_logarithmic")
        
        test_adversarial.test_adaptive_perturbation_with_epsilon()
        print("✅ test_adaptive_perturbation_with_epsilon")
        
        test_adversarial.test_attack_vector_defaults()
        print("✅ test_attack_vector_defaults")
        
        test_adversarial.test_adversarial_config_validation()
        print("✅ test_adversarial_config_validation")
        
        test_adversarial.test_engine_generates_attack()
        print("✅ test_engine_generates_attack")
        
        return True
        
    except Exception as e:
        print(f"❌ Function test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_instantiation():
    """Test that test classes can be created"""
    try:
        import test_adversarial
        
        print("Testing class instantiation...")
        
        # Test basic classes
        config_test = test_adversarial.TestAdversarialConfig()
        print("✅ TestAdversarialConfig")
        
        evasion_test = test_adversarial.TestEvasionTechniques()
        evasion_test.setup_method()
        print("✅ TestEvasionTechniques with setup")
        
        poisoning_test = test_adversarial.TestPoisoningAttackGenerator()
        poisoning_test.setup_method()
        print("✅ TestPoisoningAttackGenerator with setup")
        
        return True
        
    except Exception as e:
        print(f"❌ Class test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("QUICK VALIDATION OF test_adversarial.py FIXES")
    print("=" * 60)
    
    results = []
    
    print("\n1. Testing imports...")
    results.append(test_imports())
    
    print("\n2. Testing standalone functions...")
    results.append(test_basic_functions())
    
    print("\n3. Testing class instantiation...")
    results.append(test_class_instantiation())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL QUICK TESTS PASSED ({passed}/{total})")
        print("✅ Major fixes are working!")
        return True
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)