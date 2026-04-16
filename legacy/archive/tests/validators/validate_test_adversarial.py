#!/usr/bin/env python3
"""
Final validation script to ensure test_adversarial.py works correctly
"""

def main():
    print("=== Validating test_adversarial.py ===\n")
    
    try:
        # Step 1: Test basic imports
        print("1. Testing imports...")
        import test_adversarial
        print("   ✓ Main module imported successfully")
        
        # Step 2: Test fixture creation
        print("\n2. Testing TestFixtures...")
        sample_record = test_adversarial.TestFixtures.create_sample_telemetry()
        print(f"   ✓ Sample telemetry created: {sample_record.event_id}")
        
        config = test_adversarial.TestFixtures.create_test_config()
        print(f"   ✓ Test config created: {config.adversarial_mode}")
        
        dataset = test_adversarial.TestFixtures.create_test_dataset(5)
        print(f"   ✓ Test dataset created with {len(dataset)} records")
        
        # Step 3: Test basic record creation
        print("\n3. Testing make_record function...")
        record = test_adversarial.make_record()
        print(f"   ✓ Record created: {record.event_id}")
        
        # Step 4: Test standalone functions
        print("\n4. Testing standalone test functions...")
        
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
            func = getattr(test_adversarial, func_name)
            func()
            print(f"   ✓ {func_name}() passed")
        
        # Step 5: Test class instantiation
        print("\n5. Testing test class instantiation...")
        
        # Test AdversarialConfig class
        test_config_class = test_adversarial.TestAdversarialConfig()
        print("   ✓ TestAdversarialConfig instantiated")
        
        # Test EvasionTechniques class
        test_evasion = test_adversarial.TestEvasionTechniques()
        test_evasion.setup_method()
        print("   ✓ TestEvasionTechniques instantiated and setup")
        
        # Test PoisoningAttackGenerator class
        test_poisoning = test_adversarial.TestPoisoningAttackGenerator()
        test_poisoning.setup_method()
        print("   ✓ TestPoisoningAttackGenerator instantiated and setup")
        
        print(f"\n✅ ALL VALIDATIONS PASSED! test_adversarial.py is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)