#!/usr/bin/env python3
"""
Quick test check to see current status of test_adversarial.py fixes
"""

def check_imports():
    """Check if imports are working"""
    print("=== IMPORT CHECK ===")
    try:
        import test_adversarial
        print("✅ test_adversarial module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_basic_functionality():
    """Check basic functionality"""
    print("\n=== BASIC FUNCTIONALITY CHECK ===")
    try:
        import test_adversarial
        
        # Test make_record function
        record = test_adversarial.make_record()
        print(f"✅ make_record() works: {record.event_id}")
        
        # Test fixtures
        sample = test_adversarial.TestFixtures.create_sample_telemetry()
        print(f"✅ create_sample_telemetry() works: {sample.event_id}")
        
        config = test_adversarial.TestFixtures.create_test_config()
        print(f"✅ create_test_config() works: {config.adversarial_mode}")
        
        dataset = test_adversarial.TestFixtures.create_test_dataset(3)
        print(f"✅ create_test_dataset() works: {len(dataset)} records")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_class_instantiation():
    """Check if test classes can be instantiated"""
    print("\n=== CLASS INSTANTIATION CHECK ===")
    try:
        import test_adversarial
        
        classes_to_test = [
            'TestAdversarialConfig',
            'TestEvasionTechniques', 
            'TestPoisoningAttackGenerator',
            'TestEconomicAttackSimulator',
        ]
        
        for class_name in classes_to_test:
            try:
                cls = getattr(test_adversarial, class_name)
                instance = cls()
                
                # Try setup_method if it exists
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                    print(f"✅ {class_name} with setup_method")
                else:
                    print(f"✅ {class_name} (no setup needed)")
                    
            except Exception as e:
                print(f"❌ {class_name}: {str(e)[:60]}...")
        
        return True
    except Exception as e:
        print(f"❌ Class instantiation check failed: {e}")
        return False

def check_standalone_functions():
    """Check standalone test functions"""
    print("\n=== STANDALONE FUNCTIONS CHECK ===")
    try:
        import test_adversarial
        
        functions_to_test = [
            'test_noise_injection_does_not_mutate_original',
            'test_gradient_masking_adds_time_jitter',
            'test_input_transformation_logarithmic',
            'test_attack_vector_defaults',
            'test_engine_generates_attack'
        ]
        
        passed = 0
        for func_name in functions_to_test:
            try:
                func = getattr(test_adversarial, func_name)
                func()
                print(f"✅ {func_name}")
                passed += 1
            except Exception as e:
                print(f"❌ {func_name}: {str(e)[:50]}...")
        
        print(f"\nStandalone functions: {passed}/{len(functions_to_test)} passed")
        return passed > 0
        
    except Exception as e:
        print(f"❌ Standalone function check failed: {e}")
        return False

def main():
    """Main test runner"""
    print("QUICK TEST STATUS CHECK FOR test_adversarial.py")
    print("=" * 60)
    
    results = []
    results.append(check_imports())
    
    if results[0]:  # Only proceed if imports work
        results.append(check_basic_functionality())
        results.append(check_class_instantiation()) 
        results.append(check_standalone_functions())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 Great! Major functionality is working")
        print("✅ test_adversarial.py appears to be in good shape")
    elif passed > 0:
        print("⚠️ Some functionality working, some issues remain")
    else:
        print("❌ Major issues still present")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)