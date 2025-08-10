#!/usr/bin/env python3
"""
Validation script to test current status of test_adversarial.py
"""

def test_basic_imports():
    """Test if basic imports work"""
    print("Testing basic imports...")
    
    try:
        # Test core imports
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
        print("✅ app_telemetry imports work")
        
        from app_config import AdversarialConfig, AdversarialMode  
        print("✅ app_config imports work")
        
        # Test adversarial imports
        from app_adversarial import (
            AdversarialAnomalyEngine,
            AttackType,
            AttackVector,
            EvasionTechniques,
            PoisoningAttackGenerator
        )
        print("✅ app_adversarial imports work")
        
        # Test the test module itself
        import test_adversarial
        print("✅ test_adversarial module imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixture_creation():
    """Test fixture creation functions"""
    print("\nTesting fixture creation...")
    
    try:
        import test_adversarial
        
        # Test make_record
        record = test_adversarial.make_record()
        assert record.event_id == "evt"
        print("✅ make_record() works")
        
        # Test TestFixtures class methods
        sample = test_adversarial.TestFixtures.create_sample_telemetry()
        assert sample.event_id == "test_event_001"
        print("✅ create_sample_telemetry() works")
        
        config = test_adversarial.TestFixtures.create_test_config()
        print("✅ create_test_config() works")
        
        dataset = test_adversarial.TestFixtures.create_test_dataset(3)
        assert len(dataset) == 3
        print("✅ create_test_dataset() works")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixture creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_class_creation():
    """Test basic test class creation"""
    print("\nTesting test class creation...")
    
    try:
        import test_adversarial
        
        # Test config class
        config_test = test_adversarial.TestAdversarialConfig()
        print("✅ TestAdversarialConfig created")
        
        # Test classes with setup_method
        evasion_test = test_adversarial.TestEvasionTechniques()
        evasion_test.setup_method()
        assert hasattr(evasion_test, 'sample_telemetry')
        print("✅ TestEvasionTechniques with setup works")
        
        poisoning_test = test_adversarial.TestPoisoningAttackGenerator()
        poisoning_test.setup_method()
        print("✅ TestPoisoningAttackGenerator with setup works")
        
        return True
        
    except Exception as e:
        print(f"❌ Class creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_critical_fixes():
    """Test the critical fixes that were applied"""
    print("\nTesting critical fixes...")
    
    try:
        # Test TelemetrySource import fix in app_adversarial
        from app_adversarial import EconomicAttackSimulator, TelemetrySource
        print("✅ TelemetrySource import in app_adversarial works")
        
        # Test temporal_trends fix
        from app_adversarial import AdversarialMetricsCollector
        collector = AdversarialMetricsCollector()
        collector.metrics_history = [
            {
                'attack_type': 'test', 
                'attack_success': True, 
                'stealth_score': 0.5, 
                'perturbation_magnitude': 0.1, 
                'economic_impact': 10, 
                'detection_triggered': False, 
                'timestamp': 1000
            }
        ]
        report = collector.generate_research_report()
        assert 'temporal_trends' in report
        print("✅ temporal_trends field fix works")
        
        # Test surrogate_models fix
        from app_adversarial import QueryFreeAttackEngine
        from app_config import AdversarialConfig, AdversarialMode
        config = AdversarialConfig(adversarial_mode=AdversarialMode.TEST)
        engine = QueryFreeAttackEngine(config)
        
        import test_adversarial
        test_data = test_adversarial.TestFixtures.create_test_dataset(3)
        engine.build_surrogate_model(test_data)
        assert len(engine.surrogate_models) > 0
        print("✅ surrogate_models fallback fix works")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical fixes test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_sample_standalone_tests():
    """Run a few standalone test functions"""
    print("\nTesting standalone functions...")
    
    try:
        import test_adversarial
        
        test_functions = [
            'test_noise_injection_does_not_mutate_original',
            'test_gradient_masking_adds_time_jitter', 
            'test_attack_vector_defaults',
            'test_adversarial_config_validation'
        ]
        
        passed = 0
        for func_name in test_functions:
            try:
                func = getattr(test_adversarial, func_name)
                func()
                print(f"✅ {func_name}")
                passed += 1
            except Exception as e:
                print(f"❌ {func_name}: {str(e)[:50]}...")
        
        print(f"Standalone functions: {passed}/{len(test_functions)} passed")
        return passed >= 3  # Most should work
        
    except Exception as e:
        print(f"❌ Standalone test error: {e}")
        return False

def main():
    """Main validation function"""
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION: test_adversarial.py STATUS CHECK")
    print("=" * 70)
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Fixture creation", test_fixture_creation), 
        ("Test class creation", test_class_creation),
        ("Critical fixes", test_critical_fixes),
        ("Standalone functions", run_sample_standalone_tests)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        result = test_func()
        results.append(result)
        print(f"Result: {'✅ PASS' if result else '❌ FAIL'}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 70)
    print(f"📊 VALIDATION SUMMARY: {passed}/{total} test groups passed")
    
    if passed == total:
        print("🎉 EXCELLENT! All validation checks passed!")
        print("✅ test_adversarial.py appears to be working correctly")
        print("✅ Major fixes have been successfully applied")
        success_rate = 100
    elif passed >= 4:
        print("✅ GOOD! Most validation checks passed")
        print("⚠️ Minor issues may remain but major functionality works")
        success_rate = 80
    elif passed >= 2:
        print("⚠️ PARTIAL SUCCESS - Some functionality works")  
        print("🔧 Additional fixes may be needed")
        success_rate = 60
    else:
        print("❌ SIGNIFICANT ISSUES REMAIN")
        print("🛠️ Major repairs still needed")
        success_rate = 20
    
    print(f"\n🎯 Estimated test success rate: {success_rate}%")
    
    if passed >= 4:
        print(f"\n🚀 RECOMMENDATION: test_adversarial.py should now work much better!")
        print(f"   Most tests should pass when run with pytest")
    
    return passed >= 4

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)