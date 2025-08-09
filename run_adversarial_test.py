#!/usr/bin/env python3

import sys
import traceback

def run_test():
    """Run a simple validation test for test_adversarial.py"""
    try:
        # Try to import the main components from test_adversarial
        print("Testing imports...")
        from test_adversarial import TestFixtures, make_record
        print("✓ TestFixtures and make_record imported")
        
        from app_adversarial import AttackType, AdversarialConfig, AdversarialMode, EvasionTechniques
        print("✓ Adversarial components imported")
        
        from app_telemetry import TelemetryRecord, ExecutionPhase, AnomalyType, TelemetrySource
        print("✓ Telemetry components imported")
        
        # Test basic record creation
        record = make_record()
        print(f"✓ Basic record created: {record.event_id}")
        
        # Test TestFixtures
        sample_telemetry = TestFixtures.create_sample_telemetry()
        print(f"✓ Sample telemetry created: {sample_telemetry.event_id}")
        
        # Test evasion techniques
        original = make_record()
        modified = EvasionTechniques.noise_injection(original, noise_level=0.1)
        print(f"✓ Noise injection test passed: original duration={original.duration}, modified duration={modified.duration}")
        
        # Test configuration
        config = TestFixtures.create_test_config()
        print(f"✓ Test config created: mode={config.adversarial_mode}")
        
        # Test configuration validation
        issues = config.validate()
        print(f"✓ Config validation returned {len(issues)} issues")
        
        # Test individual test functions
        print("\nTesting individual test functions...")
        
        # Test noise injection function
        import test_adversarial
        test_adversarial.test_noise_injection_does_not_mutate_original()
        print("✓ test_noise_injection_does_not_mutate_original passed")
        
        test_adversarial.test_gradient_masking_adds_time_jitter() 
        print("✓ test_gradient_masking_adds_time_jitter passed")
        
        test_adversarial.test_input_transformation_logarithmic()
        print("✓ test_input_transformation_logarithmic passed")
        
        test_adversarial.test_adaptive_perturbation_with_epsilon()
        print("✓ test_adaptive_perturbation_with_epsilon passed")
        
        test_adversarial.test_attack_vector_defaults()
        print("✓ test_attack_vector_defaults passed")
        
        test_adversarial.test_adversarial_config_validation()
        print("✓ test_adversarial_config_validation passed")
        
        test_adversarial.test_engine_generates_attack()
        print("✓ test_engine_generates_attack passed")
        
        print("\n✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)