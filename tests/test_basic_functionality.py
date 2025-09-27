#!/usr/bin/env python3
"""Basic functionality test to validate interconnections"""

import sys
import time
import asyncio
from typing import Dict, Any

def test_imports():
    """Test that all modules can be imported"""
    try:
        from app_config import AdversarialConfig, AdversarialMode
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
        from app_adversarial import (
            AdversarialAnomalyEngine, AttackType, AttackResult,
            EvasionTechniques, PoisoningAttackGenerator, 
            EconomicAttackSimulator, AdversarialTestSuite
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_instantiation():
    """Test basic class instantiation"""
    try:
        from app_config import AdversarialConfig, AdversarialMode
        from app_adversarial import AdversarialAnomalyEngine
        
        config = AdversarialConfig(adversarial_mode=AdversarialMode.TEST)
        engine = AdversarialAnomalyEngine(config)
        print("✓ Basic instantiation successful")
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_telemetry_record_creation():
    """Test TelemetryRecord creation"""
    try:
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
        
        record = TelemetryRecord(
            event_id="test_001",
            timestamp=time.time(),
            function_id="test_function",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=1.5,
            memory_spike_kb=128000,
            cpu_utilization=45.2,
            network_io_bytes=2048,
            fallback_mode=False,
            source=TelemetrySource.PRIMARY,
            concurrency_id="test_concurrency_001"
        )
        print("✓ TelemetryRecord creation successful")
        return True
    except Exception as e:
        print(f"✗ TelemetryRecord creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evasion_techniques():
    """Test EvasionTechniques static methods"""
    try:
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
        from app_adversarial import EvasionTechniques
        
        # Create sample record
        record = TelemetryRecord(
            event_id="test_001",
            timestamp=time.time(),
            function_id="test_function",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=1.5,
            memory_spike_kb=128000,
            cpu_utilization=45.2,
            network_io_bytes=2048,
            fallback_mode=False,
            source=TelemetrySource.PRIMARY,
            concurrency_id="test_concurrency_001"
        )
        
        # Test noise injection
        noisy_record = EvasionTechniques.noise_injection(record, noise_level=0.1)
        assert noisy_record.duration != record.duration
        
        # Test gradient masking
        masked_record = EvasionTechniques.gradient_masking(record, masking_strength=0.2)
        assert masked_record.timestamp >= record.timestamp
        
        print("✓ EvasionTechniques methods work")
        return True
    except Exception as e:
        print(f"✗ EvasionTechniques test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_adversarial_config_validation():
    """Test AdversarialConfig validation"""
    try:
        from app_config import AdversarialConfig, AdversarialMode
        
        # Test valid config
        config = AdversarialConfig(adversarial_mode=AdversarialMode.TEST)
        issues = config.validate()
        
        # Test invalid config
        bad_config = AdversarialConfig(
            gan_latent_dim=5,  # Too small
            max_evasion_budget=1.5  # Invalid range
        )
        bad_issues = bad_config.validate()
        assert len(bad_issues) > 0
        
        print("✓ AdversarialConfig validation works")
        return True
    except Exception as e:
        print(f"✗ AdversarialConfig validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_attack_generation():
    """Test async attack generation"""
    try:
        from app_config import AdversarialConfig, AdversarialMode
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
        from app_adversarial import AdversarialAnomalyEngine, AttackType
        
        config = AdversarialConfig(
            adversarial_mode=AdversarialMode.TEST,
            enable_gan_generation=False  # Disable GAN for simpler test
        )
        engine = AdversarialAnomalyEngine(config)
        
        record = TelemetryRecord(
            event_id="test_001",
            timestamp=time.time(),
            function_id="test_function",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=1.5,
            memory_spike_kb=128000,
            cpu_utilization=45.2,
            network_io_bytes=2048,
            fallback_mode=False,
            source=TelemetrySource.PRIMARY,
            concurrency_id="test_concurrency_001"
        )
        
        # Generate a simple attack
        attack_result = await engine.generate_adversarial_anomaly(record, AttackType.NOISE_INJECTION)
        assert attack_result.attack_type == AttackType.NOISE_INJECTION
        assert len(attack_result.generated_telemetry) > 0
        
        print("✓ Async attack generation successful")
        return True
    except Exception as e:
        print(f"✗ Async attack generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🧪 Running basic functionality tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_instantiation,
        test_telemetry_record_creation,
        test_evasion_techniques,
        test_adversarial_config_validation,
        test_async_attack_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print(f"\n📋 Running {test_func.__name__}...")
        
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        
        if result:
            passed += 1
        
        print("-" * 30)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(result)