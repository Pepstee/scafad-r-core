#!/usr/bin/env python3
"""
Clean Integration Test for SCAFAD Components
===========================================

This test verifies that all the adversarial components work together seamlessly.
"""

import pytest
import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any

# Import all the components we need to test
try:
    from app_config import AdversarialConfig, AdversarialMode
    from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
    from app_adversarial import (
        AdversarialAnomalyEngine, AttackType, AttackResult,
        EvasionTechniques, PoisoningAttackGenerator, 
        EconomicAttackSimulator, AdversarialTestSuite,
        AdversarialMetricsCollector, AdversarialValidationFramework,
        QueryFreeAttackEngine, TransferAttackEngine,
        AdversarialRobustnessAnalyzer, MultiStepCampaignOrchestrator
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    raise

class TestHelper:
    """Helper class for creating test data"""
    
    @staticmethod
    def create_sample_telemetry() -> TelemetryRecord:
        """Create a sample telemetry record"""
        return TelemetryRecord(
            event_id="test_event_001",
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
    
    @staticmethod
    def create_test_config() -> AdversarialConfig:
        """Create a test configuration"""
        return AdversarialConfig(
            adversarial_mode=AdversarialMode.TEST,
            enable_evasion_techniques=True,
            enable_poisoning_attacks=True,
            max_evasion_budget=0.1,
            max_poisoning_rate=0.05,
            enable_gan_generation=False,  # Disable for testing
            gan_epochs=10,
            gan_batch_size=16
        )
    
    @staticmethod
    def create_test_dataset(size: int) -> List[TelemetryRecord]:
        """Create a test dataset"""
        dataset = []
        for i in range(size):
            record = TelemetryRecord(
                event_id=f"test_event_{i}",
                timestamp=time.time() - (size - i) * 60,
                function_id=f"test_function_{i % 3}",
                execution_phase=ExecutionPhase.INVOKE,
                anomaly_type=AnomalyType.BENIGN if i % 4 != 0 else AnomalyType.CPU_BURST,
                duration=random.uniform(0.1, 5.0),
                memory_spike_kb=random.randint(64000, 256000),
                cpu_utilization=random.uniform(10.0, 90.0),
                network_io_bytes=random.randint(1000, 50000),
                fallback_mode=False,
                source=TelemetrySource.SCAFAD_LAYER0,
                concurrency_id=f"test_concurrency_{i}"
            )
            dataset.append(record)
        return dataset

def test_basic_config():
    """Test basic configuration creation and validation"""
    print("🧪 Testing basic configuration...")
    
    # Test valid config
    config = TestHelper.create_test_config()
    assert config.adversarial_mode == AdversarialMode.TEST
    assert config.enable_gan_generation is False
    
    # Test validation
    issues = config.validate()
    assert isinstance(issues, list)
    
    print("✓ Configuration test passed")

def test_telemetry_record():
    """Test telemetry record creation"""
    print("🧪 Testing telemetry record creation...")
    
    record = TestHelper.create_sample_telemetry()
    assert record.event_id == "test_event_001"
    assert record.anomaly_type == AnomalyType.BENIGN
    assert record.execution_phase == ExecutionPhase.INVOKE
    assert record.source == TelemetrySource.PRIMARY
    
    print("✓ TelemetryRecord test passed")

def test_evasion_techniques():
    """Test evasion techniques"""
    print("🧪 Testing evasion techniques...")
    
    original = TestHelper.create_sample_telemetry()
    
    # Test noise injection
    noisy = EvasionTechniques.noise_injection(original, noise_level=0.1)
    assert noisy.duration != original.duration
    assert noisy.duration > 0
    assert 0 <= noisy.cpu_utilization <= 100
    
    # Test gradient masking
    masked = EvasionTechniques.gradient_masking(original, masking_strength=0.2)
    assert masked.timestamp >= original.timestamp
    
    # Test input transformation
    transformed = EvasionTechniques.input_transformation(original, "logarithmic")
    assert transformed.duration != original.duration
    
    # Test adaptive perturbation
    perturbed = EvasionTechniques.adaptive_perturbation(original, epsilon=0.1)
    assert perturbed.duration != original.duration
    
    print("✓ EvasionTechniques test passed")

def test_poisoning_attacks():
    """Test poisoning attack generation"""
    print("🧪 Testing poisoning attacks...")
    
    generator = PoisoningAttackGenerator(max_poison_rate=0.05)
    test_data = TestHelper.create_test_dataset(20)
    
    # Test label flip attack
    poisoned_data = generator.generate_label_flip_attack(test_data, 0.03)
    assert len(poisoned_data) == len(test_data)
    
    # Count poisoned records
    poisoned_count = sum(1 for record in poisoned_data 
                        if hasattr(record, 'custom_fields') and 
                           record.custom_fields.get('poisoned', False))
    assert poisoned_count > 0
    
    # Test backdoor attack
    trigger_pattern = {'cpu_utilization': 85.5, 'memory_spike_kb': 256000}
    backdoor_data = generator.generate_backdoor_attack(test_data, trigger_pattern)
    
    backdoor_count = sum(1 for record in backdoor_data 
                        if hasattr(record, 'custom_fields') and 
                           record.custom_fields.get('backdoor', False))
    assert backdoor_count > 0
    
    print("✓ Poisoning attacks test passed")

def test_economic_attacks():
    """Test economic attack simulation"""
    print("🧪 Testing economic attacks...")
    
    config = TestHelper.create_test_config()
    simulator = EconomicAttackSimulator(config)
    
    # Test DoW attack
    dow_result = simulator.simulate_denial_of_wallet_attack(duration_minutes=1, intensity="low")
    assert dow_result.attack_type == AttackType.DENIAL_OF_WALLET
    assert dow_result.economic_impact > 0
    assert len(dow_result.generated_telemetry) > 0
    
    # Test cryptomining attack
    mining_result = simulator.simulate_cryptomining_attack(duration_minutes=1)
    assert mining_result.attack_type == AttackType.CRYPTOMINING
    assert mining_result.economic_impact > 0
    
    print("✓ Economic attacks test passed")

@pytest.mark.asyncio
async def test_adversarial_engine():
    """Test main adversarial engine"""
    print("🧪 Testing adversarial engine...")
    
    config = TestHelper.create_test_config()
    engine = AdversarialAnomalyEngine(config)
    
    # Verify initialization
    assert engine.config == config
    assert hasattr(engine, 'evasion_techniques')
    assert hasattr(engine, 'poisoning_generator')
    assert hasattr(engine, 'economic_simulator')
    assert hasattr(engine, 'attack_history')
    
    # Test attack generation
    sample_telemetry = TestHelper.create_sample_telemetry()
    
    attack_result = await engine.generate_adversarial_anomaly(
        sample_telemetry, AttackType.NOISE_INJECTION
    )
    
    assert isinstance(attack_result, AttackResult)
    assert attack_result.attack_type == AttackType.NOISE_INJECTION
    assert len(attack_result.generated_telemetry) > 0
    
    # Check attack history
    assert len(engine.attack_history) > 0
    
    print("✓ Adversarial engine test passed")

def test_query_free_engine():
    """Test query-free attack engine"""
    print("🧪 Testing query-free engine...")
    
    config = TestHelper.create_test_config()
    engine = QueryFreeAttackEngine(config)
    
    # Test surrogate model building
    training_data = TestHelper.create_test_dataset(20)
    engine.build_surrogate_model(training_data)
    
    assert hasattr(engine, 'feature_statistics')
    assert 'mean' in engine.feature_statistics
    assert 'std' in engine.feature_statistics
    
    # Test adversarial generation
    target_record = TestHelper.create_sample_telemetry()
    adversarial_record = engine.generate_query_free_adversarial(target_record)
    
    assert hasattr(adversarial_record, 'custom_fields')
    assert adversarial_record.custom_fields.get('query_free_adversarial') is True
    
    print("✓ Query-free engine test passed")

def test_transfer_attacks():
    """Test transfer attack engine"""
    print("🧪 Testing transfer attacks...")
    
    config = TestHelper.create_test_config()
    engine = TransferAttackEngine(config)
    
    source_record = TestHelper.create_sample_telemetry()
    
    # Test transfer attack generation
    adversarial_record = engine.generate_transfer_attack(
        source_record, target_model_type="test_model"
    )
    
    assert hasattr(adversarial_record, 'custom_fields')
    assert adversarial_record.custom_fields.get('transfer_attack') is True
    assert adversarial_record.custom_fields.get('target_model_type') == "test_model"
    
    print("✓ Transfer attacks test passed")

def test_metrics_collection():
    """Test metrics collection"""
    print("🧪 Testing metrics collection...")
    
    collector = AdversarialMetricsCollector()
    
    # Create sample attack result
    attack_result = AttackResult(
        attack_id="test_attack_001",
        attack_type=AttackType.NOISE_INJECTION,
        start_time=time.time(),
        end_time=time.time() + 1,
        evasion_success=True,
        stealth_score=0.8,
        economic_impact=15.0,
        perturbation_magnitude=0.12
    )
    
    detection_response = {
        'response_time_ms': 150,
        'confidence': 0.75,
        'false_positive_risk': 0.1
    }
    
    # Record metrics
    collector.record_attack_metrics(attack_result, detection_response)
    
    assert len(collector.metrics_history) > 0
    recorded = collector.metrics_history[-1]
    assert recorded['attack_id'] == attack_result.attack_id
    assert recorded['attack_type'] == attack_result.attack_type.value
    
    print("✓ Metrics collection test passed")

def test_validation_framework():
    """Test adversarial validation framework"""
    print("🧪 Testing validation framework...")
    
    framework = AdversarialValidationFramework()
    baseline_data = TestHelper.create_test_dataset(10)
    
    # Create mock attack result
    attack_result = AttackResult(
        attack_id="validation_test_001",
        attack_type=AttackType.NOISE_INJECTION,
        start_time=time.time(),
        end_time=time.time() + 1,
        evasion_success=True,
        perturbation_magnitude=0.1,
        generated_telemetry=[TestHelper.create_sample_telemetry()],
        stealth_score=0.7
    )
    
    # Test validation
    validation_scores = framework.validate_attack_realism(attack_result, baseline_data)
    
    required_metrics = [
        'statistical_realism',
        'temporal_consistency', 
        'resource_feasibility',
        'behavioral_plausibility',
        'overall_realism'
    ]
    
    for metric in required_metrics:
        assert metric in validation_scores
        assert 0.0 <= validation_scores[metric] <= 1.0
    
    print("✓ Validation framework test passed")

@pytest.mark.asyncio
async def test_complete_workflow():
    """Test complete adversarial workflow"""
    print("🧪 Testing complete workflow...")
    
    # Setup components
    config = TestHelper.create_test_config()
    engine = AdversarialAnomalyEngine(config)
    metrics_collector = AdversarialMetricsCollector()
    
    # Generate test data
    test_data = TestHelper.create_test_dataset(5)
    
    # Mock detection system
    def mock_detection_system(telemetry):
        return {
            'anomaly_detected': telemetry.cpu_utilization > 80,
            'confidence': min(1.0, telemetry.cpu_utilization / 100.0),
            'response_time_ms': random.randint(10, 100)
        }
    
    # Run attacks and collect metrics
    for sample in test_data:
        attack_result = await engine.generate_adversarial_anomaly(
            sample, AttackType.NOISE_INJECTION
        )
        
        # Test against detection system
        for telemetry in attack_result.generated_telemetry:
            detection_response = mock_detection_system(telemetry)
            metrics_collector.record_attack_metrics(attack_result, detection_response)
    
    # Generate reports
    effectiveness_report = engine.get_attack_effectiveness_report()
    research_report = metrics_collector.generate_research_report()
    
    # Verify reports
    assert 'generation_timestamp' in effectiveness_report or 'total_attack_types_tested' in effectiveness_report
    assert 'summary' in research_report
    assert research_report['summary']['total_attacks'] > 0
    
    print("✓ Complete workflow test passed")

async def main():
    """Run all integration tests"""
    print("🎯 SCAFAD Adversarial Components Integration Test")
    print("=" * 60)
    
    # List of test functions
    tests = [
        test_basic_config,
        test_telemetry_record,
        test_evasion_techniques,
        test_poisoning_attacks,
        test_economic_attacks,
        test_adversarial_engine,
        test_query_free_engine,
        test_transfer_attacks,
        test_metrics_collection,
        test_validation_framework,
        test_complete_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            print(f"\n📋 Running {test_func.__name__}...")
            
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            
            passed += 1
            print(f"✅ {test_func.__name__} PASSED")
            
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print(f"\n📊 FINAL RESULTS")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ All components work seamlessly together")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. See error details above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🌟 Integration successful - all components are interconnected properly!")
    else:
        print("\n🔧 Some issues found - check the error messages above")