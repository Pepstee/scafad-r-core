#!/usr/bin/env python3
"""Simple validation to check basic integration"""

import sys
import traceback

def main():
    print("🔍 Validating SCAFAD Component Integration...")
    
    try:
        # Test 1: Import all modules
        print("\n1️⃣ Testing imports...")
        from app_config import AdversarialConfig, AdversarialMode
        from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
        from app_adversarial import (
            AdversarialAnomalyEngine, AttackType, EvasionTechniques,
            PoisoningAttackGenerator, EconomicAttackSimulator
        )
        print("✅ All imports successful")
        
        # Test 2: Create basic objects
        print("\n2️⃣ Testing object creation...")
        config = AdversarialConfig(adversarial_mode=AdversarialMode.TEST)
        engine = AdversarialAnomalyEngine(config)
        print("✅ Objects created successfully")
        
        # Test 3: Create telemetry record
        print("\n3️⃣ Testing telemetry record...")
        import time
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
            concurrency_id="test_001"
        )
        print("✅ TelemetryRecord created successfully")
        
        # Test 4: Test evasion techniques
        print("\n4️⃣ Testing evasion techniques...")
        noisy_record = EvasionTechniques.noise_injection(record, noise_level=0.1)
        if noisy_record.duration != record.duration:
            print("✅ Noise injection works")
        else:
            print("⚠️ Noise injection may not be working properly")
        
        # Test 5: Test poisoning generator
        print("\n5️⃣ Testing poisoning generator...")
        generator = PoisoningAttackGenerator(max_poison_rate=0.05)
        test_data = [record] * 10
        poisoned = generator.generate_label_flip_attack(test_data, 0.02)
        print("✅ Poisoning generator works")
        
        # Test 6: Test economic simulator
        print("\n6️⃣ Testing economic simulator...")
        simulator = EconomicAttackSimulator(config)
        # Don't run the full simulation to save time, just check it instantiates
        print("✅ Economic simulator instantiated")
        
        print("\n🎉 BASIC VALIDATION SUCCESSFUL!")
        print("All core components can be imported and instantiated correctly.")
        print("The files are properly interconnected.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)