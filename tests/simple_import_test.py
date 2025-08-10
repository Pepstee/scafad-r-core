#!/usr/bin/env python3

print("Testing imports...")

try:
    # Test basic imports first
    from app_telemetry import TelemetryRecord, TelemetrySource
    print("✅ app_telemetry imports OK")
    
    from app_config import AdversarialConfig, AdversarialMode
    print("✅ app_config imports OK")
    
    # Test app_adversarial
    from app_adversarial import AdversarialAnomalyEngine
    print("✅ app_adversarial imports OK")
    
    # Try to import test_adversarial
    import test_adversarial
    print("✅ test_adversarial imports OK")
    
    # Try basic functionality
    record = test_adversarial.make_record()
    print(f"✅ make_record() works: {record.event_id}")
    
    sample = test_adversarial.TestFixtures.create_sample_telemetry()
    print(f"✅ TestFixtures works: {sample.event_id}")
    
    print("\n🎉 BASIC FUNCTIONALITY WORKS!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()