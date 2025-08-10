#!/usr/bin/env python3
"""Quick import test to check dependencies"""

try:
    print("Testing app_config imports...")
    from app_config import AdversarialConfig, AdversarialMode
    print(f"✓ AdversarialMode.TEST = {AdversarialMode.TEST}")
    print(f"✓ AdversarialMode.DISABLED = {AdversarialMode.DISABLED}")
    
    print("\nTesting app_telemetry imports...")
    from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
    print(f"✓ ExecutionPhase.INVOKE = {ExecutionPhase.INVOKE}")
    print(f"✓ AnomalyType.BENIGN = {AnomalyType.BENIGN}")
    print(f"✓ AnomalyType.CPU_BURST = {AnomalyType.CPU_BURST}")
    
    print("\nTesting app_adversarial imports...")
    from app_adversarial import (
        AdversarialAnomalyEngine, AttackType, AttackResult,
        EvasionTechniques, AdversarialTestSuite
    )
    print(f"✓ AttackType.NOISE_INJECTION = {AttackType.NOISE_INJECTION}")
    
    print("\nTesting basic instantiation...")
    config = AdversarialConfig(adversarial_mode=AdversarialMode.TEST)
    print("✓ AdversarialConfig created")
    
    engine = AdversarialAnomalyEngine(config)
    print("✓ AdversarialAnomalyEngine created")
    
    print("\n🎉 All basic imports and instantiation successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()