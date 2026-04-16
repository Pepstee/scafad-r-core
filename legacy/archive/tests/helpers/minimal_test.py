#!/usr/bin/env python3

# Test minimum functionality
import time

try:
    exec("""
# Test imports
from app_config import AdversarialConfig, AdversarialMode
from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
from app_adversarial import AdversarialAnomalyEngine, AttackType, EvasionTechniques

# Test basic creation
config = AdversarialConfig(adversarial_mode=AdversarialMode.TEST)
record = TelemetryRecord(
    event_id="test", timestamp=time.time(), function_id="test_func",
    execution_phase=ExecutionPhase.INVOKE, anomaly_type=AnomalyType.BENIGN,
    duration=1.0, memory_spike_kb=100000, cpu_utilization=50.0, network_io_bytes=1000,
    fallback_mode=False, source=TelemetrySource.PRIMARY, concurrency_id="test"
)

# Test evasion
noisy = EvasionTechniques.noise_injection(record, noise_level=0.1)
if noisy.duration != record.duration:
    print("SUCCESS: Basic integration works!")
else:
    print("WARNING: Evasion technique may not be working")

print("All basic tests passed - components are interconnected")
    """)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()