#!/usr/bin/env python3
"""Quick test to verify all imports work"""

import sys
import traceback

# Test core modules
test_modules = [
    'app_config',
    'app_telemetry', 
    'app_main',
    'app_adversarial',
    'app_economic',
    'app_formal',
    'app_provenance',
    'app_schema',
    'app_silent_failure',
    'core.ignn_model',
    'utils.helpers'
]

print("SCAFAD Import Verification Test")
print("=" * 40)

failed_imports = []
successful_imports = []

for module in test_modules:
    try:
        __import__(module)
        print(f"✅ {module}")
        successful_imports.append(module)
    except Exception as e:
        print(f"❌ {module}: {str(e)}")
        failed_imports.append((module, str(e)))

print("\n" + "=" * 40)
print(f"Results: {len(successful_imports)}/{len(test_modules)} modules imported successfully")

if failed_imports:
    print("\nFAILED IMPORTS:")
    for module, error in failed_imports:
        print(f"  {module}: {error}")
    sys.exit(1)
else:
    print("\n✅ ALL IMPORTS SUCCESSFUL!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    try:
        from app_config import get_default_config, ConfigurationFactory
        from core.ignn_model import iGNNAnomalyDetector
        
        # Test config creation
        config = get_default_config()
        print(f"✅ Configuration loaded: {config.version['version']}")
        
        # Test i-GNN detector
        detector = iGNNAnomalyDetector()
        print(f"✅ i-GNN detector initialized")
        
        # Test basic detection
        test_records = [{
            'event_id': 'test_001',
            'timestamp': 1641859200,
            'function_id': 'test_function',
            'execution_phase': 'INVOKE',
            'duration': 1.5,
            'memory_spike_kb': 2048,
            'cpu_utilization': 45.0,
            'network_io_bytes': 1024,
            'is_cold_start': False,
            'error_occurred': False
        }]
        
        result = detector.detect_anomalies(test_records)
        print(f"✅ Basic anomaly detection works: {result.get('anomaly_detected')}")
        
        print("\n🎉 SCAFAD SYSTEM IS READY!")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        sys.exit(1)