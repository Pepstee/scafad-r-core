#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspace')

print("🚀 SCAFAD Layer 0 - Direct Integration Test Runner")
print("=" * 60)

# Import all required components directly
try:
    import asyncio
    import json
    import time
    from complete_layer0_integration_test import run_complete_integration_test
    
    print("✅ All imports successful!")
    
    # Run the test
    print("\n🔧 Starting Layer 0 Integration Tests...")
    result = asyncio.run(run_complete_integration_test())
    
    print("\n📊 FINAL TEST SUMMARY")
    print("=" * 40)
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"Readiness Status: {result['readiness_status']}")
    print(f"Tests Passed: {result['tests_passed']}/{result['total_tests']}")
    print(f"Execution Time: {result['execution_time_seconds']:.1f}s")
    print(f"Recommendation: {result['recommendation']}")
    
except Exception as e:
    print(f"❌ Error running tests: {e}")
    import traceback
    traceback.print_exc()