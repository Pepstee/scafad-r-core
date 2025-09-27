#!/usr/bin/env python3
"""
Direct execution of complete Layer 0 integration test
"""
import asyncio
import sys
import traceback

# Add workspace to path
sys.path.insert(0, '/workspace')

def run_integration_test():
    """Run the complete integration test"""
    try:
        print('🚀 SCAFAD Layer 0 - Complete Integration Test Suite Execution')
        print('=' * 70)
        
        # Import the test function
        from complete_layer0_integration_test import run_complete_integration_test
        
        # Execute the integration test
        results = asyncio.run(run_complete_integration_test())
        
        print('=' * 70)
        print('✅ Integration Test Results Summary:')
        print(f"   Overall Score: {results['overall_score']:.3f}/1.0")
        print(f"   Tests Passed: {results['tests_passed']}/{results['total_tests']}")
        print(f"   Component Availability: {results['availability_score']:.3f}")
        print(f"   Average Test Score: {results['average_test_score']:.3f}")
        print(f"   Readiness Status: {results['readiness_status']}")
        print(f"   Recommendation: {results['recommendation']}")
        print(f"   Execution Time: {results['execution_time_seconds']:.1f}s")
        
        print('=' * 70)
        print('🏁 Complete Layer 0 Integration Test Execution Finished!')
        
        return results
        
    except Exception as e:
        print(f"❌ Integration test execution failed: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_integration_test()