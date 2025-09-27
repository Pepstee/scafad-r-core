#!/usr/bin/env python3
"""Execute the complete Layer 0 acceptance test suite"""

import sys
import asyncio
import traceback

# Add workspace to path
sys.path.insert(0, '/workspace')

async def main():
    """Execute the complete Layer 0 acceptance test suite"""
    try:
        from complete_layer0_test_suite import CompleteLayer0TestSuite
        
        print("🚀 Executing Complete SCAFAD Layer 0 Acceptance Test Suite...")
        print("   This will test ALL Layer 0 components and scenarios")
        
        test_suite = CompleteLayer0TestSuite()
        results = await test_suite.run_complete_acceptance_suite()
        
        if results:
            print(f"\n🎉 Complete test suite execution finished!")
            print(f"📊 Results Summary:")
            print(f"   Overall Success Rate: {results['overall_success_rate']*100:.1f}%")
            print(f"   Tests Executed: {results['total_tests']}")
            print(f"   Deployment Status: {results['deployment_recommendation']}")
            print(f"   Component Coverage: {results['component_coverage']*100:.1f}%")
        else:
            print("❌ Test suite execution failed")
        
        return results
        
    except Exception as e:
        print(f"💥 Test execution error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(main())