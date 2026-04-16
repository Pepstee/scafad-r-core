#!/usr/bin/env python3
"""Execute the comprehensive system test"""

import sys
import asyncio
import traceback

# Add workspace to path
sys.path.insert(0, '/workspace')

# Import the test module
from run_system_test import run_comprehensive_system_test

def main():
    """Execute the comprehensive system test"""
    try:
        print("Initiating SCAFAD Layer 0 comprehensive system test...")
        result = asyncio.run(run_comprehensive_system_test())
        
        if result:
            print(f"\n🎉 Test execution completed!")
            print(f"Overall Success Rate: {result['overall_success_rate']*100:.1f}%")
            print(f"Assessment: {result['assessment']}")
        else:
            print("❌ Test execution failed")
            
    except Exception as e:
        print(f"💥 Test execution error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()