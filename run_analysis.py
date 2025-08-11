#!/usr/bin/env python3
"""Run the focused system analysis"""

import sys
sys.path.insert(0, '/workspace')

try:
    # Import and run the focused system test
    from focused_system_test import run_real_world_simulation
    
    print("🚀 Starting SCAFAD Layer 0 Real-World Analysis...")
    result = run_real_world_simulation()
    
    if result:
        print(f"\n🎉 Analysis Complete!")
        print(f"Final Assessment: {result['final_rating']}")
    else:
        print("❌ Analysis failed to complete")

except Exception as e:
    print(f"💥 Error running analysis: {e}")
    import traceback
    traceback.print_exc()