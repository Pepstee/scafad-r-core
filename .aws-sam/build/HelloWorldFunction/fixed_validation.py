#!/usr/bin/env python3
"""
SCAFAD Layer 0 - Fixed Final Validation Suite
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Any
from datetime import datetime
import traceback

def quick_validation():
    """Quick validation to check basic functionality"""
    print("🔍 Running Quick SCAFAD Validation")
    print("-" * 40)
    
    issues = []
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        issues.append("app.py file not found")
        return False, issues
    
    try:
        # Try to import the app
        sys.path.insert(0, '.')
        import app
        
        # Check for required functions/classes
        required_items = ['lambda_handler', 'Layer0_AdaptiveTelemetryController', 'TelemetryRecord']
        for item in required_items:
            if not hasattr(app, item):
                issues.append(f"Missing required component: {item}")
            else:
                print(f"  ✅ {item}")
        
        # Test lambda_handler
        from unittest.mock import Mock
        context = Mock()
        context.aws_request_id = "test-123"
        context.function_name = "test-function"
        context.memory_limit_in_mb = 128
        
        test_event = {
            'anomaly': 'cpu_burst',
            'execution_phase': 'invoke',
            'test_mode': True
        }
        
        response = app.lambda_handler(test_event, context)
        
        if isinstance(response, dict) and 'statusCode' in response:
            print(f"  ✅ Lambda handler test passed (Status: {response['statusCode']})")
        else:
            issues.append("Lambda handler returned invalid response format")
        
        # Parse response body
        try:
            body_data = json.loads(response.get('body', '{}'))
            required_fields = ['telemetry_id', 'processing_time_ms']
            
            for field in required_fields:
                if field in body_data:
                    print(f"  ✅ Response field: {field}")
                else:
                    issues.append(f"Missing response field: {field}")
        except json.JSONDecodeError:
            issues.append("Response body is not valid JSON")
        
    except ImportError as e:
        issues.append(f"Cannot import app module: {e}")
    except Exception as e:
        issues.append(f"Validation error: {e}")
    
    success = len(issues) == 0
    if success:
        print("  🎉 Quick validation PASSED!")
    else:
        print("  ❌ Quick validation FAILED:")
        for issue in issues:
            print(f"    • {issue}")
    
    return success, issues

def main():
    """Main validation function"""
    print("🏁 SCAFAD Layer 0 - Fixed Validation Suite")
    print("=" * 50)
    
    success, issues = quick_validation()
    
    print(f"\n📊 Validation Results")
    print("-" * 30)
    print(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"Issues: {len(issues)}")
    
    if success:
        print("\n🎯 Layer 1 Readiness: ✅ READY")
        print("\n📋 Next Steps:")
        print("  1. Deploy: sam build && sam deploy")
        print("  2. Test: sam local invoke")
        print("  3. Monitor: Check CloudWatch logs")
    else:
        print("\n🎯 Layer 1 Readiness: ❌ NOT READY")
        print("\n🔧 Issues to fix:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'validation_passed': success,
        'issues_count': len(issues),
        'issues': issues,
        'layer1_ready': success
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: validation_results.json")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
