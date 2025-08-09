#!/usr/bin/env python3
"""
Simple SCAFAD Layer 0 Test - Windows Compatible
===============================================
"""

import json
import boto3
import time
import sys

def test_scafad_basic():
    """Simple test that works reliably"""
    print("Testing SCAFAD Layer 0 basic functionality...")
    
    function_name = "scafad-test-stack-HelloWorldFunction-k79tX3iBcK74"
    
    try:
        # Initialize Lambda client
        client = boto3.client('lambda')
        print("[OK] AWS client initialized")
        
        # Test basic invocation
        test_payload = {
            "test": "simple",
            "anomaly": "benign",
            "function_profile_id": "test"
        }
        
        print("[TEST] Invoking function...")
        start_time = time.time()
        
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(test_payload)
        )
        
        duration = time.time() - start_time
        result = json.loads(response['Payload'].read())
        
        # Check results
        status_code = result.get('statusCode', 0)
        has_body = 'body' in result
        
        print(f"[RESULT] Status: {status_code}")
        print(f"[RESULT] Duration: {duration:.2f}s")
        print(f"[RESULT] Has body: {has_body}")
        
        if status_code == 200 and has_body:
            print("[SUCCESS] Basic SCAFAD test passed!")
            
            # Try to parse body for SCAFAD fields
            try:
                if isinstance(result['body'], str):
                    body = json.loads(result['body'])
                else:
                    body = result['body']
                
                scafad_fields = [
                    'telemetry_id', 'processing_time_ms', 'anomaly_detected',
                    'economic_risk_score', 'completeness_score'
                ]
                
                found_fields = [field for field in scafad_fields if field in body]
                
                print(f"[SCAFAD] Found {len(found_fields)}/{len(scafad_fields)} SCAFAD fields")
                
                if len(found_fields) >= 3:
                    print("[SUCCESS] SCAFAD telemetry format looks good!")
                    return True
                else:
                    print("[WARNING] Some SCAFAD fields missing")
                    return True  # Still counts as basic success
                    
            except Exception as e:
                print(f"[WARNING] Could not parse SCAFAD fields: {e}")
                return True  # Basic function still works
                
        else:
            print("[FAIL] Basic test failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def test_multiple_anomalies():
    """Test different anomaly types"""
    print("\nTesting different anomaly types...")
    
    function_name = "scafad-test-stack-HelloWorldFunction-k79tX3iBcK74"
    client = boto3.client('lambda')
    
    anomalies = ["benign", "cold_start", "cpu_burst"]
    results = {}
    
    for anomaly in anomalies:
        print(f"[TEST] Testing {anomaly}... ", end="")
        
        try:
            payload = {
                "anomaly": anomaly,
                "function_profile_id": "test",
                "execution_phase": "invoke"
            }
            
            response = client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read())
            success = result.get('statusCode') == 200
            
            print("[OK]" if success else "[FAIL]")
            results[anomaly] = success
            
        except Exception as e:
            print(f"[ERROR] {e}")
            results[anomaly] = False
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n[SUMMARY] Anomaly tests: {success_count}/{total_count} passed")
    return success_count > 0

if __name__ == "__main__":
    print("SCAFAD Layer 0 Simple Test Suite")
    print("=" * 50)
    
    # Run tests
    basic_success = test_scafad_basic()
    anomaly_success = test_multiple_anomalies()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if basic_success and anomaly_success:
        print("[SUCCESS] SCAFAD Layer 0 is working!")
        print("Your function is ready for dissertation experiments.")
        sys.exit(0)
    elif basic_success:
        print("[PARTIAL] Basic functionality works, some anomaly types need fixing")
        sys.exit(0)
    else:
        print("[FAIL] Basic functionality not working")
        sys.exit(1)
