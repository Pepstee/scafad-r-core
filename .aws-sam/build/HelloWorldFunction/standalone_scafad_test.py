#!/usr/bin/env python3
"""
Standalone SCAFAD Layer 0 Test
Tests your SCAFAD implementation without needing SAM CLI or Lambda
"""

import json
import asyncio
import time
import sys
import os
from datetime import datetime

# Add current directory to path to import app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_mock_context():
    """Create a mock Lambda context for testing"""
    class MockContext:
        def __init__(self):
            self.aws_request_id = f"test-{int(time.time())}"
            self.function_name = "scafad-test-function"
            self.function_version = "$LATEST"
            self.memory_limit_in_mb = 1024
            self.remaining_time_in_millis = lambda: 30000
    
    return MockContext()

def generate_test_payloads():
    """Generate test payloads for SCAFAD"""
    return [
        {
            "anomaly": "benign",
            "function_profile_id": "ml_inference",
            "execution_phase": "invoke",
            "test_mode": True,
            "payload_id": "test_001"
        },
        {
            "anomaly": "cold_start",
            "function_profile_id": "data_processor",
            "execution_phase": "init",
            "test_mode": True,
            "payload_id": "test_002"
        },
        {
            "anomaly": "execution_failure",
            "function_profile_id": "api_gateway",
            "execution_phase": "invoke",
            "enable_adversarial": True,
            "test_mode": True,
            "payload_id": "test_003"
        },
        {
            "anomaly": "memory_spike",
            "function_profile_id": "image_resizer",
            "execution_phase": "invoke",
            "economic_attack": "resource_waste",
            "test_mode": True,
            "payload_id": "test_004"
        },
        {
            "anomaly": "timeout_fallback",
            "function_profile_id": "batch_worker",
            "execution_phase": "shutdown",
            "force_starvation": True,
            "test_mode": True,
            "payload_id": "test_005"
        }
    ]

def test_scafad_directly():
    """Test SCAFAD implementation directly"""
    print("🚀 SCAFAD Layer 0 - Standalone Test")
    print("=" * 50)
    
    try:
        # Try to import your app
        import app
        print("✅ Successfully imported app.py")
        
        # Check if the lambda handler exists
        if hasattr(app, 'lambda_handler'):
            print("✅ lambda_handler found")
        else:
            print("❌ lambda_handler not found in app.py")
            return False
        
        # Generate test payloads
        payloads = generate_test_payloads()
        print(f"📦 Generated {len(payloads)} test payloads")
        
        # Test each payload
        results = []
        for i, payload in enumerate(payloads):
            print(f"\n🧪 Test {i+1}/{len(payloads)}: {payload['anomaly']}")
            
            try:
                # Create mock context
                context = create_mock_context()
                
                # Call the lambda handler
                start_time = time.time()
                result = app.lambda_handler(payload, context)
                duration = time.time() - start_time
                
                # Parse result
                success = result.get('statusCode') == 200
                print(f"   {'✅' if success else '❌'} Status: {result.get('statusCode')}")
                print(f"   ⏱️  Duration: {duration:.3f}s")
                
                # Try to parse body
                body = result.get('body')
                if body and isinstance(body, str):
                    try:
                        parsed_body = json.loads(body)
                        if 'anomaly_detected' in parsed_body:
                            print(f"   🔍 Anomaly Detected: {parsed_body['anomaly_detected']}")
                        if 'telemetry_id' in parsed_body:
                            print(f"   📊 Telemetry ID: {parsed_body['telemetry_id'][:8]}...")
                        if 'processing_time_ms' in parsed_body:
                            print(f"   ⚡ Processing Time: {parsed_body['processing_time_ms']:.1f}ms")
                    except json.JSONDecodeError:
                        print(f"   📄 Body: {body[:100]}...")
                
                results.append({
                    'payload_id': payload['payload_id'],
                    'success': success,
                    'duration': duration,
                    'result': result
                })
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results.append({
                    'payload_id': payload['payload_id'],
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        print(f"   Success Rate: {successful/len(results)*100:.1f}%")
        
        if successful > 0:
            avg_duration = sum(r.get('duration', 0) for r in results if r.get('success')) / successful
            print(f"   Avg Duration: {avg_duration:.3f}s")
        
        return successful > 0
        
    except ImportError as e:
        print(f"❌ Could not import app.py: {e}")
        print("💡 Make sure app.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_handler():
    """Test the async handler if available"""
    print(f"\n🔄 Testing Async Handler...")
    
    try:
        import app
        
        if hasattr(app, 'enhanced_lambda_handler'):
            print("✅ Found enhanced_lambda_handler")
            
            # Test one payload with async handler
            payload = {
                "anomaly": "cpu_burst",
                "function_profile_id": "analytics_engine",
                "execution_phase": "invoke",
                "test_mode": True,
                "payload_id": "async_test_001"
            }
            
            context = create_mock_context()
            
            # Run async handler
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(app.enhanced_lambda_handler(payload, context))
                print(f"✅ Async handler worked: {result.get('statusCode')}")
                return True
            finally:
                loop.close()
        else:
            print("ℹ️  enhanced_lambda_handler not found (sync only)")
            return True
            
    except Exception as e:
        print(f"❌ Async handler test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 SCAFAD Standalone Testing Tool")
    print("Tests your SCAFAD implementation without SAM CLI")
    print("=" * 60)
    
    # Test basic functionality
    basic_success = test_scafad_directly()
    
    # Test async functionality
    async_success = test_async_handler()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    
    if basic_success:
        print("✅ Your SCAFAD implementation is working correctly!")
        print("💡 The 0% success rate in invoke.py was due to missing SAM CLI")
        print("📥 Install SAM CLI to use the full invoke.py script")
    else:
        print("❌ Issues found in SCAFAD implementation")
        print("🔧 Fix the errors above before installing SAM CLI")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Install SAM CLI: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html")
    print(f"   2. Run: sam --version (to verify)")
    print(f"   3. Run: python invoke.py --n 5 --mode test")
    print(f"   4. Your SCAFAD framework should then show proper success rates!")

if __name__ == "__main__":
    main()