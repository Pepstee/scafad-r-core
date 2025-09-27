# run_tests.py
"""
Simple test runner for SCAFAD Layer 0
Run this to validate your implementation
"""

import sys
import os
import json
import time
import traceback
from unittest.mock import Mock

def test_import():
    """Test if SCAFAD Layer 0 can be imported"""
    print("🔍 Testing SCAFAD Layer 0 imports...")
    
    try:
        # Test basic imports
        from app import (
            Layer0_AdaptiveTelemetryController,
            lambda_handler,
            TelemetryRecord,
            AnomalyType,
            ExecutionPhase,
            generate_test_payloads
        )
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure app.py contains the complete SCAFAD implementation")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        traceback.print_exc()
        return False

def test_controller_creation():
    """Test creating the main controller"""
    print("\n🔍 Testing controller creation...")
    
    try:
        from app import Layer0_AdaptiveTelemetryController
        controller = Layer0_AdaptiveTelemetryController()
        
        # Check all components exist
        components = [
            'graph_builder', 'adversarial_simulator', 'provenance_tracker',
            'schema_registry', 'formal_verifier', 'economic_monitor',
            'silent_failure_detector', 'telemetry_channels'
        ]
        
        for component in components:
            if not hasattr(controller, component):
                print(f"❌ Missing component: {component}")
                return False
        
        print("✅ Controller created with all components")
        return True
        
    except Exception as e:
        print(f"❌ Controller creation failed: {e}")
        traceback.print_exc()
        return False

def test_lambda_handler():
    """Test the Lambda handler function"""
    print("\n🔍 Testing Lambda handler...")
    
    try:
        from app import lambda_handler, AnomalyType, ExecutionPhase
        
        # Create mock context
        context = Mock()
        context.aws_request_id = "test-123"
        context.function_name = "test-function"
        context.function_version = "$LATEST"
        context.memory_limit_in_mb = 128
        context.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"
        
        # Test event
        event = {
            "test_mode": True,
            "anomaly": "benign",
            "execution_phase": "invoke",
            "function_profile_id": "test-function"
        }
        
        # Call handler
        result = lambda_handler(event, context)
        
        # Validate response
        if not isinstance(result, dict):
            print("❌ Handler should return a dict")
            return False
        
        required_fields = ['statusCode', 'body']
        for field in required_fields:
            if field not in result:
                print(f"❌ Missing field in response: {field}")
                return False
        
        print("✅ Lambda handler working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Lambda handler test failed: {e}")
        traceback.print_exc()
        return False

def test_telemetry_record():
    """Test TelemetryRecord creation"""
    print("\n🔍 Testing TelemetryRecord...")
    
    try:
        from app import TelemetryRecord, AnomalyType, ExecutionPhase
        
        record = TelemetryRecord(
            event_id="test-record-123",
            timestamp=time.time(),
            function_id="test-function",
            execution_phase=ExecutionPhase.INVOKE,
            anomaly_type=AnomalyType.BENIGN,
            duration=0.1,
            memory_spike_kb=8192,
            cpu_utilization=15.0,
            network_io_bytes=1024,
            fallback_mode=False,
            source="test",
            concurrency_id="TST"
        )
        
        # Validate record
        if record.event_id != "test-record-123":
            print("❌ TelemetryRecord event_id not set correctly")
            return False
        
        if record.anomaly_type != AnomalyType.BENIGN:
            print("❌ TelemetryRecord anomaly_type not set correctly")
            return False
        
        if record.log_version is None:
            print("❌ TelemetryRecord log_version not set")
            return False
        
        print("✅ TelemetryRecord creation working")
        return True
        
    except Exception as e:
        print(f"❌ TelemetryRecord test failed: {e}")
        traceback.print_exc()
        return False

def test_payload_generation():
    """Test payload generation"""
    print("\n🔍 Testing payload generation...")
    
    try:
        from app import generate_test_payloads
        
        payloads = generate_test_payloads(3, 42)
        
        if not isinstance(payloads, list):
            print("❌ generate_test_payloads should return a list")
            return False
        
        if len(payloads) != 3:
            print(f"❌ Expected 3 payloads, got {len(payloads)}")
            return False
        
        # Check payload structure
        required_fields = ['anomaly', 'function_profile_id', 'execution_phase']
        for i, payload in enumerate(payloads):
            for field in required_fields:
                if field not in payload:
                    print(f"❌ Payload {i} missing field: {field}")
                    return False
        
        print("✅ Payload generation working")
        return True
        
    except Exception as e:
        print(f"❌ Payload generation test failed: {e}")
        traceback.print_exc()
        return False

def test_environment_validation():
    """Test environment validation"""
    print("\n🔍 Testing environment validation...")
    
    try:
        from app import validate_environment
        
        validation_result = validate_environment()
        
        if not isinstance(validation_result, dict):
            print("❌ validate_environment should return a dict")
            return False
        
        required_checks = ['python_version', 'required_modules']
        for check in required_checks:
            if check not in validation_result:
                print(f"❌ Missing validation check: {check}")
                return False
        
        print("✅ Environment validation working")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation test failed: {e}")
        traceback.print_exc()
        return False

def test_full_integration():
    """Test full integration with multiple payloads"""
    print("\n🔍 Testing full integration...")
    
    try:
        from app import lambda_handler, generate_test_payloads
        
        # Generate test payloads
        payloads = generate_test_payloads(2, 42)
        
        # Create mock context
        context = Mock()
        context.aws_request_id = "integration-test"
        context.function_name = "integration-test-function"
        context.function_version = "$LATEST"
        context.memory_limit_in_mb = 256
        context.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:integration"
        
        success_count = 0
        
        for i, payload in enumerate(payloads):
            try:
                context.aws_request_id = f"integration-test-{i}"
                result = lambda_handler(payload, context)
                
                if isinstance(result, dict) and 'statusCode' in result:
                    success_count += 1
                    print(f"  ✅ Payload {i+1}: {payload.get('anomaly', 'unknown')}")
                else:
                    print(f"  ❌ Payload {i+1}: Invalid response")
                    
            except Exception as e:
                print(f"  ❌ Payload {i+1}: {e}")
        
        if success_count == len(payloads):
            print("✅ Full integration test passed")
            return True
        else:
            print(f"❌ Integration test partial failure: {success_count}/{len(payloads)} succeeded")
            return False
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_test():
    """Run a simple performance test"""
    print("\n🔍 Running performance test...")
    
    try:
        from app import lambda_handler
        
        # Create mock context
        context = Mock()
        context.aws_request_id = "perf-test"
        context.function_name = "perf-test-function"
        context.function_version = "$LATEST"
        context.memory_limit_in_mb = 128
        
        # Simple event
        event = {
            "test_mode": True,
            "anomaly": "benign",
            "execution_phase": "invoke"
        }
        
        # Measure execution time
        start_time = time.time()
        result = lambda_handler(event, context)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # ms
        
        print(f"✅ Performance test completed in {execution_time:.2f}ms")
        
        if execution_time > 5000:  # 5 seconds
            print("⚠️  Execution time is quite high - consider optimization")
        elif execution_time > 1000:  # 1 second
            print("⚠️  Execution time is moderate")
        else:
            print("✅ Execution time is good")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("🚀 SCAFAD Layer 0 Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Controller Creation", test_controller_creation),
        ("Lambda Handler", test_lambda_handler),
        ("TelemetryRecord", test_telemetry_record),
        ("Payload Generation", test_payload_generation),
        ("Environment Validation", test_environment_validation),
        ("Full Integration", test_full_integration),
        ("Performance Test", run_performance_test)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} encountered unexpected error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Your SCAFAD Layer 0 is working perfectly!")
        print("🚀 Ready for deployment!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        print("💡 Common issues:")
        print("   - Missing imports in app.py")
        print("   - Syntax errors in the code")
        print("   - Missing dependencies")
    
    print("\n" + "=" * 50)
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# validate_deployment.py
"""
Deployment validation script for SCAFAD Layer 0
"""

import json
import os
import sys
from typing import Dict, List

def check_file_structure():
    """Check if required files exist"""
    print("🔍 Checking file structure...")
    
    required_files = {
        'app.py': 'Main SCAFAD Layer 0 implementation',
        'template.yaml': 'SAM template',
        'samconfig.toml': 'SAM configuration'
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (MISSING)")
            missing_files.append(filename)
    
    return len(missing_files) == 0

def check_sam_template():
    """Check SAM template configuration"""
    print("\n🔍 Checking SAM template...")
    
    try:
        with open('template.yaml', 'r') as f:
            template_content = f.read()
        
        required_sections = [
            'AWSTemplateFormatVersion',
            'Transform',
            'Resources'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in template_content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing sections in template.yaml: {missing_sections}")
            return False
        else:
            print("✅ SAM template structure looks good")
            return True
            
    except FileNotFoundError:
        print("❌ template.yaml not found")
        return False
    except Exception as e:
        print(f"❌ Error reading template.yaml: {e}")
        return False

def check_dependencies():
    """Check Python dependencies"""
    print("\n🔍 Checking Python dependencies...")
    
    required_modules = [
        'json', 'time', 'random', 'uuid', 'hashlib',
        'asyncio', 'logging', 'sys', 'os', 'typing',
        'collections', 'dataclasses', 'datetime', 'enum', 'math'
    ]
    
    optional_modules = [
        'networkx', 'numpy', 'tenacity'
    ]
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} (required)")
        except ImportError:
            print(f"❌ {module} (required) - MISSING")
            missing_required.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} (optional)")
        except ImportError:
            print(f"⚠️  {module} (optional) - missing (will use fallbacks)")
            missing_optional.append(module)
    
    if missing_required:
        print(f"\n❌ Missing required modules: {missing_required}")
        print("💡 These are standard Python modules and should be available")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional modules: {missing_optional}")
        print("💡 Install with: pip install networkx numpy tenacity")
        print("   (SCAFAD will use fallback implementations)")
    
    return True

def check_environment_variables():
    """Check environment configuration"""
    print("\n🔍 Checking environment variables...")
    
    optional_vars = {
        'SCAFAD_TEMPORAL_WINDOW': '300',
        'SCAFAD_TIMEOUT_THRESHOLD': '0.6',
        'SCAFAD_VERBOSITY': 'NORMAL',
        'SCAFAD_ADVERSARIAL_MODE': 'DISABLED',
        'AWS_REGION': 'us-east-1'
    }
    
    for var, default in optional_vars.items():
        value = os.environ.get(var, default)
        print(f"✅ {var}={value}")
    
    return True

def check_aws_configuration():
    """Check AWS configuration"""
    print("\n🔍 Checking AWS configuration...")
    
    try:
        # Check if AWS CLI is configured
        aws_region = os.environ.get('AWS_REGION')
        aws_profile = os.environ.get('AWS_PROFILE')
        
        if aws_region:
            print(f"✅ AWS Region: {aws_region}")
        else:
            print("⚠️  AWS_REGION not set (will use default)")
        
        if aws_profile:
            print(f"✅ AWS Profile: {aws_profile}")
        else:
            print("⚠️  AWS_PROFILE not set (will use default)")
        
        # Check for AWS credentials
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        if aws_access_key:
            print("✅ AWS credentials found")
        else:
            print("⚠️  AWS credentials not found in environment")
            print("💡 Make sure to configure AWS CLI or set credentials")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS configuration check failed: {e}")
        return False

def generate_deployment_checklist():
    """Generate deployment checklist"""
    checklist = {
        "pre_deployment": [
            "✅ Run `python run_tests.py` to validate implementation",
            "✅ Run `python validate_deployment.py` to check configuration",
            "✅ Ensure AWS CLI is configured with appropriate permissions",
            "✅ Review template.yaml for correct resource configuration",
            "✅ Set environment variables if needed"
        ],
        "deployment": [
            "📦 Run `sam build` to build the application",
            "🚀 Run `sam deploy --guided` for first deployment",
            "🔄 Run `sam deploy` for subsequent deployments",
            "📊 Monitor CloudWatch logs for execution",
            "🧪 Test with sample payloads"
        ],
        "post_deployment": [
            "📈 Check CloudWatch metrics",
            "📋 Review telemetry output",
            "🔍 Validate anomaly detection",
            "⚡ Test performance under load",
            "🛡️  Verify security configurations"
        ]
    }
    
    print("\n📋 Deployment Checklist")
    print("=" * 30)
    
    for phase, items in checklist.items():
        print(f"\n{phase.replace('_', ' ').title()}:")
        for item in items:
            print(f"  {item}")
    
    return checklist

def main():
    """Main validation function"""
    print("🔧 SCAFAD Layer 0 Deployment Validation")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("SAM Template", check_sam_template),
        ("Dependencies", check_dependencies),
        ("Environment Variables", check_environment_variables),
        ("AWS Configuration", check_aws_configuration)
    ]
    
    passed = 0
    failed = 0
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 Validation Results")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All validation checks passed!")
        print("🚀 Your SCAFAD Layer 0 is ready for deployment!")
    else:
        print(f"\n⚠️  {failed} validation check(s) failed.")
        print("💡 Please address the issues before deployment.")
    
    # Always show deployment checklist
    generate_deployment_checklist()
    
    print("\n" + "=" * 50)
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# quick_test.py
"""
Ultra-quick test to verify SCAFAD Layer 0 is working
"""

def quick_test():
    """Run the quickest possible test"""
    print("⚡ SCAFAD Layer 0 Quick Test")
    print("=" * 30)
    
    try:
        # Test 1: Basic import
        print("1. Testing import...", end=" ")
        from app import lambda_handler, AnomalyType
        print("✅")
        
        # Test 2: Create simple event
        print("2. Creating test event...", end=" ")
        event = {"test_mode": True, "anomaly": "benign"}
        print("✅")
        
        # Test 3: Mock context
        print("3. Creating mock context...", end=" ")
        from unittest.mock import Mock
        context = Mock()
        context.aws_request_id = "quick-test"
        context.function_name = "quick-test"
        context.function_version = "$LATEST"
        context.memory_limit_in_mb = 128
        print("✅")
        
        # Test 4: Call handler
        print("4. Calling lambda_handler...", end=" ")
        result = lambda_handler(event, context)
        print("✅")
        
        # Test 5: Validate response
        print("5. Validating response...", end=" ")
        assert isinstance(result, dict)
        assert 'statusCode' in result
        assert 'body' in result
        print("✅")
        
        print("\n🎉 Quick test PASSED! SCAFAD Layer 0 is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)