#!/usr/bin/env python3
"""
SCAFAD Layer 0 Comprehensive Test Suite
======================================

Systematic testing to verify Layer 0 works as intended for dissertation validation.
"""

import json
import subprocess
import time
import os
import boto3
from datetime import datetime
from typing import Dict, List, Any, Tuple
import sys

class SCAFADTester:
    """Comprehensive SCAFAD Layer 0 testing framework"""
    
    def __init__(self):
        self.function_name = "scafad-test-stack-HelloWorldFunction-k79tX3iBcK74"
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        self.lambda_client = None
        
    def setup(self):
        """Setup test environment"""
        print("🔧 Setting up SCAFAD Layer 0 test environment...")
        
        try:
            self.lambda_client = boto3.client('lambda')
            print("   ✅ AWS Lambda client initialized")
        except Exception as e:
            print(f"   ❌ Failed to initialize AWS client: {e}")
            return False
        
        # Verify function exists
        try:
            response = self.lambda_client.get_function(FunctionName=self.function_name)
            print(f"   ✅ Function verified: {self.function_name}")
            print(f"      Runtime: {response['Configuration']['Runtime']}")
            print(f"      Memory: {response['Configuration']['MemorySize']}MB")
            return True
        except Exception as e:
            print(f"   ❌ Function not found: {e}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Test 1: Basic Lambda function response"""
        print("\n📋 Test 1: Basic Functionality")
        print("-" * 40)
        
        test_payload = {
            "test": "basic_functionality",
            "anomaly": "benign",
            "function_profile_id": "test_function",
            "execution_phase": "invoke",
            "timestamp": time.time()
        }
        
        try:
            start_time = time.time()
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(test_payload)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse response
            result = json.loads(response['Payload'].read())
            
            # Validate basic response structure
            checks = {
                "status_code_200": result.get("statusCode") == 200,
                "has_body": "body" in result,
                "has_headers": "headers" in result,
                "response_time_reasonable": duration < 30.0,
                "no_function_error": response.get("FunctionError") is None
            }
            
            success = all(checks.values())
            
            print(f"   Status Code: {result.get('statusCode')} {'✅' if checks['status_code_200'] else '❌'}")
            print(f"   Response Time: {duration:.2f}s {'✅' if checks['response_time_reasonable'] else '❌'}")
            print(f"   Has Body: {'✅' if checks['has_body'] else '❌'}")
            print(f"   Has Headers: {'✅' if checks['has_headers'] else '❌'}")
            print(f"   No Errors: {'✅' if checks['no_function_error'] else '❌'}")
            
            self.test_results["tests"]["basic_functionality"] = {
                "success": success,
                "duration": duration,
                "checks": checks,
                "response": result
            }
            
            return success
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results["tests"]["basic_functionality"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_scafad_telemetry_format(self) -> bool:
        """Test 2: SCAFAD telemetry format validation"""
        print("\n📋 Test 2: SCAFAD Telemetry Format")
        print("-" * 40)
        
        test_payload = {
            "anomaly": "cold_start",
            "function_profile_id": "ml_inference", 
            "execution_phase": "init",
            "concurrency_id": "TST",
            "test_telemetry_format": True
        }
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=self.function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(test_payload)
            )
            
            result = json.loads(response['Payload'].read())
            
            # Parse response body
            if result.get("statusCode") == 200 and "body" in result:
                try:
                    body = json.loads(result["body"])
                except:
                    body = result["body"]
                
                # Check SCAFAD-specific fields
                scafad_checks = {
                    "has_telemetry_id": "telemetry_id" in body,
                    "has_processing_time": "processing_time_ms" in body,
                    "has_anomaly_detected": "anomaly_detected" in body,
                    "has_economic_risk_score": "economic_risk_score" in body,
                    "has_completeness_score": "completeness_score" in body,
                    "has_performance_summary": "performance_summary" in body,
                    "has_node_id": "node_id" in body,
                    "has_provenance_id": "provenance_id" in body
                }
                
                # Check headers for SCAFAD metadata
                headers = result.get("headers", {})
                header_checks = {
                    "has_scafad_version": "X-SCAFAD-Version" in headers,
                    "has_scafad_layer": "X-SCAFAD-Layer" in headers,
                    "has_processing_time_header": "X-Processing-Time" in headers,
                    "has_telemetry_id_header": "X-Telemetry-Id" in headers
                }
                
                all_checks = {**scafad_checks, **header_checks}
                success = all(all_checks.values())
                
                print("   SCAFAD Body Fields:")
                for field, passed in scafad_checks.items():
                    print(f"      {field}: {'✅' if passed else '❌'}")
                
                print("   SCAFAD Headers:")
                for field, passed in header_checks.items():
                    print(f"      {field}: {'✅' if passed else '❌'}")
                
                # Additional validation
                if body.get("schema_version"):
                    print(f"   Schema Version: {body['schema_version']} ✅")
                
                self.test_results["tests"]["scafad_telemetry"] = {
                    "success": success,
                    "checks": all_checks,
                    "telemetry_sample": body
                }
                
                return success
            else:
                print("   ❌ Invalid response format")
                return False
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            self.test_results["tests"]["scafad_telemetry"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_anomaly_types(self) -> bool:
        """Test 3: Different anomaly types"""
        print("\n📋 Test 3: Anomaly Type Processing")
        print("-" * 40)
        
        anomaly_types = [
            "benign", "cold_start", "cpu_burst", "memory_spike",
            "timeout_fallback", "execution_failure", "adversarial"
        ]
        
        results = {}
        
        for anomaly in anomaly_types:
            print(f"   Testing {anomaly}...", end=" ")
            
            test_payload = {
                "anomaly": anomaly,
                "function_profile_id": "test_function",
                "execution_phase": "invoke"
            }
            
            try:
                response = self.lambda_client.invoke(
                    FunctionName=self.function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(test_payload)
                )
                
                result = json.loads(response['Payload'].read())
                success = result.get("statusCode") == 200
                
                print("✅" if success else "❌")
                results[anomaly] = success
                
            except Exception as e:
                print(f"❌ ({e})")
                results[anomaly] = False
        
        overall_success = all(results.values())
        success_rate = sum(results.values()) / len(results) * 100
        
        print(f"\n   Overall Success Rate: {success_rate:.1f}% {'✅' if overall_success else '❌'}")
        
        self.test_results["tests"]["anomaly_types"] = {
            "success": overall_success,
            "success_rate": success_rate,
            "individual_results": results
        }
        
        return overall_success
    
    def test_performance_benchmarks(self) -> bool:
        """Test 4: Performance benchmarks"""
        print("\n📋 Test 4: Performance Benchmarks")
        print("-" * 40)
        
        # Run multiple invocations to test performance
        iterations = 5
        durations = []
        
        test_payload = {
            "anomaly": "benign",
            "function_profile_id": "performance_test",
            "execution_phase": "invoke",
            "performance_test": True
        }
        
        print(f"   Running {iterations} performance tests...")
        
        for i in range(iterations):
            try:
                start_time = time.time()
                response = self.lambda_client.invoke(
                    FunctionName=self.function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(test_payload)
                )
                end_time = time.time()
                
                duration = end_time - start_time
                durations.append(duration)
                
                result = json.loads(response['Payload'].read())
                success = result.get("statusCode") == 200
                
                print(f"      Iteration {i+1}: {duration:.3f}s {'✅' if success else '❌'}")
                
            except Exception as e:
                print(f"      Iteration {i+1}: Failed ({e}) ❌")
                return False
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            # Performance criteria
            performance_checks = {
                "avg_under_5s": avg_duration < 5.0,
                "max_under_10s": max_duration < 10.0,
                "consistent_performance": (max_duration - min_duration) < 3.0
            }
            
            success = all(performance_checks.values())
            
            print(f"\n   Performance Summary:")
            print(f"      Average: {avg_duration:.3f}s {'✅' if performance_checks['avg_under_5s'] else '❌'}")
            print(f"      Min: {min_duration:.3f}s")
            print(f"      Max: {max_duration:.3f}s {'✅' if performance_checks['max_under_10s'] else '❌'}")
            print(f"      Consistency: {'✅' if performance_checks['consistent_performance'] else '❌'}")
            
            self.test_results["tests"]["performance"] = {
                "success": success,
                "avg_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "checks": performance_checks,
                "raw_durations": durations
            }
            
            return success
        
        return False
    
    def test_error_handling(self) -> bool:
        """Test 5: Error handling and resilience"""
        print("\n📋 Test 5: Error Handling & Resilience")
        print("-" * 40)
        
        # Test various error scenarios
        error_tests = [
            {
                "name": "malformed_json",
                "payload": "invalid json",
                "expected": "handles gracefully"
            },
            {
                "name": "missing_fields", 
                "payload": {"test": "minimal"},
                "expected": "provides defaults"
            },
            {
                "name": "invalid_anomaly_type",
                "payload": {"anomaly": "invalid_type"},
                "expected": "handles gracefully"
            }
        ]
        
        results = {}
        
        for test_case in error_tests:
            print(f"   Testing {test_case['name']}...", end=" ")
            
            try:
                if isinstance(test_case['payload'], str):
                    # Test malformed JSON
                    response = self.lambda_client.invoke(
                        FunctionName=self.function_name,
                        InvocationType='RequestResponse',
                        Payload=test_case['payload']
                    )
                else:
                    response = self.lambda_client.invoke(
                        FunctionName=self.function_name,
                        InvocationType='RequestResponse', 
                        Payload=json.dumps(test_case['payload'])
                    )
                
                result = json.loads(response['Payload'].read())
                
                # Function should handle errors gracefully (not crash)
                success = response.get("FunctionError") is None
                print("✅" if success else "❌")
                results[test_case['name']] = success
                
            except Exception as e:
                print(f"❌ ({e})")
                results[test_case['name']] = False
        
        overall_success = all(results.values())
        
        self.test_results["tests"]["error_handling"] = {
            "success": overall_success,
            "individual_results": results
        }
        
        return overall_success
    
    def test_invoke_script_integration(self) -> bool:
        """Test 6: Integration with invoke.py script"""
        print("\n📋 Test 6: Invoke Script Integration")
        print("-" * 40)
        
        # Test if invoke.py works with correct function name
        try:
            print("   Testing invoke.py with small batch...")
            
            # Check if invoke.py exists
            if not os.path.exists("invoke.py"):
                print("   ❌ invoke.py not found")
                return False
            
            # Run invoke script with minimal test
            cmd = [
                sys.executable, "invoke.py", 
                "--n", "3",
                "--mode", "test",
                "--function-name", self.function_name
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check if script ran successfully
            success = result.returncode == 0
            
            if success:
                # Look for success indicators in output
                output = result.stdout
                success_indicators = [
                    "✅" in output,
                    "completed" in output.lower(),
                    "success" in output.lower()
                ]
                
                script_success = any(success_indicators)
                print(f"   Script execution: {'✅' if script_success else '❌'}")
                
                # Check for telemetry output directory
                telemetry_created = os.path.exists("telemetry")
                print(f"   Telemetry directory: {'✅' if telemetry_created else '❌'}")
                
                overall_success = script_success and telemetry_created
                
            else:
                print(f"   ❌ Script failed with return code: {result.returncode}")
                print(f"   Error: {result.stderr}")
                overall_success = False
            
            self.test_results["tests"]["invoke_script"] = {
                "success": overall_success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            return overall_success
            
        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")
            self.test_results["tests"]["invoke_script"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        print("🧪 SCAFAD Layer 0 Comprehensive Test Suite")
        print("=" * 60)
        
        if not self.setup():
            print("❌ Setup failed. Cannot continue with tests.")
            return self.test_results
        
        # Run all tests
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("SCAFAD Telemetry Format", self.test_scafad_telemetry_format),
            ("Anomaly Types", self.test_anomaly_types),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling", self.test_error_handling),
            ("Invoke Script Integration", self.test_invoke_script_integration)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                if success:
                    passed_tests += 1
            except Exception as e:
                print(f"   ❌ Test '{test_name}' crashed: {e}")
        
        # Generate summary
        success_rate = passed_tests / total_tests * 100
        overall_success = passed_tests == total_tests
        
        self.test_results["summary"] = {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "overall_success": overall_success
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"🎯 Overall Result: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"scafad_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📋 Detailed results saved: {results_file}")
        
        if overall_success:
            print("\n🎉 SCAFAD Layer 0 is working perfectly for your dissertation!")
        else:
            print("\n🔧 Some tests failed. Review the results above for issues to fix.")
        
        return self.test_results

def main():
    """Main test execution"""
    tester = SCAFADTester()
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    if results["summary"]["overall_success"]:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()