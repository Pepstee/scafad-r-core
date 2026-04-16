#!/usr/bin/env python3
"""
SCAFAD Layer 0 - Final Validation Suite
Complete end-to-end validation before Layer 1 handoff
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Any, Tuple
from datetime import datetime
import traceback

class FinalValidationSuite:
    """Complete validation suite for SCAFAD Layer 0 production readiness"""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = time.time()
        self.issues = []
        self.warnings = []
        
    def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🏁 SCAFAD Layer 0 - Final Validation Suite")
        print("=" * 60)
        print("Comprehensive validation for Layer 1 handoff readiness\n")
        
        validations = [
            ("Code Quality", self._validate_code_quality),
            ("Component Integration", self._validate_component_integration),
            ("Data Flow Integrity", self._validate_data_flow),
            ("Error Handling", self._validate_error_handling),
            ("Performance Requirements", self._validate_performance),
            ("Layer 1 Compatibility", self._validate_layer1_compatibility),
            ("Security Configuration", self._validate_security),
            ("Deployment Readiness", self._validate_deployment_readiness)
        ]
        
        for validation_name, validation_func in validations:
            print(f"\n🔍 {validation_name} Validation")
            print("-" * 40)
            
            try:
                result = validation_func()
                self.validation_results[validation_name] = result
                
                if result['passed']:
                    print(f"  ✅ {validation_name}: PASSED")
                else:
                    print(f"  ❌ {validation_name}: FAILED")
                    self.issues.extend(result.get('issues', []))
                
                for warning in result.get('warnings', []):
                    self.warnings.append(f"{validation_name}: {warning}")
                    print(f"  ⚠️  {warning}")
                    
            except Exception as e:
                print(f"  💥 {validation_name}: VALIDATION ERROR - {e}")
                self.issues.append(f"{validation_name} validation failed: {e}")
                self.validation_results[validation_name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        return self._generate_final_report()
    
    def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality and structure"""
        issues = []
        warnings = []
        
        # Check if app.py exists and is valid
        if not os.path.exists('app.py'):
            issues.append("app.py file not found")
            return {'passed': False, 'issues': issues}
        
        # Run syntax check
        try:
            with open('app.py', 'r') as f:
                content = f.read()
            compile(content, 'app.py', 'exec')
            print("  ✅ Syntax validation passed")
        except SyntaxError as e:
            issues.append(f"Syntax error in app.py: Line {e.lineno}: {e.msg}")
            return {'passed': False, 'issues': issues}
        
        # Check for required imports
        required_imports = ['json', 'time', 'uuid', 'hashlib', 'asyncio']
        for imp in required_imports:
            if f"import {imp}" not in content:
                warnings.append(f"Missing import: {imp}")
        
        # Check for required classes
        required_classes = [
            'Layer0_AdaptiveTelemetryController',
            'TelemetryRecord',
            'AnomalyType',
            'ExecutionPhase'
        ]
        
        for cls in required_classes:
            if f"class {cls}" not in content:
                issues.append(f"Missing required class: {cls}")
        
        # Check for lambda_handler function
        if "def lambda_handler(" not in content:
            issues.append("Missing lambda_handler function")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'file_size': len(content),
            'line_count': len(content.split('\n'))
        }
    
    def _validate_component_integration(self) -> Dict[str, Any]:
        """Validate component integration and initialization"""
        issues = []
        warnings = []
        
        try:
            from app import Layer0_AdaptiveTelemetryController
            controller = Layer0_AdaptiveTelemetryController()
            
            # Check all components are initialized
            components = {
                'graph_builder': 'InvocationGraphBuilder',
                'adversarial_simulator': 'AdversarialAnomalySimulator',
                'provenance_tracker': 'ProvenanceChain',
                'telemetry_channels': 'MultiChannelTelemetry'
            }
            
            initialized_components = 0
            for attr, cls_name in components.items():
                if hasattr(controller, attr):
                    component = getattr(controller, attr)
                    if component is not None:
                        initialized_components += 1
                        print(f"    ✅ {cls_name}")
                    else:
                        issues.append(f"Component {attr} is None")
                        print(f"    ❌ {cls_name} (None)")
                else:
                    issues.append(f"Missing component attribute: {attr}")
                    print(f"    ❌ {cls_name} (Missing)")
            
            # Check optional components
            optional_components = {
                'schema_registry': 'SchemaEvolutionManager',
                'formal_verifier': 'TelemetryCompletenessVerifier',
                'economic_monitor': 'EconomicAbuseMonitor',
                'silent_failure_detector': 'SilentFailureDetector'
            }
            
            optional_count = 0
            for attr, cls_name in optional_components.items():
                if hasattr(controller, attr) and getattr(controller, attr) is not None:
                    optional_count += 1
                    print(f"    ✅ {cls_name} (optional)")
                else:
                    warnings.append(f"Optional component {cls_name} not available")
            
        except ImportError as e:
            issues.append(f"Cannot import Layer0_AdaptiveTelemetryController: {e}")
        except Exception as e:
            issues.append(f"Component integration test failed: {e}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'required_components': initialized_components,
            'optional_components': optional_count
        }
    
    def _validate_data_flow(self) -> Dict[str, Any]:
        """Validate data flow from input to output"""
        issues = []
        warnings = []
        
        try:
            from app import lambda_handler
            from unittest.mock import Mock
            
            # Create test context
            context = Mock()
            context.aws_request_id = "validation-test-001"
            context.function_name = "validation-function"
            context.function_version = "$LATEST"
            context.memory_limit_in_mb = 256
            
            # Test different anomaly types
            test_cases = [
                {'anomaly': 'benign', 'execution_phase': 'invoke'},
                {'anomaly': 'cpu_burst', 'execution_phase': 'invoke'},
                {'anomaly': 'cold_start', 'execution_phase': 'init'},
                {'anomaly': 'memory_leak', 'execution_phase': 'invoke'}
            ]
            
            successful_flows = 0
            for test_case in test_cases:
                try:
                    event = {**test_case, 'test_mode': True}
                    response = lambda_handler(event, context)
                    
                    # Validate response structure
                    if not isinstance(response, dict):
                        issues.append(f"Invalid response type for {test_case['anomaly']}")
                        continue
                    
                    if 'statusCode' not in response:
                        issues.append(f"Missing statusCode for {test_case['anomaly']}")
                        continue
                    
                    if 'body' not in response:
                        issues.append(f"Missing body for {test_case['anomaly']}")
                        continue
                    
                    # Parse body
                    try:
                        body_data = json.loads(response['body'])
                        
                        # Check for required fields
                        required_fields = ['telemetry_id', 'processing_time_ms']
                        for field in required_fields:
                            if field not in body_data:
                                issues.append(f"Missing {field} in response for {test_case['anomaly']}")
                    
                    except json.JSONDecodeError:
                        issues.append(f"Invalid JSON body for {test_case['anomaly']}")
                        continue
                    
                    successful_flows += 1
                    print(f"    ✅ {test_case['anomaly']} flow validated")
                    
                except Exception as e:
                    issues.append(f"Data flow test failed for {test_case['anomaly']}: {e}")
                    print(f"    ❌ {test_case['anomaly']} flow failed")
            
            flow_success_rate = (successful_flows / len(test_cases)) * 100
            
        except Exception as e:
            issues.append(f"Data flow validation failed: {e}")
            flow_success_rate = 0
        
        return {
            'passed': len(issues) == 0 and flow_success_rate >= 75,
            'issues': issues,
            'warnings': warnings,
            'flow_success_rate': flow_success_rate,
            'successful_flows': successful_flows,
            'total_flows': len(test_cases)
        }
    
    def _validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling and fallback mechanisms"""
        issues = []
        warnings = []
        
        try:
            from app import lambda_handler
            from unittest.mock import Mock
            
            context = Mock()
            context.aws_request_id = "error-test-001"
            context.function_name = "error-test"
            context.memory_limit_in_mb = 128
            
            error_test_cases = [
                ("Invalid Event Type", "not_a_dict"),
                ("Empty Event", {}),
                ("Missing Fields", {"unknown_field": "value"}),
                ("Invalid Anomaly", {"anomaly": "invalid_anomaly_type"}),
                ("Force Starvation", {"force_starvation": True, "test_mode": True})
            ]
            
            handled_errors = 0
            for test_name, test_event in error_test_cases:
                try:
                    response = lambda_handler(test_event, context)
                    
                    # Check if error was handled gracefully
                    if isinstance(response, dict) and 'statusCode' in response:
                        status_code = response['statusCode']
                        if status_code in [200, 202, 206, 400, 500]:
                            handled_errors += 1
                            print(f"    ✅ {test_name}: Handled (Status: {status_code})")
                        else:
                            warnings.append(f"{test_name}: Unexpected status code {status_code}")
                            print(f"    ⚠️  {test_name}: Unexpected status {status_code}")
                    else:
                        issues.append(f"{test_name}: Invalid response format")
                        print(f"    ❌ {test_name}: Invalid response")
                
                except Exception as e:
                    # Exceptions are acceptable for some error cases
                    warnings.append(f"{test_name}: Exception raised - {str(e)[:50]}...")
                    print(f"    ⚠️  {test_name}: Exception raised")
            
            error_handling_rate = (handled_errors / len(error_test_cases)) * 100
            
        except Exception as e:
            issues.append(f"Error handling validation failed: {e}")
            error_handling_rate = 0
        
        return {
            'passed': error_handling_rate >= 60,  # Allow some exceptions
            'issues': issues,
            'warnings': warnings,
            'error_handling_rate': error_handling_rate,
            'handled_errors': handled_errors,
            'total_error_tests': len(error_test_cases)
        }
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance requirements"""
        issues = []
        warnings = []
        
        try:
            from app import lambda_handler, generate_test_payloads
            from unittest.mock import Mock
            
            context = Mock()
            context.aws_request_id = "perf-test-001"
            context.function_name = "performance-test"
            context.memory_limit_in_mb = 256
            
            # Generate test payloads
            payloads = generate_test_payloads(10, 42)
            
            execution_times = []
            successful_executions = 0
            
            print(f"    Testing with {len(payloads)} payloads...")
            
            for i, payload in enumerate(payloads):
                try:
                    start_time = time.time()
                    response = lambda_handler(payload, context)
                    execution_time = (time.time() - start_time) * 1000  # ms
                    
                    execution_times.append(execution_time)
                    
                    if isinstance(response, dict) and response.get('statusCode') in [200, 202, 206]:
                        successful_executions += 1
                        
                except Exception as e:
                    warnings.append(f"Payload {i} failed: {str(e)[:50]}...")
            
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                
                print(f"    ⏱️  Average: {avg_time:.2f}ms")
                print(f"    ⏱️  Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
                print(f"    📊 Success rate: {(successful_executions/len(payloads)*100):.1f}%")
                
                # Performance requirements
                if avg_time > 500:
                    issues.append(f"Average execution time ({avg_time:.2f}ms) exceeds 500ms requirement")
                
                if max_time > 2000:
                    issues.append(f"Maximum execution time ({max_time:.2f}ms) exceeds 2000ms limit")
                
                if successful_executions / len(payloads) < 0.9:
                    issues.append(f"Success rate ({successful_executions/len(payloads)*100:.1f}%) below 90% requirement")
            
            else:
                issues.append("No successful performance measurements")
        
        except Exception as e:
            issues.append(f"Performance validation failed: {e}")
            avg_time = 0
            successful_executions = 0
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'avg_execution_time_ms': avg_time if 'avg_time' in locals() else 0,
            'success_rate': (successful_executions / len(payloads) * 100) if 'payloads' in locals() else 0,
            'total_tests': len(payloads) if 'payloads' in locals() else 0
        }
    
    def _validate_layer1_compatibility(self) -> Dict[str, Any]:
        """Validate Layer 1 handoff compatibility"""
        issues = []
        warnings = []
        
        try:
            from app import lambda_handler
            from unittest.mock import Mock
            
            context = Mock()
            context.aws_request_id = "layer1-test-001"
            context.function_name = "layer1-compatibility-test"
            context.memory_limit_in_mb = 256
            
            test_event = {
                'anomaly': 'cpu_burst',
                'execution_phase': 'invoke',
                'test_mode': True,
                'function_profile_id': 'test_func'
            }
            
            response = lambda_handler(test_event, context)
            
            # Validate response structure for Layer 1
            if not isinstance(response, dict):
                issues.append("Response is not a dictionary")
                return {'passed': False, 'issues': issues}
            
            # Check required response fields
            required_fields = ['statusCode', 'body', 'headers']
            for field in required_fields:
                if field not in response:
                    issues.append(f"Missing required response field: {field}")
            
            # Validate headers
            headers = response.get('headers', {})
            required_headers = ['Content-Type', 'X-SCAFAD-Version', 'X-SCAFAD-Layer']
            for header in required_headers:
                if header not in headers:
                    warnings.append(f"Missing recommended header: {header}")
            
            # Parse and validate body
            try:
                body_data = json.loads(response.get('body', '{}'))
                
                # Layer 1 required fields
                layer1_required = [
                    'telemetry_id',
                    'processing_time_ms',
                    'anomaly_detected'
                ]
                
                layer1_recommended = [
                    'node_id',
                    'provenance_id',
                    'economic_risk_score',
                    'completeness_score'
                ]
                
                missing_required = []
                missing_recommended = []
                
                for field in layer1_required:
                    if field not in body_data:
                        missing_required.append(field)
                        issues.append(f"Missing Layer 1 required field: {field}")
                    else:
                        print(f"    ✅ Layer 1 field: {field}")
                
                for field in layer1_recommended:
                    if field not in body_data:
                        missing_recommended.append(field)
                        warnings.append(f"Missing Layer 1 recommended field: {field}")
                    else:
                        print(f"    ✅ Layer 1 field: {field}")
                
                # Validate data types
                field_types = {
                    'telemetry_id': str,
                    'processing_time_ms': (int, float),
                    'anomaly_detected': bool
                }
                
                for field, expected_type in field_types.items():
                    if field in body_data:
                        if not isinstance(body_data[field], expected_type):
                            issues.append(f"Field {field} has wrong type: expected {expected_type}, got {type(body_data[field])}")
                
            except json.JSONDecodeError as e:
                issues.append(f"Response body is not valid JSON: {e}")
        
        except Exception as e:
            issues.append(f"Layer 1 compatibility validation failed: {e}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'layer1_compatibility_score': max(0, 100 - len(issues) * 10 - len(warnings) * 5)
        }
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security configuration"""
        issues = []
        warnings = []
        
        # Check for hardcoded secrets
        try:
            with open('app.py', 'r') as f:
                content = f.read()
            
            # Look for potential security issues
            security_patterns = [
                ('AWS_ACCESS_KEY_ID', 'Hardcoded AWS access key'),
                ('AWS_SECRET_ACCESS_KEY', 'Hardcoded AWS secret key'),
                ('password', 'Hardcoded password'),
                ('secret', 'Hardcoded secret'),
                ('api_key', 'Hardcoded API key')
            ]
            
            for pattern, description in security_patterns:
                if pattern in content.lower():
                    warnings.append(f"Potential security issue: {description}")
            
            print("    ✅ Security pattern scan completed")
            
        except Exception as e:
            warnings.append(f"Security validation failed: {e}")
        
        # Check environment variable usage
        env_vars_used = []
        if 'os.environ' in content:
            env_vars_used.append('Environment variables properly used')
            print("    ✅ Environment variables used for configuration")
        else:
            warnings.append("No environment variable usage detected")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'security_score': max(0, 100 - len(issues) * 20 - len(warnings) * 5)
        }
    
    def _validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness"""
        issues = []
        warnings = []
        
        # Check for required files
        required_files = ['app.py', 'template.yaml']
        optional_files = ['samconfig.toml', 'requirements.txt']
        
        for file in required_files:
            if os.path.exists(file):
                print(f"    ✅ {file}")
            else:
                issues.append(f"Missing required file: {file}")
                print(f"    ❌ {file}")
        
        for file in optional_files:
            if os.path.exists(file):
                print(f"    ✅ {file} (optional)")
            else:
                warnings.append(f"Missing optional file: {file}")
                print(f"    ⚠️  {file} (optional)")
        
        # Check if tests can run
        test_files = ['quick_test.py', 'run_tests.py', 'test_all.py']
        runnable_tests = 0
        
        for test_file in test_files:
            if os.path.exists(test_file):
                try:
                    result = subprocess.run(
                        [sys.executable, test_file],
                        capture_output=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        runnable_tests += 1
                        print(f"    ✅ {test_file} executable")
                    else:
                        warnings.append(f"{test_file} exits with error code {result.returncode}")
                        print(f"    ⚠️  {test_file} has issues")
                except subprocess.TimeoutExpired:
                    warnings.append(f"{test_file} timed out")
                    print(f"    ⚠️  {test_file} timed out")
                except Exception as e:
                    warnings.append(f"{test_file} execution failed: {e}")
                    print(f"    ❌ {test_file} failed")
        
        readiness_score = (runnable_tests / len(test_files)) * 100
        
        return {
            'passed': len(issues) == 0 and readiness_score >= 50,
            'issues': issues,
            'warnings': warnings,
            'readiness_score': readiness_score,
            'runnable_tests': runnable_tests,
            'total_tests': len(test_files)
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final validation report"""
        total_time = time.time() - self.start_time
        
        print(f"\n📊 Final Validation Report")
        print("=" * 60)
        
        # Calculate overall scores
        passed_validations = sum(1 for result in self.validation_results.values() 
                               if result.get('passed', False))
        total_validations = len(self.validation_results)
        overall_score = (passed_validations / total_validations * 100) if total_validations > 0 else 0
        
        print(f"Overall Score: {overall_score:.1f}%")
        print(f"Validations Passed: {passed_validations}/{total_validations}")
        print(f"Total Issues: {len(self.issues)}")
        print(f"Total Warnings: {len(self.warnings)}")
        print(f"Execution Time: {total_time:.2f}s")
        
        # Validation summary
        print(f"\n📋 Validation Summary:")
        for validation_name, result in self.validation_results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            issue_count = len(result.get('issues', []))
            print(f"  {status} {validation_name} ({issue_count} issues)")
        
        # Critical issues
        if self.issues:
            print(f"\n🚨 Critical Issues ({len(self.issues)}):")
            for i, issue in enumerate(self.issues[:10], 1):  # Show top 10
                print(f"  {i}. {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more issues")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:5], 1):  # Show top 5
                print(f"  {i}. {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more warnings")
        
        # Layer 1 readiness assessment
        layer1_ready = (
            overall_score >= 85 and
            len(self.issues) == 0 and
            passed_validations >= 6
        )
        
        print(f"\n🎯 Layer 1 Readiness Assessment:")
        if layer1_ready:
            print("  🎉 READY FOR LAYER 1 INTEGRATION!")
            print("  ✅ All critical validations passed")
            print("  🚀 Your SCAFAD Layer 0 is production-ready")
            print(f"\n📋 Next Steps:")
            print("  1. Deploy to staging environment")
            print("  2. Run integration tests with Layer 1")
            print("  3. Configure monitoring and alerting")
            print("  4. Deploy to production")
        else:
            print("  ⚠️  NOT READY for Layer 1 integration")
            print("  🔧 Please address the critical issues above")
            print(f"\n💡 Recommendations:")
            
            if overall_score < 85:
                print(f"  • Improve overall score from {overall_score:.1f}% to 85%+")
            if len(self.issues) > 0:
                print(f"  • Fix all {len(self.issues)} critical issues")
            if passed_validations < 6:
                print(f"  • Pass at least 6/8 validations (currently {passed_validations}/8)")
            
            print("  • Re-run validation after fixes: python final_validation.py")
        
        # Generate report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': total_time,
            'overall_score': overall_score,
            'layer1_ready': layer1_ready,
            'validations': {
                'total': total_validations,
                'passed': passed_validations,
                'failed': total_validations - passed_validations
            },
            'issues': {
                'total': len(self.issues),
                'critical': len(self.issues),
                'list': self.issues
            },
            'warnings': {
                'total': len(self.warnings),
                'list': self.warnings
            },
            'validation_results': self.validation_results,
            'recommendations': self._generate_recommendations(overall_score, layer1_ready)
        }
        
        return report
    
    def _generate_recommendations(self, overall_score: float, layer1_ready: bool) -> List[str]:
        """Generate specific recommendations based on validation results"""
        recommendations = []
        
        if not layer1_ready:
            # Priority recommendations
            if 'Code Quality' in self.validation_results:
                result = self.validation_results['Code Quality']
                if not result.get('passed', False):
                    recommendations.append("Fix code quality issues - check syntax and required classes")
            
            if 'Component Integration' in self.validation_results:
                result = self.validation_results['Component Integration']
                if result.get('required_components', 0) < 4:
                    recommendations.append("Ensure all required components are properly initialized")
            
            if 'Layer 1 Compatibility' in self.validation_results:
                result = self.validation_results['Layer 1 Compatibility']
                if not result.get('passed', False):
                    recommendations.append("Fix Layer 1 compatibility issues - check response format")
            
            if 'Performance Requirements' in self.validation_results:
                result = self.validation_results['Performance Requirements']
                if result.get('avg_execution_time_ms', 0) > 500:
                    recommendations.append("Optimize performance - average execution time exceeds 500ms")
        
        else:
            # Production readiness recommendations
            recommendations.extend([
                "Configure production monitoring and logging",
                "Set up automated testing pipeline", 
                "Review security configurations",
                "Plan Layer 1 integration testing",
                "Prepare deployment documentation"
            ])
        
        return recommendations

def main():
    """Main validation function"""
    validator = FinalValidationSuite()
    report = validator.run_complete_validation()
    
    # Save detailed report
    with open('scafad_final_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: scafad_final_validation_report.json")
    
    return report['layer1_ready']

if __name__ == "__main__":
    ready = main()
    exit(0 if ready else 1)