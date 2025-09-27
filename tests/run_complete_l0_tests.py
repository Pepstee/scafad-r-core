"""
Master Test Runner for Complete L0 Testing

This module orchestrates all high-impact test suites to make L0 "complete":
- Fallback orchestration matrix
- Adaptive buffer/backpressure
- Vendor adapters
- Privacy/compliance
- State machine tests
- Health/heartbeat
- L0→L1 contract

Replaces placeholders and provides comprehensive coverage.
"""

import sys
import os
import time
import traceback
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add the parent directory to the path to import test modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test suites
try:
    from tests.test_fallback_orchestration_matrix import run_fallback_matrix_tests
    from tests.test_adaptive_buffer_backpressure import run_adaptive_buffer_tests
    from tests.test_vendor_adapters import run_vendor_adapter_tests
    from tests.test_privacy_compliance import run_privacy_compliance_tests
    from tests.test_state_machine import run_state_machine_tests
    from tests.test_health_heartbeat import run_health_heartbeat_tests
    from tests.test_l0_l1_contract import run_l0_l1_contract_tests
except ImportError as e:
    print(f"Warning: Could not import test modules: {e}")
    print("Some tests may not be available")


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped'
    duration_ms: float
    error_message: str = ""
    details: Dict[str, Any] = None


@dataclass
class TestSuiteResult:
    """Test suite execution result"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration_ms: float
    test_results: List[TestResult]
    summary: Dict[str, Any] = None


class CompleteL0TestRunner:
    """Master test runner for complete L0 testing"""
    
    def __init__(self):
        self.test_suites = {
            'fallback_orchestration_matrix': {
                'runner': run_fallback_matrix_tests,
                'description': 'Fallback orchestration matrix (OTLP→Vendor→HTTP→Local)',
                'critical': True
            },
            'adaptive_buffer_backpressure': {
                'runner': run_adaptive_buffer_tests,
                'description': 'Adaptive buffer/backpressure with watermarks and LOSS_POLICY',
                'critical': True
            },
            'vendor_adapters': {
                'runner': run_vendor_adapter_tests,
                'description': 'Vendor adapter conformance (rate limits, retries, idempotency)',
                'critical': True
            },
            'privacy_compliance': {
                'runner': run_privacy_compliance_tests,
                'description': 'Privacy/compliance with PII redaction and audit trail',
                'critical': True
            },
            'state_machine': {
                'runner': run_state_machine_tests,
                'description': 'State machine tests (PROBING/ACTIVE/SHADOW/STANDBY/QUARANTINED)',
                'critical': True
            },
            'health_heartbeat': {
                'runner': run_health_heartbeat_tests,
                'description': 'Health/heartbeat with MTTR assertions and crash detection',
                'critical': True
            },
            'l0_l1_contract': {
                'runner': run_l0_l1_contract_tests,
                'description': 'L0→L1 contract with schema versioning and compatibility',
                'critical': True
            }
        }
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, TestSuiteResult]:
        """Run all test suites"""
        print("\n" + "="*80)
        print("COMPLETE L0 TESTING - MAKING L0 'COMPLETE'")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total test suites: {len(self.test_suites)}")
        print("="*80)
        
        self.start_time = time.time()
        
        for suite_name, suite_info in self.test_suites.items():
            print(f"\n{'='*20} RUNNING: {suite_name.upper()} {'='*20}")
            print(f"Description: {suite_info['description']}")
            print(f"Critical: {'YES' if suite_info['critical'] else 'NO'}")
            
            try:
                suite_result = self._run_test_suite(suite_name, suite_info)
                self.results[suite_name] = suite_result
                
                # Print suite summary
                self._print_suite_summary(suite_result)
                
            except Exception as e:
                print(f"ERROR: Test suite {suite_name} failed to run: {e}")
                traceback.print_exc()
                
                # Create failed result
                failed_result = TestSuiteResult(
                    suite_name=suite_name,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    skipped_tests=0,
                    duration_ms=0,
                    test_results=[TestResult(
                        test_name=suite_name,
                        status='failed',
                        duration_ms=0,
                        error_message=str(e)
                    )]
                )
                self.results[suite_name] = failed_result
        
        self.end_time = time.time()
        
        # Print final summary
        self._print_final_summary()
        
        return self.results
    
    def _run_test_suite(self, suite_name: str, suite_info: Dict[str, Any]) -> TestSuiteResult:
        """Run a single test suite"""
        start_time = time.time()
        
        # Capture stdout to analyze test results
        import io
        from contextlib import redirect_stdout
        
        output_capture = io.StringIO()
        
        try:
            with redirect_stdout(output_capture):
                suite_info['runner']()
            
            # Analyze output to determine test results
            output = output_capture.getvalue()
            test_results = self._analyze_test_output(output, suite_name)
            
            # Calculate statistics
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results if r.status == 'passed')
            failed_tests = sum(1 for r in test_results if r.status == 'failed')
            skipped_tests = sum(1 for r in test_results if r.status == 'skipped')
            
            duration_ms = (time.time() - start_time) * 1000
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                duration_ms=duration_ms,
                test_results=test_results,
                summary={'output': output}
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                duration_ms=duration_ms,
                test_results=[TestResult(
                    test_name=suite_name,
                    status='failed',
                    duration_ms=duration_ms,
                    error_message=str(e)
                )],
                summary={'error': str(e)}
            )
    
    def _analyze_test_output(self, output: str, suite_name: str) -> List[TestResult]:
        """Analyze test output to determine individual test results"""
        test_results = []
        
        # Parse output for test results
        lines = output.split('\n')
        current_test = None
        
        for line in lines:
            line = line.strip()
            
            # Look for test method names
            if line.startswith('def test_'):
                test_name = line.split('def ')[1].split('(')[0]
                current_test = TestResult(
                    test_name=test_name,
                    status='unknown',
                    duration_ms=0
                )
            
            # Look for test completion indicators
            elif line.startswith('✓') and current_test:
                current_test.status = 'passed'
                test_results.append(current_test)
                current_test = None
            
            elif line.startswith('✗') and current_test:
                current_test.status = 'failed'
                test_results.append(current_test)
                current_test = None
        
        # If no specific test results found, create a summary result
        if not test_results:
            test_results.append(TestResult(
                test_name=f"{suite_name}_summary",
                status='passed' if 'completed successfully' in output else 'failed',
                duration_ms=0
            ))
        
        return test_results
    
    def _print_suite_summary(self, suite_result: TestSuiteResult):
        """Print summary for a test suite"""
        status_icon = "✅" if suite_result.failed_tests == 0 else "❌"
        
        print(f"\n{status_icon} {suite_result.suite_name.upper()} COMPLETED")
        print(f"   Duration: {suite_result.duration_ms:.2f}ms")
        print(f"   Total: {suite_result.total_tests}")
        print(f"   Passed: {suite_result.passed_tests} ✅")
        print(f"   Failed: {suite_result.failed_tests} ❌")
        print(f"   Skipped: {suite_result.skipped_tests} ⏭️")
        
        if suite_result.failed_tests > 0:
            print("   Failed tests:")
            for result in suite_result.test_results:
                if result.status == 'failed':
                    print(f"     - {result.test_name}: {result.error_message}")
    
    def _print_final_summary(self):
        """Print final test execution summary"""
        total_duration = (self.end_time - self.start_time) * 1000
        
        print("\n" + "="*80)
        print("COMPLETE L0 TESTING - FINAL SUMMARY")
        print("="*80)
        
        # Calculate overall statistics
        total_suites = len(self.results)
        total_tests = sum(r.total_tests for r in self.results.values())
        total_passed = sum(r.passed_tests for r in self.results.values())
        total_failed = sum(r.failed_tests for r in self.results.values())
        total_skipped = sum(r.skipped_tests for r in self.results.values())
        
        # Determine overall status
        critical_suites = [name for name, info in self.test_suites.items() if info['critical']]
        critical_failures = [name for name in critical_suites if self.results.get(name, {}).get('failed_tests', 0) > 0]
        
        overall_status = "PASSED" if not critical_failures else "FAILED"
        status_icon = "✅" if overall_status == "PASSED" else "❌"
        
        print(f"\n{status_icon} OVERALL STATUS: {overall_status}")
        print(f"   Total Duration: {total_duration:.2f}ms")
        print(f"   Test Suites: {total_suites}")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed} ✅")
        print(f"   Failed: {total_failed} ❌")
        print(f"   Skipped: {total_skipped} ⏭️")
        
        if critical_failures:
            print(f"\n❌ CRITICAL SUITE FAILURES:")
            for suite_name in critical_failures:
                print(f"   - {suite_name}")
        
        # Print suite-by-suite summary
        print(f"\n📊 SUITE-BY-SUITE SUMMARY:")
        for suite_name, suite_info in self.test_suites.items():
            result = self.results.get(suite_name)
            if result:
                status = "✅" if result.failed_tests == 0 else "❌"
                critical = "🔴" if suite_info['critical'] else "🟡"
                print(f"   {critical} {status} {suite_name}: {result.passed_tests}/{result.total_tests} passed")
        
        # Print completion message
        if overall_status == "PASSED":
            print(f"\n🎉 CONGRATULATIONS! L0 is now 'COMPLETE'")
            print(f"   All critical test suites passed successfully")
            print(f"   High-impact test coverage achieved")
            print(f"   Placeholders replaced with comprehensive tests")
        else:
            print(f"\n⚠️  L0 is NOT yet 'COMPLETE'")
            print(f"   Critical test suites failed: {', '.join(critical_failures)}")
            print(f"   Please fix failures and re-run tests")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'test_execution': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'total_duration_ms': (self.end_time - self.start_time) * 1000 if self.end_time else 0
            },
            'overall_results': {
                'total_suites': len(self.results),
                'total_tests': sum(r.total_tests for r in self.results.values()),
                'total_passed': sum(r.passed_tests for r in self.results.values()),
                'total_failed': sum(r.failed_tests for r in self.results.values()),
                'total_skipped': sum(r.skipped_tests for r in self.results.values())
            },
            'suite_results': {},
            'recommendations': []
        }
        
        # Add suite results
        for suite_name, result in self.results.items():
            report['suite_results'][suite_name] = {
                'total_tests': result.total_tests,
                'passed_tests': result.passed_tests,
                'failed_tests': result.failed_tests,
                'skipped_tests': result.skipped_tests,
                'duration_ms': result.duration_ms,
                'success_rate': (result.passed_tests / result.total_tests * 100) if result.total_tests > 0 else 0
            }
        
        # Generate recommendations
        critical_failures = [name for name, info in self.test_suites.items() 
                           if info['critical'] and self.results.get(name, {}).get('failed_tests', 0) > 0]
        
        if critical_failures:
            report['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Fix critical test suite failures',
                'suites': critical_failures
            })
        
        # Check for any failures
        failed_suites = [name for name, result in self.results.items() if result.failed_tests > 0]
        if failed_suites:
            report['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'Investigate and fix test failures',
                'suites': failed_suites
            })
        
        # Success recommendations
        if not critical_failures:
            report['recommendations'].append({
                'priority': 'LOW',
                'action': 'L0 is complete - consider additional edge case testing',
                'suites': []
            })
        
        return report


def main():
    """Main entry point for complete L0 testing"""
    print("🚀 Starting Complete L0 Testing Suite")
    print("This will run all high-impact tests to make L0 'complete'")
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test-mode':
        print("🧪 Running in test mode (simulated execution)")
        # Simulate test execution for demonstration
        runner = CompleteL0TestRunner()
        runner.start_time = time.time()
        runner.end_time = time.time() + 1
        
        # Simulate results
        runner.results = {
            'fallback_orchestration_matrix': TestSuiteResult(
                suite_name='fallback_orchestration_matrix',
                total_tests=8,
                passed_tests=8,
                failed_tests=0,
                skipped_tests=0,
                duration_ms=1500,
                test_results=[]
            ),
            'adaptive_buffer_backpressure': TestSuiteResult(
                suite_name='adaptive_buffer_backpressure',
                total_tests=8,
                passed_tests=8,
                failed_tests=0,
                skipped_tests=0,
                duration_ms=1200,
                test_results=[]
            )
        }
        
        runner._print_final_summary()
        return
    
    try:
        # Run all tests
        runner = CompleteL0TestRunner()
        results = runner.run_all_tests()
        
        # Generate report
        report = runner.generate_report()
        
        # Save report to file
        report_file = f"l0_complete_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        # Exit with appropriate code
        critical_failures = [name for name, info in runner.test_suites.items() 
                           if info['critical'] and runner.results.get(name, {}).get('failed_tests', 0) > 0]
        
        if critical_failures:
            print(f"\n❌ Exiting with failure code due to critical suite failures")
            sys.exit(1)
        else:
            print(f"\n✅ Exiting with success code - L0 is complete!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during test execution: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
