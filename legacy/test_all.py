# test_all.py
"""
Automated test runner for SCAFAD Layer 0
Runs all tests in sequence and provides comprehensive report
"""

import subprocess
import sys
import time
import os
from datetime import datetime
import json

class TestRunner:
    """Automated test runner for SCAFAD Layer 0"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.test_files = [
            ("Quick Test", "quick_test.py", "Ultra-fast validation"),
            ("Basic Tests", "run_tests.py", "Comprehensive functionality test"),
            ("Full Test Suite", "test_scafad_layer0.py", "Complete pytest-based validation"),
            ("Deployment Validation", "validate_deployment.py", "Deployment readiness check")
        ]
    
    def print_header(self):
        """Print test suite header"""
        print("🧪" + "=" * 70 + "🧪")
        print("  SCAFAD Layer 0: Automated Test Suite")
        print("  Comprehensive validation of all components")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("🧪" + "=" * 70 + "🧪")
    
    def run_test(self, test_name, test_file, description):
        """Run a single test file"""
        print(f"\n📋 {test_name}")
        print(f"   {description}")
        print(f"   Running: {test_file}")
        print("   " + "-" * 50)
        
        if not os.path.exists(test_file):
            print(f"   ❌ Test file not found: {test_file}")
            return False
        
        start_time = time.time()
        
        try:
            # Run the test
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results
            self.results[test_name] = {
                'file': test_file,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"   ✅ PASSED ({duration:.2f}s)")
                return True
            else:
                print(f"   ❌ FAILED ({duration:.2f}s)")
                print(f"   Exit code: {result.returncode}")
                if result.stderr:
                    print("   Error output:")
                    print("   " + "\n   ".join(result.stderr.split('\n')[:5]))
                return False
        
        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT (> 300s)")
            self.results[test_name] = {
                'file': test_file,
                'duration': 300,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out',
                'success': False
            }
            return False
        
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            self.results[test_name] = {
                'file': test_file,
                'duration': 0,
                'returncode': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
            return False
    
    def check_prerequisites(self):
        """Check if all test files exist"""
        print("\n🔍 Checking Prerequisites...")
        
        missing_files = []
        for test_name, test_file, description in self.test_files:
            if os.path.exists(test_file):
                print(f"   ✅ {test_file}")
            else:
                print(f"   ❌ {test_file} (MISSING)")
                missing_files.append(test_file)
        
        # Check for main implementation
        if os.path.exists('app.py'):
            print(f"   ✅ app.py (main implementation)")
        else:
            print(f"   ❌ app.py (MISSING - CRITICAL)")
            missing_files.append('app.py')
        
        return len(missing_files) == 0, missing_files
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        passed = 0
        failed = 0
        
        for test_name, test_file, description in self.test_files:
            if self.run_test(test_name, test_file, description):
                passed += 1
            else:
                failed += 1
        
        return passed, failed
    
    def generate_summary_report(self, passed, failed):
        """Generate comprehensive summary report"""
        total_time = time.time() - self.start_time
        
        print("\n📊" + "=" * 70 + "📊")
        print("  TEST EXECUTION SUMMARY")
        print("📊" + "=" * 70 + "📊")
        
        print(f"\n⏱️  Total Execution Time: {total_time:.2f} seconds")
        print(f"✅ Tests Passed: {passed}")
        print(f"❌ Tests Failed: {failed}")
        print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        print("   " + "-" * 50)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            print(f"   {status} {test_name:<20} ({duration:6.2f}s)")
        
        return passed, failed, total_time
    
    def generate_detailed_report(self):
        """Generate detailed test report"""
        print(f"\n📝 Detailed Test Output:")
        print("   " + "=" * 50)
        
        for test_name, result in self.results.items():
            print(f"\n📄 {test_name}")
            print("   " + "-" * 30)
            print(f"   File: {result['file']}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Exit Code: {result['returncode']}")
            print(f"   Success: {result['success']}")
            
            if result['stderr']:
                print("   Error Output:")
                error_lines = result['stderr'].split('\n')
                for line in error_lines[:10]:  # Show first 10 lines
                    if line.strip():
                        print(f"     {line}")
                if len(error_lines) > 10:
                    print(f"     ... ({len(error_lines) - 10} more lines)")
    
    def save_results_to_file(self):
        """Save test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': time.time() - self.start_time,
            'results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results.values() if r['success']),
                'failed': sum(1 for r in self.results.values() if not r['success']),
                'success_rate': sum(1 for r in self.results.values() if r['success']) / len(self.results) if self.results else 0
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n💾 Test results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")
            return None
    
    def provide_recommendations(self, passed, failed):
        """Provide recommendations based on test results"""
        print(f"\n💡 Recommendations:")
        print("   " + "-" * 30)
        
        if failed == 0:
            print("   🎉 Excellent! All tests passed.")
            print("   🚀 Your SCAFAD Layer 0 is ready for deployment!")
            print("   📋 Next steps:")
            print("      1. Run 'sam build' to build the application")
            print("      2. Run 'sam deploy --guided' for first deployment")
            print("      3. Test with 'python invoke.py --n 5'")
            print("      4. Monitor with 'python fetch_logs.py'")
        
        elif failed == 1:
            print("   ⚠️  One test failed - investigate and retry.")
            failed_test = next(name for name, result in self.results.items() if not result['success'])
            print(f"   🔍 Failed test: {failed_test}")
            print("   💡 Common solutions:")
            print("      - Check for missing dependencies")
            print("      - Verify app.py has complete implementation")
            print("      - Ensure Python version >= 3.8")
        
        else:
            print(f"   ❌ Multiple tests failed ({failed}/{passed+failed})")
            print("   🔧 Troubleshooting steps:")
            print("      1. Verify app.py contains the complete SCAFAD implementation")
            print("      2. Check for syntax errors in the code")
            print("      3. Install missing dependencies:")
            print("         pip install networkx numpy tenacity pytest")
            print("      4. Run individual tests to isolate issues")
            print("      5. Check the detailed error output above")
    
    def run(self):
        """Main test runner execution"""
        self.print_header()
        
        # Check prerequisites
        prereq_ok, missing_files = self.check_prerequisites()
        if not prereq_ok:
            print(f"\n❌ Prerequisites check failed!")
            print(f"   Missing files: {missing_files}")
            print(f"   Cannot proceed with testing.")
            return False
        
        print("   ✅ All prerequisites satisfied")
        
        # Run all tests
        print(f"\n🚀 Starting test execution...")
        passed, failed = self.run_all_tests()
        
        # Generate reports
        self.generate_summary_report(passed, failed)
        
        if failed > 0:
            self.generate_detailed_report()
        
        # Save results
        self.save_results_to_file()
        
        # Provide recommendations
        self.provide_recommendations(passed, failed)
        
        print("\n🏁 Test execution completed!")
        print("🧪" + "=" * 70 + "🧪")
        
        return failed == 0

def main():
    """Main function"""
    print("Starting SCAFAD Layer 0 automated test suite...")
    
    runner = TestRunner()
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


# create_test_files.py
"""
Script to create all necessary test files if they don't exist
"""

import os

def create_quick_test():
    """Create quick_test.py if it doesn't exist"""
    content = '''# quick_test.py
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
        
        print("\\n🎉 Quick test PASSED! SCAFAD Layer 0 is working!")
        return True
        
    except Exception as e:
        print(f"\\n❌ Quick test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
'''
    
    if not os.path.exists('quick_test.py'):
        with open('quick_test.py', 'w') as f:
            f.write(content)
        print("✅ Created quick_test.py")
    else:
        print("ℹ️  quick_test.py already exists")

def create_missing_files():
    """Create any missing test files"""
    print("🔧 Creating missing test files...")
    
    create_quick_test()
    
    # Note: Other files are already created as artifacts
    print("✅ Test file creation completed")

if __name__ == "__main__":
    create_missing_files()