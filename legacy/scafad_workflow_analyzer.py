#!/usr/bin/env python3
"""
SCAFAD Layer 0 Debug Helper
Specific debugging utilities for your SCAFAD implementation
"""

import os
import sys
import json
import traceback
import importlib.util
from typing import Dict, Any, List
import subprocess

class SCAFADDebugger:
    """Debug helper specifically for SCAFAD Layer 0 components"""
    
    def __init__(self):
        self.debug_results = {}
        
    def check_app_py_structure(self) -> Dict[str, Any]:
        """Check if app.py has all required SCAFAD components"""
        print("🔍 Checking app.py structure...")
        
        required_components = [
            'lambda_handler',
            'AnomalyType', 
            'InvocationGraphBuilder',
            'AdversarialAnomalySimulator',
            'ProvenanceChain',
            'TelemetryRecord',
            'Layer0_AdaptiveTelemetryController'
        ]
        
        optional_components = [
            'SchemaEvolutionManager',
            'TelemetryCompletenessVerifier', 
            'EconomicAbuseMonitor',
            'SilentFailureDetector',
            'MultiChannelTelemetry'
        ]
        
        results = {
            'file_exists': os.path.exists('app.py'),
            'required_found': {},
            'optional_found': {},
            'issues': []
        }
        
        if not results['file_exists']:
            results['issues'].append("app.py file not found!")
            return results
        
        try:
            with open('app.py', 'r') as f:
                content = f.read()
            
            # Check for required components
            for component in required_components:
                found = component in content
                results['required_found'][component] = found
                if found:
                    print(f"  ✅ {component}")
                else:
                    print(f"  ❌ {component} - MISSING")
                    results['issues'].append(f"Missing required component: {component}")
            
            # Check for optional components
            for component in optional_components:
                found = component in content
                results['optional_found'][component] = found
                if found:
                    print(f"  ✅ {component} (optional)")
                else:
                    print(f"  ⚠️  {component} (optional) - not found")
            
            # Check for common patterns
            patterns = {
                'lambda_handler function': 'def lambda_handler(',
                'Class definitions': 'class ',
                'Import statements': 'import ',
                'Exception handling': 'try:',
                'Environment variables': 'os.environ'
            }
            
            for pattern_name, pattern in patterns.items():
                if pattern in content:
                    print(f"  ✅ {pattern_name} found")
                else:
                    print(f"  ⚠️  {pattern_name} not found")
            
        except Exception as e:
            results['issues'].append(f"Error reading app.py: {str(e)}")
        
        return results
    
    def test_import_compatibility(self) -> Dict[str, Any]:
        """Test if app.py can be imported without errors"""
        print("\n🔍 Testing app.py import compatibility...")
        
        results = {
            'can_import': False,
            'import_error': None,
            'available_functions': [],
            'available_classes': []
        }
        
        try:
            # Try to import app.py as a module
            spec = importlib.util.spec_from_file_location("app", "app.py")
            if spec and spec.loader:
                app_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(app_module)
                
                results['can_import'] = True
                print("  ✅ app.py imported successfully")
                
                # List available functions and classes
                for attr_name in dir(app_module):
                    if not attr_name.startswith('_'):
                        attr = getattr(app_module, attr_name)
                        if callable(attr):
                            if hasattr(attr, '__self__'):  # Method
                                continue
                            elif str(type(attr)) == "<class 'type'>":  # Class
                                results['available_classes'].append(attr_name)
                                print(f"  📦 Class: {attr_name}")
                            else:  # Function
                                results['available_functions'].append(attr_name)
                                print(f"  🔧 Function: {attr_name}")
                
            else:
                results['import_error'] = "Could not create module spec"
                print("  ❌ Failed to create module spec")
                
        except Exception as e:
            results['can_import'] = False
            results['import_error'] = str(e)
            print(f"  ❌ Import failed: {str(e)}")
            
        return results
    
    def test_lambda_handler(self) -> Dict[str, Any]:
        """Test the lambda_handler function specifically"""
        print("\n🔍 Testing lambda_handler function...")
        
        results = {
            'handler_exists': False,
            'handler_callable': False,
            'test_execution': False,
            'execution_error': None
        }
        
        try:
            # Import and test lambda_handler
            spec = importlib.util.spec_from_file_location("app", "app.py")
            if spec and spec.loader:
                app_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(app_module)
                
                if hasattr(app_module, 'lambda_handler'):
                    results['handler_exists'] = True
                    print("  ✅ lambda_handler function found")
                    
                    handler = getattr(app_module, 'lambda_handler')
                    if callable(handler):
                        results['handler_callable'] = True
                        print("  ✅ lambda_handler is callable")
                        
                        # Test with simple event
                        try:
                            from unittest.mock import Mock
                            test_event = {"test_mode": True, "anomaly": "benign"}
                            test_context = Mock()
                            test_context.aws_request_id = "debug-test"
                            test_context.function_name = "debug-test"
                            test_context.function_version = "$LATEST"
                            test_context.memory_limit_in_mb = 128
                            
                            response = handler(test_event, test_context)
                            
                            if isinstance(response, dict) and 'statusCode' in response:
                                results['test_execution'] = True
                                print("  ✅ lambda_handler test execution successful")
                                print(f"  📊 Response: {response.get('statusCode', 'No status')}")
                            else:
                                results['execution_error'] = "Invalid response format"
                                print(f"  ⚠️  lambda_handler returned unexpected format: {type(response)}")
                                
                        except Exception as e:
                            results['execution_error'] = str(e)
                            print(f"  ❌ lambda_handler execution failed: {str(e)}")
                    else:
                        print("  ❌ lambda_handler is not callable")
                else:
                    print("  ❌ lambda_handler function not found")
                    
        except Exception as e:
            results['execution_error'] = str(e)
            print(f"  ❌ Error during lambda_handler test: {str(e)}")
            
        return results
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check for required and optional dependencies"""
        print("\n🔍 Checking dependencies...")
        
        required_deps = [
            'json', 'time', 'random', 'hashlib', 'uuid', 'os', 'sys'
        ]
        
        optional_deps = [
            ('boto3', 'AWS SDK for Python'),
            ('networkx', 'Graph processing library'),
            ('tenacity', 'Retry library'),
            ('psutil', 'System monitoring')
        ]
        
        results = {
            'required_available': {},
            'optional_available': {},
            'missing_required': [],
            'missing_optional': []
        }
        
        # Check required
        for dep in required_deps:
            try:
                __import__(dep)
                results['required_available'][dep] = True
                print(f"  ✅ {dep} (required)")
            except ImportError:
                results['required_available'][dep] = False
                results['missing_required'].append(dep)
                print(f"  ❌ {dep} (required) - MISSING")
        
        # Check optional
        for dep, description in optional_deps:
            try:
                __import__(dep)
                results['optional_available'][dep] = True
                print(f"  ✅ {dep} (optional) - {description}")
            except ImportError:
                results['optional_available'][dep] = False
                results['missing_optional'].append(dep)
                print(f"  ⚠️  {dep} (optional) - {description} - Install with: pip install {dep}")
        
        return results
    
    def analyze_test_files(self) -> Dict[str, Any]:
        """Analyze the test files for common issues"""
        print("\n🔍 Analyzing test files...")
        
        test_files = [
            'test_all.py',
            'run_tests.py', 
            'quick_test.py',
            'validate_deployment.py'
        ]
        
        results = {
            'files_found': {},
            'syntax_valid': {},
            'issues': []
        }
        
        for test_file in test_files:
            exists = os.path.exists(test_file)
            results['files_found'][test_file] = exists
            
            if exists:
                print(f"  ✅ {test_file} found")
                
                # Quick syntax check
                try:
                    with open(test_file, 'r') as f:
                        content = f.read()
                    compile(content, test_file, 'exec')
                    results['syntax_valid'][test_file] = True
                    print(f"    ✅ Syntax valid")
                except SyntaxError as e:
                    results['syntax_valid'][test_file] = False
                    results['issues'].append(f"{test_file}: Syntax error at line {e.lineno}")
                    print(f"    ❌ Syntax error at line {e.lineno}")
                except Exception as e:
                    results['syntax_valid'][test_file] = False
                    results['issues'].append(f"{test_file}: {str(e)}")
                    print(f"    ❌ Error: {str(e)}")
            else:
                print(f"  ⚠️  {test_file} not found")
        
        return results
    
    def run_quick_diagnostic(self) -> Dict[str, Any]:
        """Run a quick diagnostic of the entire SCAFAD setup"""
        print("🏥 SCAFAD Layer 0 - Quick Diagnostic")
        print("=" * 50)
        
        diagnostic_results = {}
        
        # Run all checks
        diagnostic_results['app_structure'] = self.check_app_py_structure()
        diagnostic_results['import_test'] = self.test_import_compatibility()
        diagnostic_results['lambda_test'] = self.test_lambda_handler()
        diagnostic_results['dependencies'] = self.check_dependencies()
        diagnostic_results['test_files'] = self.analyze_test_files()
        
        # Summary
        print(f"\n📋 Diagnostic Summary")
        print("-" * 30)
        
        total_issues = 0
        total_warnings = 0
        
        # Count issues
        for check_name, check_result in diagnostic_results.items():
            if isinstance(check_result, dict):
                issues = check_result.get('issues', [])
                total_issues += len(issues)
        
        if diagnostic_results['app_structure']['file_exists']:
            required_found = diagnostic_results['app_structure']['required_found']
            missing_required = sum(1 for found in required_found.values() if not found)
            total_issues += missing_required
        
        if not diagnostic_results['import_test']['can_import']:
            total_issues += 1
        
        if not diagnostic_results['lambda_test']['test_execution']:
            total_issues += 1
        
        missing_deps = len(diagnostic_results['dependencies']['missing_required'])
        total_issues += missing_deps
        
        missing_optional = len(diagnostic_results['dependencies']['missing_optional'])
        total_warnings += missing_optional
        
        print(f"Issues found: {total_issues}")
        print(f"Warnings: {total_warnings}")
        
        if total_issues == 0:
            print("🎉 All critical checks passed!")
            print("🚀 Your SCAFAD Layer 0 appears to be working correctly")
        else:
            print("🔧 Issues need to be addressed before deployment")
        
        return diagnostic_results

def main():
    """Run the SCAFAD debugger"""
    debugger = SCAFADDebugger()
    results = debugger.run_quick_diagnostic()
    
    # Save results for later analysis
    with open('debug_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Debug results saved to debug_results.json")
    
    return len(results.get('issues', [])) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)