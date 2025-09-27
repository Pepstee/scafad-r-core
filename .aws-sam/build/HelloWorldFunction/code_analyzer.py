#!/usr/bin/env python3
"""
SCAFAD Layer 0 - Complete Workflow Analyzer and Logical Error Checker
Validates the entire Layer 0 pipeline from input to Layer 1 handoff
"""

import json
import time
from typing import Dict, List, Any, Tuple
import traceback
from datetime import datetime

class SCAFADWorkflowAnalyzer:
    """Analyzes the complete SCAFAD Layer 0 workflow for logical errors"""
    
    def __init__(self):
        self.workflow_issues = []
        self.performance_metrics = {}
        self.data_integrity_issues = []
        
    def analyze_input_validation(self, event: Dict, context: Any) -> Dict[str, Any]:
        """Analyze Layer 0 input validation logic"""
        print("🔍 Phase 1: Input Validation Analysis")
        print("-" * 50)
        
        issues = []
        
        # Test 1: Event structure validation
        if not isinstance(event, dict):
            issues.append("CRITICAL: Event must be a dictionary")
        
        # Test 2: Required fields for SCAFAD processing
        required_fields = {
            'anomaly': 'string - type of anomaly to simulate',
            'execution_phase': 'string - phase of serverless execution'
        }
        
        recommended_fields = {
            'function_profile_id': 'string - function identifier',
            'test_mode': 'boolean - enables test mode',
            'layer0_enabled': 'boolean - Layer 0 processing flag'
        }
        
        for field, description in required_fields.items():
            if field not in event:
                issues.append(f"MISSING REQUIRED: {field} ({description})")
        
        for field, description in recommended_fields.items():
            if field not in event:
                issues.append(f"MISSING RECOMMENDED: {field} ({description})")
        
        # Test 3: Context validation
        context_attrs = ['aws_request_id', 'function_name', 'memory_limit_in_mb']
        for attr in context_attrs:
            if not hasattr(context, attr):
                issues.append(f"CONTEXT MISSING: {attr}")
        
        print(f"  ✅ Event type: {type(event)}")
        print(f"  📊 Event fields: {len(event)}")
        print(f"  🆔 Context attributes: {len([a for a in context_attrs if hasattr(context, a)])}/3")
        
        if issues:
            print("  ⚠️  Input validation issues found:")
            for issue in issues:
                print(f"    • {issue}")
        else:
            print("  ✅ Input validation passed")
        
        return {
            'phase': 'input_validation',
            'passed': len(issues) == 0,
            'issues': issues,
            'event_fields': list(event.keys()) if isinstance(event, dict) else [],
            'context_valid': all(hasattr(context, attr) for attr in context_attrs)
        }
    
    def analyze_controller_initialization(self) -> Dict[str, Any]:
        """Analyze Layer0_AdaptiveTelemetryController initialization logic"""
        print("\n🔍 Phase 2: Controller Initialization Analysis")
        print("-" * 50)
        
        issues = []
        components_status = {}
        
        try:
            from app import Layer0_AdaptiveTelemetryController
            controller = Layer0_AdaptiveTelemetryController()
            
            # Check all required components
            required_components = {
                'graph_builder': 'InvocationGraphBuilder',
                'adversarial_simulator': 'AdversarialAnomalySimulator', 
                'provenance_tracker': 'ProvenanceChain',
                'schema_registry': 'SchemaEvolutionManager',
                'formal_verifier': 'TelemetryCompletenessVerifier',
                'economic_monitor': 'EconomicAbuseMonitor',
                'silent_failure_detector': 'SilentFailureDetector',
                'telemetry_channels': 'MultiChannelTelemetry'
            }
            
            for component_attr, component_class in required_components.items():
                if hasattr(controller, component_attr):
                    component = getattr(controller, component_attr)
                    if component is not None:
                        components_status[component_attr] = 'initialized'
                        print(f"  ✅ {component_class}")
                    else:
                        components_status[component_attr] = 'null'
                        issues.append(f"Component {component_attr} is None")
                        print(f"  ❌ {component_class} (None)")
                else:
                    components_status[component_attr] = 'missing'
                    issues.append(f"Missing component: {component_attr}")
                    print(f"  ❌ {component_class} (Missing)")
            
            # Check state variables
            state_vars = ['active_invocations', 'performance_metrics', 'schema_version']
            for var in state_vars:
                if hasattr(controller, var):
                    print(f"  ✅ State variable: {var}")
                else:
                    issues.append(f"Missing state variable: {var}")
                    print(f"  ❌ State variable: {var} (Missing)")
            
        except ImportError as e:
            issues.append(f"Cannot import Layer0_AdaptiveTelemetryController: {e}")
            print(f"  ❌ Import failed: {e}")
        except Exception as e:
            issues.append(f"Controller initialization failed: {e}")
            print(f"  ❌ Initialization failed: {e}")
        
        return {
            'phase': 'controller_initialization',
            'passed': len(issues) == 0,
            'issues': issues,
            'components_status': components_status,
            'total_components': len(required_components),
            'initialized_components': len([s for s in components_status.values() if s == 'initialized'])
        }
    
    def analyze_telemetry_processing(self, event: Dict, context: Any) -> Dict[str, Any]:
        """Analyze the core telemetry processing logic"""
        print("\n🔍 Phase 3: Telemetry Processing Analysis")
        print("-" * 50)
        
        issues = []
        processing_steps = {}
        
        try:
            from app import Layer0_AdaptiveTelemetryController, TelemetryRecord
            controller = Layer0_AdaptiveTelemetryController()
            
            # Step 1: Test telemetry record creation
            try:
                record = TelemetryRecord.create(
                    invocation_id=context.aws_request_id,
                    function_name=getattr(context, 'function_name', 'test'),
                    anomaly_type=event.get('anomaly', 'benign'),
                    execution_phase=event.get('execution_phase', 'invoke')
                )
                processing_steps['telemetry_creation'] = 'success'
                print("  ✅ TelemetryRecord creation")
                
                # Validate record structure
                required_fields = ['telemetry_id', 'timestamp', 'invocation_id', 'anomaly_type']
                for field in required_fields:
                    if not hasattr(record, field):
                        issues.append(f"TelemetryRecord missing field: {field}")
                        print(f"    ❌ Missing field: {field}")
                    else:
                        print(f"    ✅ Field: {field}")
                
            except Exception as e:
                processing_steps['telemetry_creation'] = 'failed'
                issues.append(f"TelemetryRecord creation failed: {e}")
                print(f"  ❌ TelemetryRecord creation failed: {e}")
            
            # Step 2: Test graph construction
            try:
                if hasattr(controller, 'graph_builder') and controller.graph_builder:
                    # Simulate graph building
                    node_id = controller.graph_builder.add_invocation_node(
                        invocation_id=context.aws_request_id,
                        function_name=getattr(context, 'function_name', 'test'),
                        timestamp=time.time(),
                        anomaly_type=event.get('anomaly', 'benign')
                    )
                    processing_steps['graph_construction'] = 'success'
                    print("  ✅ Graph node construction")
                else:
                    processing_steps['graph_construction'] = 'skipped'
                    issues.append("Graph builder not available")
                    print("  ⚠️  Graph builder not available")
            except Exception as e:
                processing_steps['graph_construction'] = 'failed'
                issues.append(f"Graph construction failed: {e}")
                print(f"  ❌ Graph construction failed: {e}")
            
            # Step 3: Test anomaly simulation
            try:
                if hasattr(controller, 'adversarial_simulator') and controller.adversarial_simulator:
                    anomaly_result = controller.adversarial_simulator.simulate_anomaly(
                        anomaly_type=event.get('anomaly', 'benign'),
                        execution_phase=event.get('execution_phase', 'invoke'),
                        intensity=0.5
                    )
                    processing_steps['anomaly_simulation'] = 'success'
                    print("  ✅ Anomaly simulation")
                else:
                    processing_steps['anomaly_simulation'] = 'skipped'
                    issues.append("Adversarial simulator not available")
                    print("  ⚠️  Adversarial simulator not available")
            except Exception as e:
                processing_steps['anomaly_simulation'] = 'failed'
                issues.append(f"Anomaly simulation failed: {e}")
                print(f"  ❌ Anomaly simulation failed: {e}")
            
            # Step 4: Test provenance tracking
            try:
                if hasattr(controller, 'provenance_tracker') and controller.provenance_tracker:
                    provenance_id = controller.provenance_tracker.create_provenance_entry(
                        invocation_id=context.aws_request_id,
                        parent_chain=None,
                        operation_type='anomaly_detection',
                        metadata={'test': True}
                    )
                    processing_steps['provenance_tracking'] = 'success'
                    print("  ✅ Provenance tracking")
                else:
                    processing_steps['provenance_tracking'] = 'skipped'
                    issues.append("Provenance tracker not available")
                    print("  ⚠️  Provenance tracker not available")
            except Exception as e:
                processing_steps['provenance_tracking'] = 'failed'
                issues.append(f"Provenance tracking failed: {e}")
                print(f"  ❌ Provenance tracking failed: {e}")
            
        except Exception as e:
            issues.append(f"Overall processing analysis failed: {e}")
            print(f"  ❌ Processing analysis failed: {e}")
        
        return {
            'phase': 'telemetry_processing',
            'passed': len(issues) == 0,
            'issues': issues,
            'processing_steps': processing_steps,
            'successful_steps': len([s for s in processing_steps.values() if s == 'success']),
            'total_steps': len(processing_steps)
        }
    
    def analyze_output_generation(self, event: Dict, context: Any) -> Dict[str, Any]:
        """Analyze the output generation and Layer 1 handoff logic"""
        print("\n🔍 Phase 4: Output Generation & Layer 1 Handoff Analysis")
        print("-" * 50)
        
        issues = []
        output_analysis = {}
        
        try:
            from app import lambda_handler
            
            # Test actual lambda_handler execution
            start_time = time.time()
            response = lambda_handler(event, context)
            execution_time = (time.time() - start_time) * 1000
            
            output_analysis['execution_time_ms'] = execution_time
            print(f"  ⏱️  Execution time: {execution_time:.2f}ms")
            
            # Validate response structure
            if not isinstance(response, dict):
                issues.append("Response is not a dictionary")
                print("  ❌ Response is not a dictionary")
                return {
                    'phase': 'output_generation',
                    'passed': False,
                    'issues': issues,
                    'output_analysis': output_analysis
                }
            
            # Check required response fields
            required_fields = ['statusCode', 'body', 'headers']
            for field in required_fields:
                if field in response:
                    print(f"  ✅ Response field: {field}")
                else:
                    issues.append(f"Missing response field: {field}")
                    print(f"  ❌ Missing response field: {field}")
            
            # Validate status code
            status_code = response.get('statusCode')
            if status_code in [200, 202, 206]:
                print(f"  ✅ Valid status code: {status_code}")
                output_analysis['status_code'] = status_code
            else:
                issues.append(f"Invalid status code: {status_code}")
                print(f"  ❌ Invalid status code: {status_code}")
            
            # Parse and validate body
            try:
                body = response.get('body', '{}')
                if isinstance(body, str):
                    body_data = json.loads(body)
                else:
                    body_data = body
                
                # Check for Layer 1 handoff data
                layer1_fields = [
                    'telemetry_id', 'node_id', 'provenance_id', 
                    'anomaly_detected', 'processing_time_ms'
                ]
                
                found_fields = []
                missing_fields = []
                
                for field in layer1_fields:
                    if field in body_data:
                        found_fields.append(field)
                        print(f"  ✅ Layer 1 field: {field}")
                    else:
                        missing_fields.append(field)
                        print(f"  ⚠️  Layer 1 field missing: {field}")
                
                output_analysis['layer1_fields_found'] = len(found_fields)
                output_analysis['layer1_fields_total'] = len(layer1_fields)
                output_analysis['body_data'] = body_data
                
                if len(found_fields) >= 3:  # At least 3 critical fields
                    print("  ✅ Sufficient Layer 1 handoff data")
                else:
                    issues.append("Insufficient Layer 1 handoff data")
                    print("  ⚠️  Insufficient Layer 1 handoff data")
                
            except json.JSONDecodeError as e:
                issues.append(f"Response body is not valid JSON: {e}")
                print(f"  ❌ Invalid JSON body: {e}")
            
            # Check headers for SCAFAD metadata
            headers = response.get('headers', {})
            scafad_headers = [h for h in headers.keys() if h.startswith('X-SCAFAD')]
            if scafad_headers:
                print(f"  ✅ SCAFAD headers found: {len(scafad_headers)}")
                output_analysis['scafad_headers'] = len(scafad_headers)
            else:
                print("  ⚠️  No SCAFAD headers found")
                output_analysis['scafad_headers'] = 0
            
        except Exception as e:
            issues.append(f"Output generation test failed: {e}")
            print(f"  ❌ Output generation failed: {e}")
            traceback.print_exc()
        
        return {
            'phase': 'output_generation',
            'passed': len(issues) == 0,
            'issues': issues,
            'output_analysis': output_analysis
        }
    
    def analyze_data_flow_integrity(self, event: Dict, context: Any) -> Dict[str, Any]:
        """Analyze data flow integrity from input to output"""
        print("\n🔍 Phase 5: Data Flow Integrity Analysis")
        print("-" * 50)
        
        issues = []
        flow_analysis = {}
        
        try:
            from app import lambda_handler
            
            # Test with different anomaly types
            test_anomalies = ['benign', 'cpu_burst', 'cold_start', 'memory_leak']
            flow_results = {}
            
            for anomaly in test_anomalies:
                test_event = {**event, 'anomaly': anomaly, 'test_mode': True}
                
                try:
                    response = lambda_handler(test_event, context)
                    body_data = json.loads(response.get('body', '{}'))
                    
                    # Check if anomaly type is preserved in response
                    if 'anomaly_detected' in body_data:
                        flow_results[anomaly] = 'preserved'
                        print(f"  ✅ {anomaly}: Data flow preserved")
                    else:
                        flow_results[anomaly] = 'lost'
                        issues.append(f"Anomaly type {anomaly}: Data not preserved in output")
                        print(f"  ⚠️  {anomaly}: Data flow incomplete")
                        
                except Exception as e:
                    flow_results[anomaly] = 'error'
                    issues.append(f"Anomaly type {anomaly}: Processing error - {e}")
                    print(f"  ❌ {anomaly}: Processing error")
            
            flow_analysis['anomaly_preservation'] = flow_results
            flow_analysis['successful_flows'] = len([r for r in flow_results.values() if r == 'preserved'])
            flow_analysis['total_flows'] = len(flow_results)
            
            # Check ID consistency
            consistent_ids = True
            try:
                response = lambda_handler(event, context)
                body_data = json.loads(response.get('body', '{}'))
                
                # Check if IDs are properly formatted
                id_fields = ['telemetry_id', 'node_id', 'provenance_id']
                for field in id_fields:
                    if field in body_data:
                        id_value = body_data[field]
                        if isinstance(id_value, str) and len(id_value) > 10:
                            print(f"  ✅ {field}: Well-formed ID")
                        else:
                            consistent_ids = False
                            issues.append(f"{field}: Malformed ID - {id_value}")
                            print(f"  ❌ {field}: Malformed ID")
            
            except Exception as e:
                consistent_ids = False
                issues.append(f"ID consistency check failed: {e}")
            
            flow_analysis['id_consistency'] = consistent_ids
            
        except Exception as e:
            issues.append(f"Data flow integrity analysis failed: {e}")
            print(f"  ❌ Data flow analysis failed: {e}")
        
        return {
            'phase': 'data_flow_integrity',
            'passed': len(issues) == 0,
            'issues': issues,
            'flow_analysis': flow_analysis
        }
    
    def analyze_error_handling(self, context: Any) -> Dict[str, Any]:
        """Analyze error handling and fallback mechanisms"""
        print("\n🔍 Phase 6: Error Handling & Fallback Analysis")
        print("-" * 50)
        
        issues = []
        error_tests = {}
        
        try:
            from app import lambda_handler
            
            # Test 1: Invalid event structure
            try:
                invalid_event = "not a dictionary"
                response = lambda_handler(invalid_event, context)
                if response.get('statusCode') in [400, 500, 206]:
                    error_tests['invalid_event'] = 'handled'
                    print("  ✅ Invalid event handled gracefully")
                else:
                    error_tests['invalid_event'] = 'unhandled'
                    issues.append("Invalid event not handled properly")
                    print("  ❌ Invalid event not handled")
            except Exception as e:
                error_tests['invalid_event'] = 'exception'
                print(f"  ⚠️  Invalid event caused exception: {e}")
            
            # Test 2: Missing required fields
            try:
                minimal_event = {}  # No required fields
                response = lambda_handler(minimal_event, context)
                if response.get('statusCode') in [200, 202, 206]:
                    error_tests['missing_fields'] = 'handled'
                    print("  ✅ Missing fields handled with defaults")
                else:
                    error_tests['missing_fields'] = 'rejected'
                    print("  ⚠️  Missing fields rejected (this may be correct)")
            except Exception as e:
                error_tests['missing_fields'] = 'exception'
                issues.append(f"Missing fields caused exception: {e}")
                print(f"  ❌ Missing fields caused exception: {e}")
            
            # Test 3: Extreme values
            try:
                extreme_event = {
                    'anomaly': 'x' * 1000,  # Very long string
                    'execution_phase': 'invalid_phase',
                    'test_mode': True
                }
                response = lambda_handler(extreme_event, context)
                error_tests['extreme_values'] = 'handled'
                print("  ✅ Extreme values handled")
            except Exception as e:
                error_tests['extreme_values'] = 'exception'
                issues.append(f"Extreme values caused exception: {e}")
                print(f"  ❌ Extreme values caused exception: {e}")
            
            # Test 4: Force starvation scenario
            try:
                starvation_event = {
                    'force_starvation': True,
                    'anomaly': 'cpu_burst',
                    'test_mode': True
                }
                response = lambda_handler(starvation_event, context)
                body_data = json.loads(response.get('body', '{}'))
                
                if 'fallback' in str(body_data).lower():
                    error_tests['starvation_fallback'] = 'activated'
                    print("  ✅ Starvation fallback activated")
                else:
                    error_tests['starvation_fallback'] = 'not_activated'
                    print("  ⚠️  Starvation fallback not clearly activated")
            
            except Exception as e:
                error_tests['starvation_fallback'] = 'exception'
                issues.append(f"Starvation test caused exception: {e}")
                print(f"  ❌ Starvation test failed: {e}")
        
        except Exception as e:
            issues.append(f"Error handling analysis failed: {e}")
            print(f"  ❌ Error handling analysis failed: {e}")
        
        return {
            'phase': 'error_handling',
            'passed': len(issues) == 0,
            'issues': issues,
            'error_tests': error_tests,
            'handled_errors': len([t for t in error_tests.values() if t in ['handled', 'activated']]),
            'total_error_tests': len(error_tests)
        }
    
    def generate_workflow_report(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive workflow analysis report"""
        print("\n📊 Workflow Analysis Report")
        print("=" * 50)
        
        total_issues = sum(len(analysis.get('issues', [])) for analysis in analyses)
        total_phases = len(analyses)
        passed_phases = sum(1 for analysis in analyses if analysis.get('passed', False))
        
        # Calculate overall health score
        health_score = (passed_phases / total_phases) * 100 if total_phases > 0 else 0
        
        print(f"Overall Health Score: {health_score:.1f}%")
        print(f"Phases Passed: {passed_phases}/{total_phases}")
        print(f"Total Issues: {total_issues}")
        
        # Phase-by-phase summary
        print(f"\n📋 Phase Summary:")
        for analysis in analyses:
            phase = analysis['phase']
            status = "✅ PASS" if analysis.get('passed', False) else "❌ FAIL"
            issue_count = len(analysis.get('issues', []))
            print(f"  {status} {phase.replace('_', ' ').title()} ({issue_count} issues)")
        
        # Critical issues
        critical_issues = []
        for analysis in analyses:
            for issue in analysis.get('issues', []):
                if any(keyword in issue.upper() for keyword in ['CRITICAL', 'MISSING REQUIRED', 'FAILED']):
                    critical_issues.append(f"{analysis['phase']}: {issue}")
        
        if critical_issues:
            print(f"\n🚨 Critical Issues:")
            for issue in critical_issues:
                print(f"  • {issue}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if health_score >= 90:
            print("  🎉 Excellent! Your SCAFAD Layer 0 is ready for production")
            print("  🚀 Next steps: Deploy and test with Layer 1 integration")
        elif health_score >= 70:
            print("  👍 Good foundation, but address the issues above")
            print("  🔧 Focus on fixing critical issues first")
        else:
            print("  ⚠️  Significant issues detected - development needed")
            print("  📖 Review implementation against SCAFAD specification")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'total_phases': total_phases,
            'passed_phases': passed_phases,
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'analyses': analyses,
            'ready_for_production': health_score >= 90 and total_issues == 0
        }

def main():
    """Main workflow analysis function"""
    print("🔍 SCAFAD Layer 0 - Complete Workflow Analysis")
    print("=" * 60)
    print("Analyzing the complete pipeline from input to Layer 1 handoff\n")
    
    # Create test event and context
    test_event = {
        'anomaly': 'cpu_burst',
        'execution_phase': 'invoke',
        'function_profile_id': 'test_function',
        'test_mode': True,
        'layer0_enabled': True,
        'payload_id': 'workflow_test_001'
    }
    
    class MockContext:
        aws_request_id = "workflow-test-12345"
        function_name = "scafad-layer0-test"
        function_version = "$LATEST"
        memory_limit_in_mb = 256
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:scafad-test"
    
    context = MockContext()
    
    # Run comprehensive analysis
    analyzer = SCAFADWorkflowAnalyzer()
    
    analyses = [
        analyzer.analyze_input_validation(test_event, context),
        analyzer.analyze_controller_initialization(),
        analyzer.analyze_telemetry_processing(test_event, context),
        analyzer.analyze_output_generation(test_event, context),
        analyzer.analyze_data_flow_integrity(test_event, context),
        analyzer.analyze_error_handling(context)
    ]
    
    # Generate final report
    report = analyzer.generate_workflow_report(analyses)
    
    # Save detailed report
    with open('scafad_workflow_analysis.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: scafad_workflow_analysis.json")
    
    return report['ready_for_production']

if __name__ == "__main__":
    ready = main()
    if ready:
        print(f"\n🎉 SCAFAD Layer 0 is ready for Layer 1 integration!")
    else:
        print(f"\n🔧 Please address the issues before proceeding to Layer 1")
    
    exit(0 if ready else 1)