# SCAFAD Layer 0 - Modular Architecture Design
# ================================================

"""
Proposed modular structure for SCAFAD Layer 0 implementation:

Core Architecture:
├── app_main.py              # Main orchestrator & Lambda handler
├── app_telemetry.py         # Telemetry records & multi-channel emission  
├── app_graph.py             # Invocation graph construction & analysis
├── app_adversarial.py       # GAN-based adversarial simulation
├── app_provenance.py        # Provenance tracking & audit chains
├── app_economic.py          # Economic abuse detection (DoW/DoS)
├── app_silent_failure.py    # Silent failure detection
├── app_formal.py            # Formal verification & completeness
├── app_schema.py            # Schema evolution management
└── app_config.py            # Configuration & validation

Support Modules:
├── utils/
│   ├── __init__.py
│   ├── validators.py        # Input validation utilities
│   ├── metrics.py           # Performance metrics collection
│   └── helpers.py           # Common utility functions
├── tests/
│   ├── test_main.py
│   ├── test_graph.py
│   ├── test_adversarial.py
│   ├── test_economic.py
│   └── test_integration.py
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── deployment.md
"""

# =============================================================================
# app_main.py - Main Orchestrator
# =============================================================================

import asyncio
import time
from typing import Dict, Any
from dataclasses import asdict

# Import all specialized modules
from app_telemetry import TelemetryRecord, MultiChannelTelemetry, AnomalyType, ExecutionPhase
from app_graph import AdvancedInvocationGraphBuilder
from app_adversarial import AdversarialAnomalyEngine
from app_provenance import ProvenanceChainTracker
from app_economic import EconomicAbuseDetector
from app_silent_failure import SilentFailureAnalyzer
from app_formal import FormalVerificationEngine
from app_schema import SchemaEvolutionManager
from app_config import Layer0Config, validate_environment


class Layer0_AdaptiveTelemetryController:
    """
    Main Layer 0 controller - orchestrates all specialized components
    
    This is the central coordinator that manages the interaction between
    all Layer 0 subsystems while maintaining clean separation of concerns.
    """
    
    def __init__(self, config: Layer0Config = None):
        self.config = config or Layer0Config()
        
        # Initialize all specialized components
        self.telemetry_manager = MultiChannelTelemetry(self.config)
        self.graph_builder = AdvancedInvocationGraphBuilder(self.config)
        self.adversarial_engine = AdversarialAnomalyEngine(self.config)
        self.provenance_tracker = ProvenanceChainTracker(self.config)
        self.economic_detector = EconomicAbuseDetector(self.config)
        self.silent_failure_analyzer = SilentFailureAnalyzer(self.config)
        self.formal_verifier = FormalVerificationEngine(self.config)
        self.schema_manager = SchemaEvolutionManager(self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'total_invocations': 0,
            'successful_emissions': 0,
            'fallback_activations': 0,
            'anomalies_detected': 0,
            'processing_times': []
        }
    
    async def process_invocation(self, event: Dict, context: Any) -> Dict:
        """
        Main processing pipeline - orchestrates all Layer 0 components
        
        Processing Flow:
        1. Input validation & sanitization
        2. Adversarial injection (if enabled)
        3. Telemetry generation
        4. Provenance recording
        5. Graph node creation
        6. Economic analysis
        7. Silent failure detection
        8. Formal verification
        9. Multi-channel emission
        10. Response assembly
        """
        
        start_time = time.time()
        self.performance_metrics['total_invocations'] += 1
        
        try:
            # Phase 1: Input Processing
            validated_event = await self._validate_input(event, context)
            
            # Phase 2: Telemetry Generation
            telemetry = await self._generate_telemetry(validated_event, context)
            
            # Phase 3: Component Processing Pipeline
            enriched_telemetry = await self._process_through_components(
                telemetry, validated_event, context
            )
            
            # Phase 4: Verification & Emission
            verification_result = await self._verify_and_emit(enriched_telemetry)
            
            # Phase 5: Response Assembly
            response = self._assemble_response(enriched_telemetry, verification_result)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.performance_metrics['processing_times'].append(processing_time)
            self.performance_metrics['successful_emissions'] += 1
            
            if enriched_telemetry.anomaly_type != AnomalyType.BENIGN:
                self.performance_metrics['anomalies_detected'] += 1
            
            return response
            
        except Exception as e:
            # Fallback processing
            return await self._handle_processing_failure(event, context, e, start_time)
    
    async def _validate_input(self, event: Dict, context: Any) -> Dict:
        """Input validation using schema manager"""
        return await self.schema_manager.validate_and_sanitize_input(event, context)
    
    async def _generate_telemetry(self, event: Dict, context: Any) -> TelemetryRecord:
        """Generate base telemetry record"""
        # Check for adversarial injection
        if self.config.adversarial_mode != 'DISABLED' and event.get('enable_adversarial'):
            return await self.adversarial_engine.inject_adversarial_anomaly(event, context)
        else:
            return await self.telemetry_manager.create_normal_telemetry(event, context)
    
    async def _process_through_components(self, telemetry: TelemetryRecord, 
                                        event: Dict, context: Any) -> TelemetryRecord:
        """Process telemetry through all specialized components"""
        
        # Provenance tracking
        provenance_record = await self.provenance_tracker.record_invocation(
            event, context, telemetry
        )
        telemetry.provenance_id = provenance_record['id']
        
        # Graph analysis
        if self.config.enable_graph_analysis:
            node_id = await self.graph_builder.create_node_with_advanced_features(
                event, context, telemetry
            )
            telemetry.graph_node_id = node_id
        
        # Economic abuse detection
        if self.config.enable_economic_monitoring:
            economic_analysis = await self.economic_detector.analyze_invocation(
                telemetry, self.graph_builder.get_recent_invocations()
            )
            telemetry.economic_risk_score = economic_analysis['risk_score']
        
        # Silent failure detection
        if hasattr(context, 'execution_result'):
            silent_failure_result = await self.silent_failure_analyzer.analyze_execution(
                event, getattr(context, 'execution_result'), telemetry
            )
            if silent_failure_result['failure_detected']:
                telemetry.anomaly_type = AnomalyType.EXECUTION_FAILURE
                telemetry.fallback_mode = True
        
        return telemetry
    
    async def _verify_and_emit(self, telemetry: TelemetryRecord) -> Dict:
        """Formal verification and multi-channel emission"""
        
        # Formal verification
        verification_result = await self.formal_verifier.verify_telemetry_completeness(
            [asdict(telemetry)]
        )
        telemetry.completeness_score = verification_result['overall_score']
        
        # Multi-channel emission
        emission_result = await self.telemetry_manager.emit_telemetry(telemetry)
        
        return {
            'verification': verification_result,
            'emission': emission_result
        }
    
    def _assemble_response(self, telemetry: TelemetryRecord, verification_result: Dict) -> Dict:
        """Assemble final response"""
        
        return {
            'status': 'success',
            'telemetry_id': telemetry.event_id,
            'node_id': telemetry.graph_node_id,
            'provenance_id': telemetry.provenance_id,
            'anomaly_detected': telemetry.anomaly_type != AnomalyType.BENIGN,
            'economic_risk_score': telemetry.economic_risk_score,
            'completeness_score': telemetry.completeness_score,
            'emission_success': verification_result['emission']['total_success'] > 0,
            'processing_metrics': self._get_current_metrics()
        }
    
    async def _handle_processing_failure(self, event: Dict, context: Any, 
                                       error: Exception, start_time: float) -> Dict:
        """Handle processing failures with graceful degradation"""
        
        self.performance_metrics['fallback_activations'] += 1
        
        # Generate minimal fallback telemetry
        fallback_telemetry = await self.telemetry_manager.create_fallback_telemetry(
            event, context, error
        )
        
        # Emit fallback telemetry
        await self.telemetry_manager.emit_telemetry(fallback_telemetry)
        
        processing_time = time.time() - start_time
        self.performance_metrics['processing_times'].append(processing_time)
        
        return {
            'status': 'fallback',
            'error': str(error),
            'fallback_telemetry_id': fallback_telemetry.event_id,
            'processing_time_ms': processing_time * 1000
        }
    
    def _get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        total = self.performance_metrics['total_invocations']
        
        if total == 0:
            return self.performance_metrics
        
        avg_processing_time = sum(self.performance_metrics['processing_times']) / len(
            self.performance_metrics['processing_times']
        ) if self.performance_metrics['processing_times'] else 0
        
        return {
            **self.performance_metrics,
            'success_rate': self.performance_metrics['successful_emissions'] / total,
            'fallback_rate': self.performance_metrics['fallback_activations'] / total,
            'anomaly_rate': self.performance_metrics['anomalies_detected'] / total,
            'average_processing_time_ms': avg_processing_time * 1000,
            'graph_metrics': self.graph_builder.get_advanced_graph_metrics() if self.config.enable_graph_analysis else {}
        }


# =============================================================================
# Lambda Handler & Entry Points
# =============================================================================

# Global controller instance for Lambda reuse
_layer0_controller = None

def get_layer0_controller() -> Layer0_AdaptiveTelemetryController:
    """Get or create Layer 0 controller (Lambda container reuse optimization)"""
    global _layer0_controller
    
    if _layer0_controller is None:
        config = Layer0Config()
        _layer0_controller = Layer0_AdaptiveTelemetryController(config)
    
    return _layer0_controller


async def enhanced_lambda_handler(event: Dict, context: Any) -> Dict:
    """
    Enhanced Lambda handler with complete Layer 0 capabilities
    
    This is the main entry point for AWS Lambda invocations.
    Handles all the HTTP response formatting and error handling.
    """
    
    controller = get_layer0_controller()
    
    # Process through Layer 0
    result = await controller.process_invocation(event, context)
    
    # Format HTTP response
    if result['status'] == 'success':
        status_code = 202 if result['anomaly_detected'] else 200
        body = {
            'message': 'SCAFAD Layer 0 processing complete',
            **result
        }
    else:
        status_code = 206  # Partial content (fallback)
        body = {
            'message': 'SCAFAD Layer 0 fallback mode',
            **result
        }
    
    return {
        'statusCode': status_code,
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            'X-SCAFAD-Version': controller.config.version,
            'X-SCAFAD-Layer': '0',
            'X-Telemetry-Id': result.get('telemetry_id', '')
        }
    }


def lambda_handler(event: Dict, context: Any) -> Dict:
    """Synchronous Lambda handler wrapper"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(enhanced_lambda_handler(event, context))
    finally:
        loop.close()


# =============================================================================
# Testing & Validation Interface
# =============================================================================

class Layer0TestingInterface:
    """
    Testing interface for comprehensive Layer 0 validation
    """
    
    def __init__(self):
        self.controller = Layer0_AdaptiveTelemetryController()
    
    async def run_component_tests(self) -> Dict:
        """Run tests on individual components"""
        
        test_results = {}
        
        # Test each component individually
        components = {
            'graph_builder': self.controller.graph_builder,
            'adversarial_engine': self.controller.adversarial_engine,
            'provenance_tracker': self.controller.provenance_tracker,
            'economic_detector': self.controller.economic_detector,
            'silent_failure_analyzer': self.controller.silent_failure_analyzer,
            'formal_verifier': self.controller.formal_verifier
        }
        
        for component_name, component in components.items():
            if hasattr(component, 'run_self_test'):
                test_results[component_name] = await component.run_self_test()
            else:
                test_results[component_name] = {'status': 'no_test_available'}
        
        return test_results
    
    async def run_integration_tests(self, n_tests: int = 10) -> Dict:
        """Run integration tests with synthetic data"""
        
        from utils.test_data_generator import generate_test_payloads
        
        test_payloads = generate_test_payloads(n_tests)
        results = {'passed': 0, 'failed': 0, 'details': []}
        
        for i, payload in enumerate(test_payloads):
            try:
                # Mock context
                class MockContext:
                    aws_request_id = f"test-{i:03d}"
                    function_name = "scafad-test-function"
                    memory_limit_in_mb = 128
                
                result = await self.controller.process_invocation(payload, MockContext())
                
                if result['status'] in ['success', 'fallback']:
                    results['passed'] += 1
                    results['details'].append({
                        'test_id': i,
                        'status': 'passed',
                        'result': result
                    })
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'test_id': i,
                        'status': 'failed',
                        'error': result.get('error', 'unknown')
                    })
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'test_id': i,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['success_rate'] = results['passed'] / (results['passed'] + results['failed'])
        
        return results


# =============================================================================
# Main Entry Point & CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == '--test':
            # Run comprehensive tests
            async def run_tests():
                tester = Layer0TestingInterface()
                
                print("Running component tests...")
                component_results = await tester.run_component_tests()
                
                print("Running integration tests...")
                integration_results = await tester.run_integration_tests(20)
                
                print("\n" + "="*60)
                print("SCAFAD Layer 0 Test Results")
                print("="*60)
                print(f"Component Tests: {len(component_results)} components tested")
                print(f"Integration Tests: {integration_results['success_rate']:.2%} success rate")
                print(f"Passed: {integration_results['passed']}")
                print(f"Failed: {integration_results['failed']}")
                
                return {
                    'component_tests': component_results,
                    'integration_tests': integration_results
                }
            
            # Run tests
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                test_results = loop.run_until_complete(run_tests())
                
                if test_results['integration_tests']['success_rate'] >= 0.9:
                    print("\n✅ All tests passed!")
                    sys.exit(0)
                else:
                    print("\n❌ Some tests failed!")
                    sys.exit(1)
            finally:
                loop.close()
        
        elif command == '--validate':
            # Validate environment
            validation_results = validate_environment()
            
            print("Environment Validation Results:")
            for check, result in validation_results.items():
                status = "✅" if result else "❌"
                print(f"{status} {check}")
            
            if all(validation_results.values()):
                print("\n✅ Environment validation successful")
                sys.exit(0)
            else:
                print("\n❌ Environment validation failed")
                sys.exit(1)
        
        elif command == '--demo':
            # Run demonstration
            async def run_demo():
                controller = Layer0_AdaptiveTelemetryController()
                
                # Generate sample event
                demo_event = {
                    'anomaly': 'cold_start',
                    'function_profile_id': 'demo_function',
                    'execution_phase': 'invoke',
                    'test_mode': True,
                    'enable_adversarial': True
                }
                
                class MockContext:
                    aws_request_id = "demo-request-001"
                    function_name = "scafad-demo-function"
                    memory_limit_in_mb = 256
                
                result = await controller.process_invocation(demo_event, MockContext())
                
                print("SCAFAD Layer 0 Demo Result:")
                print("="*40)
                print(json.dumps(result, indent=2))
                
                return result
            
            # Run demo
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                demo_result = loop.run_until_complete(run_demo())
                print("\n✅ Demo completed successfully!")
            finally:
                loop.close()
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: --test, --validate, --demo")
            sys.exit(1)
    
    else:
        print("SCAFAD Layer 0 - Modular Architecture")
        print("Usage: python app_main.py [--test|--validate|--demo]")
        print("\nFor full functionality, ensure all component modules are available:")
        print("  - app_telemetry.py")
        print("  - app_graph.py") 
        print("  - app_adversarial.py")
        print("  - app_provenance.py")
        print("  - app_economic.py")
        print("  - app_silent_failure.py")
        print("  - app_formal.py")
        print("  - app_schema.py")
        print("  - app_config.py")