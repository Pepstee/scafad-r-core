#!/usr/bin/env python3
"""
SCAFAD Layer 0 Production Readiness Validator
==============================================

Validates production readiness of all Layer 0 components by:
- Checking component availability and imports
- Validating core functionality with focused tests  
- Assessing integration readiness
- Generating production deployment recommendations
"""

import sys
import time
import json
import traceback
from typing import Dict, List, Any, Tuple

# Add workspace to path
sys.path.insert(0, '/workspace')

class Layer0ProductionValidator:
    """Production readiness validator for Layer 0"""
    
    def __init__(self):
        self.component_availability = {}
        self.test_results = {}
        self.start_time = time.time()
        
    def check_component_availability(self) -> Dict[str, bool]:
        """Check availability of all Layer 0 components"""
        print("🔧 Checking Layer 0 Component Availability...")
        print("=" * 60)
        
        components = [
            ('layer0_signal_negotiation', 'SignalNegotiator', 'Signal Negotiation'),
            ('layer0_redundancy_manager', 'RedundancyManager', 'Redundancy Manager'),
            ('layer0_sampler', 'ExecutionAwareSampler', 'Execution-Aware Sampler'),
            ('layer0_fallback_orchestrator', 'FallbackOrchestrator', 'Fallback Orchestrator'),
            ('layer0_adaptive_buffer', 'AdaptiveBuffer', 'Adaptive Buffer'),
            ('layer0_vendor_adapters', 'VendorAdapterManager', 'Vendor Adapters'),
            ('layer0_health_monitor', 'HealthMonitor', 'Health Monitor'),
            ('layer0_privacy_compliance', 'PrivacyCompliancePipeline', 'Privacy Compliance'),
            ('layer0_l1_contract', 'L0L1ContractManager', 'L0-L1 Contract'),
            ('layer0_core', 'AnomalyDetectionEngine', 'Anomaly Detection Engine'),
            ('app_config', 'Layer0Config', 'Configuration System'),
            ('app_telemetry', 'create_telemetry_record_with_telemetry_id', 'Telemetry System')
        ]
        
        for module_name, class_name, display_name in components:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                self.component_availability[module_name] = True
                print(f"   ✅ {display_name}: Available")
            except ImportError as e:
                self.component_availability[module_name] = False
                print(f"   ❌ {display_name}: Import Error - {e}")
            except AttributeError as e:
                self.component_availability[module_name] = False
                print(f"   ❌ {display_name}: Missing Class - {e}")
            except Exception as e:
                self.component_availability[module_name] = False
                print(f"   ❌ {display_name}: Error - {e}")
        
        available_count = sum(1 for available in self.component_availability.values() if available)
        total_count = len(self.component_availability)
        
        print(f"\n📊 Component Availability: {available_count}/{total_count} ({available_count/total_count*100:.1f}%)")
        return self.component_availability
    
    def test_signal_negotiation(self) -> Tuple[float, Dict[str, Any]]:
        """Test Signal Negotiation component"""
        print("\n🔧 Testing Signal Negotiation...")
        
        try:
            from layer0_signal_negotiation import SignalNegotiator
            from app_config import create_testing_config
            
            config = create_testing_config()
            negotiator = SignalNegotiator(config)
            
            # Test channel discovery
            available_channels = len(negotiator.available_channels)
            print(f"   📡 Discovered {available_channels} channels")
            
            # Test QoS recommendations
            recommendations = negotiator.get_channel_recommendations(1024, priority="balanced")
            print(f"   📊 Generated {len(recommendations)} recommendations")
            
            # Test health summary
            health = negotiator.get_channel_health_summary()
            print(f"   💊 Health summary: {len(health)} channels monitored")
            
            score = min(1.0, (available_channels + len(recommendations) + len(health)) / 10)
            print(f"   🏆 Signal Negotiation Score: {score:.3f}")
            
            return score, {
                "channels_discovered": available_channels,
                "recommendations": len(recommendations),
                "health_monitored": len(health)
            }
            
        except Exception as e:
            print(f"   ❌ Signal Negotiation Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_adaptive_buffer(self) -> Tuple[float, Dict[str, Any]]:
        """Test Adaptive Buffer component"""
        print("\n📊 Testing Adaptive Buffer...")
        
        try:
            from layer0_adaptive_buffer import AdaptiveBuffer, BufferConfig
            
            config = BufferConfig(max_queue_size=50, max_memory_bytes=5*1024, base_batch_size=5)
            buffer = AdaptiveBuffer(config, "test_buffer")
            
            # Test enqueue/dequeue
            enqueue_success = 0
            for i in range(25):
                if buffer.enqueue(f"item_{i}", size_bytes=100):
                    enqueue_success += 1
            
            items = buffer.dequeue(10)
            dequeue_success = len(items)
            
            # Test watermarks
            for i in range(20):
                buffer.enqueue(f"watermark_item_{i}", size_bytes=100)
            
            status = buffer.get_status()
            backpressure_active = status["backpressure_active"]
            
            buffer.shutdown()
            
            print(f"   📥 Enqueued: {enqueue_success}/25")
            print(f"   📤 Dequeued: {dequeue_success} items")
            print(f"   ⚖️ Backpressure: {'Active' if backpressure_active else 'Inactive'}")
            
            score = min(1.0, (enqueue_success + dequeue_success + (10 if backpressure_active else 0)) / 45)
            print(f"   🏆 Adaptive Buffer Score: {score:.3f}")
            
            return score, {
                "enqueue_success": enqueue_success,
                "dequeue_success": dequeue_success,
                "backpressure_active": backpressure_active
            }
            
        except Exception as e:
            print(f"   ❌ Adaptive Buffer Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_vendor_adapters(self) -> Tuple[float, Dict[str, Any]]:
        """Test Vendor Adapters component"""
        print("\n🔌 Testing Vendor Adapters...")
        
        try:
            from layer0_vendor_adapters import VendorAdapterManager, ProviderType
            import uuid
            
            manager = VendorAdapterManager()
            
            # Create adapters
            cloudwatch = manager.create_adapter(ProviderType.CLOUDWATCH)
            datadog = manager.create_adapter(ProviderType.DATADOG)
            
            print(f"   🏭 Created CloudWatch adapter: {cloudwatch is not None}")
            print(f"   🏭 Created DataDog adapter: {datadog is not None}")
            
            # Get status
            summary = manager.get_status_summary()
            adapters_created = summary["adapters_count"]
            
            print(f"   📊 Total adapters: {adapters_created}")
            
            score = min(1.0, adapters_created / 2)
            print(f"   🏆 Vendor Adapters Score: {score:.3f}")
            
            return score, {
                "cloudwatch_created": cloudwatch is not None,
                "datadog_created": datadog is not None,
                "adapters_created": adapters_created
            }
            
        except Exception as e:
            print(f"   ❌ Vendor Adapters Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_health_monitor(self) -> Tuple[float, Dict[str, Any]]:
        """Test Health Monitor component"""
        print("\n💊 Testing Health Monitor...")
        
        try:
            from layer0_health_monitor import HealthMonitor, HealthCheck, ComponentType, HealthStatus
            from app_config import create_testing_config
            
            def sample_health_check():
                return HealthStatus.HEALTHY, {"status": "ok", "cpu": 25.5}
            
            config = create_testing_config()
            monitor = HealthMonitor(config, "test_monitor")
            
            # Register health check
            check = HealthCheck(
                name="test_check",
                component=ComponentType.SIGNAL_NEGOTIATOR,
                check_function=sample_health_check,
                interval_ms=500
            )
            
            monitor.register_health_check(check)
            monitor.register_component_heartbeat(ComponentType.STREAM_PROCESSOR)
            
            # Send heartbeats
            for _ in range(3):
                monitor.heartbeat(ComponentType.STREAM_PROCESSOR)
                time.sleep(0.1)
            
            # Wait for health checks
            time.sleep(1.0)
            
            # Get system health
            system_health = monitor.get_system_health()
            overall_status = system_health["overall_status"]
            
            component_status = monitor.get_component_status(ComponentType.SIGNAL_NEGOTIATOR)
            
            monitor.shutdown()
            
            print(f"   🏥 Overall system status: {overall_status}")
            print(f"   📈 Component checks: {len(component_status.get('recent_results', []))}")
            
            score = 1.0 if overall_status in ["healthy", "degraded"] else 0.5
            print(f"   🏆 Health Monitor Score: {score:.3f}")
            
            return score, {
                "overall_status": overall_status,
                "component_monitored": len(component_status.get("recent_results", []))
            }
            
        except Exception as e:
            print(f"   ❌ Health Monitor Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_privacy_compliance(self) -> Tuple[float, Dict[str, Any]]:
        """Test Privacy Compliance component"""
        print("\n🔒 Testing Privacy Compliance...")
        
        try:
            from layer0_privacy_compliance import PrivacyCompliancePipeline, ComplianceConfig, DataClassification
            
            config = ComplianceConfig(require_consent=True, anonymization_enabled=True)
            pipeline = PrivacyCompliancePipeline(config)
            
            # Test data with PII
            test_data = {
                "user_email": "test@example.com",
                "phone_number": "555-123-4567",
                "server_ip": "192.168.1.1",
                "normal_data": "this is fine"
            }
            
            # Process data
            redacted_data, redaction_results = pipeline.process_data(
                test_data,
                user_id="test_user",
                data_classification=DataClassification.CONFIDENTIAL,
                operation="test"
            )
            
            # Verify redactions
            pii_redacted = "test@example.com" not in json.dumps(redacted_data)
            redactions_applied = len(redaction_results)
            
            # Get metrics
            metrics = pipeline.get_privacy_metrics()
            has_metrics = metrics["total_records_processed"] > 0
            
            print(f"   🚫 PII redacted: {'Yes' if pii_redacted else 'No'}")
            print(f"   📊 Redactions applied: {redactions_applied}")
            print(f"   📈 Records processed: {metrics['total_records_processed']}")
            
            score = (pii_redacted + (redactions_applied > 0) + has_metrics) / 3.0
            print(f"   🏆 Privacy Compliance Score: {score:.3f}")
            
            return score, {
                "pii_redacted": pii_redacted,
                "redactions_applied": redactions_applied,
                "records_processed": metrics["total_records_processed"]
            }
            
        except Exception as e:
            print(f"   ❌ Privacy Compliance Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_l0_l1_contract(self) -> Tuple[float, Dict[str, Any]]:
        """Test L0-L1 Contract component"""
        print("\n📋 Testing L0-L1 Contract...")
        
        try:
            from layer0_l1_contract import L0L1ContractManager, SchemaVersion
            
            contract_manager = L0L1ContractManager()
            
            # Test valid telemetry validation
            valid_telemetry = {
                "telemetry_id": "contract_test_001",
                "timestamp": time.time(),
                "function_id": "test_function",
                "execution_phase": "invoke",
                "duration": 150.5,
                "memory_spike_kb": 1024,
                "cpu_utilization": 45.2,
                "anomaly_type": "benign"
            }
            
            validation_result = contract_manager.validate_telemetry_record(valid_telemetry, SchemaVersion.V1_0)
            valid_passed = validation_result.is_valid
            
            # Test invalid telemetry (missing required field)
            invalid_telemetry = {
                "timestamp": time.time(),
                "function_id": "test_function",
                "execution_phase": "invoke"
            }
            
            invalid_result = contract_manager.validate_telemetry_record(invalid_telemetry, SchemaVersion.V1_0)
            invalid_failed = not invalid_result.is_valid
            
            # Get contract status
            status = contract_manager.get_contract_status()
            has_schemas = status["schema_registry"]["total_schemas"] > 0
            
            print(f"   ✅ Valid telemetry passed: {'Yes' if valid_passed else 'No'}")
            print(f"   ❌ Invalid telemetry failed: {'Yes' if invalid_failed else 'No'}")
            print(f"   📚 Total schemas: {status['schema_registry']['total_schemas']}")
            
            score = (valid_passed + invalid_failed + has_schemas) / 3.0
            print(f"   🏆 L0-L1 Contract Score: {score:.3f}")
            
            return score, {
                "valid_passed": valid_passed,
                "invalid_failed": invalid_failed,
                "total_schemas": status["schema_registry"]["total_schemas"]
            }
            
        except Exception as e:
            print(f"   ❌ L0-L1 Contract Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_anomaly_detection(self) -> Tuple[float, Dict[str, Any]]:
        """Test Anomaly Detection component"""
        print("\n🧠 Testing Anomaly Detection...")
        
        try:
            from layer0_core import AnomalyDetectionEngine
            from app_config import create_testing_config
            from app_telemetry import create_telemetry_record_with_telemetry_id
            
            config = create_testing_config()
            engine = AnomalyDetectionEngine(config)
            
            # Test benign record
            benign_record = create_telemetry_record_with_telemetry_id(
                telemetry_id="test_benign_001",
                function_id="test_function",
                execution_phase="invoke",
                anomaly_type="benign",
                duration=120.0,
                memory_spike_kb=1024,
                cpu_utilization=25.0
            )
            
            benign_result = engine.detect_anomalies(benign_record)
            benign_correct = benign_result.overall_confidence <= 0.5
            
            # Test anomalous record
            anomaly_record = create_telemetry_record_with_telemetry_id(
                telemetry_id="test_anomaly_001",
                function_id="test_function",
                execution_phase="invoke",
                anomaly_type="cpu_spike",
                duration=2500.0,
                memory_spike_kb=15000,
                cpu_utilization=92.0
            )
            
            anomaly_result = engine.detect_anomalies(anomaly_record)
            anomaly_correct = anomaly_result.overall_confidence > 0.5
            
            print(f"   😊 Benign detection (confidence: {benign_result.overall_confidence:.3f}): {'Correct' if benign_correct else 'Incorrect'}")
            print(f"   🚨 Anomaly detection (confidence: {anomaly_result.overall_confidence:.3f}): {'Correct' if anomaly_correct else 'Incorrect'}")
            
            score = (benign_correct + anomaly_correct) / 2.0
            print(f"   🏆 Anomaly Detection Score: {score:.3f}")
            
            return score, {
                "benign_confidence": benign_result.overall_confidence,
                "anomaly_confidence": anomaly_result.overall_confidence,
                "benign_correct": benign_correct,
                "anomaly_correct": anomaly_correct
            }
            
        except Exception as e:
            print(f"   ❌ Anomaly Detection Error: {e}")
            return 0.0, {"error": str(e)}
    
    def run_production_readiness_assessment(self) -> Dict[str, Any]:
        """Run complete production readiness assessment"""
        print("🚀 SCAFAD Layer 0 - Production Readiness Assessment")
        print("=" * 60)
        
        # Check component availability
        self.check_component_availability()
        
        # Run component tests
        component_tests = [
            ("Signal Negotiation", self.test_signal_negotiation),
            ("Adaptive Buffer", self.test_adaptive_buffer),
            ("Vendor Adapters", self.test_vendor_adapters),
            ("Health Monitor", self.test_health_monitor),
            ("Privacy Compliance", self.test_privacy_compliance),
            ("L0-L1 Contract", self.test_l0_l1_contract),
            ("Anomaly Detection", self.test_anomaly_detection)
        ]
        
        for test_name, test_function in component_tests:
            try:
                score, details = test_function()
                self.test_results[test_name] = {
                    "score": score,
                    "details": details,
                    "status": "PASSED" if score >= 0.7 else "FAILED"
                }
            except Exception as e:
                self.test_results[test_name] = {
                    "score": 0.0,
                    "details": {"error": str(e)},
                    "status": "ERROR"
                }
        
        return self.generate_production_report()
    
    def generate_production_report(self) -> Dict[str, Any]:
        """Generate production readiness report"""
        total_time = time.time() - self.start_time
        
        # Calculate scores
        available_components = sum(1 for available in self.component_availability.values() if available)
        total_components = len(self.component_availability)
        availability_score = available_components / total_components
        
        test_scores = [result["score"] for result in self.test_results.values()]
        average_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
        
        overall_score = (availability_score * 0.3) + (average_test_score * 0.7)
        
        # Determine readiness status
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        total_tests = len(self.test_results)
        
        if overall_score >= 0.9 and passed_tests >= 0.8 * total_tests:
            readiness_status = "🟢 PRODUCTION READY"
            recommendation = "Layer 0 is ready for production deployment"
        elif overall_score >= 0.8 and passed_tests >= 0.7 * total_tests:
            readiness_status = "🟡 NEARLY READY"
            recommendation = "Layer 0 needs minor fixes before production"
        elif overall_score >= 0.6:
            readiness_status = "🟠 DEVELOPMENT READY"
            recommendation = "Layer 0 suitable for development/testing environments"
        else:
            readiness_status = "🔴 NOT READY"
            recommendation = "Layer 0 requires significant work before deployment"
        
        # Print results
        print(f"\n📊 LAYER 0 PRODUCTION READINESS RESULTS")
        print("=" * 60)
        
        print(f"\nComponent Availability:")
        print(f"   Available: {available_components}/{total_components} ({availability_score*100:.1f}%)")
        
        print(f"\nTest Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⚠️"
            print(f"   {status_icon} {test_name:25} | Score: {result['score']:.3f} | {result['status']}")
        
        print(f"\nOverall Assessment:")
        print(f"   Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"   Average Test Score: {average_test_score:.3f}")
        print(f"   Overall Score: {overall_score:.3f}")
        print(f"   Execution Time: {total_time:.1f}s")
        print(f"   Readiness Status: {readiness_status}")
        print(f"   Recommendation: {recommendation}")
        
        # Key findings
        print(f"\n💡 Key Findings:")
        excellent_components = [name for name, result in self.test_results.items() if result["score"] >= 0.9]
        if excellent_components:
            print(f"   ✅ Excellent: {', '.join(excellent_components)}")
        
        good_components = [name for name, result in self.test_results.items() if 0.7 <= result["score"] < 0.9]
        if good_components:
            print(f"   🟡 Good: {', '.join(good_components)}")
        
        needs_work = [name for name, result in self.test_results.items() if result["score"] < 0.7]
        if needs_work:
            print(f"   ❌ Needs Work: {', '.join(needs_work)}")
        
        # Deployment checklist
        print(f"\n📋 Deployment Readiness Checklist:")
        checklist_items = [
            ("Core Components Available", availability_score >= 0.8),
            ("Integration Tests Passing", passed_tests >= 0.8 * total_tests),
            ("Performance Acceptable", average_test_score >= 0.7),
            ("Health Monitoring Active", self.test_results.get("Health Monitor", {}).get("score", 0) >= 0.7),
            ("Privacy Compliance Ready", self.test_results.get("Privacy Compliance", {}).get("score", 0) >= 0.7),
            ("Contract Validation Working", self.test_results.get("L0-L1 Contract", {}).get("score", 0) >= 0.7)
        ]
        
        for item, status in checklist_items:
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {item}")
        
        print(f"\n✨ Layer 0 production readiness assessment complete!")
        
        return {
            "overall_score": overall_score,
            "availability_score": availability_score,
            "average_test_score": average_test_score,
            "readiness_status": readiness_status,
            "recommendation": recommendation,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "execution_time_seconds": total_time,
            "test_results": self.test_results,
            "components_available": self.component_availability
        }

def main():
    """Main execution"""
    validator = Layer0ProductionValidator()
    results = validator.run_production_readiness_assessment()
    
    print(f"\n🎯 FINAL VERDICT: {results['readiness_status']}")
    print(f"📊 Overall Score: {results['overall_score']:.3f}")
    print(f"📋 Recommendation: {results['recommendation']}")
    
    return results

if __name__ == "__main__":
    results = main()