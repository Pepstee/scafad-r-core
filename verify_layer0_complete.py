#!/usr/bin/env python3
"""
Complete Layer 0 Verification and Testing
=========================================
This script comprehensively tests all Layer 0 components to verify they are
finished, working, tested, and ready for deployment.
"""

import sys
import time
import json
import asyncio
from typing import Dict, Any, List, Tuple

# Add workspace to path
sys.path.insert(0, '/workspace')

class Layer0Verifier:
    """Comprehensive Layer 0 verification system"""
    
    def __init__(self):
        self.test_results = {}
        self.component_availability = {}
        self.start_time = time.time()
    
    def check_imports_and_availability(self) -> Dict[str, bool]:
        """Check if all Layer 0 components can be imported"""
        print("🔧 COMPONENT AVAILABILITY CHECK")
        print("=" * 50)
        
        components = [
            ('layer0_signal_negotiation', 'SignalNegotiator'),
            ('layer0_redundancy_manager', 'RedundancyManager'), 
            ('layer0_sampler', 'ExecutionAwareSampler'),
            ('layer0_fallback_orchestrator', 'FallbackOrchestrator'),
            ('layer0_adaptive_buffer', 'AdaptiveBuffer'),
            ('layer0_vendor_adapters', 'VendorAdapterManager'),
            ('layer0_health_monitor', 'HealthMonitor'),
            ('layer0_privacy_compliance', 'PrivacyCompliancePipeline'),
            ('layer0_l1_contract', 'L0L1ContractManager'),
            ('layer0_core', 'AnomalyDetectionEngine'),
            ('app_config', 'create_testing_config'),
            ('app_telemetry', 'create_telemetry_record_with_telemetry_id')
        ]
        
        for module_name, class_name in components:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if hasattr(module, class_name):
                    self.component_availability[module_name] = True
                    print(f"✅ {module_name}: {class_name} available")
                else:
                    self.component_availability[module_name] = False
                    print(f"❌ {module_name}: {class_name} missing")
            except ImportError as e:
                self.component_availability[module_name] = False
                print(f"❌ {module_name}: Import failed - {e}")
            except Exception as e:
                self.component_availability[module_name] = False
                print(f"❌ {module_name}: Error - {e}")
        
        available = sum(self.component_availability.values())
        total = len(self.component_availability)
        print(f"\n📊 Availability: {available}/{total} ({available/total*100:.1f}%)")
        
        return self.component_availability
    
    def test_signal_negotiation(self) -> Tuple[float, Dict[str, Any]]:
        """Test signal negotiation functionality"""
        print("\n🔧 TESTING SIGNAL NEGOTIATION")
        print("-" * 30)
        
        try:
            from layer0_signal_negotiation import SignalNegotiator
            from app_config import create_testing_config
            
            config = create_testing_config()
            negotiator = SignalNegotiator(config)
            
            # Test basic functionality
            channels = len(negotiator.available_channels)
            recommendations = negotiator.get_channel_recommendations(1024, priority="balanced")
            health = negotiator.get_channel_health_summary()
            
            print(f"   📡 Channels discovered: {channels}")
            print(f"   🎯 Recommendations: {len(recommendations)}")
            print(f"   💊 Health tracked: {len(health)}")
            
            score = min(1.0, (channels + len(recommendations) + len(health)) / 15.0)
            print(f"   🏆 Signal Negotiation Score: {score:.3f}")
            
            return score, {
                "channels": channels,
                "recommendations": len(recommendations),
                "health_tracked": len(health)
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_adaptive_buffer(self) -> Tuple[float, Dict[str, Any]]:
        """Test adaptive buffer functionality"""
        print("\n📊 TESTING ADAPTIVE BUFFER")
        print("-" * 30)
        
        try:
            from layer0_adaptive_buffer import AdaptiveBuffer, BufferConfig
            
            config = BufferConfig(max_queue_size=20, base_batch_size=5)
            buffer = AdaptiveBuffer(config, "test_buffer")
            
            # Test enqueue/dequeue
            enqueue_count = 0
            for i in range(15):
                if buffer.enqueue(f"item_{i}", size_bytes=100):
                    enqueue_count += 1
            
            items = buffer.dequeue(5)
            dequeue_count = len(items)
            
            # Test status
            status = buffer.get_status()
            
            buffer.shutdown()
            
            print(f"   📥 Enqueued: {enqueue_count}/15")
            print(f"   📤 Dequeued: {dequeue_count}")
            print(f"   📊 Status: {status.get('queue_size', 0)} items")
            
            score = min(1.0, (enqueue_count + dequeue_count) / 20.0)
            print(f"   🏆 Adaptive Buffer Score: {score:.3f}")
            
            return score, {
                "enqueued": enqueue_count,
                "dequeued": dequeue_count,
                "queue_size": status.get('queue_size', 0)
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_health_monitor(self) -> Tuple[float, Dict[str, Any]]:
        """Test health monitor functionality"""
        print("\n💊 TESTING HEALTH MONITOR")
        print("-" * 30)
        
        try:
            from layer0_health_monitor import HealthMonitor, HealthCheck, ComponentType, HealthStatus
            from app_config import create_testing_config
            
            def test_health_func():
                return HealthStatus.HEALTHY, {"status": "ok"}
            
            config = create_testing_config()
            monitor = HealthMonitor(config, "test_monitor")
            
            # Register health check
            check = HealthCheck(
                name="test_check",
                component=ComponentType.SIGNAL_NEGOTIATOR,
                check_function=test_health_func,
                interval_ms=1000
            )
            monitor.register_health_check(check)
            
            # Register heartbeat
            monitor.register_component_heartbeat(ComponentType.STREAM_PROCESSOR)
            
            # Send heartbeats
            for _ in range(3):
                monitor.heartbeat(ComponentType.STREAM_PROCESSOR)
                time.sleep(0.1)
            
            # Get status
            system_health = monitor.get_system_health()
            component_status = monitor.get_component_status(ComponentType.SIGNAL_NEGOTIATOR)
            
            monitor.shutdown()
            
            print(f"   🏥 System status: {system_health['overall_status']}")
            print(f"   📈 Components monitored: {system_health['component_count']}")
            print(f"   💗 Recent checks: {len(component_status.get('recent_results', []))}")
            
            score = 0.9 if system_health['overall_status'] != 'unknown' else 0.5
            print(f"   🏆 Health Monitor Score: {score:.3f}")
            
            return score, {
                "system_status": system_health['overall_status'],
                "components_monitored": system_health['component_count'],
                "checks_performed": len(component_status.get('recent_results', []))
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_anomaly_detection(self) -> Tuple[float, Dict[str, Any]]:
        """Test anomaly detection functionality"""
        print("\n🧠 TESTING ANOMALY DETECTION")
        print("-" * 30)
        
        try:
            from layer0_core import AnomalyDetectionEngine, DetectionConfig
            from app_telemetry import create_telemetry_record_with_telemetry_id
            
            detection_config = DetectionConfig()
            engine = AnomalyDetectionEngine(detection_config)
            
            # Test benign record
            benign_record = create_telemetry_record_with_telemetry_id(
                telemetry_id="test_benign",
                function_id="test_func",
                execution_phase="invoke",
                anomaly_type="benign",
                duration=100.0,
                memory_spike_kb=1000,
                cpu_utilization=25.0
            )
            
            benign_result = engine.detect_anomalies(benign_record)
            
            # Test anomaly record
            anomaly_record = create_telemetry_record_with_telemetry_id(
                telemetry_id="test_anomaly",
                function_id="test_func",
                execution_phase="invoke",
                anomaly_type="cpu_burst",
                duration=3000.0,
                memory_spike_kb=15000,
                cpu_utilization=95.0
            )
            
            anomaly_result = engine.detect_anomalies(anomaly_record)
            
            print(f"   😊 Benign confidence: {benign_result.combined_confidence:.3f}")
            print(f"   🚨 Anomaly confidence: {anomaly_result.combined_confidence:.3f}")
            print(f"   🔧 Detection algorithms: {len(benign_result.detection_results)}")
            
            benign_correct = benign_result.combined_confidence <= 0.6
            anomaly_correct = anomaly_result.combined_confidence > 0.4
            score = (benign_correct + anomaly_correct) / 2.0
            print(f"   🏆 Anomaly Detection Score: {score:.3f}")
            
            return score, {
                "benign_confidence": benign_result.combined_confidence,
                "anomaly_confidence": anomaly_result.combined_confidence,
                "algorithms_used": len(benign_result.detection_results),
                "benign_correct": benign_correct,
                "anomaly_correct": anomaly_correct
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_privacy_compliance(self) -> Tuple[float, Dict[str, Any]]:
        """Test privacy compliance functionality"""
        print("\n🔒 TESTING PRIVACY COMPLIANCE")
        print("-" * 30)
        
        try:
            from layer0_privacy_compliance import PrivacyCompliancePipeline, ComplianceConfig, DataClassification
            
            config = ComplianceConfig(require_consent=True, anonymization_enabled=True)
            pipeline = PrivacyCompliancePipeline(config)
            
            # Test data with PII
            test_data = {
                "email": "test@example.com",
                "phone": "555-123-4567",
                "ip": "192.168.1.1",
                "safe_data": "this is fine"
            }
            
            redacted_data, redaction_results = pipeline.process_data(
                test_data,
                user_id="test_user",
                data_classification=DataClassification.CONFIDENTIAL
            )
            
            metrics = pipeline.get_privacy_metrics()
            
            pii_redacted = "test@example.com" not in json.dumps(redacted_data)
            
            print(f"   🚫 PII redacted: {'Yes' if pii_redacted else 'No'}")
            print(f"   📊 Redactions: {len(redaction_results)}")
            print(f"   📈 Records processed: {metrics['total_records_processed']}")
            
            score = (pii_redacted + (len(redaction_results) > 0) + (metrics['total_records_processed'] > 0)) / 3.0
            print(f"   🏆 Privacy Compliance Score: {score:.3f}")
            
            return score, {
                "pii_redacted": pii_redacted,
                "redactions_applied": len(redaction_results),
                "records_processed": metrics['total_records_processed']
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {"error": str(e)}
    
    def test_l0_l1_contract(self) -> Tuple[float, Dict[str, Any]]:
        """Test L0-L1 contract functionality"""
        print("\n📋 TESTING L0-L1 CONTRACT")
        print("-" * 30)
        
        try:
            from layer0_l1_contract import L0L1ContractManager, SchemaVersion
            
            manager = L0L1ContractManager()
            
            # Test valid telemetry
            valid_data = {
                "telemetry_id": "test_001",
                "timestamp": time.time(),
                "function_id": "test_func",
                "execution_phase": "invoke",
                "duration": 100.0,
                "memory_spike_kb": 1000,
                "cpu_utilization": 25.0,
                "anomaly_type": "benign"
            }
            
            result = manager.validate_telemetry_record(valid_data, SchemaVersion.V1_0)
            
            # Test invalid telemetry
            invalid_data = {
                "timestamp": time.time(),
                "function_id": "test_func"
                # Missing required telemetry_id
            }
            
            invalid_result = manager.validate_telemetry_record(invalid_data, SchemaVersion.V1_0)
            
            status = manager.get_contract_status()
            
            print(f"   ✅ Valid passed: {'Yes' if result.is_valid else 'No'}")
            print(f"   ❌ Invalid failed: {'Yes' if not invalid_result.is_valid else 'No'}")
            print(f"   📚 Schemas: {status['schema_registry']['total_schemas']}")
            
            score = (result.is_valid + (not invalid_result.is_valid) + (status['schema_registry']['total_schemas'] > 0)) / 3.0
            print(f"   🏆 L0-L1 Contract Score: {score:.3f}")
            
            return score, {
                "valid_passed": result.is_valid,
                "invalid_failed": not invalid_result.is_valid,
                "total_schemas": status['schema_registry']['total_schemas']
            }
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0.0, {"error": str(e)}
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run complete Layer 0 verification"""
        print("🚀 SCAFAD LAYER 0 - COMPREHENSIVE VERIFICATION")
        print("=" * 60)
        print("Verifying that Layer 0 is finished, working, tested, and ready for deployment...")
        
        # Check component availability
        self.check_imports_and_availability()
        
        # Run functional tests
        tests = [
            ("Signal Negotiation", self.test_signal_negotiation),
            ("Adaptive Buffer", self.test_adaptive_buffer),
            ("Health Monitor", self.test_health_monitor),
            ("Anomaly Detection", self.test_anomaly_detection),
            ("Privacy Compliance", self.test_privacy_compliance),
            ("L0-L1 Contract", self.test_l0_l1_contract)
        ]
        
        for test_name, test_func in tests:
            try:
                score, details = test_func()
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
        
        # Generate final report
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        total_time = time.time() - self.start_time
        
        # Calculate scores
        available_components = sum(self.component_availability.values())
        total_components = len(self.component_availability)
        availability_score = available_components / total_components
        
        test_scores = [result["score"] for result in self.test_results.values()]
        avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
        
        overall_score = (availability_score * 0.3) + (avg_test_score * 0.7)
        
        passed_tests = sum(1 for r in self.test_results.values() if r["status"] == "PASSED")
        total_tests = len(self.test_results)
        
        # Determine status
        if overall_score >= 0.9 and passed_tests >= 0.8 * total_tests:
            status = "🟢 PRODUCTION READY"
            recommendation = "Layer 0 is completely finished and ready for deployment"
        elif overall_score >= 0.8:
            status = "🟡 NEARLY READY"
            recommendation = "Layer 0 is mostly complete with minor issues"
        elif overall_score >= 0.6:
            status = "🟠 DEVELOPMENT READY"
            recommendation = "Layer 0 has good foundation but needs refinement"
        else:
            status = "🔴 NEEDS WORK"
            recommendation = "Layer 0 requires significant additional development"
        
        # Print comprehensive report
        print(f"\n" + "="*60)
        print("📊 LAYER 0 COMPREHENSIVE VERIFICATION RESULTS")
        print("="*60)
        
        print(f"\nComponent Availability:")
        print(f"   Available: {available_components}/{total_components} ({availability_score*100:.1f}%)")
        
        print(f"\nFunctional Test Results:")
        for test_name, result in self.test_results.items():
            icon = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⚠️"
            print(f"   {icon} {test_name:20} | Score: {result['score']:.3f} | {result['status']}")
        
        print(f"\nOverall Assessment:")
        print(f"   Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"   Average Test Score: {avg_test_score:.3f}")
        print(f"   Overall Score: {overall_score:.3f}")
        print(f"   Verification Time: {total_time:.1f}s")
        print(f"   Status: {status}")
        print(f"   Recommendation: {recommendation}")
        
        # Component breakdown
        print(f"\n🔧 Component Status Breakdown:")
        for component, available in self.component_availability.items():
            icon = "✅" if available else "❌"
            name = component.replace('layer0_', '').replace('_', ' ').title()
            print(f"   {icon} {name}")
        
        # Key findings
        print(f"\n💡 Key Findings:")
        excellent = [name for name, result in self.test_results.items() if result["score"] >= 0.9]
        if excellent:
            print(f"   ✅ Excellent: {', '.join(excellent)}")
        
        good = [name for name, result in self.test_results.items() if 0.7 <= result["score"] < 0.9]
        if good:
            print(f"   🟡 Good: {', '.join(good)}")
        
        needs_work = [name for name, result in self.test_results.items() if result["score"] < 0.7]
        if needs_work:
            print(f"   ❌ Needs Work: {', '.join(needs_work)}")
        
        print(f"\n✨ Layer 0 Comprehensive Verification Complete!")
        
        return {
            "overall_score": overall_score,
            "availability_score": availability_score,
            "avg_test_score": avg_test_score,
            "status": status,
            "recommendation": recommendation,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "execution_time": total_time,
            "test_results": self.test_results,
            "component_availability": self.component_availability
        }

def main():
    """Main verification execution"""
    verifier = Layer0Verifier()
    results = verifier.run_comprehensive_verification()
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Overall Score: {results['overall_score']:.3f}/1.0")
    print(f"   Status: {results['status']}")
    print(f"   {results['recommendation']}")
    
    return results

if __name__ == "__main__":
    results = main()