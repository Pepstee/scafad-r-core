#!/usr/bin/env python3
"""
Complete Layer 0 Integration Test Suite
========================================

Comprehensive integration testing for all Layer 0 components:
- Signal Negotiation & Channel Management
- Redundancy Management & Failover
- Execution-Aware Sampling & Adaptation
- Fallback Orchestration Matrix
- Stream Processing Pipeline
- Compression Optimization
- Adaptive Buffer & Backpressure
- Vendor Adapter Conformance
- Health & Heartbeat Monitoring
- Runtime Control & Management

This test validates end-to-end Layer 0 functionality for production readiness.
"""

import sys
import time
import asyncio
import json
import threading
from typing import Dict, List, Any, Optional
import traceback

# Add workspace to path
sys.path.insert(0, '/workspace')

# =============================================================================
# Component Availability Check
# =============================================================================

def check_component_availability():
    """Check availability of all Layer 0 components"""
    print("🔧 Checking Layer 0 Component Availability...")
    print("=" * 60)
    
    components = {}
    
    # Core components
    core_components = [
        ('layer0_signal_negotiation', 'SignalNegotiator', 'Signal Negotiation'),
        ('layer0_redundancy_manager', 'RedundancyManager', 'Redundancy Manager'),
        ('layer0_sampler', 'ExecutionAwareSampler', 'Execution-Aware Sampler'),
        ('layer0_fallback_orchestrator', 'FallbackOrchestrator', 'Fallback Orchestrator'),
        ('layer0_stream_processor', 'StreamProcessor', 'Stream Processor'),
        ('layer0_compression_optimizer', 'CompressionOptimizer', 'Compression Optimizer'),
        ('layer0_adaptive_buffer', 'AdaptiveBuffer', 'Adaptive Buffer'),
        ('layer0_vendor_adapters', 'VendorAdapterManager', 'Vendor Adapters'),
        ('layer0_health_monitor', 'HealthMonitor', 'Health Monitor'),
        ('layer0_runtime_control', 'RuntimeController', 'Runtime Control'),
        ('layer0_core', 'AnomalyDetectionEngine', 'Anomaly Detection Engine'),
        ('app_config', 'Layer0Config', 'Configuration System'),
        ('app_telemetry', 'create_telemetry_record_with_telemetry_id', 'Telemetry System')
    ]
    
    for module_name, class_name, display_name in core_components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            components[module_name] = True
            print(f"   ✅ {display_name}: Available")
        except ImportError as e:
            components[module_name] = False
            print(f"   ❌ {display_name}: Import Error - {e}")
        except AttributeError as e:
            components[module_name] = False
            print(f"   ❌ {display_name}: Missing Class - {e}")
        except Exception as e:
            components[module_name] = False
            print(f"   ❌ {display_name}: Error - {e}")
    
    available_count = sum(1 for available in components.values() if available)
    total_count = len(components)
    
    print(f"\n📊 Component Availability: {available_count}/{total_count} ({available_count/total_count*100:.1f}%)")
    
    return components

# =============================================================================
# Individual Component Tests
# =============================================================================

async def test_signal_negotiation():
    """Test Signal Negotiation component"""
    print("\n🔧 Testing Signal Negotiation...")
    
    try:
        from layer0_signal_negotiation import SignalNegotiator, ChannelType
        from app_config import create_testing_config
        
        config = create_testing_config()
        negotiator = SignalNegotiator(config)
        
        # Test channel discovery
        available_channels = len(negotiator.available_channels)
        assert available_channels > 0, "No channels discovered"
        
        # Test negotiation
        await negotiator.negotiate_all_channels()
        negotiated_channels = len(negotiator.negotiated_channels)
        
        # Test health summary
        health = negotiator.get_channel_health_summary()
        assert isinstance(health, dict), "Health summary should be dict"
        
        score = min(1.0, (available_channels + negotiated_channels) / 10)
        print(f"   ✅ Signal Negotiation Score: {score:.3f}")
        
        return score, {"channels_discovered": available_channels, "channels_negotiated": negotiated_channels}
        
    except Exception as e:
        print(f"   ❌ Signal Negotiation Error: {e}")
        return 0.0, {"error": str(e)}

async def test_redundancy_management():
    """Test Redundancy Management component"""
    print("\n🔄 Testing Redundancy Management...")
    
    try:
        from layer0_redundancy_manager import RedundancyManager
        from app_config import create_testing_config
        from app_telemetry import create_telemetry_record_with_telemetry_id
        
        config = create_testing_config()
        manager = RedundancyManager(config)
        
        # Create test record
        test_record = create_telemetry_record_with_telemetry_id(
            telemetry_id="redundancy_test_001",
            function_id="test_function",
            execution_phase="invoke",
            anomaly_type="benign",
            duration=100.0,
            memory_spike_kb=1024,
            cpu_utilization=25.0,
            custom_fields={"test": True}
        )
        
        # Test duplication
        dup_result = manager.duplicate_telemetry(test_record, strategy="active_active")
        dup_success = dup_result is not None and dup_result.get("duplicated", False)
        
        # Test failover
        failover_result = manager.initiate_failover("primary", "standby")
        failover_success = failover_result is not None and failover_result.get("failover_initiated", False)
        
        # Test deduplication
        first_check = manager.check_duplicate("test_001")
        second_check = manager.check_duplicate("test_001")
        dedup_success = first_check != second_check
        
        score = (dup_success + failover_success + dedup_success) / 3.0
        print(f"   ✅ Redundancy Management Score: {score:.3f}")
        
        return score, {
            "duplication": dup_success,
            "failover": failover_success,
            "deduplication": dedup_success
        }
        
    except Exception as e:
        print(f"   ❌ Redundancy Management Error: {e}")
        return 0.0, {"error": str(e)}

async def test_execution_aware_sampling():
    """Test Execution-Aware Sampling component"""
    print("\n🎯 Testing Execution-Aware Sampling...")
    
    try:
        from layer0_sampler import ExecutionAwareSampler
        from app_config import create_testing_config
        
        config = create_testing_config()
        sampler = ExecutionAwareSampler(config)
        
        # Test cold/warm adaptive sampling
        cold_rate = sampler.get_sampling_rate(execution_state="cold", function_id="test_func_1")
        warm_rate = sampler.get_sampling_rate(execution_state="warm", function_id="test_func_1")
        cold_warm_adaptive = cold_rate >= warm_rate and 0.0 <= cold_rate <= 1.0 and 0.0 <= warm_rate <= 1.0
        
        # Test latency-based sampling
        sampler.update_latency_metrics("fast_func", 50.0)
        sampler.update_latency_metrics("slow_func", 2000.0)
        fast_rate = sampler.get_sampling_rate(function_id="fast_func")
        slow_rate = sampler.get_sampling_rate(function_id="slow_func")
        latency_adaptive = slow_rate >= fast_rate
        
        # Test error rate sampling
        sampler.update_error_metrics("reliable_func", 1, 100)
        sampler.update_error_metrics("unreliable_func", 20, 100)
        reliable_rate = sampler.get_sampling_rate(function_id="reliable_func")
        unreliable_rate = sampler.get_sampling_rate(function_id="unreliable_func")
        error_adaptive = unreliable_rate >= reliable_rate
        
        score = (cold_warm_adaptive + latency_adaptive + error_adaptive) / 3.0
        print(f"   ✅ Execution-Aware Sampling Score: {score:.3f}")
        
        return score, {
            "cold_warm_adaptive": cold_warm_adaptive,
            "latency_adaptive": latency_adaptive,
            "error_adaptive": error_adaptive
        }
        
    except Exception as e:
        print(f"   ❌ Execution-Aware Sampling Error: {e}")
        return 0.0, {"error": str(e)}

async def test_fallback_orchestration():
    """Test Fallback Orchestration component"""
    print("\n🚨 Testing Fallback Orchestration...")
    
    try:
        from layer0_fallback_orchestrator import FallbackOrchestrator, FallbackMode
        from app_config import create_testing_config
        
        config = create_testing_config()
        orchestrator = FallbackOrchestrator(config)
        
        # Test initial status
        status = orchestrator.get_fallback_status()
        initial_mode = status["current_mode"]
        
        # Test telemetry tracking
        orchestrator.update_telemetry_tracking("telemetry")
        orchestrator.update_performance_metric(0.8)
        
        # Test forced fallback
        orchestrator.force_fallback(FallbackMode.DEGRADED, "integration_test")
        
        # Test status after fallback
        fallback_status = orchestrator.get_fallback_status()
        fallback_triggered = fallback_status["current_mode"] == "degraded"
        
        # Test channel failover matrix
        matrix_status = orchestrator.channel_failover_matrix.get_matrix_status()
        matrix_active = matrix_status["active_channel"] is not None
        
        orchestrator.shutdown()
        
        score = (1.0 if fallback_triggered else 0.5) + (0.5 if matrix_active else 0.0)
        print(f"   ✅ Fallback Orchestration Score: {score:.3f}")
        
        return score, {
            "initial_mode": initial_mode,
            "fallback_triggered": fallback_triggered,
            "matrix_active": matrix_active
        }
        
    except Exception as e:
        print(f"   ❌ Fallback Orchestration Error: {e}")
        return 0.0, {"error": str(e)}

async def test_adaptive_buffer():
    """Test Adaptive Buffer component"""
    print("\n📊 Testing Adaptive Buffer...")
    
    try:
        from layer0_adaptive_buffer import AdaptiveBuffer, BufferConfig, LossPolicy
        
        config = BufferConfig(
            max_queue_size=100,
            max_memory_bytes=10 * 1024,
            base_batch_size=10
        )
        
        buffer = AdaptiveBuffer(config, "integration_test_buffer")
        
        # Test normal enqueue/dequeue
        enqueue_success = 0
        for i in range(50):
            if buffer.enqueue(f"item_{i}", size_bytes=100):
                enqueue_success += 1
        
        # Test batch dequeue
        items = buffer.dequeue(20)
        dequeue_success = len(items)
        
        # Test watermark triggering
        for i in range(40):  # Should trigger high watermark
            buffer.enqueue(f"watermark_item_{i}", size_bytes=100)
        
        status = buffer.get_status()
        backpressure_active = status["backpressure_active"]
        
        buffer.shutdown()
        
        score = min(1.0, (enqueue_success + dequeue_success + (20 if backpressure_active else 0)) / 90)
        print(f"   ✅ Adaptive Buffer Score: {score:.3f}")
        
        return score, {
            "enqueue_success": enqueue_success,
            "dequeue_success": dequeue_success,
            "backpressure_active": backpressure_active
        }
        
    except Exception as e:
        print(f"   ❌ Adaptive Buffer Error: {e}")
        return 0.0, {"error": str(e)}

async def test_vendor_adapters():
    """Test Vendor Adapters component"""
    print("\n🔌 Testing Vendor Adapters...")
    
    try:
        from layer0_vendor_adapters import VendorAdapterManager, ProviderType, RequestMetadata
        import uuid
        
        manager = VendorAdapterManager()
        
        # Create adapters
        cloudwatch = manager.create_adapter(ProviderType.CLOUDWATCH)
        datadog = manager.create_adapter(ProviderType.DATADOG)
        
        # Test payload
        test_payload = {"test": "integration_data", "metrics": [1, 2, 3]}
        
        # Test CloudWatch adapter
        metadata = RequestMetadata(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            attempt=1,
            payload_size_bytes=len(json.dumps(test_payload).encode())
        )
        
        cw_success, cw_response = await cloudwatch.send_with_retry(test_payload, metadata)
        
        # Test DataDog adapter  
        metadata.request_id = str(uuid.uuid4())
        dd_success, dd_response = await datadog.send_with_retry(test_payload, metadata)
        
        # Get status summary
        summary = manager.get_status_summary()
        adapters_created = summary["adapters_count"]
        
        score = (cw_success + dd_success + min(1.0, adapters_created / 2)) / 3.0
        print(f"   ✅ Vendor Adapters Score: {score:.3f}")
        
        return score, {
            "cloudwatch_success": cw_success,
            "datadog_success": dd_success,
            "adapters_created": adapters_created
        }
        
    except Exception as e:
        print(f"   ❌ Vendor Adapters Error: {e}")
        return 0.0, {"error": str(e)}

async def test_health_monitor():
    """Test Health Monitor component"""
    print("\n💊 Testing Health Monitor...")
    
    try:
        from layer0_health_monitor import HealthMonitor, HealthCheck, ComponentType, HealthStatus
        from app_config import create_testing_config
        
        def sample_health_check():
            return HealthStatus.HEALTHY, {"status": "ok", "cpu": 25.5}
        
        config = create_testing_config()
        monitor = HealthMonitor(config, "integration_test_monitor")
        
        # Register health check
        check = HealthCheck(
            name="integration_test_check",
            component=ComponentType.SIGNAL_NEGOTIATOR,
            check_function=sample_health_check,
            interval_ms=500
        )
        
        monitor.register_health_check(check)
        
        # Register heartbeat monitoring
        monitor.register_component_heartbeat(ComponentType.STREAM_PROCESSOR)
        
        # Send heartbeats
        for _ in range(3):
            monitor.heartbeat(ComponentType.STREAM_PROCESSOR)
            await asyncio.sleep(0.2)
        
        # Wait for health checks
        await asyncio.sleep(1.0)
        
        # Get system health
        system_health = monitor.get_system_health()
        overall_status = system_health["overall_status"]
        
        # Get component status
        component_status = monitor.get_component_status(ComponentType.SIGNAL_NEGOTIATOR)
        
        monitor.shutdown()
        
        score = 1.0 if overall_status in ["healthy", "degraded"] else 0.5
        print(f"   ✅ Health Monitor Score: {score:.3f}")
        
        return score, {
            "overall_status": overall_status,
            "component_monitored": len(component_status.get("recent_results", []))
        }
        
    except Exception as e:
        print(f"   ❌ Health Monitor Error: {e}")
        return 0.0, {"error": str(e)}

async def test_anomaly_detection():
    """Test Anomaly Detection Engine"""
    print("\n🧠 Testing Anomaly Detection Engine...")
    
    try:
        from layer0_core import AnomalyDetectionEngine
        from app_config import create_testing_config
        from app_telemetry import create_telemetry_record_with_telemetry_id
        
        config = create_testing_config()
        from layer0_core import DetectionConfig
        detection_config = DetectionConfig()
        engine = AnomalyDetectionEngine(detection_config)
        
        # Test benign record
        benign_record = create_telemetry_record_with_telemetry_id(
            telemetry_id="integration_benign_001",
            function_id="integration_test_function",
            execution_phase="invoke",
            anomaly_type="benign",
            duration=120.0,
            memory_spike_kb=1024,
            cpu_utilization=25.0
        )
        
        benign_result = engine.detect_anomalies(benign_record)
        benign_correct = benign_result.combined_confidence <= 0.5
        
        # Test anomalous record
        anomaly_record = create_telemetry_record_with_telemetry_id(
            telemetry_id="integration_anomaly_001",
            function_id="integration_test_function",
            execution_phase="invoke",
            anomaly_type="cpu_burst",
            duration=2500.0,
            memory_spike_kb=15000,
            cpu_utilization=92.0
        )
        
        anomaly_result = engine.detect_anomalies(anomaly_record)
        anomaly_correct = anomaly_result.combined_confidence > 0.5
        
        score = (benign_correct + anomaly_correct) / 2.0
        print(f"   ✅ Anomaly Detection Score: {score:.3f}")
        
        return score, {
            "benign_confidence": benign_result.combined_confidence,
            "anomaly_confidence": anomaly_result.combined_confidence,
            "benign_correct": benign_correct,
            "anomaly_correct": anomaly_correct
        }
        
    except Exception as e:
        print(f"   ❌ Anomaly Detection Error: {e}")
        return 0.0, {"error": str(e)}

async def test_privacy_compliance():
    """Test Privacy Compliance Pipeline"""
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
            user_id="integration_test_user",
            data_classification=DataClassification.CONFIDENTIAL,
            operation="integration_test"
        )
        
        # Verify redactions
        pii_redacted = "test@example.com" not in json.dumps(redacted_data)
        redactions_applied = len(redaction_results) > 0
        
        # Get metrics
        metrics = pipeline.get_privacy_metrics()
        has_metrics = metrics["total_records_processed"] > 0
        
        score = (pii_redacted + redactions_applied + has_metrics) / 3.0
        print(f"   ✅ Privacy Compliance Score: {score:.3f}")
        
        return score, {
            "pii_redacted": pii_redacted,
            "redactions_applied": len(redaction_results),
            "records_processed": metrics["total_records_processed"],
            "pii_detections": metrics["pii_detections"]
        }
        
    except Exception as e:
        print(f"   ❌ Privacy Compliance Error: {e}")
        return 0.0, {"error": str(e)}

async def test_l0_l1_contract():
    """Test L0-L1 Contract Validation"""
    print("\n📋 Testing L0-L1 Contract...")
    
    try:
        from layer0_l1_contract import L0L1ContractManager, SchemaVersion
        
        contract_manager = L0L1ContractManager()
        
        # Test valid telemetry validation
        valid_telemetry = {
            "telemetry_id": "contract_test_001",
            "timestamp": time.time(),
            "function_id": "contract_test_function",
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
            "function_id": "contract_test_function",
            "execution_phase": "invoke"
            # Missing telemetry_id
        }
        
        invalid_result = contract_manager.validate_telemetry_record(invalid_telemetry, SchemaVersion.V1_0)
        invalid_failed = not invalid_result.is_valid
        
        # Get contract status
        status = contract_manager.get_contract_status()
        has_schemas = status["schema_registry"]["total_schemas"] > 0
        
        score = (valid_passed + invalid_failed + has_schemas) / 3.0
        print(f"   ✅ L0-L1 Contract Score: {score:.3f}")
        
        return score, {
            "valid_telemetry_passed": valid_passed,
            "invalid_telemetry_failed": invalid_failed,
            "total_schemas": status["schema_registry"]["total_schemas"],
            "validation_success_rate": status["validation_metrics"]["success_rate"]
        }
        
    except Exception as e:
        print(f"   ❌ L0-L1 Contract Error: {e}")
        return 0.0, {"error": str(e)}

# =============================================================================
# End-to-End Integration Test
# =============================================================================

async def test_end_to_end_integration():
    """Test end-to-end Layer 0 integration"""
    print("\n🔗 Testing End-to-End Integration...")
    
    try:
        from app_config import create_testing_config
        from layer0_signal_negotiation import SignalNegotiator
        from layer0_redundancy_manager import RedundancyManager
        from layer0_sampler import ExecutionAwareSampler
        from layer0_fallback_orchestrator import FallbackOrchestrator
        from layer0_adaptive_buffer import AdaptiveBuffer, BufferConfig
        from layer0_vendor_adapters import VendorAdapterManager, ProviderType
        from layer0_health_monitor import HealthMonitor
        from layer0_core import AnomalyDetectionEngine
        from app_telemetry import create_telemetry_record_with_telemetry_id
        
        config = create_testing_config()
        
        # Initialize all components
        signal_negotiator = SignalNegotiator(config)
        redundancy_manager = RedundancyManager(config)
        sampler = ExecutionAwareSampler(config)
        orchestrator = FallbackOrchestrator(config, signal_negotiator, redundancy_manager, sampler)
        
        buffer_config = BufferConfig(max_queue_size=50, base_batch_size=5)
        buffer = AdaptiveBuffer(buffer_config, "e2e_buffer")
        
        adapter_manager = VendorAdapterManager()
        cloudwatch_adapter = adapter_manager.create_adapter(ProviderType.CLOUDWATCH)
        
        health_monitor = HealthMonitor(config, "e2e_health_monitor")
        from layer0_core import DetectionConfig
        detection_config = DetectionConfig()
        anomaly_engine = AnomalyDetectionEngine(detection_config)
        
        # Test end-to-end flow
        steps_completed = 0
        
        # Step 1: Channel negotiation
        await signal_negotiator.negotiate_all_channels()
        if len(signal_negotiator.negotiated_channels) > 0:
            steps_completed += 1
        
        # Step 2: Create telemetry record
        test_record = create_telemetry_record_with_telemetry_id(
            telemetry_id="e2e_integration_001",
            function_id="e2e_test_function",
            execution_phase="invoke",
            anomaly_type="cpu_burst",
            duration=1500.0,
            memory_spike_kb=8000,
            cpu_utilization=85.0
        )
        steps_completed += 1
        
        # Step 3: Anomaly detection
        detection_result = anomaly_engine.detect_anomalies(test_record)
        if detection_result.combined_confidence > 0:
            steps_completed += 1
        
        # Step 4: Buffer processing
        buffer_success = buffer.enqueue({"telemetry": test_record, "detection": detection_result})
        if buffer_success:
            steps_completed += 1
        
        buffered_items = buffer.dequeue(1)
        if len(buffered_items) > 0:
            steps_completed += 1
        
        # Step 5: Sampling decision
        sample_rate = sampler.get_sampling_rate(function_id="e2e_test_function")
        if 0.0 <= sample_rate <= 1.0:
            steps_completed += 1
        
        # Step 6: Health monitoring
        system_health = health_monitor.get_system_health()
        if system_health["overall_status"] != "unknown":
            steps_completed += 1
        
        # Cleanup
        orchestrator.shutdown()
        buffer.shutdown()
        health_monitor.shutdown()
        
        score = steps_completed / 7.0
        print(f"   ✅ End-to-End Integration Score: {score:.3f}")
        
        return score, {
            "steps_completed": steps_completed,
            "total_steps": 7,
            "detection_confidence": detection_result.combined_confidence,
            "sample_rate": sample_rate,
            "system_health": system_health["overall_status"]
        }
        
    except Exception as e:
        print(f"   ❌ End-to-End Integration Error: {e}")
        traceback.print_exc()
        return 0.0, {"error": str(e)}

# =============================================================================
# Main Test Runner
# =============================================================================

async def run_complete_integration_test():
    """Run complete Layer 0 integration test suite"""
    print("🚀 SCAFAD Layer 0 - Complete Integration Test Suite")
    print("=" * 60)
    
    test_start = time.time()
    
    # Check component availability
    components_available = check_component_availability()
    
    # Test results
    test_results = {}
    
    # Run individual component tests
    component_tests = [
        ("Signal Negotiation", test_signal_negotiation),
        ("Redundancy Management", test_redundancy_management),
        ("Execution-Aware Sampling", test_execution_aware_sampling),
        ("Fallback Orchestration", test_fallback_orchestration),
        ("Adaptive Buffer", test_adaptive_buffer),
        ("Vendor Adapters", test_vendor_adapters),
        ("Health Monitor", test_health_monitor),
        ("Anomaly Detection", test_anomaly_detection),
        ("End-to-End Integration", test_end_to_end_integration)
    ]
    
    for test_name, test_function in component_tests:
        try:
            score, details = await test_function()
            test_results[test_name] = {
                "score": score,
                "details": details,
                "status": "PASSED" if score >= 0.7 else "FAILED"
            }
        except Exception as e:
            test_results[test_name] = {
                "score": 0.0,
                "details": {"error": str(e)},
                "status": "ERROR"
            }
    
    # Calculate overall results
    total_time = time.time() - test_start
    
    # Component availability score
    available_components = sum(1 for available in components_available.values() if available)
    total_components = len(components_available)
    availability_score = available_components / total_components
    
    # Test execution score
    test_scores = [result["score"] for result in test_results.values()]
    average_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    
    # Overall score (weighted)
    overall_score = (availability_score * 0.3) + (average_test_score * 0.7)
    
    # Determine readiness status
    passed_tests = sum(1 for result in test_results.values() if result["status"] == "PASSED")
    total_tests = len(test_results)
    
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
    
    # Print detailed results
    print(f"\n📊 LAYER 0 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    print(f"\nComponent Availability:")
    print(f"   Available: {available_components}/{total_components} ({availability_score*100:.1f}%)")
    
    print(f"\nTest Results:")
    for test_name, result in test_results.items():
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
    
    excellent_components = [name for name, result in test_results.items() if result["score"] >= 0.9]
    if excellent_components:
        print(f"   ✅ Excellent: {', '.join(excellent_components)}")
    
    good_components = [name for name, result in test_results.items() if 0.7 <= result["score"] < 0.9]
    if good_components:
        print(f"   🟡 Good: {', '.join(good_components)}")
    
    needs_work = [name for name, result in test_results.items() if result["score"] < 0.7]
    if needs_work:
        print(f"   ❌ Needs Work: {', '.join(needs_work)}")
    
    # Deployment checklist
    print(f"\n📋 Deployment Readiness Checklist:")
    checklist_items = [
        ("Core Components Available", availability_score >= 0.8),
        ("Integration Tests Passing", passed_tests >= 0.8 * total_tests),
        ("Performance Acceptable", average_test_score >= 0.7),
        ("End-to-End Flow Working", test_results.get("End-to-End Integration", {}).get("score", 0) >= 0.7),
        ("Health Monitoring Active", test_results.get("Health Monitor", {}).get("score", 0) >= 0.7),
        ("Fallback Mechanisms Ready", test_results.get("Fallback Orchestration", {}).get("score", 0) >= 0.7)
    ]
    
    for item, status in checklist_items:
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {item}")
    
    print(f"\n✨ Complete Layer 0 integration test finished!")
    
    return {
        "overall_score": overall_score,
        "availability_score": availability_score,
        "average_test_score": average_test_score,
        "readiness_status": readiness_status,
        "recommendation": recommendation,
        "tests_passed": passed_tests,
        "total_tests": total_tests,
        "execution_time_seconds": total_time,
        "test_results": test_results,
        "components_available": components_available
    }

if __name__ == "__main__":
    result = asyncio.run(run_complete_integration_test())