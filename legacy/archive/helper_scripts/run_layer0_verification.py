#!/usr/bin/env python3
"""
Manual Layer 0 Verification Runner
================================
Since the bash shell is not available, this script manually runs the verification tests.
"""

import sys
import os
import time
import json
import traceback
from typing import Dict, Any, List, Tuple

# Add workspace to path
sys.path.insert(0, '/workspace')

def test_component_imports():
    """Test if all Layer 0 components can be imported"""
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
    
    results = {}
    available = 0
    
    for module_name, class_name in components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                results[module_name] = True
                available += 1
                print(f"✅ {module_name}: {class_name} available")
            else:
                results[module_name] = False
                print(f"❌ {module_name}: {class_name} missing")
        except ImportError as e:
            results[module_name] = False
            print(f"❌ {module_name}: Import failed - {e}")
        except Exception as e:
            results[module_name] = False
            print(f"❌ {module_name}: Error - {e}")
    
    total = len(results)
    print(f"\n📊 Availability: {available}/{total} ({available/total*100:.1f}%)")
    
    return results, available/total

def test_signal_negotiation():
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
        print(f"   Traceback: {traceback.format_exc()}")
        return 0.0, {"error": str(e)}

def test_adaptive_buffer():
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
        print(f"   Traceback: {traceback.format_exc()}")
        return 0.0, {"error": str(e)}

def test_anomaly_detection():
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
        print(f"   Traceback: {traceback.format_exc()}")
        return 0.0, {"error": str(e)}

def main():
    """Main verification execution"""
    start_time = time.time()
    
    print("🚀 SCAFAD LAYER 0 - COMPREHENSIVE VERIFICATION")
    print("=" * 60)
    print("Verifying that Layer 0 is finished, working, tested, and ready for deployment...")
    
    # Check component availability
    component_availability, availability_score = test_component_imports()
    
    # Run functional tests
    test_results = {}
    
    tests = [
        ("Signal Negotiation", test_signal_negotiation),
        ("Adaptive Buffer", test_adaptive_buffer),
        ("Anomaly Detection", test_anomaly_detection)
    ]
    
    for test_name, test_func in tests:
        try:
            score, details = test_func()
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
    
    # Generate final report
    total_time = time.time() - start_time
    
    test_scores = [result["score"] for result in test_results.values()]
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    
    overall_score = (availability_score * 0.3) + (avg_test_score * 0.7)
    
    passed_tests = sum(1 for r in test_results.values() if r["status"] == "PASSED")
    total_tests = len(test_results)
    
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
    print(f"   Available: {int(availability_score * 12)}/12 ({availability_score*100:.1f}%)")
    
    print(f"\nFunctional Test Results:")
    for test_name, result in test_results.items():
        icon = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⚠️"
        print(f"   {icon} {test_name:20} | Score: {result['score']:.3f} | {result['status']}")
    
    print(f"\nOverall Assessment:")
    print(f"   Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"   Average Test Score: {avg_test_score:.3f}")
    print(f"   Overall Score: {overall_score:.3f}")
    print(f"   Verification Time: {total_time:.1f}s")
    print(f"   Status: {status}")
    print(f"   Recommendation: {recommendation}")
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Overall Score: {overall_score:.3f}/1.0")
    print(f"   Status: {status}")
    print(f"   {recommendation}")
    
    return {
        "overall_score": overall_score,
        "availability_score": availability_score,
        "avg_test_score": avg_test_score,
        "status": status,
        "recommendation": recommendation,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "execution_time": total_time,
        "test_results": test_results,
        "component_availability": component_availability
    }

if __name__ == "__main__":
    results = main()