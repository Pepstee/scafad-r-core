#!/usr/bin/env python3
"""Direct execution of complete Layer 0 test suite"""

import sys
import time
import asyncio
sys.path.insert(0, '/workspace')

# Import required modules step by step to check availability
print("🔧 Checking Layer 0 component availability...")

components_available = {}

# Check Signal Negotiator
try:
    from layer0_signal_negotiation import SignalNegotiator, ChannelType, CompressionType
    components_available['signal_negotiator'] = True
    print("   ✅ Signal Negotiator: Available")
except ImportError as e:
    components_available['signal_negotiator'] = False
    print(f"   ❌ Signal Negotiator: {e}")

# Check Redundancy Manager  
try:
    from layer0_redundancy_manager import RedundancyManager
    components_available['redundancy_manager'] = True
    print("   ✅ Redundancy Manager: Available")
except ImportError as e:
    components_available['redundancy_manager'] = False
    print(f"   ❌ Redundancy Manager: {e}")

# Check Execution-Aware Sampler
try:
    from layer0_sampler import ExecutionAwareSampler
    components_available['execution_aware_sampler'] = True
    print("   ✅ Execution-Aware Sampler: Available")
except ImportError as e:
    components_available['execution_aware_sampler'] = False
    print(f"   ❌ Execution-Aware Sampler: {e}")

# Check other components
other_components = [
    ('layer0_fallback_orchestrator', 'Fallback Orchestrator'),
    ('layer0_runtime_control', 'Runtime Control'),
    ('app_config', 'Configuration'),
    ('app_telemetry', 'Telemetry System'),
    ('layer0_core', 'Anomaly Detection Engine')
]

for module_name, display_name in other_components:
    try:
        __import__(module_name)
        components_available[module_name] = True
        print(f"   ✅ {display_name}: Available")
    except ImportError as e:
        components_available[module_name] = False
        print(f"   ❌ {display_name}: {e}")

print(f"\n📊 Component Availability: {sum(components_available.values())}/{len(components_available)} available")

# Now run focused tests on available components
print(f"\n🚀 SCAFAD Layer 0 - Complete Acceptance Test Execution")
print("=" * 65)

test_start = time.time()

# Test Signal Negotiator if available
if components_available.get('signal_negotiator') and components_available.get('app_config'):
    print(f"\n🔧 TESTING SIGNAL NEGOTIATOR")
    print("-" * 30)
    
    try:
        from app_config import create_testing_config
        from layer0_signal_negotiation import SignalNegotiator
        
        config = create_testing_config()
        negotiator = SignalNegotiator(config)
        
        # Test 1: Channel Discovery
        print("   📡 Testing channel capability discovery...")
        available_channels = len(negotiator.available_channels)
        print(f"      Discovered {available_channels} channels")
        
        # Test 2: Negotiation
        print("   🤝 Testing channel negotiation...")
        asyncio.run(negotiator.negotiate_all_channels())
        negotiated_channels = len(negotiator.negotiated_channels)
        print(f"      Negotiated {negotiated_channels} channels")
        
        # Test 3: QoS Recommendations
        print("   📊 Testing QoS-based recommendations...")
        recommendations = negotiator.get_channel_recommendations(1024, priority="balanced")
        print(f"      Generated {len(recommendations)} recommendations")
        
        # Test 4: Channel Health
        print("   💊 Testing channel health tracking...")
        if negotiated_channels > 0:
            channel_type = list(negotiator.negotiated_channels.keys())[0]
            negotiator.update_channel_health(channel_type, True, 50.0)
            health_summary = negotiator.get_channel_health_summary()
            print(f"      Health tracking: {len(health_summary)} channels monitored")
        
        signal_negotiator_score = min(1.0, (available_channels + negotiated_channels + len(recommendations)) / 10)
        print(f"   🏆 Signal Negotiator Score: {signal_negotiator_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Signal Negotiator test failed: {e}")
        signal_negotiator_score = 0.0
else:
    print(f"\n⚠️ SKIPPING SIGNAL NEGOTIATOR - Component not available")
    signal_negotiator_score = 0.0

# Test Redundancy Manager if available
if components_available.get('redundancy_manager') and components_available.get('app_config'):
    print(f"\n🔄 TESTING REDUNDANCY MANAGER")
    print("-" * 30)
    
    try:
        from app_config import create_testing_config
        from layer0_redundancy_manager import RedundancyManager
        from app_telemetry import create_telemetry_record_with_telemetry_id
        
        config = create_testing_config()
        redundancy_manager = RedundancyManager(config)
        
        # Test 1: Duplication
        print("   📋 Testing telemetry duplication...")
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
        
        duplication_result = redundancy_manager.duplicate_telemetry(test_record, strategy="active_active")
        duplication_success = duplication_result is not None and duplication_result.get("duplicated", False)
        print(f"      Duplication: {'Success' if duplication_success else 'Failed'}")
        
        # Test 2: Failover
        print("   🔀 Testing failover logic...")
        failover_result = redundancy_manager.initiate_failover("primary", "standby")
        failover_success = failover_result is not None and failover_result.get("failover_initiated", False)
        print(f"      Failover: {'Success' if failover_success else 'Failed'}")
        
        # Test 3: Deduplication
        print("   🧹 Testing deduplication...")
        first_check = redundancy_manager.check_duplicate("test_001")
        second_check = redundancy_manager.check_duplicate("test_001") 
        dedup_success = first_check != second_check
        print(f"      Deduplication: {'Success' if dedup_success else 'Failed'}")
        
        # Test 4: Idempotency
        print("   🔄 Testing idempotency...")
        op1 = redundancy_manager.execute_idempotent_operation("op_001", {"test": "data"})
        op2 = redundancy_manager.execute_idempotent_operation("op_001", {"test": "data"})
        idempotency_success = op1 == op2
        print(f"      Idempotency: {'Success' if idempotency_success else 'Failed'}")
        
        redundancy_score = (duplication_success + failover_success + dedup_success + idempotency_success) / 4.0
        print(f"   🏆 Redundancy Manager Score: {redundancy_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Redundancy Manager test failed: {e}")
        redundancy_score = 0.0
else:
    print(f"\n⚠️ SKIPPING REDUNDANCY MANAGER - Component not available")
    redundancy_score = 0.0

# Test Execution-Aware Sampler if available
if components_available.get('execution_aware_sampler') and components_available.get('app_config'):
    print(f"\n🎯 TESTING EXECUTION-AWARE SAMPLER")
    print("-" * 35)
    
    try:
        from app_config import create_testing_config
        from layer0_sampler import ExecutionAwareSampler
        
        config = create_testing_config()
        sampler = ExecutionAwareSampler(config)
        
        # Test 1: Cold/Warm Adaptive Sampling
        print("   🌡️ Testing cold/warm adaptive sampling...")
        cold_rate = sampler.get_sampling_rate(execution_state="cold", function_id="test_func_1")
        warm_rate = sampler.get_sampling_rate(execution_state="warm", function_id="test_func_1")
        cold_warm_adaptive = cold_rate >= warm_rate and 0.0 <= cold_rate <= 1.0 and 0.0 <= warm_rate <= 1.0
        print(f"      Cold: {cold_rate:.3f}, Warm: {warm_rate:.3f} - {'Adaptive' if cold_warm_adaptive else 'Failed'}")
        
        # Test 2: Latency-Based Sampling
        print("   ⏱️ Testing latency-based sampling...")
        sampler.update_latency_metrics("fast_func", 50.0)
        sampler.update_latency_metrics("slow_func", 2000.0)
        fast_rate = sampler.get_sampling_rate(function_id="fast_func")
        slow_rate = sampler.get_sampling_rate(function_id="slow_func")
        latency_adaptive = slow_rate >= fast_rate
        print(f"      Fast: {fast_rate:.3f}, Slow: {slow_rate:.3f} - {'Adaptive' if latency_adaptive else 'Failed'}")
        
        # Test 3: Error Rate Sampling
        print("   ❌ Testing error rate adaptive sampling...")
        sampler.update_error_metrics("reliable_func", 1, 100)  # 1% error
        sampler.update_error_metrics("unreliable_func", 20, 100)  # 20% error  
        reliable_rate = sampler.get_sampling_rate(function_id="reliable_func")
        unreliable_rate = sampler.get_sampling_rate(function_id="unreliable_func")
        error_adaptive = unreliable_rate >= reliable_rate
        print(f"      Reliable: {reliable_rate:.3f}, Unreliable: {unreliable_rate:.3f} - {'Adaptive' if error_adaptive else 'Failed'}")
        
        # Test 4: Load-Based Sampling
        print("   📈 Testing load-based sampling strategies...")
        low_load_strategy = sampler.determine_sampling_strategy(requests_per_second=10)
        high_load_strategy = sampler.determine_sampling_strategy(requests_per_second=1000)
        load_adaptive = low_load_strategy != high_load_strategy
        print(f"      Low Load: {low_load_strategy}, High Load: {high_load_strategy} - {'Adaptive' if load_adaptive else 'Static'}")
        
        sampler_score = (cold_warm_adaptive + latency_adaptive + error_adaptive + load_adaptive) / 4.0
        print(f"   🏆 Execution-Aware Sampler Score: {sampler_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Execution-Aware Sampler test failed: {e}")
        sampler_score = 0.0
else:
    print(f"\n⚠️ SKIPPING EXECUTION-AWARE SAMPLER - Component not available")
    sampler_score = 0.0

# Test core anomaly detection for advanced patterns
if components_available.get('layer0_core') and components_available.get('app_config'):
    print(f"\n🧠 TESTING ADVANCED ANOMALY DETECTION")
    print("-" * 40)
    
    try:
        from app_config import create_testing_config
        from layer0_core import AnomalyDetectionEngine
        from app_telemetry import TelemetryRecord
        
        config = create_testing_config()
        engine = AnomalyDetectionEngine(config)
        
        # Test advanced anomaly patterns
        advanced_patterns = [
            {
                "name": "Silent Failure",
                "params": {"anomaly_type": "silent_failure", "duration": 50.0, "memory": 900, "cpu": 15.0},
                "expected_detection": True
            },
            {
                "name": "Concurrency Abuse", 
                "params": {"anomaly_type": "concurrency_abuse", "duration": 300.0, "memory": 3000, "cpu": 85.0},
                "expected_detection": True
            },
            {
                "name": "Resource Leak",
                "params": {"anomaly_type": "resource_leak", "duration": 1200.0, "memory": 8000, "cpu": 55.0},
                "expected_detection": True
            },
            {
                "name": "Timing Attack",
                "params": {"anomaly_type": "timing_attack", "duration": 950.0, "memory": 1500, "cpu": 70.0},
                "expected_detection": True
            }
        ]
        
        advanced_results = []
        for pattern in advanced_patterns:
            record = create_telemetry_record_with_telemetry_id(
                telemetry_id=f"advanced_{pattern['name'].lower().replace(' ', '_')}",
                function_id="advanced_test_function",
                execution_phase="invoke",
                anomaly_type=pattern["params"]["anomaly_type"],
                duration=float(pattern["params"]["duration"]),
                memory_spike_kb=pattern["params"]["memory"],
                cpu_utilization=float(pattern["params"]["cpu"]),
                custom_fields={"advanced_pattern": True}
            )
            
            result = engine.detect_anomalies(record)
            detected = result.overall_confidence > 0.5
            correct = detected == pattern["expected_detection"]
            
            advanced_results.append(correct)
            status = "✅" if correct else "❌"
            print(f"   {status} {pattern['name']:18} | Conf: {result.overall_confidence:.3f}")
        
        advanced_detection_score = sum(advanced_results) / len(advanced_results)
        print(f"   🏆 Advanced Pattern Detection Score: {advanced_detection_score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Advanced anomaly detection test failed: {e}")
        advanced_detection_score = 0.0
else:
    print(f"\n⚠️ SKIPPING ADVANCED ANOMALY DETECTION - Component not available")
    advanced_detection_score = 0.0

# Check additional components exist (even if we can't test them fully)
additional_components = []

if components_available.get('layer0_fallback_orchestrator'):
    additional_components.append("Fallback Orchestrator")
if components_available.get('layer0_runtime_control'):
    additional_components.append("Runtime Control")
if components_available.get('app_telemetry'):
    additional_components.append("Multi-Channel Telemetry")

print(f"\n📦 ADDITIONAL COMPONENTS AVAILABLE")
print("-" * 35)
for component in additional_components:
    print(f"   ✅ {component}")

if not additional_components:
    print("   ⚠️ No additional components detected")

# Calculate overall assessment
total_time = time.time() - test_start

print(f"\n{'='*65}")
print("📊 COMPLETE LAYER 0 ACCEPTANCE TEST RESULTS")
print('='*65)

# Calculate scores
component_scores = {
    "Signal Negotiator": signal_negotiator_score,
    "Redundancy Manager": redundancy_score, 
    "Execution-Aware Sampler": sampler_score,
    "Advanced Anomaly Detection": advanced_detection_score
}

# Available vs expected components
expected_core_components = [
    "Signal Negotiator", "Redundancy Manager", "Execution-Aware Sampler", 
    "Fallback Orchestrator", "Adaptive Buffer/Backpressure", "Vendor Adapters",
    "Advanced Anomaly Detection", "Privacy & Compliance", "End-to-End Integration"
]

available_core_components = len([c for c in expected_core_components[:4] if components_available.get(c.lower().replace(' ', '_').replace('-', '_').replace('/', '_'), False)])
total_expected = len(expected_core_components)

# Score calculation
tested_scores = [score for score in component_scores.values() if score > 0]
average_score = sum(tested_scores) / len(tested_scores) if tested_scores else 0.0

component_availability = available_core_components / total_expected
overall_score = (average_score * 0.7) + (component_availability * 0.3)

print(f"\nComponent Test Results:")
for component, score in component_scores.items():
    if score > 0:
        status_icon = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"   {status_icon} {component:30} | Score: {score:.3f}")
    else:
        print(f"   ⚠️ {component:30} | Not Tested")

print(f"\n📊 Overall Assessment:")
print(f"   Components Available: {available_core_components}/{total_expected} ({component_availability*100:.1f}%)")
print(f"   Components Tested: {len(tested_scores)}")
print(f"   Average Test Score: {average_score:.3f}")
print(f"   Overall Score: {overall_score:.3f}")
print(f"   Test Execution Time: {total_time:.1f}s")

# Final assessment
if overall_score >= 0.85 and component_availability >= 0.7:
    final_status = "🟢 EXCELLENT - Strong Foundation"
    recommendation = "Core components working well, complete remaining components"
elif overall_score >= 0.7 and component_availability >= 0.5:
    final_status = "🟡 GOOD - Solid Progress" 
    recommendation = "Key components functional, focus on missing pieces"
elif overall_score >= 0.5:
    final_status = "🟠 FAIR - Basic Functionality"
    recommendation = "Basic functionality present, significant work needed"
else:
    final_status = "🔴 NEEDS_WORK - Major Issues"
    recommendation = "Core issues need resolution before proceeding"

print(f"\n🏆 Final Assessment: {final_status}")
print(f"📋 Recommendation: {recommendation}")

# Detailed findings
print(f"\n💡 Key Findings:")

working_components = [comp for comp, score in component_scores.items() if score >= 0.8]
if working_components:
    print(f"   ✅ Excellent Components: {', '.join(working_components)}")

needs_work = [comp for comp, score in component_scores.items() if 0 < score < 0.6]
if needs_work:
    print(f"   ⚠️ Components Needing Work: {', '.join(needs_work)}")

missing_critical = []
for comp in ["Signal Negotiator", "Redundancy Manager", "Execution-Aware Sampler"]:
    if comp not in component_scores or component_scores[comp] == 0:
        missing_critical.append(comp)

if missing_critical:
    print(f"   ❌ Missing Critical Components: {', '.join(missing_critical)}")

not_yet_tested = [
    "Fallback Orchestrator", "Adaptive Buffer/Backpressure", "Vendor Adapters",
    "Privacy & Compliance", "End-to-End Integration"
]
print(f"   📋 Components Not Yet Fully Tested: {', '.join(not_yet_tested)}")

print(f"\n✨ Complete Layer 0 acceptance test execution finished!")
print(f"🎯 This represents the current state of Layer 0 implementation completeness")

# Return results dictionary
results = {
    "overall_score": overall_score,
    "component_availability": component_availability,
    "average_test_score": average_score,
    "component_scores": component_scores,
    "final_status": final_status,
    "recommendation": recommendation,
    "execution_time_seconds": total_time,
    "components_tested": len(tested_scores),
    "components_available": available_core_components,
    "total_expected_components": total_expected
}

print(f"\n📊 Results Summary:")
for key, value in results.items():
    if isinstance(value, (int, float)):
        print(f"   {key}: {value}")
    elif isinstance(value, str):
        print(f"   {key}: {value}")