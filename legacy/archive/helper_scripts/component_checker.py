#!/usr/bin/env python3
"""
Component Availability Checker for Layer 0
Checks if all Layer 0 components are available and importable
"""
import sys
sys.path.insert(0, '/workspace')

def check_layer0_components():
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
        ('layer0_privacy_compliance', 'PrivacyCompliancePipeline', 'Privacy Compliance'),
        ('layer0_l1_contract', 'L0L1ContractManager', 'L0-L1 Contract'),
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
    
    return components, available_count, total_count

if __name__ == "__main__":
    components, available_count, total_count = check_layer0_components()
    
    print(f"\n💡 Component Status Summary:")
    print(f"   Available Components: {available_count}")
    print(f"   Total Components: {total_count}")
    print(f"   Availability Rate: {available_count/total_count*100:.1f}%")
    
    if available_count == total_count:
        print("   🟢 All components are available!")
    elif available_count >= 0.8 * total_count:
        print("   🟡 Most components are available")
    else:
        print("   🔴 Several components are missing")
        
    # List missing components
    missing = [name for name, available in components.items() if not available]
    if missing:
        print(f"\n❌ Missing Components: {', '.join(missing)}")