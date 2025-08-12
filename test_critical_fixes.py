#!/usr/bin/env python3
"""
Test script to verify all critical fixes are working
"""

import sys
import traceback

sys.path.insert(0, '/workspace')

def test_imports():
    """Test that all critical imports work"""
    print("🧪 Testing Critical Import Fixes...")
    
    try:
        # Test defaultdict import in app_telemetry
        from app_telemetry import MultiChannelTelemetry
        print("   ✅ app_telemetry defaultdict import: Fixed")
        
        # Test time import in app_config
        from app_config import create_testing_config
        print("   ✅ app_config time import: Fixed")
        
        # Test TelemetrySource enum values
        from app_telemetry import TelemetrySource
        assert hasattr(TelemetrySource, 'FALLBACK_GENERATOR')
        print("   ✅ TelemetrySource.FALLBACK_GENERATOR: Available")
        
        return True
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_class_initialization():
    """Test that classes can be initialized without missing attributes"""
    print("\n🏗️ Testing Class Initialization...")
    
    try:
        # Test MultiChannelTelemetry can be created and has required attributes
        from app_config import create_testing_config
        from app_telemetry import MultiChannelTelemetry
        
        config = create_testing_config()
        telemetry = MultiChannelTelemetry(config)
        
        # Check for previously missing attributes
        assert hasattr(telemetry, 'channel_health'), "Missing channel_health attribute"
        assert hasattr(telemetry, 'performance_history'), "Missing performance_history attribute"
        assert isinstance(telemetry.channel_health, dict), "channel_health should be dict"
        assert isinstance(telemetry.performance_history, list), "performance_history should be list"
        
        print("   ✅ MultiChannelTelemetry initialization: Fixed")
        
        return True
    except Exception as e:
        print(f"   ❌ Class initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_method_existence():
    """Test that required methods exist and are callable"""
    print("\n🔧 Testing Method Existence...")
    
    try:
        from app_telemetry import TelemetryGenerator, TelemetryConfig
        
        # Create TelemetryGenerator
        config = TelemetryConfig()
        generator = TelemetryGenerator(config)
        
        # Check that create_normal_telemetry method exists (not generate_telemetry)
        assert hasattr(generator, 'create_normal_telemetry'), "Missing create_normal_telemetry method"
        assert callable(generator.create_normal_telemetry), "create_normal_telemetry should be callable"
        
        print("   ✅ TelemetryGenerator.create_normal_telemetry: Available")
        
        return True
    except Exception as e:
        print(f"   ❌ Method existence test failed: {e}")
        traceback.print_exc()
        return False

def test_no_duplicate_methods():
    """Test that duplicate methods have been resolved"""
    print("\n🔍 Testing Method Conflicts...")
    
    try:
        from app_telemetry import MultiChannelTelemetry
        from app_config import create_testing_config
        
        config = create_testing_config()
        telemetry = MultiChannelTelemetry(config)
        
        # Check that both methods exist with different names
        assert hasattr(telemetry, '_emit_to_channel'), "_emit_to_channel should exist"
        assert hasattr(telemetry, '_emit_to_channel_with_timeout'), "_emit_to_channel_with_timeout should exist"
        
        print("   ✅ Method naming conflicts: Resolved")
        
        return True
    except Exception as e:
        print(f"   ❌ Method conflict test failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality to ensure fixes don't break core features"""
    print("\n⚙️ Testing Basic Functionality...")
    
    try:
        from app_config import create_testing_config
        from app_telemetry import MultiChannelTelemetry, TelemetryGenerator
        
        # Create config and components
        config = create_testing_config()
        telemetry = MultiChannelTelemetry(config)
        
        # Test performance metrics access (previously would fail due to missing attributes)
        metrics = telemetry.get_performance_metrics()
        assert isinstance(metrics, dict), "Performance metrics should be dict"
        assert 'channel_health_score' in metrics, "Should have channel_health_score"
        assert 'average_latency_ms' in metrics, "Should have average_latency_ms"
        
        print("   ✅ Basic functionality: Working")
        
        return True
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all critical fix tests"""
    print("🚀 SCAFAD Layer 0 - Critical Fixes Verification")
    print("=" * 60)
    
    tests = [
        ("Import Fixes", test_imports),
        ("Class Initialization", test_class_initialization),
        ("Method Existence", test_method_existence),
        ("Method Conflicts", test_no_duplicate_methods),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n❌ {test_name}: EXCEPTION - {e}")
    
    print(f"\n" + "=" * 60)
    print("📊 CRITICAL FIXES VERIFICATION RESULTS")
    print("=" * 60)
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        status = "🟢 ALL FIXES VERIFIED"
        recommendation = "Layer 0 critical issues have been resolved"
        ready = True
    elif passed >= total * 0.8:
        status = "🟡 MOSTLY FIXED"
        recommendation = "Most critical issues resolved, minor issues remain"
        ready = False
    else:
        status = "🔴 FIXES INCOMPLETE"
        recommendation = "Significant issues remain, additional fixes needed"
        ready = False
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    print(f"Ready for Deployment: {'Yes' if ready else 'No'}")
    
    if ready:
        print(f"\n🎉 All critical fixes have been successfully applied!")
        print(f"   SCAFAD Layer 0 is now ready for deployment.")
    else:
        print(f"\n🚧 Additional fixes are needed before deployment.")
        print(f"   Review failed tests and apply necessary corrections.")
    
    return ready

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)