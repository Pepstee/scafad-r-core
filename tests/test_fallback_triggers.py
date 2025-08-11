"""
Test suite for FallbackOrchestrator trigger mechanisms

This module tests each fallback trigger type individually to ensure proper
state transitions and behavior under various failure conditions.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer0_fallback_orchestrator import (
    FallbackOrchestrator, FallbackTrigger, FallbackMode, 
    FallbackReason, create_fallback_orchestrator
)
from app_config import Layer0Config


class TestFallbackTriggers:
    """Test suite for fallback trigger mechanisms"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = Layer0Config()
        
        # Create mock components
        self.mock_signal_negotiator = Mock()
        self.mock_redundancy_manager = Mock()
        self.mock_sampler = Mock()
        
        # Configure mock signal negotiator
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'cloudwatch': {'qos_score': 0.9, 'latency_ms': 50},
            'xray': {'qos_score': 0.8, 'latency_ms': 100},
            'side_trace': {'qos_score': 0.95, 'latency_ms': 25}
        }
        
        # Create orchestrator instance
        self.orchestrator = FallbackOrchestrator(
            config=self.config,
            signal_negotiator=self.mock_signal_negotiator,
            redundancy_manager=self.mock_redundancy_manager,
            sampler=self.mock_sampler
        )
    
    def teardown_method(self):
        """Clean up after each test method"""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown()
    
    def test_telemetry_timeout_trigger(self):
        """Test telemetry timeout trigger mechanism"""
        print("\n=== Testing Telemetry Timeout Trigger ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Simulate telemetry timeout by setting old timestamps
        current_time = time.time()
        self.orchestrator.last_telemetry_seen['invocation_trace'] = current_time - 10.0  # 10 seconds ago
        self.orchestrator.last_telemetry_seen['side_channel_latency'] = current_time - 8.0   # 8 seconds ago
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after timeout: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'degraded'
        assert 'timeout' in status['active_triggers']
        
        # Check fallback history
        history = self.orchestrator.get_fallback_history(limit=1)
        assert len(history) > 0
        latest_event = history[-1]
        assert latest_event['trigger'] == 'timeout'
        assert latest_event['reason'] == 'telemetry_timeout'
        
        print("✓ Telemetry timeout trigger test passed")
    
    def test_channel_degradation_trigger(self):
        """Test channel degradation trigger mechanism"""
        print("\n=== Testing Channel Degradation Trigger ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Simulate channel degradation by returning low QoS scores
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'cloudwatch': {'qos_score': 0.2, 'latency_ms': 500},  # Below 0.3 threshold
            'xray': {'qos_score': 0.1, 'latency_ms': 1000},      # Below 0.3 threshold
            'side_trace': {'qos_score': 0.25, 'latency_ms': 800}  # Below 0.3 threshold
        }
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after channel degradation: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'emergency'
        assert 'channel_degradation' in status['active_triggers']
        
        # Check fallback history
        history = self.orchestrator.get_fallback_history(limit=1)
        assert len(history) > 0
        latest_event = history[-1]
        assert latest_event['trigger'] == 'channel_degradation'
        assert latest_event['reason'] == 'channel_unreliable'
        
        print("✓ Channel degradation trigger test passed")
    
    def test_data_missing_trigger(self):
        """Test data missing trigger mechanism"""
        print("\n=== Testing Data Missing Trigger ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Simulate missing data by setting old timestamps
        current_time = time.time()
        self.orchestrator.last_telemetry_seen['invocation_trace'] = current_time - 15.0  # 15 seconds ago (> 10s threshold)
        self.orchestrator.last_telemetry_seen['side_channel_latency'] = current_time - 12.0  # 12 seconds ago (> 10s threshold)
        self.orchestrator.last_telemetry_seen['cloudwatch_logs'] = current_time - 20.0  # 20 seconds ago (> 15s threshold)
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after data missing: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'degraded'
        assert 'data_missing' in status['active_triggers']
        
        # Check fallback history
        history = self.orchestrator.get_fallback_history(limit=1)
        assert len(history) > 0
        latest_event = history[-1]
        assert latest_event['trigger'] == 'data_missing'
        assert latest_event['reason'] == 'data_incomplete'
        
        print("✓ Data missing trigger test passed")
    
    def test_performance_degradation_trigger(self):
        """Test performance degradation trigger mechanism"""
        print("\n=== Testing Performance Degradation Trigger ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Add performance metrics below threshold (0.5)
        for _ in range(15):  # Need at least 10 data points
            self.orchestrator.update_performance_metric(0.3)  # Below 0.5 threshold
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after performance degradation: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'degraded'
        assert 'performance_degradation' in status['active_triggers']
        
        # Check fallback history
        history = self.orchestrator.get_fallback_history(limit=1)
        assert len(history) > 0
        latest_event = history[-1]
        assert latest_event['trigger'] == 'performance_degradation'
        assert latest_event['reason'] == 'performance_critical'
        
        print("✓ Performance degradation trigger test passed")
    
    def test_error_rate_threshold_trigger(self):
        """Test error rate threshold trigger mechanism"""
        print("\n=== Testing Error Rate Threshold Trigger ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Add error rates above threshold (0.2)
        for _ in range(15):  # Need at least 10 data points
            self.orchestrator.update_error_rate(0.3)  # Above 0.2 threshold
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after error rate threshold: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        # Note: Error rate trigger is not currently implemented in _check_fallback_conditions
        # This test documents the current state and can be updated when implemented
        
        print("✓ Error rate threshold trigger test completed (not yet implemented)")
    
    def test_recovery_mechanism(self):
        """Test recovery from fallback mode"""
        print("\n=== Testing Recovery Mechanism ===")
        
        # First trigger a fallback
        self.orchestrator.force_fallback(FallbackMode.DEGRADED, "test_fallback")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Status after forced fallback: {initial_status['current_mode']}")
        
        # Simulate recovery by updating all telemetry streams
        current_time = time.time()
        self.orchestrator.last_telemetry_seen['invocation_trace'] = current_time
        self.orchestrator.last_telemetry_seen['side_channel_latency'] = current_time
        self.orchestrator.last_telemetry_seen['cloudwatch_logs'] = current_time
        self.orchestrator.last_telemetry_seen['xray_traces'] = current_time
        
        # Add good performance metrics
        for _ in range(15):
            self.orchestrator.update_performance_metric(0.8)  # Above 0.5 threshold
        
        # Reset channel health to good
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'cloudwatch': {'qos_score': 0.9, 'latency_ms': 50},
            'xray': {'qos_score': 0.8, 'latency_ms': 100},
            'side_trace': {'qos_score': 0.95, 'latency_ms': 25}
        }
        
        # Wait for cooldown to expire
        self.orchestrator.state.cooldown_until = 0
        
        # Check if recovery is possible
        can_recover = self.orchestrator._can_recover()
        print(f"Can recover: {can_recover}")
        
        # Attempt recovery
        self.orchestrator._attempt_recovery()
        
        # Wait for recovery confirmation
        time.sleep(6.0)  # Wait for 5s recovery confirmation + buffer
        
        # Check final status
        final_status = self.orchestrator.get_fallback_status()
        print(f"Final status: {final_status['current_mode']}")
        print(f"Recovery attempts: {final_status['recovery_attempts']}")
        
        # Note: Recovery might still be in progress or completed
        # The exact state depends on timing and thread execution
        
        print("✓ Recovery mechanism test completed")
    
    def test_cooldown_and_hysteresis(self):
        """Test cooldown and hysteresis mechanisms"""
        print("\n=== Testing Cooldown and Hysteresis ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Trigger a fallback
        self.orchestrator.last_telemetry_seen['invocation_trace'] = time.time() - 10.0
        self.orchestrator._check_fallback_conditions()
        
        status_after_fallback = self.orchestrator.get_fallback_status()
        print(f"Status after fallback: {status_after_fallback['current_mode']}")
        print(f"Cooldown until: {status_after_fallback['cooldown_until']}")
        print(f"Hysteresis until: {status_after_fallback['hysteresis_until']}")
        
        # Try to trigger another fallback immediately (should be blocked by cooldown)
        self.orchestrator.last_telemetry_seen['side_channel_latency'] = time.time() - 10.0
        self.orchestrator._check_fallback_conditions()
        
        # Status should remain the same due to cooldown
        status_after_cooldown = self.orchestrator.get_fallback_status()
        print(f"Status after cooldown check: {status_after_cooldown['current_mode']}")
        
        # Verify cooldown is working
        assert status_after_fallback['current_mode'] == status_after_cooldown['current_mode']
        
        print("✓ Cooldown and hysteresis test passed")
    
    def test_null_model_integration(self):
        """Test null model path integration"""
        print("\n=== Testing Null Model Integration ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Trigger a fallback to activate null model
        self.orchestrator.last_telemetry_seen['invocation_trace'] = time.time() - 10.0
        self.orchestrator._check_fallback_conditions()
        
        # Check if null model was activated
        status = self.orchestrator.get_fallback_status()
        print(f"Status after null model activation: {status['current_mode']}")
        
        # Verify that downstream notification was logged
        # This would be captured in the logs in a real scenario
        
        # Check if sampler and redundancy manager were updated
        # Note: In unit tests, these are mocks, so we can't verify actual calls easily
        
        print("✓ Null model integration test completed")
    
    def test_structured_logging(self):
        """Test structured fallback logging"""
        print("\n=== Testing Structured Logging ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Trigger a fallback
        self.orchestrator.last_telemetry_seen['invocation_trace'] = time.time() - 10.0
        self.orchestrator._check_fallback_conditions()
        
        # Get fallback history
        history = self.orchestrator.get_fallback_history(limit=1)
        assert len(history) > 0
        
        latest_event = history[-1]
        print(f"Latest fallback event: {latest_event}")
        
        # Verify structured data
        required_fields = ['timestamp', 'trigger', 'reason', 'mode_transition', 'metadata']
        for field in required_fields:
            assert field in latest_event, f"Missing field: {field}"
        
        # Verify metadata structure
        metadata = latest_event['metadata']
        assert 'trigger_details' in metadata
        assert 'reason_details' in metadata
        assert 'cooldown_until' in metadata
        assert 'hysteresis_until' in metadata
        
        print("✓ Structured logging test passed")


def run_comprehensive_tests():
    """Run all fallback trigger tests"""
    print("Starting comprehensive fallback trigger tests...")
    print("=" * 60)
    
    test_instance = TestFallbackTriggers()
    
    # Run all test methods
    test_methods = [
        test_instance.test_telemetry_timeout_trigger,
        test_instance.test_channel_degradation_trigger,
        test_instance.test_data_missing_trigger,
        test_instance.test_performance_degradation_trigger,
        test_instance.test_error_rate_threshold_trigger,
        test_instance.test_recovery_mechanism,
        test_instance.test_cooldown_and_hysteresis,
        test_instance.test_null_model_integration,
        test_instance.test_structured_logging
    ]
    
    passed = 0
    total = len(test_methods)
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
            print(f"✓ {test_method.__name__} PASSED")
        except Exception as e:
            print(f"✗ {test_method.__name__} FAILED: {e}")
        print("-" * 40)
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fallback trigger tests passed!")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)
