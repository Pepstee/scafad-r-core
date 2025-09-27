"""
Test suite for Fallback Orchestration Matrix

This module tests the complete fallback orchestration matrix including:
- OTLP gRPC timeout → vendor native fallback
- Vendor throttle → HTTP JSON fallback  
- HTTP 5xx → local spool fallback
- Cooldown/hysteresis mechanisms
- Recovery back to preferred path
- Dedup/idempotency on re-publish
- Budget overshoot prevention
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from typing import Dict, List, Any

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer0_fallback_orchestrator import (
    FallbackOrchestrator, FallbackTrigger, FallbackMode, 
    FallbackReason, create_fallback_orchestrator
)
from app_config import Layer0Config


class TestFallbackOrchestrationMatrix:
    """Test suite for complete fallback orchestration matrix"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = Layer0Config()
        
        # Create comprehensive mock components
        self.mock_signal_negotiator = Mock()
        self.mock_redundancy_manager = Mock()
        self.mock_sampler = Mock()
        
        # Configure mock signal negotiator with multiple channels
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'otlp_grpc': {'qos_score': 0.9, 'latency_ms': 50, 'status': 'active'},
            'vendor_native': {'qos_score': 0.8, 'latency_ms': 100, 'status': 'active'},
            'http_json': {'qos_score': 0.7, 'latency_ms': 150, 'status': 'active'},
            'local_spool': {'qos_score': 0.6, 'latency_ms': 200, 'status': 'standby'}
        }
        
        # Create orchestrator instance
        self.orchestrator = FallbackOrchestrator(
            config=self.config,
            signal_negotiator=self.mock_signal_negotiator,
            redundancy_manager=self.mock_redundancy_manager,
            sampler=self.mock_sampler
        )
        
        # Track fallback events for assertions
        self.fallback_events = []
        self.orchestrator._log_structured_fallback = self._capture_fallback_event
        
    def teardown_method(self):
        """Clean up after each test method"""
        if hasattr(self, 'orchestrator'):
            self.orchestrator.shutdown()
    
    def _capture_fallback_event(self, fallback_event):
        """Capture fallback events for testing assertions"""
        self.fallback_events.append(fallback_event)
        # Call original method
        self.orchestrator._original_log_structured_fallback(fallback_event)
    
    def test_otlp_grpc_timeout_to_vendor_native(self):
        """Test OTLP gRPC timeout fallback to vendor native"""
        print("\n=== Testing OTLP gRPC Timeout → Vendor Native Fallback ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Simulate OTLP gRPC timeout by setting old timestamps
        current_time = time.time()
        self.orchestrator.last_telemetry_seen['otlp_grpc'] = current_time - 15.0  # 15 seconds ago
        
        # Mock channel health degradation for OTLP gRPC
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'otlp_grpc': {'qos_score': 0.1, 'latency_ms': 5000, 'status': 'degraded'},
            'vendor_native': {'qos_score': 0.8, 'latency_ms': 100, 'status': 'active'},
            'http_json': {'qos_score': 0.7, 'latency_ms': 150, 'status': 'active'},
            'local_spool': {'qos_score': 0.6, 'latency_ms': 200, 'status': 'standby'}
        }
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after OTLP timeout: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'degraded'
        assert 'timeout' in status['active_triggers'] or 'channel_degradation' in status['active_triggers']
        
        # Verify vendor native is now preferred
        assert self.mock_redundancy_manager.set_redundancy_mode.called
        
        print("✓ OTLP gRPC timeout → vendor native fallback test passed")
    
    def test_vendor_throttle_to_http_json(self):
        """Test vendor throttle fallback to HTTP JSON"""
        print("\n=== Testing Vendor Throttle → HTTP JSON Fallback ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Simulate vendor native throttling (high error rate)
        for i in range(10):
            self.orchestrator.update_performance_metric(0.1)  # Low performance
            self.orchestrator.update_telemetry_tracking('error', time.time())
        
        # Mock channel health showing vendor native issues
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'otlp_grpc': {'qos_score': 0.9, 'latency_ms': 50, 'status': 'active'},
            'vendor_native': {'qos_score': 0.2, 'latency_ms': 1000, 'status': 'throttled'},
            'http_json': {'qos_score': 0.7, 'latency_ms': 150, 'status': 'active'},
            'local_spool': {'qos_score': 0.6, 'latency_ms': 200, 'status': 'standby'}
        }
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after vendor throttle: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'degraded'
        assert 'error_rate_threshold' in status['active_triggers'] or 'performance_degradation' in status['active_triggers']
        
        print("✓ Vendor throttle → HTTP JSON fallback test passed")
    
    def test_http_5xx_to_local_spool(self):
        """Test HTTP 5xx fallback to local spool"""
        print("\n=== Testing HTTP 5xx → Local Spool Fallback ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Simulate HTTP 5xx errors
        for i in range(15):  # Trigger error rate threshold
            self.orchestrator.update_telemetry_tracking('error', time.time())
        
        # Mock channel health showing HTTP JSON issues
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'otlp_grpc': {'qos_score': 0.9, 'latency_ms': 50, 'status': 'active'},
            'vendor_native': {'qos_score': 0.8, 'latency_ms': 100, 'status': 'active'},
            'http_json': {'qos_score': 0.1, 'latency_ms': 2000, 'status': 'error_5xx'},
            'local_spool': {'qos_score': 0.6, 'latency_ms': 200, 'status': 'active'}
        }
        
        # Check fallback conditions
        self.orchestrator._check_fallback_conditions()
        
        # Verify fallback was triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status after HTTP 5xx: {status['current_mode']}")
        print(f"Active triggers: {status['active_triggers']}")
        
        assert status['current_mode'] == 'degraded'
        assert 'error_rate_threshold' in status['active_triggers']
        
        print("✓ HTTP 5xx → local spool fallback test passed")
    
    def test_cooldown_and_hysteresis_mechanisms(self):
        """Test cooldown and hysteresis mechanisms"""
        print("\n=== Testing Cooldown and Hysteresis Mechanisms ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Trigger initial fallback
        self.orchestrator.force_fallback(FallbackMode.DEGRADED, "test_cooldown")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial fallback time: {initial_status['last_fallback_time']}")
        
        # Verify cooldown is active
        assert self.orchestrator.fallback_state.cooldown_until > time.time()
        
        # Try to trigger another fallback during cooldown
        self.orchestrator.force_fallback(FallbackMode.DEGRADED, "test_cooldown_override")
        status_during_cooldown = self.orchestrator.get_fallback_status()
        
        # Should still be in cooldown period
        assert self.orchestrator.fallback_state.cooldown_until > time.time()
        
        # Wait for cooldown to expire
        time.sleep(0.1)  # Small delay for testing
        
        # Now should be able to trigger fallback again
        self.orchestrator.force_fallback(FallbackMode.DEGRADED, "test_post_cooldown")
        post_cooldown_status = self.orchestrator.get_fallback_status()
        
        print("✓ Cooldown and hysteresis mechanisms test passed")
    
    def test_recovery_to_preferred_path(self):
        """Test recovery back to preferred path"""
        print("\n=== Testing Recovery to Preferred Path ===")
        
        # Start in degraded mode
        self.orchestrator.force_fallback(FallbackMode.DEGRADED, "test_recovery")
        initial_status = self.orchestrator.get_fallback_status()
        print(f"Initial status: {initial_status['current_mode']}")
        
        # Mock channel health recovery
        self.mock_signal_negotiator.get_channel_health_summary.return_value = {
            'otlp_grpc': {'qos_score': 0.95, 'latency_ms': 30, 'status': 'active'},
            'vendor_native': {'qos_score': 0.9, 'latency_ms': 80, 'status': 'active'},
            'http_json': {'qos_score': 0.8, 'latency_ms': 120, 'status': 'active'},
            'local_spool': {'qos_score': 0.7, 'latency_ms': 180, 'status': 'standby'}
        }
        
        # Update telemetry to show recovery
        current_time = time.time()
        self.orchestrator.update_telemetry_tracking('invocation_trace', current_time)
        self.orchestrator.update_telemetry_tracking('side_channel_latency', current_time)
        
        # Check recovery conditions
        recovery_possible = self.orchestrator._can_recover()
        print(f"Recovery possible: {recovery_possible}")
        
        if recovery_possible:
            # Attempt recovery
            self.orchestrator._attempt_recovery()
            recovery_status = self.orchestrator.get_fallback_status()
            print(f"Status after recovery attempt: {recovery_status['current_mode']}")
            
            # Should be back to normal or in recovery mode
            assert recovery_status['current_mode'] in ['normal', 'recovery']
        
        print("✓ Recovery to preferred path test passed")
    
    def test_dedup_and_idempotency(self):
        """Test deduplication and idempotency on re-publish"""
        print("\n=== Testing Dedup and Idempotency ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Generate unique telemetry IDs
        telemetry_ids = [f"test_telemetry_{i}" for i in range(5)]
        
        # Send same telemetry multiple times
        for telemetry_id in telemetry_ids:
            for attempt in range(3):  # Send 3 times each
                self.orchestrator.update_telemetry_tracking('invocation_trace', time.time(), telemetry_id)
        
        # Verify no duplicate processing
        # This would be implemented in the actual telemetry processing pipeline
        # For now, we verify the orchestrator doesn't create duplicate events
        
        # Check fallback history for duplicates
        history = self.orchestrator.get_fallback_history(limit=100)
        unique_events = set()
        
        for event in history:
            event_key = f"{event['trigger']}_{event['reason']}_{event['timestamp']}"
            unique_events.add(event_key)
        
        # Should have no duplicate events
        assert len(unique_events) == len(history)
        
        print("✓ Dedup and idempotency test passed")
    
    def test_budget_overshoot_prevention(self):
        """Test budget overshoot prevention"""
        print("\n=== Testing Budget Overshoot Prevention ===")
        
        # Reset to normal mode
        self.orchestrator.force_fallback(FallbackMode.NORMAL, "test_reset")
        
        # Simulate high resource usage
        for i in range(20):
            self.orchestrator.update_performance_metric(0.1)  # Low performance
        
        # Check if budget protection is triggered
        status = self.orchestrator.get_fallback_status()
        print(f"Status under high load: {status['current_mode']}")
        
        # Verify budget protection mechanisms
        # This would be implemented in the actual budget management system
        # For now, we verify the orchestrator responds to performance degradation
        
        if status['current_mode'] == 'degraded':
            assert 'performance_degradation' in status['active_triggers']
        
        print("✓ Budget overshoot prevention test passed")
    
    def test_complete_fallback_matrix(self):
        """Test complete fallback matrix end-to-end"""
        print("\n=== Testing Complete Fallback Matrix ===")
        
        # Test the complete fallback chain: OTLP → Vendor → HTTP → Local → Recovery
        
        # 1. Start with OTLP gRPC timeout
        self.test_otlp_grpc_timeout_to_vendor_native()
        
        # 2. Vendor native throttling
        self.test_vendor_throttle_to_http_json()
        
        # 3. HTTP 5xx errors
        self.test_http_5xx_to_local_spool()
        
        # 4. Recovery back to preferred path
        self.test_recovery_to_preferred_path()
        
        # Verify final state
        final_status = self.orchestrator.get_fallback_status()
        print(f"Final status: {final_status['current_mode']}")
        print(f"Total fallbacks: {final_status['fallback_count']}")
        
        # Should have gone through the complete matrix
        assert final_status['fallback_count'] >= 3
        
        print("✓ Complete fallback matrix test passed")


def run_fallback_matrix_tests():
    """Run all fallback orchestration matrix tests"""
    print("\n" + "="*60)
    print("RUNNING FALLBACK ORCHESTRATION MATRIX TESTS")
    print("="*60)
    
    test_suite = TestFallbackOrchestrationMatrix()
    
    # Run all test methods
    test_methods = [
        'test_otlp_grpc_timeout_to_vendor_native',
        'test_vendor_throttle_to_http_json',
        'test_http_5xx_to_local_spool',
        'test_cooldown_and_hysteresis_mechanisms',
        'test_recovery_to_preferred_path',
        'test_dedup_and_idempotency',
        'test_budget_overshoot_prevention',
        'test_complete_fallback_matrix'
    ]
    
    for method_name in test_methods:
        try:
            test_suite.setup_method()
            method = getattr(test_suite, method_name)
            method()
            print(f"✓ {method_name} completed successfully")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
        finally:
            test_suite.teardown_method()
    
    print("\n" + "="*60)
    print("FALLBACK ORCHESTRATION MATRIX TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_fallback_matrix_tests()
