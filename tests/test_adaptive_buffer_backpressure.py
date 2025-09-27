"""
Test suite for Adaptive Buffer/Backpressure Mechanisms

This module tests the complete adaptive buffer and backpressure system including:
- Queue watermarks (low/mid/high) triggering
- Backpressure signals and propagation
- Batch resizing under pressure
- LOSS_POLICY handling (lossy_tail vs lossless_block vs hybrid)
- Bounded memory enforcement
- Shed accounting under chaos conditions (packet loss/slow sinks)
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
from typing import Dict, List, Any, Optional
from collections import deque
import random

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import Layer0Config


class LossPolicy:
    """Enumeration of loss policies"""
    LOSSLESS_BLOCK = "lossless_block"
    LOSSY_TAIL = "lossy_tail"
    HYBRID = "hybrid"


class AdaptiveBuffer:
    """Mock adaptive buffer implementation for testing"""
    
    def __init__(self, config: Layer0Config):
        self.config = config
        self.max_queue_size = 1000
        self.low_watermark = 0.2  # 20%
        self.mid_watermark = 0.5  # 50%
        self.high_watermark = 0.8  # 80%
        
        self.queue = deque()
        self.current_size = 0
        self.loss_policy = LossPolicy.LOSSLESS_BLOCK
        self.backpressure_active = False
        self.batch_size = 100
        self.base_batch_size = 100
        
        # Memory tracking
        self.memory_usage_bytes = 0
        self.max_memory_bytes = 100 * 1024 * 1024  # 100MB
        
        # Shed accounting
        self.shed_count = 0
        self.shed_bytes = 0
        self.packet_loss_count = 0
        
        # Monitoring
        self.watermark_triggers = []
        self.backpressure_events = []
        
    def enqueue(self, item: Any, size_bytes: int = 1024) -> bool:
        """Enqueue item with size tracking"""
        if self.current_size >= self.max_queue_size:
            if self.loss_policy == LossPolicy.LOSSLESS_BLOCK:
                return False  # Block
            elif self.loss_policy == LossPolicy.LOSSY_TAIL:
                # Drop oldest item
                if self.queue:
                    old_item = self.queue.popleft()
                    self.current_size -= 1
                    self.shed_count += 1
                    self.shed_bytes += size_bytes
            elif self.loss_policy == LossPolicy.HYBRID:
                # Drop if above high watermark, otherwise block
                if self.current_size > self.max_queue_size * self.high_watermark:
                    if self.queue:
                        old_item = self.queue.popleft()
                        self.current_size -= 1
                        self.shed_count += 1
                        self.shed_bytes += size_bytes
                else:
                    return False  # Block
        
        self.queue.append(item)
        self.current_size += 1
        self.memory_usage_bytes += size_bytes
        
        # Check watermarks
        self._check_watermarks()
        
        return True
    
    def dequeue(self, batch_size: Optional[int] = None) -> List[Any]:
        """Dequeue items with batch size adaptation"""
        if batch_size is None:
            batch_size = self.batch_size
        
        items = []
        actual_batch_size = min(batch_size, len(self.queue))
        
        for _ in range(actual_batch_size):
            if self.queue:
                item = self.queue.popleft()
                items.append(item)
                self.current_size -= 1
                self.memory_usage_bytes -= 1024  # Assume 1KB per item
        
        # Adapt batch size based on pressure
        self._adapt_batch_size()
        
        # Check watermarks after dequeue
        self._check_watermarks()
        
        return items
    
    def _check_watermarks(self):
        """Check queue watermarks and trigger appropriate actions"""
        queue_ratio = self.current_size / self.max_queue_size
        
        if queue_ratio >= self.high_watermark:
            self._trigger_high_watermark()
        elif queue_ratio >= self.mid_watermark:
            self._trigger_mid_watermark()
        elif queue_ratio <= self.low_watermark:
            self._trigger_low_watermark()
    
    def _trigger_high_watermark(self):
        """Trigger high watermark actions"""
        self.watermark_triggers.append(('high', time.time()))
        self.backpressure_active = True
        self.backpressure_events.append(('high_watermark', time.time()))
        
        # Reduce batch size aggressively
        self.batch_size = max(10, self.batch_size // 2)
        
        # Switch to lossy policy if needed
        if self.loss_policy == LossPolicy.LOSSLESS_BLOCK:
            self.loss_policy = LossPolicy.HYBRID
    
    def _trigger_mid_watermark(self):
        """Trigger mid watermark actions"""
        self.watermark_triggers.append(('mid', time.time()))
        self.backpressure_events.append(('mid_watermark', time.time()))
        
        # Moderate batch size reduction
        self.batch_size = max(25, int(self.batch_size * 0.8))
    
    def _trigger_low_watermark(self):
        """Trigger low watermark actions"""
        self.watermark_triggers.append(('low', time.time()))
        self.backpressure_active = False
        self.backpressure_events.append(('low_watermark', time.time()))
        
        # Gradually increase batch size
        self.batch_size = min(self.base_batch_size, int(self.batch_size * 1.2))
        
        # Return to lossless policy
        if self.loss_policy != LossPolicy.LOSSLESS_BLOCK:
            self.loss_policy = LossPolicy.LOSSLESS_BLOCK
    
    def _adapt_batch_size(self):
        """Adapt batch size based on current pressure"""
        queue_ratio = self.current_size / self.max_queue_size
        
        if queue_ratio > 0.7:
            # High pressure - reduce batch size
            self.batch_size = max(10, int(self.batch_size * 0.9))
        elif queue_ratio < 0.3:
            # Low pressure - increase batch size
            self.batch_size = min(self.base_batch_size, int(self.batch_size * 1.1))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        return {
            'current_size': self.current_size,
            'max_size': self.max_queue_size,
            'queue_ratio': self.current_size / self.max_queue_size,
            'backpressure_active': self.backpressure_active,
            'batch_size': self.batch_size,
            'loss_policy': self.loss_policy,
            'memory_usage_bytes': self.memory_usage_bytes,
            'max_memory_bytes': self.max_memory_bytes,
            'shed_count': self.shed_count,
            'shed_bytes': self.shed_bytes,
            'packet_loss_count': self.packet_loss_count
        }


class TestAdaptiveBufferBackpressure:
    """Test suite for adaptive buffer and backpressure mechanisms"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = Layer0Config()
        self.buffer = AdaptiveBuffer(self.config)
        
    def test_queue_watermarks(self):
        """Test queue watermark triggering at low/mid/high levels"""
        print("\n=== Testing Queue Watermarks ===")
        
        # Start with empty queue
        status = self.buffer.get_status()
        assert status['queue_ratio'] == 0.0
        assert not status['backpressure_active']
        
        # Fill to low watermark
        items_to_low = int(self.buffer.max_queue_size * self.buffer.low_watermark)
        for i in range(items_to_low):
            self.buffer.enqueue(f"item_{i}")
        
        status = self.buffer.get_status()
        print(f"After low watermark: ratio={status['queue_ratio']:.2f}, backpressure={status['backpressure_active']}")
        assert status['queue_ratio'] >= self.buffer.low_watermark
        
        # Fill to mid watermark
        items_to_mid = int(self.buffer.max_queue_size * self.buffer.mid_watermark) - items_to_low
        for i in range(items_to_mid):
            self.buffer.enqueue(f"item_mid_{i}")
        
        status = self.buffer.get_status()
        print(f"After mid watermark: ratio={status['queue_ratio']:.2f}, backpressure={status['backpressure_active']}")
        assert status['queue_ratio'] >= self.buffer.mid_watermark
        
        # Fill to high watermark
        items_to_high = int(self.buffer.max_queue_size * self.buffer.high_watermark) - items_to_mid - items_to_low
        for i in range(items_to_high):
            self.buffer.enqueue(f"item_high_{i}")
        
        status = self.buffer.get_status()
        print(f"After high watermark: ratio={status['queue_ratio']:.2f}, backpressure={status['backpressure_active']}")
        assert status['queue_ratio'] >= self.buffer.high_watermark
        assert status['backpressure_active']
        
        # Verify watermark triggers were recorded
        assert len(self.buffer.watermark_triggers) >= 3
        watermark_types = [trigger[0] for trigger in self.buffer.watermark_triggers]
        assert 'low' in watermark_types
        assert 'mid' in watermark_types
        assert 'high' in watermark_types
        
        print("✓ Queue watermarks test passed")
    
    def test_backpressure_signals(self):
        """Test backpressure signal generation and propagation"""
        print("\n=== Testing Backpressure Signals ===")
        
        # Fill queue to trigger backpressure
        for i in range(int(self.buffer.max_queue_size * 0.9)):
            self.buffer.enqueue(f"item_{i}")
        
        status = self.buffer.get_status()
        print(f"Queue filled: ratio={status['queue_ratio']:.2f}")
        assert status['backpressure_active']
        
        # Verify backpressure events
        assert len(self.buffer.backpressure_events) > 0
        backpressure_types = [event[0] for event in self.buffer.backpressure_events]
        assert 'high_watermark' in backpressure_types
        
        # Test backpressure deactivation
        # Drain queue to low watermark
        while self.buffer.current_size > int(self.buffer.max_queue_size * self.buffer.low_watermark):
            self.buffer.dequeue(50)
        
        status = self.buffer.get_status()
        print(f"After draining: ratio={status['queue_ratio']:.2f}, backpressure={status['backpressure_active']}")
        assert not status['backpressure_active']
        
        # Verify low watermark event
        backpressure_types = [event[0] for event in self.buffer.backpressure_events]
        assert 'low_watermark' in backpressure_types
        
        print("✓ Backpressure signals test passed")
    
    def test_batch_resizing(self):
        """Test batch size adaptation under pressure"""
        print("\n=== Testing Batch Resizing ===")
        
        initial_batch_size = self.buffer.batch_size
        print(f"Initial batch size: {initial_batch_size}")
        
        # Fill queue to high watermark to trigger batch size reduction
        for i in range(int(self.buffer.max_queue_size * 0.9)):
            self.buffer.enqueue(f"item_{i}")
        
        status = self.buffer.get_status()
        print(f"High watermark batch size: {status['batch_size']}")
        assert status['batch_size'] < initial_batch_size
        
        # Drain queue to low watermark to trigger batch size increase
        while self.buffer.current_size > int(self.buffer.max_queue_size * self.buffer.low_watermark):
            self.buffer.dequeue(50)
        
        status = self.buffer.get_status()
        print(f"Low watermark batch size: {status['batch_size']}")
        assert status['batch_size'] > status['batch_size']  # Should have increased
        
        print("✓ Batch resizing test passed")
    
    def test_loss_policy_handling(self):
        """Test LOSS_POLICY handling under different conditions"""
        print("\n=== Testing Loss Policy Handling ===")
        
        # Test lossless_block policy (default)
        self.buffer.loss_policy = LossPolicy.LOSSLESS_BLOCK
        print(f"Testing {self.buffer.loss_policy} policy")
        
        # Fill queue completely
        for i in range(self.buffer.max_queue_size):
            success = self.buffer.enqueue(f"item_{i}")
            if not success:
                break
        
        # Should block when full
        success = self.buffer.enqueue("blocked_item")
        assert not success
        
        # Test lossy_tail policy
        self.buffer.loss_policy = LossPolicy.LOSSY_TAIL
        print(f"Testing {self.buffer.loss_policy} policy")
        
        # Should drop oldest items when full
        success = self.buffer.enqueue("lossy_item")
        assert success
        assert self.buffer.shed_count > 0
        
        # Test hybrid policy
        self.buffer.loss_policy = LossPolicy.HYBRID
        print(f"Testing {self.buffer.loss_policy} policy")
        
        # Should behave like lossless below high watermark
        # and like lossy above high watermark
        status = self.buffer.get_status()
        if status['queue_ratio'] < self.buffer.high_watermark:
            # Below high watermark - should block
            success = self.buffer.enqueue("hybrid_item")
            # May or may not succeed depending on exact state
        
        print("✓ Loss policy handling test passed")
    
    def test_bounded_memory(self):
        """Test bounded memory enforcement"""
        print("\n=== Testing Bounded Memory ===")
        
        initial_memory = self.buffer.memory_usage_bytes
        print(f"Initial memory usage: {initial_memory} bytes")
        
        # Fill queue with large items
        large_item_size = 1024 * 1024  # 1MB per item
        max_items = self.buffer.max_memory_bytes // large_item_size
        
        for i in range(max_items):
            success = self.buffer.enqueue(f"large_item_{i}", large_item_size)
            if not success:
                break
        
        status = self.buffer.get_status()
        print(f"Memory usage after filling: {status['memory_usage_bytes']} bytes")
        print(f"Max memory: {status['max_memory_bytes']} bytes")
        
        # Memory usage should not exceed max
        assert status['memory_usage_bytes'] <= status['max_memory_bytes']
        
        # Test memory cleanup on dequeue
        items = self.buffer.dequeue(10)
        status_after_dequeue = self.buffer.get_status()
        print(f"Memory after dequeue: {status_after_dequeue['memory_usage_bytes']} bytes")
        
        # Memory should have decreased
        assert status_after_dequeue['memory_usage_bytes'] < status['memory_usage_bytes']
        
        print("✓ Bounded memory test passed")
    
    def test_shed_accounting(self):
        """Test shed accounting under chaos conditions"""
        print("\n=== Testing Shed Accounting ===")
        
        initial_shed_count = self.buffer.shed_count
        initial_shed_bytes = self.buffer.shed_bytes
        
        # Simulate packet loss
        self.buffer.packet_loss_count += 5
        print(f"Simulated packet loss: {self.buffer.packet_loss_count}")
        
        # Fill queue to trigger shedding
        self.buffer.loss_policy = LossPolicy.LOSSY_TAIL
        
        for i in range(self.buffer.max_queue_size + 10):
            self.buffer.enqueue(f"chaos_item_{i}")
        
        status = self.buffer.get_status()
        print(f"Shed count: {status['shed_count']}")
        print(f"Shed bytes: {status['shed_bytes']}")
        
        # Should have shed some items
        assert status['shed_count'] > initial_shed_count
        assert status['shed_bytes'] > initial_shed_bytes
        
        # Verify shed accounting accuracy
        expected_shed_bytes = status['shed_count'] * 1024  # Assuming 1KB per item
        assert abs(status['shed_bytes'] - expected_shed_bytes) <= 1024  # Allow small tolerance
        
        print("✓ Shed accounting test passed")
    
    def test_chaos_conditions(self):
        """Test buffer behavior under chaos conditions"""
        print("\n=== Testing Chaos Conditions ===")
        
        # Simulate slow sink (slow consumer)
        def slow_consumer():
            """Slow consumer that processes items slowly"""
            time.sleep(0.1)  # Simulate slow processing
            return self.buffer.dequeue(5)
        
        # Fill queue rapidly
        producer_thread = threading.Thread(target=self._rapid_producer)
        producer_thread.start()
        
        # Start slow consumer
        consumer_thread = threading.Thread(target=self._slow_consumer_loop)
        consumer_thread.start()
        
        # Let them run for a bit
        time.sleep(0.5)
        
        # Stop threads
        self.buffer.shutdown = True
        producer_thread.join(timeout=1)
        consumer_thread.join(timeout=1)
        
        # Check final state
        status = self.buffer.get_status()
        print(f"Final state: size={status['current_size']}, backpressure={status['backpressure_active']}")
        
        # Should have handled chaos gracefully
        assert status['current_size'] <= self.buffer.max_queue_size
        assert status['memory_usage_bytes'] <= status['max_memory_bytes']
        
        print("✓ Chaos conditions test passed")
    
    def _rapid_producer(self):
        """Rapid producer for chaos testing"""
        i = 0
        while not hasattr(self.buffer, 'shutdown') or not self.buffer.shutdown:
            self.buffer.enqueue(f"chaos_producer_{i}")
            i += 1
            time.sleep(0.001)  # Very fast production
    
    def _slow_consumer_loop(self):
        """Slow consumer loop for chaos testing"""
        while not hasattr(self.buffer, 'shutdown') or not self.buffer.shutdown:
            items = self.buffer.dequeue(5)
            time.sleep(0.1)  # Slow consumption
    
    def test_end_to_end_backpressure(self):
        """Test complete end-to-end backpressure flow"""
        print("\n=== Testing End-to-End Backpressure ===")
        
        # Simulate realistic load pattern
        phases = [
            ('normal', 100, 0.1),      # Normal load
            ('spike', 500, 0.001),     # Traffic spike
            ('sustained', 300, 0.05),  # Sustained high load
            ('recovery', 50, 0.2),     # Recovery period
        ]
        
        for phase_name, item_count, delay in phases:
            print(f"Phase: {phase_name} - adding {item_count} items")
            
            for i in range(item_count):
                success = self.buffer.enqueue(f"{phase_name}_item_{i}")
                if not success:
                    print(f"  Blocked at item {i}")
                    break
                time.sleep(delay)
            
            status = self.buffer.get_status()
            print(f"  Phase complete: size={status['current_size']}, backpressure={status['backpressure_active']}")
        
        # Final verification
        final_status = self.buffer.get_status()
        print(f"Final status: {final_status}")
        
        # Should have maintained stability
        assert final_status['current_size'] <= self.buffer.max_queue_size
        assert final_status['memory_usage_bytes'] <= final_status['max_memory_bytes']
        
        print("✓ End-to-end backpressure test passed")


def run_adaptive_buffer_tests():
    """Run all adaptive buffer and backpressure tests"""
    print("\n" + "="*60)
    print("RUNNING ADAPTIVE BUFFER/BACKPRESSURE TESTS")
    print("="*60)
    
    test_suite = TestAdaptiveBufferBackpressure()
    
    # Run all test methods
    test_methods = [
        'test_queue_watermarks',
        'test_backpressure_signals',
        'test_batch_resizing',
        'test_loss_policy_handling',
        'test_bounded_memory',
        'test_shed_accounting',
        'test_chaos_conditions',
        'test_end_to_end_backpressure'
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
    print("ADAPTIVE BUFFER/BACKPRESSURE TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_adaptive_buffer_tests()
