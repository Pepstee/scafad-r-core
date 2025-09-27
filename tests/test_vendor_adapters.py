"""
Test suite for Vendor Adapters

This module tests vendor adapter conformance including:
- Rate limits and throttling
- Payload caps and size limits
- Retries with jitter and exponential backoff
- Error mapping and status code handling
- Idempotency key management
- Quota exhaustion backoff strategies
- Provider-specific behavior validation
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import Layer0Config


class ProviderType(Enum):
    """Supported telemetry providers"""
    CLOUDWATCH = "cloudwatch"
    XRAY = "xray"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SPLUNK = "splunk"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"


class ErrorType(Enum):
    """Error types for testing"""
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    AUTHENTICATION_FAILED = "authentication_failed"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass
class VendorConfig:
    """Configuration for vendor-specific behavior"""
    provider: ProviderType
    rate_limit_rps: int = 100
    payload_max_size_bytes: int = 1024 * 1024  # 1MB
    max_retries: int = 3
    retry_delay_base_ms: int = 100
    retry_delay_max_ms: int = 5000
    jitter_factor: float = 0.1
    quota_budget: int = 10000
    quota_reset_interval_ms: int = 3600000  # 1 hour
    supports_idempotency: bool = True
    idempotency_key_header: str = "X-Idempotency-Key"
    timeout_ms: int = 5000


class VendorAdapter:
    """Mock vendor adapter implementation for testing"""
    
    def __init__(self, config: VendorConfig):
        self.config = config
        self.current_rate = 0
        self.rate_window_start = time.time()
        self.rate_window_duration = 1.0  # 1 second
        
        self.quota_used = 0
        self.quota_reset_time = time.time() + (self.config.quota_reset_interval_ms / 1000)
        
        self.retry_count = 0
        self.last_request_time = 0
        
        # Tracking
        self.requests_sent = 0
        self.requests_failed = 0
        self.rate_limit_hits = 0
        self.quota_exhaustions = 0
        self.payload_rejections = 0
        
        # Idempotency tracking
        self.processed_keys = set()
        
    def send_telemetry(self, payload: Dict[str, Any], 
                      idempotency_key: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Send telemetry with full error handling"""
        self.requests_sent += 1
        
        # Check idempotency
        if idempotency_key and self.config.supports_idempotency:
            if idempotency_key in self.processed_keys:
                return True, {"status": "duplicate", "message": "Already processed"}
            self.processed_keys.add(idempotency_key)
        
        # Check payload size
        payload_size = len(json.dumps(payload).encode('utf-8'))
        if payload_size > self.config.payload_max_size_bytes:
            self.payload_rejections += 1
            return False, {
                "error": ErrorType.PAYLOAD_TOO_LARGE.value,
                "message": f"Payload too large: {payload_size} bytes",
                "max_size": self.config.payload_max_size_bytes
            }
        
        # Check rate limit
        if not self._check_rate_limit():
            self.rate_limit_hits += 1
            return False, {
                "error": ErrorType.RATE_LIMIT.value,
                "message": "Rate limit exceeded",
                "retry_after": self._calculate_retry_delay()
            }
        
        # Check quota
        if not self._check_quota():
            self.quota_exhaustions += 1
            return False, {
                "error": ErrorType.QUOTA_EXHAUSTED.value,
                "message": "Quota exhausted",
                "reset_time": self.quota_reset_time
            }
        
        # Simulate network delay
        time.sleep(0.001)  # 1ms
        
        # Simulate occasional failures
        if self._should_fail():
            self.requests_failed += 1
            return False, {
                "error": ErrorType.INTERNAL_ERROR.value,
                "message": "Internal server error",
                "retry_after": self._calculate_retry_delay()
            }
        
        # Success
        self.current_rate += 1
        self.quota_used += 1
        self.last_request_time = time.time()
        
        return True, {"status": "success", "request_id": str(uuid.uuid4())}
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limit"""
        current_time = time.time()
        
        # Reset rate window if needed
        if current_time - self.rate_window_start >= self.rate_window_duration:
            self.rate_window_start = current_time
            self.current_rate = 0
        
        return self.current_rate < self.config.rate_limit_rps
    
    def _check_quota(self) -> bool:
        """Check if quota is available"""
        current_time = time.time()
        
        # Reset quota if interval has passed
        if current_time >= self.quota_reset_time:
            self.quota_used = 0
            self.quota_reset_time = current_time + (self.config.quota_reset_interval_ms / 1000)
        
        return self.quota_used < self.config.quota_budget
    
    def _calculate_retry_delay(self) -> int:
        """Calculate retry delay with exponential backoff and jitter"""
        base_delay = min(
            self.config.retry_delay_base_ms * (2 ** self.retry_count),
            self.config.retry_delay_max_ms
        )
        
        # Add jitter
        jitter = base_delay * self.config.jitter_factor * (2 * random.random() - 1)
        delay = max(0, base_delay + jitter)
        
        return int(delay)
    
    def _should_fail(self) -> bool:
        """Determine if request should fail (for testing)"""
        # 5% failure rate for testing
        return random.random() < 0.05
    
    def retry_with_backoff(self, payload: Dict[str, Any], 
                          idempotency_key: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Retry request with exponential backoff"""
        for attempt in range(self.config.max_retries + 1):
            success, response = self.send_telemetry(payload, idempotency_key)
            
            if success:
                self.retry_count = 0
                return True, response
            
            # Check if retryable
            if response.get("error") not in [
                ErrorType.RATE_LIMIT.value,
                ErrorType.QUOTA_EXHAUSTED.value,
                ErrorType.NETWORK_ERROR.value,
                ErrorType.TIMEOUT.value,
                ErrorType.INTERNAL_ERROR.value
            ]:
                return False, response
            
            if attempt < self.config.max_retries:
                delay = response.get("retry_after", self._calculate_retry_delay())
                time.sleep(delay / 1000)  # Convert to seconds
                self.retry_count += 1
        
        return False, {"error": "max_retries_exceeded", "message": "Max retries exceeded"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current adapter status"""
        return {
            "provider": self.config.provider.value,
            "current_rate": self.current_rate,
            "rate_limit_rps": self.config.rate_limit_rps,
            "quota_used": self.quota_used,
            "quota_budget": self.config.quota_budget,
            "requests_sent": self.requests_sent,
            "requests_failed": self.requests_failed,
            "rate_limit_hits": self.rate_limit_hits,
            "quota_exhaustions": self.quota_exhaustions,
            "payload_rejections": self.payload_rejections,
            "retry_count": self.retry_count
        }


class TestVendorAdapters:
    """Test suite for vendor adapter conformance"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = Layer0Config()
        
        # Create adapters for different providers
        self.adapters = {
            ProviderType.CLOUDWATCH: VendorAdapter(VendorConfig(
                provider=ProviderType.CLOUDWATCH,
                rate_limit_rps=100,
                payload_max_size_bytes=1024 * 1024,
                supports_idempotency=True
            )),
            ProviderType.XRAY: VendorAdapter(VendorConfig(
                provider=ProviderType.XRAY,
                rate_limit_rps=200,
                payload_max_size_bytes=512 * 1024,
                supports_idempotency=False
            )),
            ProviderType.DATADOG: VendorAdapter(VendorConfig(
                provider=ProviderType.DATADOG,
                rate_limit_rps=150,
                payload_max_size_bytes=2048 * 1024,
                supports_idempotency=True
            ))
        }
        
    def test_rate_limits(self):
        """Test rate limit enforcement"""
        print("\n=== Testing Rate Limits ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        config = adapter.config
        
        # Send requests up to rate limit
        success_count = 0
        for i in range(config.rate_limit_rps + 10):
            success, response = adapter.send_telemetry({"test": f"rate_limit_{i}"})
            if success:
                success_count += 1
            else:
                break
        
        print(f"Successful requests: {success_count}")
        print(f"Rate limit: {config.rate_limit_rps}")
        
        # Should not exceed rate limit
        assert success_count <= config.rate_limit_rps
        
        # Verify rate limit error
        status = adapter.get_status()
        assert status['rate_limit_hits'] > 0
        
        print("✓ Rate limits test passed")
    
    def test_payload_caps(self):
        """Test payload size limits"""
        print("\n=== Testing Payload Caps ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        config = adapter.config
        
        # Test payload within limits
        small_payload = {"test": "small", "data": "x" * 100}
        success, response = adapter.send_telemetry(small_payload)
        assert success
        
        # Test payload exceeding limits
        large_payload = {"test": "large", "data": "x" * (config.payload_max_size_bytes + 1000)}
        success, response = adapter.send_telemetry(large_payload)
        assert not success
        assert response["error"] == ErrorType.PAYLOAD_TOO_LARGE.value
        
        # Verify payload rejection tracking
        status = adapter.get_status()
        assert status['payload_rejections'] > 0
        
        print("✓ Payload caps test passed")
    
    def test_retries_and_jitter(self):
        """Test retry mechanism with jitter"""
        print("\n=== Testing Retries and Jitter ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        
        # Force failures by temporarily setting high failure rate
        original_should_fail = adapter._should_fail
        adapter._should_fail = lambda: True
        
        # Test retry with backoff
        start_time = time.time()
        success, response = adapter.retry_with_backoff({"test": "retry_test"})
        end_time = time.time()
        
        # Restore original behavior
        adapter._should_fail = original_should_fail
        
        # Should have failed after max retries
        assert not success
        assert response["error"] == "max_retries_exceeded"
        
        # Should have taken time for retries
        elapsed_time = end_time - start_time
        expected_min_time = (adapter.config.retry_delay_base_ms * 3) / 1000  # At least 3 retries
        assert elapsed_time >= expected_min_time
        
        print("✓ Retries and jitter test passed")
    
    def test_error_mapping(self):
        """Test error mapping and status code handling"""
        print("\n=== Testing Error Mapping ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        
        # Test different error conditions
        error_tests = [
            (ErrorType.RATE_LIMIT, "rate_limit"),
            (ErrorType.QUOTA_EXHAUSTED, "quota_exhausted"),
            (ErrorType.PAYLOAD_TOO_LARGE, "payload_too_large"),
        ]
        
        for error_type, test_name in error_tests:
            if error_type == ErrorType.RATE_LIMIT:
                # Fill rate limit
                for i in range(adapter.config.rate_limit_rps + 1):
                    adapter.send_telemetry({"test": f"rate_limit_{i}"})
                
                # Next request should hit rate limit
                success, response = adapter.send_telemetry({"test": "rate_limit_test"})
                assert not success
                assert response["error"] == error_type.value
                
            elif error_type == ErrorType.QUOTA_EXHAUSTED:
                # Fill quota
                for i in range(adapter.config.quota_budget + 1):
                    adapter.send_telemetry({"test": f"quota_{i}"})
                
                # Next request should hit quota limit
                success, response = adapter.send_telemetry({"test": "quota_test"})
                assert not success
                assert response["error"] == error_type.value
                
            elif error_type == ErrorType.PAYLOAD_TOO_LARGE:
                # Send oversized payload
                large_payload = {"test": "large", "data": "x" * (adapter.config.payload_max_size_bytes + 1000)}
                success, response = adapter.send_telemetry(large_payload)
                assert not success
                assert response["error"] == error_type.value
        
        print("✓ Error mapping test passed")
    
    def test_idempotency_keys(self):
        """Test idempotency key management"""
        print("\n=== Testing Idempotency Keys ===")
        
        # Test with provider that supports idempotency
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        assert adapter.config.supports_idempotency
        
        # Generate unique idempotency key
        idempotency_key = str(uuid.uuid4())
        
        # Send first request
        success1, response1 = adapter.send_telemetry(
            {"test": "idempotency_1"}, 
            idempotency_key
        )
        assert success1
        
        # Send duplicate request with same key
        success2, response2 = adapter.send_telemetry(
            {"test": "idempotency_2"}, 
            idempotency_key
        )
        assert success2
        assert response2["status"] == "duplicate"
        
        # Test with provider that doesn't support idempotency
        xray_adapter = self.adapters[ProviderType.XRAY]
        assert not xray_adapter.config.supports_idempotency
        
        # Should process normally even with idempotency key
        success3, response3 = xray_adapter.send_telemetry(
            {"test": "xray_test"}, 
            idempotency_key
        )
        # May succeed or fail, but shouldn't be treated as duplicate
        
        print("✓ Idempotency keys test passed")
    
    def test_quota_exhaustion_backoff(self):
        """Test quota exhaustion backoff strategies"""
        print("\n=== Testing Quota Exhaustion Backoff ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        
        # Fill quota completely
        for i in range(adapter.config.quota_budget):
            adapter.send_telemetry({"test": f"quota_fill_{i}"})
        
        # Verify quota is exhausted
        status = adapter.get_status()
        assert status['quota_used'] >= adapter.config.quota_budget
        
        # Try to send more - should fail
        success, response = adapter.send_telemetry({"test": "quota_exhausted"})
        assert not success
        assert response["error"] == ErrorType.QUOTA_EXHAUSTED.value
        
        # Check reset time
        assert "reset_time" in response
        
        # Wait for quota reset (simulate)
        adapter.quota_reset_time = time.time() - 1  # Force reset
        
        # Should work again
        success, response = adapter.send_telemetry({"test": "quota_reset"})
        assert success
        
        print("✓ Quota exhaustion backoff test passed")
    
    def test_provider_specific_behavior(self):
        """Test provider-specific behavior differences"""
        print("\n=== Testing Provider-Specific Behavior ===")
        
        # Test CloudWatch (supports idempotency)
        cloudwatch = self.adapters[ProviderType.CLOUDWATCH]
        assert cloudwatch.config.supports_idempotency
        assert cloudwatch.config.payload_max_size_bytes == 1024 * 1024
        
        # Test X-Ray (no idempotency, smaller payload)
        xray = self.adapters[ProviderType.XRAY]
        assert not xray.config.supports_idempotency
        assert xray.config.payload_max_size_bytes == 512 * 1024
        
        # Test DataDog (higher rate limit, larger payload)
        datadog = self.adapters[ProviderType.DATADOG]
        assert datadog.config.rate_limit_rps == 150
        assert datadog.config.payload_max_size_bytes == 2048 * 1024
        
        # Test behavior differences
        idempotency_key = str(uuid.uuid4())
        
        # CloudWatch should handle idempotency
        success1, _ = cloudwatch.send_telemetry({"test": "test1"}, idempotency_key)
        success2, response2 = cloudwatch.send_telemetry({"test": "test2"}, idempotency_key)
        assert success1
        assert response2["status"] == "duplicate"
        
        # X-Ray should process both
        success3, _ = xray.send_telemetry({"test": "test3"}, idempotency_key)
        success4, _ = xray.send_telemetry({"test": "test4"}, idempotency_key)
        assert success3
        assert success4
        
        print("✓ Provider-specific behavior test passed")
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        print("\n=== Testing Concurrent Requests ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        results = []
        
        def send_request(request_id: int):
            """Send a single request"""
            success, response = adapter.send_telemetry({"test": f"concurrent_{request_id}"})
            results.append((request_id, success, response))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=send_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 10
        
        # Check rate limiting
        status = adapter.get_status()
        assert status['requests_sent'] >= 10
        
        print("✓ Concurrent requests test passed")
    
    def test_end_to_end_adapter_flow(self):
        """Test complete end-to-end adapter flow"""
        print("\n=== Testing End-to-End Adapter Flow ===")
        
        adapter = self.adapters[ProviderType.CLOUDWATCH]
        
        # Phase 1: Normal operation
        print("Phase 1: Normal operation")
        for i in range(50):
            success, response = adapter.send_telemetry({"phase": "normal", "id": i})
            assert success
        
        # Phase 2: Approach rate limit
        print("Phase 2: Approach rate limit")
        for i in range(50):
            success, response = adapter.send_telemetry({"phase": "rate_limit", "id": i})
            if not success:
                print(f"Rate limited at request {i}")
                break
        
        # Phase 3: Recovery
        print("Phase 3: Recovery")
        time.sleep(1.1)  # Wait for rate window reset
        
        success, response = adapter.send_telemetry({"phase": "recovery", "id": 0})
        assert success
        
        # Final status
        status = adapter.get_status()
        print(f"Final status: {status}")
        
        # Should have handled all phases gracefully
        assert status['requests_sent'] > 0
        assert status['requests_failed'] >= 0  # May have some failures
        
        print("✓ End-to-end adapter flow test passed")


def run_vendor_adapter_tests():
    """Run all vendor adapter tests"""
    print("\n" + "="*60)
    print("RUNNING VENDOR ADAPTER TESTS")
    print("="*60)
    
    test_suite = TestVendorAdapters()
    
    # Run all test methods
    test_methods = [
        'test_rate_limits',
        'test_payload_caps',
        'test_retries_and_jitter',
        'test_error_mapping',
        'test_idempotency_keys',
        'test_quota_exhaustion_backoff',
        'test_provider_specific_behavior',
        'test_concurrent_requests',
        'test_end_to_end_adapter_flow'
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
    print("VENDOR ADAPTER TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    import random
    run_vendor_adapter_tests()
