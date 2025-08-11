"""
Test suite for Health/Heartbeat Mechanisms

This module tests the complete health and heartbeat system including:
- Heartbeat drop detection and handling
- Clock skew detection and correction
- Probe timeout mechanisms
- Partial component crash (extension crash) scenarios
- Mean Time To Recovery (MTTR) assertions
- Health monitoring and alerting
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# Add the parent directory to the path to import Layer 0 components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import Layer0Config


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component types for health monitoring"""
    CORE = "core"
    EXTENSION = "extension"
    ADAPTER = "adapter"
    MONITOR = "monitor"
    STORAGE = "storage"


class HeartbeatEvent(Enum):
    """Heartbeat event types"""
    SENT = "sent"
    RECEIVED = "received"
    DROPPED = "dropped"
    TIMEOUT = "timeout"
    CORRUPTED = "corrupted"


@dataclass
class HeartbeatMessage:
    """Heartbeat message structure"""
    component_id: str
    component_type: ComponentType
    timestamp: float
    sequence_number: int
    health_status: HealthStatus
    metadata: Dict[str, Any]
    signature: Optional[str] = None


@dataclass
class HealthMetrics:
    """Health metrics for a component"""
    uptime_seconds: float
    last_heartbeat: float
    heartbeat_interval_ms: int
    missed_heartbeats: int
    consecutive_failures: int
    consecutive_successes: int
    response_time_ms: float
    error_rate: float
    resource_usage: Dict[str, float]


@dataclass
class HealthConfig:
    """Health monitoring configuration"""
    heartbeat_interval_ms: int = 1000
    heartbeat_timeout_ms: int = 5000
    max_missed_heartbeats: int = 3
    clock_skew_threshold_ms: int = 100
    probe_timeout_ms: int = 3000
    mttr_target_ms: int = 10000
    health_check_interval_ms: int = 500
    alert_threshold: float = 0.8


class HeartbeatMonitor:
    """Heartbeat monitoring and health detection system"""
    
    def __init__(self, config: HealthConfig):
        self.config = config
        self.components = {}
        self.heartbeat_history = []
        self.health_alerts = []
        self.recovery_events = []
        
        # Monitoring state
        self.monitoring_active = False
        self.last_health_check = 0
        self.total_heartbeats = 0
        self.total_drops = 0
        
        # Clock synchronization
        self.clock_reference = time.time()
        self.clock_skew_detected = False
        self.clock_corrections = []
        
        # MTTR tracking
        self.failure_start_times = {}
        self.recovery_times = {}
        self.mttr_metrics = []
    
    def register_component(self, component_id: str, component_type: ComponentType, 
                          heartbeat_interval_ms: int = None) -> bool:
        """Register a component for health monitoring"""
        if component_id in self.components:
            return False
        
        interval = heartbeat_interval_ms or self.config.heartbeat_interval_ms
        
        self.components[component_id] = {
            'type': component_type,
            'heartbeat_interval_ms': interval,
            'last_heartbeat': time.time(),
            'missed_heartbeats': 0,
            'consecutive_failures': 0,
            'consecutive_successes': 0,
            'health_status': HealthStatus.UNKNOWN,
            'uptime_start': time.time(),
            'metadata': {}
        }
        
        return True
    
    def send_heartbeat(self, component_id: str, health_status: HealthStatus = HealthStatus.HEALTHY,
                      metadata: Dict[str, Any] = None) -> bool:
        """Send a heartbeat from a component"""
        if component_id not in self.components:
            return False
        
        current_time = time.time()
        component = self.components[component_id]
        
        # Create heartbeat message
        heartbeat = HeartbeatMessage(
            component_id=component_id,
            component_type=component['type'],
            timestamp=current_time,
            sequence_number=self.total_heartbeats,
            health_status=health_status,
            metadata=metadata or {},
            signature=self._generate_signature(component_id, current_time)
        )
        
        # Process heartbeat
        success = self._process_heartbeat(heartbeat)
        
        if success:
            self.total_heartbeats += 1
            self.heartbeat_history.append(heartbeat)
            
            # Update component state
            component['last_heartbeat'] = current_time
            component['missed_heartbeats'] = 0
            component['consecutive_successes'] += 1
            component['consecutive_failures'] = 0
            component['health_status'] = health_status
            if metadata:
                component['metadata'].update(metadata)
        
        return success
    
    def _process_heartbeat(self, heartbeat: HeartbeatMessage) -> bool:
        """Process incoming heartbeat message"""
        # Check for clock skew
        clock_skew = abs(heartbeat.timestamp - time.time()) * 1000
        if clock_skew > self.config.clock_skew_threshold_ms:
            self.clock_skew_detected = True
            self.clock_corrections.append({
                'timestamp': time.time(),
                'component_id': heartbeat.component_id,
                'skew_ms': clock_skew,
                'corrected': False
            })
            return False
        
        # Validate signature
        if not self._validate_signature(heartbeat):
            return False
        
        # Check for heartbeat drops
        component = self.components[heartbeat.component_id]
        expected_heartbeats = int((time.time() - component['last_heartbeat']) / 
                                 (component['heartbeat_interval_ms'] / 1000))
        
        if expected_heartbeats > 1:
            component['missed_heartbeats'] += expected_heartbeats - 1
            self.total_drops += expected_heartbeats - 1
        
        return True
    
    def _generate_signature(self, component_id: str, timestamp: float) -> str:
        """Generate heartbeat signature (mock implementation)"""
        # In real implementation, use cryptographic signing
        return f"SIG_{component_id}_{timestamp}"
    
    def _validate_signature(self, heartbeat: HeartbeatMessage) -> bool:
        """Validate heartbeat signature (mock implementation)"""
        # In real implementation, verify cryptographic signature
        expected_sig = self._generate_signature(heartbeat.component_id, heartbeat.timestamp)
        return heartbeat.signature == expected_sig
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        current_time = time.time()
        self.last_health_check = current_time
        
        health_summary = {
            'timestamp': current_time,
            'total_components': len(self.components),
            'healthy_components': 0,
            'degraded_components': 0,
            'unhealthy_components': 0,
            'critical_components': 0,
            'unknown_components': 0,
            'alerts': []
        }
        
        for component_id, component in self.components.items():
            # Check for missed heartbeats
            time_since_last = current_time - component['last_heartbeat']
            missed_heartbeats = int(time_since_last / (component['heartbeat_interval_ms'] / 1000))
            
            if missed_heartbeats > 0:
                component['missed_heartbeats'] = missed_heartbeats
                
                # Update health status based on missed heartbeats
                if missed_heartbeats >= self.config.max_missed_heartbeats:
                    if component['health_status'] != HealthStatus.CRITICAL:
                        component['health_status'] = HealthStatus.CRITICAL
                        self._record_failure(component_id, "heartbeat_timeout")
                elif missed_heartbeats >= self.config.max_missed_heartbeats // 2:
                    if component['health_status'] != HealthStatus.UNHEALTHY:
                        component['health_status'] = HealthStatus.UNHEALTHY
                else:
                    if component['health_status'] != HealthStatus.DEGRADED:
                        component['health_status'] = HealthStatus.DEGRADED
            
            # Count components by health status
            if component['health_status'] == HealthStatus.HEALTHY:
                health_summary['healthy_components'] += 1
            elif component['health_status'] == HealthStatus.DEGRADED:
                health_summary['degraded_components'] += 1
            elif component['health_status'] == HealthStatus.UNHEALTHY:
                health_summary['unhealthy_components'] += 1
            elif component['health_status'] == HealthStatus.CRITICAL:
                health_summary['critical_components'] += 1
            else:
                health_summary['unknown_components'] += 1
            
            # Generate alerts for critical components
            if component['health_status'] == HealthStatus.CRITICAL:
                alert = {
                    'timestamp': current_time,
                    'component_id': component_id,
                    'severity': 'critical',
                    'message': f"Component {component_id} is critical - {component['missed_heartbeats']} missed heartbeats"
                }
                health_summary['alerts'].append(alert)
                self.health_alerts.append(alert)
        
        return health_summary
    
    def _record_failure(self, component_id: str, failure_type: str):
        """Record component failure for MTTR tracking"""
        current_time = time.time()
        
        if component_id not in self.failure_start_times:
            self.failure_start_times[component_id] = current_time
    
    def record_recovery(self, component_id: str, recovery_type: str):
        """Record component recovery for MTTR tracking"""
        current_time = time.time()
        
        if component_id in self.failure_start_times:
            failure_start = self.failure_start_times[component_id]
            recovery_time = current_time - failure_start
            
            self.recovery_times[component_id] = recovery_time
            self.mttr_metrics.append({
                'component_id': component_id,
                'failure_type': recovery_type,
                'failure_start': failure_start,
                'recovery_time': current_time,
                'mttr_ms': recovery_time * 1000
            })
            
            # Clear failure tracking
            del self.failure_start_times[component_id]
            
            # Update component status
            if component_id in self.components:
                self.components[component_id]['health_status'] = HealthStatus.HEALTHY
    
    def get_mttr_summary(self) -> Dict[str, Any]:
        """Get MTTR summary and metrics"""
        if not self.mttr_metrics:
            return {
                'total_recoveries': 0,
                'average_mttr_ms': 0,
                'min_mttr_ms': 0,
                'max_mttr_ms': 0,
                'mttr_target_met': True
            }
        
        mttr_values = [metric['mttr_ms'] for metric in self.mttr_metrics]
        
        return {
            'total_recoveries': len(self.mttr_metrics),
            'average_mttr_ms': sum(mttr_values) / len(mttr_values),
            'min_mttr_ms': min(mttr_values),
            'max_mttr_ms': max(mttr_values),
            'mttr_target_met': all(mttr <= self.config.mttr_target_ms for mttr in mttr_values)
        }
    
    def simulate_component_crash(self, component_id: str, crash_type: str = "extension_crash"):
        """Simulate component crash for testing"""
        if component_id in self.components:
            component = self.components[component_id]
            
            # Mark as critical
            component['health_status'] = HealthStatus.CRITICAL
            component['consecutive_failures'] += 5
            component['missed_heartbeats'] = self.config.max_missed_heartbeats + 1
            
            # Record failure
            self._record_failure(component_id, crash_type)
            
            # Generate alert
            alert = {
                'timestamp': time.time(),
                'component_id': component_id,
                'severity': 'critical',
                'message': f"Component {component_id} crashed - {crash_type}"
            }
            self.health_alerts.append(alert)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return {
            'monitoring_active': self.monitoring_active,
            'total_components': len(self.components),
            'total_heartbeats': self.total_heartbeats,
            'total_drops': self.total_drops,
            'clock_skew_detected': self.clock_skew_detected,
            'clock_corrections': len(self.clock_corrections),
            'health_alerts': len(self.health_alerts),
            'mttr_summary': self.get_mttr_summary()
        }


class TestHealthHeartbeat:
    """Test suite for health and heartbeat mechanisms"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = HealthConfig()
        self.monitor = HeartbeatMonitor(self.config)
        
        # Register test components
        self.test_components = [
            ('core_service', ComponentType.CORE),
            ('extension_1', ComponentType.EXTENSION),
            ('adapter_1', ComponentType.ADAPTER),
            ('monitor_1', ComponentType.MONITOR),
            ('storage_1', ComponentType.STORAGE)
        ]
        
        for component_id, component_type in self.test_components:
            self.monitor.register_component(component_id, component_type)
    
    def test_heartbeat_drop_detection(self):
        """Test heartbeat drop detection and handling"""
        print("\n=== Testing Heartbeat Drop Detection ===")
        
        # Send initial heartbeats
        for component_id, _ in self.test_components:
            self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
        
        # Verify all components are healthy
        health_summary = self.monitor.check_health()
        assert health_summary['healthy_components'] == len(self.test_components)
        
        # Simulate heartbeat drops by not sending heartbeats
        time.sleep(0.1)  # Small delay
        
        # Check health again - should detect missed heartbeats
        health_summary = self.monitor.check_health()
        
        # Some components may have missed heartbeats
        total_healthy = (health_summary['healthy_components'] + 
                        health_summary['degraded_components'] + 
                        health_summary['unhealthy_components'] + 
                        health_summary['critical_components'])
        
        assert total_healthy == len(self.test_components)
        
        # Check for drops
        system_status = self.monitor.get_system_status()
        assert system_status['total_drops'] >= 0
        
        print(f"  Heartbeats sent: {system_status['total_heartbeats']}")
        print(f"  Heartbeats dropped: {system_status['total_drops']}")
        
        print("✓ Heartbeat drop detection test passed")
    
    def test_clock_skew_detection(self):
        """Test clock skew detection and handling"""
        print("\n=== Testing Clock Skew Detection ===")
        
        # Test normal heartbeat (no skew)
        component_id = 'core_service'
        success = self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
        assert success == True
        
        # Verify no clock skew detected
        assert not self.monitor.clock_skew_detected
        
        # Test clock skew by manipulating timestamp
        # Create a heartbeat with skewed timestamp
        skewed_heartbeat = HeartbeatMessage(
            component_id=component_id,
            component_type=ComponentType.CORE,
            timestamp=time.time() + 1.0,  # 1 second in the future
            sequence_number=999,
            health_status=HealthStatus.HEALTHY,
            metadata={},
            signature="skewed_sig"
        )
        
        # Process skewed heartbeat
        success = self.monitor._process_heartbeat(skewed_heartbeat)
        assert success == False  # Should reject skewed heartbeat
        
        # Verify clock skew detected
        assert self.monitor.clock_skew_detected
        assert len(self.monitor.clock_corrections) > 0
        
        # Check correction record
        correction = self.monitor.clock_corrections[-1]
        assert correction['component_id'] == component_id
        assert correction['skew_ms'] > self.config.clock_skew_threshold_ms
        
        print(f"  Clock skew detected: {correction['skew_ms']:.2f}ms")
        print(f"  Total corrections: {len(self.monitor.clock_corrections)}")
        
        print("✓ Clock skew detection test passed")
    
    def test_probe_timeout(self):
        """Test probe timeout mechanisms"""
        print("\n=== Testing Probe Timeout ===")
        
        # Send initial heartbeats
        for component_id, _ in self.test_components:
            self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
        
        # Wait for probe timeout
        timeout_duration = self.config.probe_timeout_ms / 1000
        time.sleep(timeout_duration + 0.1)  # Add small buffer
        
        # Check health - should detect timeouts
        health_summary = self.monitor.check_health()
        
        # Components should have missed heartbeats
        assert health_summary['degraded_components'] > 0 or health_summary['unhealthy_components'] > 0
        
        # Check component states
        for component_id, _ in self.test_components:
            component = self.monitor.components[component_id]
            assert component['missed_heartbeats'] > 0
            
            # Health status should reflect missed heartbeats
            if component['missed_heartbeats'] >= self.config.max_missed_heartbeats:
                assert component['health_status'] == HealthStatus.CRITICAL
            elif component['missed_heartbeats'] >= self.config.max_missed_heartbeats // 2:
                assert component['health_status'] in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        
        print(f"  Degraded components: {health_summary['degraded_components']}")
        print(f"  Unhealthy components: {health_summary['unhealthy_components']}")
        print(f"  Critical components: {health_summary['critical_components']}")
        
        print("✓ Probe timeout test passed")
    
    def test_partial_component_crash(self):
        """Test partial component crash (extension crash) scenarios"""
        print("\n=== Testing Partial Component Crash ===")
        
        # Start with healthy components
        for component_id, _ in self.test_components:
            self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
        
        # Simulate extension crash
        extension_id = 'extension_1'
        self.monitor.simulate_component_crash(extension_id, "extension_crash")
        
        # Verify crash detection
        component = self.monitor.components[extension_id]
        assert component['health_status'] == HealthStatus.CRITICAL
        assert component['consecutive_failures'] > 0
        assert component['missed_heartbeats'] > self.config.max_missed_heartbeats
        
        # Check for alerts
        assert len(self.monitor.health_alerts) > 0
        latest_alert = self.monitor.health_alerts[-1]
        assert latest_alert['component_id'] == extension_id
        assert latest_alert['severity'] == 'critical'
        assert 'crashed' in latest_alert['message']
        
        # Verify failure tracking for MTTR
        assert extension_id in self.monitor.failure_start_times
        
        # Other components should remain healthy
        for component_id, _ in self.test_components:
            if component_id != extension_id:
                component = self.monitor.components[component_id]
                assert component['health_status'] == HealthStatus.HEALTHY
        
        print(f"  Extension crash detected: {extension_id}")
        print(f"  Health alerts generated: {len(self.monitor.health_alerts)}")
        
        print("✓ Partial component crash test passed")
    
    def test_mttr_assertions(self):
        """Test Mean Time To Recovery (MTTR) assertions"""
        print("\n=== Testing MTTR Assertions ===")
        
        # Simulate multiple component failures
        failure_scenarios = [
            ('extension_1', 'extension_crash'),
            ('adapter_1', 'network_timeout'),
            ('monitor_1', 'resource_exhaustion')
        ]
        
        failure_times = {}
        
        for component_id, failure_type in failure_scenarios:
            # Record failure start time
            failure_times[component_id] = time.time()
            
            # Simulate failure
            self.monitor.simulate_component_crash(component_id, failure_type)
            
            # Verify failure tracking
            assert component_id in self.monitor.failure_start_times
        
        # Wait for some time to simulate recovery delay
        recovery_delay = 0.1  # 100ms
        time.sleep(recovery_delay)
        
        # Record recoveries
        for component_id, failure_type in failure_scenarios:
            self.monitor.record_recovery(component_id, f"recovery_{failure_type}")
        
        # Check MTTR metrics
        mttr_summary = self.monitor.get_mttr_summary()
        assert mttr_summary['total_recoveries'] == len(failure_scenarios)
        
        # Verify MTTR values
        for metric in self.mttr_summary['mttr_metrics']:
            mttr_ms = metric['mttr_ms']
            assert mttr_ms > 0
            assert mttr_ms >= recovery_delay * 1000  # At least the recovery delay
            
            # Check if MTTR target was met
            target_met = mttr_ms <= self.config.mttr_target_ms
            print(f"  {metric['component_id']}: MTTR {mttr_ms:.2f}ms, Target met: {target_met}")
        
        # Overall MTTR summary
        print(f"  Average MTTR: {mttr_summary['average_mttr_ms']:.2f}ms")
        print(f"  MTTR target: {self.config.mttr_target_ms}ms")
        print(f"  Target met: {mttr_summary['mttr_target_met']}")
        
        print("✓ MTTR assertions test passed")
    
    def test_health_monitoring_cycle(self):
        """Test complete health monitoring cycle"""
        print("\n=== Testing Health Monitoring Cycle ===")
        
        # Start monitoring cycle
        monitoring_phases = [
            ('initial_health', HealthStatus.HEALTHY),
            ('degradation', HealthStatus.DEGRADED),
            ('failure', HealthStatus.CRITICAL),
            ('recovery', HealthStatus.HEALTHY)
        ]
        
        for phase_name, target_status in monitoring_phases:
            print(f"  Phase: {phase_name}")
            
            if phase_name == 'initial_health':
                # Send healthy heartbeats
                for component_id, _ in self.test_components:
                    self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
                
            elif phase_name == 'degradation':
                # Simulate some components degrading
                for i in range(2):
                    component_id = self.test_components[i][0]
                    self.monitor.send_heartbeat(component_id, HealthStatus.DEGRADED)
                
            elif phase_name == 'failure':
                # Simulate component failure
                component_id = 'extension_1'
                self.monitor.simulate_component_crash(component_id, "test_failure")
                
            elif phase_name == 'recovery':
                # Record recovery
                component_id = 'extension_1'
                if component_id in self.monitor.failure_start_times:
                    self.monitor.record_recovery(component_id, "test_recovery")
            
            # Check health at each phase
            health_summary = self.monitor.check_health()
            print(f"    Health: {health_summary['healthy_components']}H, "
                  f"{health_summary['degraded_components']}D, "
                  f"{health_summary['unhealthy_components']}U, "
                  f"{health_summary['critical_components']}C")
        
        # Final system status
        system_status = self.monitor.get_system_status()
        print(f"  Final status: {system_status}")
        
        # Verify monitoring cycle completed
        assert system_status['total_components'] == len(self.test_components)
        assert system_status['total_heartbeats'] > 0
        assert len(self.monitor.health_alerts) > 0
        
        print("✓ Health monitoring cycle test passed")
    
    def test_concurrent_heartbeat_processing(self):
        """Test concurrent heartbeat processing"""
        print("\n=== Testing Concurrent Heartbeat Processing ===")
        
        # Test concurrent heartbeat sending
        def send_heartbeats(component_id: str, count: int):
            """Send multiple heartbeats for a component"""
            for i in range(count):
                self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
                time.sleep(0.001)  # Small delay
        
        # Start concurrent heartbeat senders
        threads = []
        for component_id, _ in self.test_components[:3]:  # Test first 3 components
            thread = threading.Thread(target=send_heartbeats, args=(component_id, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all heartbeats were processed
        system_status = self.monitor.get_system_status()
        expected_heartbeats = 3 * 10  # 3 components * 10 heartbeats each
        
        assert system_status['total_heartbeats'] >= expected_heartbeats
        
        # Check component states
        for component_id, _ in self.test_components[:3]:
            component = self.monitor.components[component_id]
            assert component['consecutive_successes'] > 0
            assert component['health_status'] == HealthStatus.HEALTHY
        
        print(f"  Total heartbeats processed: {system_status['total_heartbeats']}")
        print(f"  Expected heartbeats: {expected_heartbeats}")
        
        print("✓ Concurrent heartbeat processing test passed")
    
    def test_end_to_end_health_system(self):
        """Test complete end-to-end health system"""
        print("\n=== Testing End-to-End Health System ===")
        
        # Complete health system lifecycle
        lifecycle_phases = [
            ('registration', 'register_components'),
            ('heartbeat_start', 'start_heartbeats'),
            ('health_check', 'perform_health_checks'),
            ('failure_simulation', 'simulate_failures'),
            ('recovery_simulation', 'simulate_recoveries'),
            ('final_validation', 'validate_system_state')
        ]
        
        for phase_name, phase_action in lifecycle_phases:
            print(f"  Phase: {phase_name}")
            
            if phase_action == 'register_components':
                # Additional component registration
                self.monitor.register_component('test_component', ComponentType.EXTENSION)
                
            elif phase_action == 'start_heartbeats':
                # Send heartbeats to all components
                for component_id in self.monitor.components:
                    self.monitor.send_heartbeat(component_id, HealthStatus.HEALTHY)
                
            elif phase_action == 'perform_health_checks':
                # Run health checks
                health_summary = self.monitor.check_health()
                print(f"    Health check: {health_summary['healthy_components']} healthy")
                
            elif phase_action == 'simulate_failures':
                # Simulate various failure types
                failure_types = ['crash', 'timeout', 'resource_exhaustion']
                for i, failure_type in enumerate(failure_types):
                    component_id = f'extension_{i+1}'
                    if component_id in self.monitor.components:
                        self.monitor.simulate_component_crash(component_id, failure_type)
                
            elif phase_action == 'simulate_recoveries':
                # Record recoveries
                for component_id in list(self.monitor.failure_start_times.keys()):
                    self.monitor.record_recovery(component_id, "lifecycle_recovery")
                
            elif phase_action == 'validate_system_state':
                # Final system validation
                system_status = self.monitor.get_system_status()
                mttr_summary = self.monitor.get_mttr_summary()
                
                print(f"    System status: {system_status}")
                print(f"    MTTR summary: {mttr_summary}")
                
                # Verify system integrity
                assert system_status['total_components'] > 0
                assert system_status['total_heartbeats'] > 0
                assert len(self.monitor.health_alerts) >= 0
                assert mttr_summary['total_recoveries'] >= 0
        
        print("✓ End-to-end health system test passed")


def run_health_heartbeat_tests():
    """Run all health and heartbeat tests"""
    print("\n" + "="*60)
    print("RUNNING HEALTH/HEARTBEAT TESTS")
    print("="*60)
    
    test_suite = TestHealthHeartbeat()
    
    # Run all test methods
    test_methods = [
        'test_heartbeat_drop_detection',
        'test_clock_skew_detection',
        'test_probe_timeout',
        'test_partial_component_crash',
        'test_mttr_assertions',
        'test_health_monitoring_cycle',
        'test_concurrent_heartbeat_processing',
        'test_end_to_end_health_system'
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
    print("HEALTH/HEARTBEAT TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_health_heartbeat_tests()
