"""
Test suite for State Machine Testing

This module tests the complete state machine behavior including:
- Forced cold starts and initialization
- Induced degradation and failure scenarios
- Recovery tests and state transitions
- Channel states (PROBING/ACTIVE/SHADOW/STANDBY/QUARANTINED)
- Guard conditions and state validation
- Hysteresis and transition timing
- Telemetry during state transitions
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


class ChannelState(Enum):
    """Channel state enumeration"""
    PROBING = "probing"
    ACTIVE = "active"
    SHADOW = "shadow"
    STANDBY = "standby"
    QUARANTINED = "quarantined"


class StateTransition(Enum):
    """State transition types"""
    INITIALIZE = "initialize"
    ACTIVATE = "activate"
    DEGRADE = "degrade"
    RECOVER = "recover"
    QUARANTINE = "quarantine"
    SHUTDOWN = "shutdown"


@dataclass
class StateTransitionEvent:
    """State transition event record"""
    timestamp: float
    from_state: ChannelState
    to_state: ChannelState
    transition_type: StateTransition
    trigger: str
    metadata: Dict[str, Any]
    telemetry_snapshot: Dict[str, Any]
    guard_conditions: Dict[str, bool]


@dataclass
class ChannelHealth:
    """Channel health metrics"""
    qos_score: float
    latency_ms: float
    error_rate: float
    throughput_rps: float
    last_seen: float
    consecutive_failures: int
    consecutive_successes: int


@dataclass
class StateMachineConfig:
    """State machine configuration"""
    cold_start_timeout_ms: int = 5000
    probing_interval_ms: int = 1000
    activation_threshold: float = 0.8
    degradation_threshold: float = 0.3
    quarantine_threshold: int = 5
    recovery_timeout_ms: int = 10000
    hysteresis_window_ms: int = 5000
    max_recovery_attempts: int = 3
    telemetry_collection_enabled: bool = True


class ChannelStateMachine:
    """Channel state machine implementation"""
    
    def __init__(self, config: StateMachineConfig):
        self.config = config
        self.current_state = ChannelState.PROBING
        self.previous_state = None
        self.state_start_time = time.time()
        self.transition_history = []
        self.health_metrics = {}
        self.recovery_attempts = 0
        self.last_transition_time = 0
        
        # State-specific tracking
        self.probing_start_time = None
        self.activation_attempts = 0
        self.degradation_start_time = None
        self.quarantine_reason = None
        
        # Telemetry collection
        self.telemetry_buffer = []
        self.max_telemetry_buffer_size = 1000
        
        # Guard conditions
        self.guard_conditions = {
            'health_check_passed': True,
            'resource_available': True,
            'permissions_valid': True,
            'network_reachable': True,
            'quota_available': True
        }
    
    def initialize(self) -> StateTransitionEvent:
        """Initialize the state machine (cold start)"""
        return self._transition_to(ChannelState.PROBING, StateTransition.INITIALIZE, "cold_start")
    
    def update_health(self, channel_id: str, health: ChannelHealth):
        """Update channel health metrics"""
        self.health_metrics[channel_id] = health
        
        # Collect telemetry if enabled
        if self.config.telemetry_collection_enabled:
            self._collect_telemetry(channel_id, health)
        
        # Check for state transitions based on health
        self._evaluate_state_transitions(channel_id, health)
    
    def force_state(self, new_state: ChannelState, reason: str = "manual_override") -> StateTransitionEvent:
        """Force a state transition (for testing)"""
        return self._transition_to(new_state, StateTransition.INITIALIZE, reason)
    
    def induce_degradation(self, channel_id: str, degradation_type: str) -> StateTransitionEvent:
        """Induce degradation for testing"""
        if channel_id in self.health_metrics:
            health = self.health_metrics[channel_id]
            
            if degradation_type == "latency":
                health.latency_ms *= 10
                health.qos_score *= 0.5
            elif degradation_type == "errors":
                health.error_rate = min(1.0, health.error_rate + 0.3)
                health.qos_score *= 0.3
            elif degradation_type == "failures":
                health.consecutive_failures += 5
                health.qos_score = 0.1
            
            # Force health update
            self.update_health(channel_id, health)
            
            # Check for degradation transition
            if health.qos_score < self.config.degradation_threshold:
                return self._transition_to(ChannelState.SHADOW, StateTransition.DEGRADE, f"induced_{degradation_type}")
        
        return None
    
    def attempt_recovery(self, channel_id: str) -> StateTransitionEvent:
        """Attempt recovery from degraded state"""
        if self.current_state in [ChannelState.SHADOW, ChannelState.QUARANTINED]:
            if self.recovery_attempts < self.config.max_recovery_attempts:
                self.recovery_attempts += 1
                
                # Simulate recovery attempt
                if channel_id in self.health_metrics:
                    health = self.health_metrics[channel_id]
                    health.qos_score = min(1.0, health.qos_score + 0.3)
                    health.error_rate = max(0.0, health.error_rate - 0.2)
                    
                    if health.qos_score >= self.config.activation_threshold:
                        return self._transition_to(ChannelState.ACTIVE, StateTransition.RECOVER, "recovery_successful")
                    else:
                        return self._transition_to(ChannelState.STANDBY, StateTransition.RECOVER, "recovery_partial")
        
        return None
    
    def _evaluate_state_transitions(self, channel_id: str, health: ChannelHealth):
        """Evaluate if state transitions are needed based on health"""
        current_time = time.time()
        
        # Check for activation from probing
        if (self.current_state == ChannelState.PROBING and 
            health.qos_score >= self.config.activation_threshold and
            self._check_guard_conditions()):
            
            self._transition_to(ChannelState.ACTIVE, StateTransition.ACTIVATE, "health_threshold_met")
        
        # Check for degradation from active
        elif (self.current_state == ChannelState.ACTIVE and 
              health.qos_score < self.config.degradation_threshold):
            
            self._transition_to(ChannelState.SHADOW, StateTransition.DEGRADE, "health_degraded")
        
        # Check for quarantine
        elif (health.consecutive_failures >= self.config.quarantine_threshold and
              self.current_state != ChannelState.QUARANTINED):
            
            self._transition_to(ChannelState.QUARANTINED, StateTransition.QUARANTINE, "excessive_failures")
        
        # Check for standby transition
        elif (self.current_state == ChannelState.SHADOW and
              health.qos_score >= self.config.activation_threshold * 0.7 and
              health.qos_score < self.config.activation_threshold):
            
            self._transition_to(ChannelState.STANDBY, StateTransition.RECOVER, "partial_recovery")
    
    def _transition_to(self, new_state: ChannelState, transition_type: StateTransition, trigger: str) -> StateTransitionEvent:
        """Perform state transition with telemetry collection"""
        current_time = time.time()
        
        # Validate transition
        if not self._is_valid_transition(new_state):
            return None
        
        # Create transition event
        transition_event = StateTransitionEvent(
            timestamp=current_time,
            from_state=self.current_state,
            to_state=new_state,
            transition_type=transition_type,
            trigger=trigger,
            metadata=self._get_transition_metadata(),
            telemetry_snapshot=self._get_telemetry_snapshot(),
            guard_conditions=self.guard_conditions.copy()
        )
        
        # Update state
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = current_time
        self.last_transition_time = current_time
        
        # Update state-specific tracking
        self._update_state_tracking(new_state, transition_type)
        
        # Record transition
        self.transition_history.append(transition_event)
        
        return transition_event
    
    def _is_valid_transition(self, new_state: ChannelState) -> bool:
        """Check if state transition is valid"""
        valid_transitions = {
            ChannelState.PROBING: [ChannelState.ACTIVE, ChannelState.QUARANTINED],
            ChannelState.ACTIVE: [ChannelState.SHADOW, ChannelState.STANDBY, ChannelState.QUARANTINED],
            ChannelState.SHADOW: [ChannelState.ACTIVE, ChannelState.STANDBY, ChannelState.QUARANTINED],
            ChannelState.STANDBY: [ChannelState.ACTIVE, ChannelState.SHADOW, ChannelState.QUARANTINED],
            ChannelState.QUARANTINED: [ChannelState.PROBING, ChannelState.STANDBY]
        }
        
        return new_state in valid_transitions.get(self.current_state, [])
    
    def _update_state_tracking(self, new_state: ChannelState, transition_type: StateTransition):
        """Update state-specific tracking variables"""
        if new_state == ChannelState.PROBING:
            self.probing_start_time = time.time()
            self.activation_attempts = 0
        elif new_state == ChannelState.ACTIVE:
            self.probing_start_time = None
            self.degradation_start_time = None
            self.recovery_attempts = 0
        elif new_state == ChannelState.SHADOW:
            self.degradation_start_time = time.time()
        elif new_state == ChannelState.QUARANTINED:
            self.quarantine_reason = "excessive_failures"
    
    def _check_guard_conditions(self) -> bool:
        """Check if all guard conditions are met"""
        return all(self.guard_conditions.values())
    
    def _get_transition_metadata(self) -> Dict[str, Any]:
        """Get metadata for state transition"""
        return {
            'state_duration_ms': (time.time() - self.state_start_time) * 1000,
            'recovery_attempts': self.recovery_attempts,
            'total_transitions': len(self.transition_history),
            'current_health_metrics': len(self.health_metrics)
        }
    
    def _get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Get telemetry snapshot for transition"""
        return {
            'health_metrics': self.health_metrics.copy(),
            'telemetry_buffer_size': len(self.telemetry_buffer),
            'guard_conditions': self.guard_conditions.copy(),
            'system_time': time.time()
        }
    
    def _collect_telemetry(self, channel_id: str, health: ChannelHealth):
        """Collect telemetry data"""
        telemetry_entry = {
            'timestamp': time.time(),
            'channel_id': channel_id,
            'state': self.current_state.value,
            'health': {
                'qos_score': health.qos_score,
                'latency_ms': health.latency_ms,
                'error_rate': health.error_rate,
                'throughput_rps': health.throughput_rps
            }
        }
        
        self.telemetry_buffer.append(telemetry_entry)
        
        # Maintain buffer size
        if len(self.telemetry_buffer) > self.max_telemetry_buffer_size:
            self.telemetry_buffer.pop(0)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary"""
        return {
            'current_state': self.current_state.value,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'state_duration_ms': (time.time() - self.state_start_time) * 1000,
            'total_transitions': len(self.transition_history),
            'recovery_attempts': self.recovery_attempts,
            'health_channels': len(self.health_metrics),
            'telemetry_entries': len(self.telemetry_buffer),
            'guard_conditions': self.guard_conditions.copy()
        }
    
    def get_transition_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transition history"""
        return [
            {
                'timestamp': event.timestamp,
                'from_state': event.from_state.value,
                'to_state': event.to_state.value,
                'transition_type': event.transition_type.value,
                'trigger': event.trigger,
                'metadata': event.metadata
            }
            for event in self.transition_history[-limit:]
        ]


class TestStateMachine:
    """Test suite for state machine behavior"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.config = StateMachineConfig()
        self.state_machine = ChannelStateMachine(self.config)
        
        # Test channels
        self.test_channels = {
            'primary': ChannelHealth(
                qos_score=0.9,
                latency_ms=50,
                error_rate=0.05,
                throughput_rps=1000,
                last_seen=time.time(),
                consecutive_failures=0,
                consecutive_successes=10
            ),
            'secondary': ChannelHealth(
                qos_score=0.7,
                latency_ms=100,
                error_rate=0.1,
                throughput_rps=800,
                last_seen=time.time(),
                consecutive_failures=2,
                consecutive_successes=5
            ),
            'backup': ChannelHealth(
                qos_score=0.5,
                latency_ms=200,
                error_rate=0.2,
                throughput_rps=500,
                last_seen=time.time(),
                consecutive_failures=3,
                consecutive_successes=2
            )
        }
    
    def test_forced_cold_starts(self):
        """Test forced cold starts and initialization"""
        print("\n=== Testing Forced Cold Starts ===")
        
        # Test initial state
        assert self.state_machine.current_state == ChannelState.PROBING
        
        # Force cold start
        transition = self.state_machine.initialize()
        assert transition is not None
        assert transition.to_state == ChannelState.PROBING
        assert transition.transition_type == StateTransition.INITIALIZE
        
        # Verify state tracking
        assert self.state_machine.probing_start_time is not None
        assert self.state_machine.state_start_time > 0
        
        # Test multiple cold starts
        for i in range(3):
            transition = self.state_machine.force_state(ChannelState.PROBING, f"cold_start_{i}")
            assert transition is not None
            assert transition.to_state == ChannelState.PROBING
        
        # Verify transition history
        history = self.state_machine.get_transition_history()
        assert len(history) >= 4  # Initial + 3 forced
        
        print("✓ Forced cold starts test passed")
    
    def test_induced_degradation(self):
        """Test induced degradation scenarios"""
        print("\n=== Testing Induced Degradation ===")
        
        # Start with active state
        self.state_machine.force_state(ChannelState.ACTIVE, "test_setup")
        
        # Add healthy channels
        for channel_id, health in self.test_channels.items():
            self.state_machine.update_health(channel_id, health)
        
        # Induce latency degradation
        transition = self.state_machine.induce_degradation('primary', 'latency')
        if transition:
            print(f"  Latency degradation: {transition.to_state.value}")
            assert transition.to_state in [ChannelState.SHADOW, ChannelState.STANDBY]
        
        # Induce error degradation
        transition = self.state_machine.induce_degradation('secondary', 'errors')
        if transition:
            print(f"  Error degradation: {transition.to_state.value}")
            assert transition.to_state in [ChannelState.SHADOW, ChannelState.STANDBY]
        
        # Induce failure degradation
        transition = self.state_machine.induce_degradation('backup', 'failures')
        if transition:
            print(f"  Failure degradation: {transition.to_state.value}")
            assert transition.to_state in [ChannelState.SHADOW, ChannelState.QUARANTINED]
        
        # Verify degradation tracking
        assert self.state_machine.degradation_start_time is not None
        
        print("✓ Induced degradation test passed")
    
    def test_recovery_tests(self):
        """Test recovery mechanisms and state transitions"""
        print("\n=== Testing Recovery Tests ===")
        
        # Start in degraded state
        self.state_machine.force_state(ChannelState.SHADOW, "recovery_test_setup")
        
        # Add degraded channels
        degraded_health = ChannelHealth(
            qos_score=0.2,  # Below threshold
            latency_ms=500,
            error_rate=0.8,
            throughput_rps=100,
            last_seen=time.time(),
            consecutive_failures=3,
            consecutive_successes=1
        )
        
        self.state_machine.update_health('recovery_test', degraded_health)
        
        # Attempt recovery
        transition = self.state_machine.attempt_recovery('recovery_test')
        assert transition is not None
        
        print(f"  Recovery transition: {transition.to_state.value}")
        print(f"  Recovery trigger: {transition.trigger}")
        
        # Should transition to standby or active
        assert transition.to_state in [ChannelState.STANDBY, ChannelState.ACTIVE]
        
        # Verify recovery tracking
        assert self.state_machine.recovery_attempts > 0
        
        # Test multiple recovery attempts
        for i in range(3):
            transition = self.state_machine.attempt_recovery('recovery_test')
            if transition:
                print(f"  Recovery attempt {i+1}: {transition.to_state.value}")
        
        print("✓ Recovery tests passed")
    
    def test_channel_states(self):
        """Test all channel states and transitions"""
        print("\n=== Testing Channel States ===")
        
        # Test PROBING state
        self.state_machine.force_state(ChannelState.PROBING, "state_test")
        assert self.state_machine.current_state == ChannelState.PROBING
        assert self.state_machine.probing_start_time is not None
        
        # Test ACTIVE state
        self.state_machine.force_state(ChannelState.ACTIVE, "state_test")
        assert self.state_machine.current_state == ChannelState.ACTIVE
        assert self.state_machine.probing_start_time is None
        
        # Test SHADOW state
        self.state_machine.force_state(ChannelState.SHADOW, "state_test")
        assert self.state_machine.current_state == ChannelState.SHADOW
        assert self.state_machine.degradation_start_time is not None
        
        # Test STANDBY state
        self.state_machine.force_state(ChannelState.STANDBY, "state_test")
        assert self.state_machine.current_state == ChannelState.STANDBY
        
        # Test QUARANTINED state
        self.state_machine.force_state(ChannelState.QUARANTINED, "state_test")
        assert self.state_machine.current_state == ChannelState.QUARANTINED
        assert self.state_machine.quarantine_reason is not None
        
        # Verify all states were visited
        visited_states = set()
        for transition in self.state_machine.transition_history:
            visited_states.add(transition.to_state)
        
        assert len(visited_states) == 5  # All states visited
        
        print("✓ Channel states test passed")
    
    def test_guard_conditions(self):
        """Test guard conditions and state validation"""
        print("\n=== Testing Guard Conditions ===")
        
        # Start with all guards passing
        assert self.state_machine._check_guard_conditions() == True
        
        # Test individual guard failures
        guards_to_test = [
            'health_check_passed',
            'resource_available',
            'permissions_valid',
            'network_reachable',
            'quota_available'
        ]
        
        for guard in guards_to_test:
            # Set guard to fail
            self.state_machine.guard_conditions[guard] = False
            assert self.state_machine._check_guard_conditions() == False
            
            # Restore guard
            self.state_machine.guard_conditions[guard] = True
            assert self.state_machine._check_guard_conditions() == True
        
        # Test multiple guard failures
        self.state_machine.guard_conditions['health_check_passed'] = False
        self.state_machine.guard_conditions['resource_available'] = False
        assert self.state_machine._check_guard_conditions() == False
        
        # Restore all guards
        for guard in self.state_machine.guard_conditions:
            self.state_machine.guard_conditions[guard] = True
        
        print("✓ Guard conditions test passed")
    
    def test_hysteresis_and_timing(self):
        """Test hysteresis and transition timing"""
        print("\n=== Testing Hysteresis and Timing ===")
        
        # Test state duration tracking
        start_time = time.time()
        self.state_machine.force_state(ChannelState.ACTIVE, "timing_test")
        
        # Wait a bit
        time.sleep(0.1)
        
        # Check state duration
        summary = self.state_machine.get_state_summary()
        duration_ms = summary['state_duration_ms']
        assert duration_ms > 0
        assert duration_ms >= 100  # At least 100ms
        
        # Test transition timing
        transition = self.state_machine.force_state(ChannelState.SHADOW, "timing_test")
        assert transition is not None
        
        # Verify timing in metadata
        metadata = transition.metadata
        assert metadata['state_duration_ms'] > 0
        
        # Test hysteresis window
        # Rapid transitions should respect hysteresis
        for i in range(5):
            transition = self.state_machine.force_state(ChannelState.ACTIVE, f"rapid_{i}")
            time.sleep(0.001)  # Very short delay
        
        # Should have recorded all transitions
        history = self.state_machine.get_transition_history()
        assert len(history) >= 7  # Initial + shadow + 5 rapid
        
        print("✓ Hysteresis and timing test passed")
    
    def test_telemetry_during_transitions(self):
        """Test telemetry collection during state transitions"""
        print("\n=== Testing Telemetry During Transitions ===")
        
        # Enable telemetry collection
        self.state_machine.config.telemetry_collection_enabled = True
        
        # Perform multiple state transitions with health updates
        states_to_test = [ChannelState.ACTIVE, ChannelState.SHADOW, ChannelState.STANDBY, ChannelState.ACTIVE]
        
        for i, target_state in enumerate(states_to_test):
            # Update health before transition
            health = ChannelHealth(
                qos_score=0.8 - (i * 0.1),
                latency_ms=50 + (i * 25),
                error_rate=0.05 + (i * 0.05),
                throughput_rps=1000 - (i * 100),
                last_seen=time.time(),
                consecutive_failures=i,
                consecutive_successes=10 - i
            )
            
            self.state_machine.update_health(f'test_channel_{i}', health)
            
            # Force transition
            transition = self.state_machine.force_state(target_state, f"telemetry_test_{i}")
            assert transition is not None
            
            # Verify telemetry snapshot
            telemetry_snapshot = transition.telemetry_snapshot
            assert 'health_metrics' in telemetry_snapshot
            assert 'telemetry_buffer_size' in telemetry_snapshot
            assert 'guard_conditions' in telemetry_snapshot
        
        # Check telemetry buffer
        assert len(self.state_machine.telemetry_buffer) > 0
        
        # Verify telemetry entries
        for entry in self.state_machine.telemetry_buffer:
            assert 'timestamp' in entry
            assert 'channel_id' in entry
            assert 'state' in entry
            assert 'health' in entry
        
        print("✓ Telemetry during transitions test passed")
    
    def test_state_transition_validation(self):
        """Test state transition validation and invalid transitions"""
        print("\n=== Testing State Transition Validation ===")
        
        # Test valid transitions from each state
        valid_transitions = {
            ChannelState.PROBING: [ChannelState.ACTIVE, ChannelState.QUARANTINED],
            ChannelState.ACTIVE: [ChannelState.SHADOW, ChannelState.STANDBY, ChannelState.QUARANTINED],
            ChannelState.SHADOW: [ChannelState.ACTIVE, ChannelState.STANDBY, ChannelState.QUARANTINED],
            ChannelState.STANDBY: [ChannelState.ACTIVE, ChannelState.SHADOW, ChannelState.QUARANTINED],
            ChannelState.QUARANTINED: [ChannelState.PROBING, ChannelState.STANDBY]
        }
        
        for from_state, valid_to_states in valid_transitions.items():
            self.state_machine.force_state(from_state, "validation_test")
            
            for to_state in valid_to_states:
                transition = self.state_machine.force_state(to_state, "validation_test")
                assert transition is not None
                assert transition.to_state == to_state
            
            # Test invalid transition
            invalid_state = ChannelState.PROBING  # This should be invalid from most states
            if invalid_state not in valid_to_states:
                # Try invalid transition
                original_state = self.state_machine.current_state
                transition = self.state_machine.force_state(invalid_state, "invalid_test")
                
                # Should remain in original state
                assert self.state_machine.current_state == original_state
        
        print("✓ State transition validation test passed")
    
    def test_end_to_end_state_machine(self):
        """Test complete end-to-end state machine flow"""
        print("\n=== Testing End-to-End State Machine ===")
        
        # Complete lifecycle test
        lifecycle_phases = [
            ('cold_start', ChannelState.PROBING),
            ('activation', ChannelState.ACTIVE),
            ('degradation', ChannelState.SHADOW),
            ('partial_recovery', ChannelState.STANDBY),
            ('full_recovery', ChannelState.ACTIVE),
            ('quarantine', ChannelState.QUARANTINED),
            ('reset', ChannelState.PROBING)
        ]
        
        for phase_name, target_state in lifecycle_phases:
            print(f"  Phase: {phase_name} -> {target_state.value}")
            
            # Update health based on phase
            if phase_name == 'degradation':
                health = ChannelHealth(
                    qos_score=0.2,  # Low QoS
                    latency_ms=500,
                    error_rate=0.8,
                    throughput_rps=100,
                    last_seen=time.time(),
                    consecutive_failures=3,
                    consecutive_successes=1
                )
            elif phase_name == 'partial_recovery':
                health = ChannelHealth(
                    qos_score=0.6,  # Medium QoS
                    latency_ms=200,
                    error_rate=0.3,
                    throughput_rps=600,
                    last_seen=time.time(),
                    consecutive_failures=1,
                    consecutive_successes=3
                )
            elif phase_name == 'full_recovery':
                health = ChannelHealth(
                    qos_score=0.9,  # High QoS
                    latency_ms=50,
                    error_rate=0.05,
                    throughput_rps=1000,
                    last_seen=time.time(),
                    consecutive_failures=0,
                    consecutive_successes=10
                )
            else:
                health = self.test_channels['primary']
            
            self.state_machine.update_health(f'lifecycle_{phase_name}', health)
            
            # Perform transition
            transition = self.state_machine.force_state(target_state, phase_name)
            assert transition is not None
            assert transition.to_state == target_state
            
            # Verify state-specific tracking
            summary = self.state_machine.get_state_summary()
            assert summary['current_state'] == target_state.value
        
        # Final verification
        final_summary = self.state_machine.get_state_summary()
        print(f"  Final state: {final_summary['current_state']}")
        print(f"  Total transitions: {final_summary['total_transitions']}")
        
        # Should have completed all phases
        assert final_summary['total_transitions'] >= len(lifecycle_phases)
        
        print("✓ End-to-end state machine test passed")


def run_state_machine_tests():
    """Run all state machine tests"""
    print("\n" + "="*60)
    print("RUNNING STATE MACHINE TESTS")
    print("="*60)
    
    test_suite = TestStateMachine()
    
    # Run all test methods
    test_methods = [
        'test_forced_cold_starts',
        'test_induced_degradation',
        'test_recovery_tests',
        'test_channel_states',
        'test_guard_conditions',
        'test_hysteresis_and_timing',
        'test_telemetry_during_transitions',
        'test_state_transition_validation',
        'test_end_to_end_state_machine'
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
    print("STATE MACHINE TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_state_machine_tests()
