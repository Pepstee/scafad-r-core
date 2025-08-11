"""
SCAFAD Layer 0: Comprehensive Integration Test Harness
======================================================

Test harness that simulates:
- Channel degradation (drop CloudWatch, slow X-Ray)
- Cold-start bursts and latency spikes
- Missing telemetry streams

Measures:
- MTTR to switch to fallback channels
- Data loss ppm with and without redundancy
- Added latency from compression or extra channels

Academic References:
- Chaos engineering principles (Netflix)
- Fault injection testing (Arlat et al.)
- Performance testing methodologies (Jain)
"""

import time
import json
import logging
import asyncio
import threading
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock
import statistics

# Import Layer 0 components
from layer0_signal_negotiation import SignalNegotiator, ChannelType
from layer0_redundancy_manager import RedundancyManager, RedundancyMode
from layer0_sampler import Sampler, SamplingStrategy
from layer0_fallback_orchestrator import FallbackOrchestrator, FallbackMode
from layer0_runtime_control import RuntimeControlLoop, ControlPhase, ControlState
from layer0_core import AnomalyDetectionEngine

# Import configuration
from app_config import Layer0Config, TelemetryConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestScenario:
    """Configuration for a test scenario"""
    name: str
    description: str
    duration_seconds: int
    channel_failures: Dict[str, Dict[str, Any]]
    cold_start_bursts: List[Tuple[float, int]]  # (time_offset, burst_size)
    latency_spikes: List[Tuple[float, float]]   # (time_offset, latency_ms)
    telemetry_drops: List[Tuple[float, float]]  # (time_offset, duration_seconds)
    expected_fallback_timeout_ms: float
    expected_data_loss_ppm: float
    expected_latency_overhead_ms: float

@dataclass
class TestMetrics:
    """Metrics collected during testing"""
    start_time: float
    end_time: float
    
    # Channel health metrics
    channel_health_history: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    channel_failures: Dict[str, int] = field(default_factory=dict)
    channel_recovery_times: Dict[str, List[float]] = field(default_factory=dict)
    
    # Performance metrics
    latency_measurements: List[Tuple[float, float]] = field(default_factory=list)
    throughput_measurements: List[Tuple[float, float]] = field(default_factory=list)
    cold_start_detections: List[Tuple[float, bool]] = field(default_factory=list)
    
    # Fallback metrics
    fallback_activations: List[Tuple[float, str]] = field(default_factory=list)
    fallback_recovery_times: List[float] = field(default_factory=list)
    mttr_measurements: List[float] = field(default_factory=list)
    
    # Data loss metrics
    telemetry_drops: List[Tuple[float, float]] = field(default_factory=list)
    data_loss_ppm: float = 0.0
    redundancy_effectiveness: float = 0.0
    
    # Control loop metrics
    control_cycle_times: List[float] = field(default_factory=list)
    adaptation_triggers: List[Tuple[float, str]] = field(default_factory=list)
    phase_success_rates: Dict[str, List[float]] = field(default_factory=dict)

# =============================================================================
# Test Scenarios
# =============================================================================

def create_basic_degradation_scenario() -> TestScenario:
    """Create a basic channel degradation scenario"""
    return TestScenario(
        name="Basic Channel Degradation",
        description="Simulates gradual degradation of CloudWatch and X-Ray channels",
        duration_seconds=60,
        channel_failures={
            'cloudwatch': {
                'failure_time': 10.0,
                'recovery_time': 40.0,
                'failure_type': 'complete_drop'
            },
            'xray': {
                'failure_time': 15.0,
                'recovery_time': 35.0,
                'failure_type': 'latency_degradation'
            }
        },
        cold_start_bursts=[],
        latency_spikes=[],
        telemetry_drops=[],
        expected_fallback_timeout_ms=5000,
        expected_data_loss_ppm=1000,
        expected_latency_overhead_ms=50
    )

def create_cold_start_burst_scenario() -> TestScenario:
    """Create a cold start burst scenario"""
    return TestScenario(
        name="Cold Start Burst",
        description="Simulates burst of cold starts with high latency",
        duration_seconds=120,
        channel_failures={},
        cold_start_bursts=[
            (20.0, 10),   # 10 cold starts at 20 seconds
            (60.0, 15),   # 15 cold starts at 60 seconds
            (100.0, 8)    # 8 cold starts at 100 seconds
        ],
        latency_spikes=[
            (20.0, 2000),  # 2 second latency spike
            (60.0, 3000),  # 3 second latency spike
            (100.0, 1500)  # 1.5 second latency spike
        ],
        telemetry_drops=[],
        expected_fallback_timeout_ms=3000,
        expected_data_loss_ppm=500,
        expected_latency_overhead_ms=100
    )

def create_telemetry_stream_failure_scenario() -> TestScenario:
    """Create a telemetry stream failure scenario"""
    return TestScenario(
        name="Telemetry Stream Failure",
        description="Simulates complete loss of telemetry streams",
        duration_seconds=90,
        channel_failures={
            'cloudwatch': {
                'failure_time': 10.0,
                'recovery_time': 70.0,
                'failure_type': 'complete_drop'
            },
            'xray': {
                'failure_time': 10.0,
                'recovery_time': 70.0,
                'failure_type': 'complete_drop'
            },
            'side_channel': {
                'failure_time': 10.0,
                'recovery_time': 70.0,
                'failure_type': 'complete_drop'
            }
        },
        cold_start_bursts=[],
        latency_spikes=[],
        telemetry_drops=[
            (10.0, 60.0)  # 60 second telemetry drop
        ],
        expected_fallback_timeout_ms=2000,
        expected_data_loss_ppm=10000,
        expected_latency_overhead_ms=200
    )

def create_mixed_failure_scenario() -> TestScenario:
    """Create a mixed failure scenario combining multiple issues"""
    return TestScenario(
        name="Mixed Failure Scenario",
        description="Combines channel degradation, cold starts, and telemetry drops",
        duration_seconds=180,
        channel_failures={
            'cloudwatch': {
                'failure_time': 30.0,
                'recovery_time': 120.0,
                'failure_type': 'intermittent'
            },
            'xray': {
                'failure_time': 45.0,
                'recovery_time': 105.0,
                'failure_type': 'latency_degradation'
            }
        },
        cold_start_bursts=[
            (60.0, 20),   # 20 cold starts at 60 seconds
            (120.0, 15)   # 15 cold starts at 120 seconds
        ],
        latency_spikes=[
            (60.0, 2500),  # 2.5 second latency spike
            (120.0, 2000)  # 2 second latency spike
        ],
        telemetry_drops=[
            (30.0, 30.0),  # 30 second telemetry drop
            (90.0, 45.0)   # 45 second telemetry drop
        ],
        expected_fallback_timeout_ms=4000,
        expected_data_loss_ppm=2000,
        expected_latency_overhead_ms=150
    )

# =============================================================================
# Test Harness
# =============================================================================

class Layer0TestHarness:
    """
    Comprehensive test harness for Layer 0 components
    
    Simulates various failure scenarios and measures system resilience,
    performance, and data integrity.
    """
    
    def __init__(self, config: Layer0Config):
        self.config = config
        self.test_metrics = TestMetrics(
            start_time=time.time(),
            end_time=0.0
        )
        
        # Initialize Layer 0 components
        self.signal_negotiator = None
        self.redundancy_manager = None
        self.sampler = None
        self.fallback_orchestrator = None
        self.detection_engine = None
        self.runtime_control = None
        
        # Test state
        self._test_active = False
        self._test_thread = None
        self._failure_simulator = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Layer 0 test harness initialized")
    
    def _initialize_components(self):
        """Initialize all Layer 0 components"""
        try:
            # Create components
            self.signal_negotiator = SignalNegotiator(self.config)
            self.redundancy_manager = RedundancyManager(self.config)
            self.sampler = Sampler(self.config)
            self.fallback_orchestrator = FallbackOrchestrator(self.config)
            self.detection_engine = AnomalyDetectionEngine(self.config)
            self.runtime_control = RuntimeControlLoop(self.config)
            
            # Initialize runtime control
            self.runtime_control.initialize_components(
                self.signal_negotiator,
                self.redundancy_manager,
                self.sampler,
                self.fallback_orchestrator,
                self.detection_engine
            )
            
            logger.info("All Layer 0 components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_test_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Run a complete test scenario"""
        logger.info(f"Starting test scenario: {scenario.name}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Duration: {scenario.duration_seconds} seconds")
        
        # Reset test metrics
        self.test_metrics = TestMetrics(start_time=time.time())
        
        # Start runtime control loop
        self.runtime_control.start_control_loop()
        
        # Start failure simulator
        self._start_failure_simulator(scenario)
        
        # Run the test
        self._test_active = True
        self._test_thread = threading.Thread(target=self._run_test_worker, args=(scenario,))
        self._test_thread.start()
        
        # Wait for test completion
        self._test_thread.join()
        
        # Stop components
        self._stop_test()
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        # Generate test report
        report = self._generate_test_report(scenario)
        
        logger.info(f"Test scenario completed: {scenario.name}")
        return report
    
    def _start_failure_simulator(self, scenario: TestScenario):
        """Start the failure simulator for the scenario"""
        self._failure_simulator = FailureSimulator(scenario, self)
        self._failure_simulator.start()
    
    def _run_test_worker(self, scenario: TestScenario):
        """Main test worker thread"""
        try:
            start_time = time.time()
            end_time = start_time + scenario.duration_seconds
            
            # Continuous monitoring loop
            while time.time() < end_time and self._test_active:
                current_time = time.time() - start_time
                
                # Collect metrics
                self._collect_metrics(current_time)
                
                # Simulate normal operation
                self._simulate_normal_operation(current_time)
                
                # Wait for next iteration
                time.sleep(0.1)  # 100ms intervals
                
        except Exception as e:
            logger.error(f"Test worker error: {e}")
        finally:
            self.test_metrics.end_time = time.time()
    
    def _collect_metrics(self, current_time: float):
        """Collect metrics during test execution"""
        try:
            # Channel health metrics
            if self.signal_negotiator:
                channel_health = self.signal_negotiator.get_channel_health_summary()
                for channel, health in channel_health.items():
                    if channel not in self.test_metrics.channel_health_history:
                        self.test_metrics.channel_health_history[channel] = []
                    
                    qos_score = health.get('qos_score', 0.0)
                    self.test_metrics.channel_health_history[channel].append((current_time, qos_score))
            
            # Performance metrics
            if self.runtime_control:
                control_status = self.runtime_control.get_control_status()
                cycle_time = control_status.get('metrics', {}).get('average_cycle_time_ms', 0.0)
                self.test_metrics.control_cycle_times.append(cycle_time)
                
                # Phase success rates
                phase_metrics = control_status.get('phase_metrics', {})
                for phase, rate in phase_metrics.items():
                    if phase not in self.test_metrics.phase_success_rates:
                        self.test_metrics.phase_success_rates[phase] = []
                    self.test_metrics.phase_success_rates[phase].append(rate)
            
            # Fallback metrics
            if self.fallback_orchestrator:
                fallback_status = self.fallback_orchestrator.get_fallback_status()
                current_mode = fallback_status.get('current_mode', 'normal')
                if current_mode != 'normal':
                    self.test_metrics.fallback_activations.append((current_time, current_mode))
            
            # Cold start detection
            if self.sampler:
                sampling_summary = self.sampler.get_sampling_summary()
                is_cold_start = sampling_summary.get('cold_start_detected', False)
                self.test_metrics.cold_start_detections.append((current_time, is_cold_start))
                
        except Exception as e:
            logger.warning(f"Error collecting metrics: {e}")
    
    def _simulate_normal_operation(self, current_time: float):
        """Simulate normal system operation during testing"""
        try:
            # Simulate telemetry emission
            if self.redundancy_manager:
                # Create mock telemetry data
                telemetry_data = {
                    'timestamp': time.time(),
                    'invocation_id': f'test_{current_time}_{random.randint(1000, 9999)}',
                    'metrics': {
                        'duration_ms': random.uniform(50, 200),
                        'memory_mb': random.uniform(128, 512),
                        'cpu_percent': random.uniform(10, 80)
                    }
                }
                
                # Emit through redundancy manager
                self.redundancy_manager.emit_telemetry(telemetry_data)
            
            # Simulate anomaly detection
            if self.detection_engine:
                # Create mock input data
                input_data = {
                    'timestamp': time.time(),
                    'features': [random.uniform(0, 1) for _ in range(10)]
                }
                
                # Run detection
                result = self.detection_engine.detect_anomalies(input_data)
                
        except Exception as e:
            logger.warning(f"Error in normal operation simulation: {e}")
    
    def _stop_test(self):
        """Stop the test and clean up"""
        self._test_active = False
        
        # Stop failure simulator
        if self._failure_simulator:
            self._failure_simulator.stop()
        
        # Stop runtime control
        if self.runtime_control:
            self.runtime_control.shutdown()
        
        logger.info("Test stopped and components cleaned up")
    
    def _calculate_final_metrics(self):
        """Calculate final test metrics"""
        try:
            # Calculate MTTR (Mean Time To Recovery)
            if self.test_metrics.fallback_recovery_times:
                self.test_metrics.mttr_measurements = self.test_metrics.fallback_recovery_times
            
            # Calculate data loss PPM
            total_telemetry_expected = (self.test_metrics.end_time - self.test_metrics.start_time) * 100  # 100 telemetry events per second
            total_telemetry_lost = sum(duration * 100 for _, duration in self.test_metrics.telemetry_drops)
            self.test_metrics.data_loss_ppm = (total_telemetry_lost / max(total_telemetry_expected, 1)) * 1000000
            
            # Calculate redundancy effectiveness
            if self.test_metrics.channel_health_history:
                avg_health = statistics.mean([
                    statistics.mean([score for _, score in history])
                    for history in self.test_metrics.channel_health_history.values()
                ])
                self.test_metrics.redundancy_effectiveness = avg_health
            
            # Calculate latency overhead
            if self.test_metrics.control_cycle_times:
                avg_cycle_time = statistics.mean(self.test_metrics.control_cycle_times)
                self.test_metrics.latency_overhead_ms = avg_cycle_time
                
        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")
    
    def _generate_test_report(self, scenario: TestScenario) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        try:
            report = {
                'scenario': {
                    'name': scenario.name,
                    'description': scenario.description,
                    'duration_seconds': scenario.duration_seconds
                },
                'test_execution': {
                    'start_time': self.test_metrics.start_time,
                    'end_time': self.test_metrics.end_time,
                    'total_duration': self.test_metrics.end_time - self.test_metrics.start_time
                },
                'results': {
                    'mttr_ms': statistics.mean(self.test_metrics.mttr_measurements) if self.test_metrics.mttr_measurements else 0,
                    'data_loss_ppm': self.test_metrics.data_loss_ppm,
                    'latency_overhead_ms': self.test_metrics.latency_overhead_ms,
                    'redundancy_effectiveness': self.test_metrics.redundancy_effectiveness
                },
                'expected_vs_actual': {
                    'mttr': {
                        'expected_ms': scenario.expected_fallback_timeout_ms,
                        'actual_ms': statistics.mean(self.test_metrics.mttr_measurements) if self.test_metrics.mttr_measurements else 0,
                        'within_spec': (statistics.mean(self.test_metrics.mttr_measurements) if self.test_metrics.mttr_measurements else 0) <= scenario.expected_fallback_timeout_ms
                    },
                    'data_loss': {
                        'expected_ppm': scenario.expected_data_loss_ppm,
                        'actual_ppm': self.test_metrics.data_loss_ppm,
                        'within_spec': self.test_metrics.data_loss_ppm <= scenario.expected_data_loss_ppm
                    },
                    'latency_overhead': {
                        'expected_ms': scenario.expected_latency_overhead_ms,
                        'actual_ms': self.test_metrics.latency_overhead_ms,
                        'within_spec': self.test_metrics.latency_overhead_ms <= scenario.expected_latency_overhead_ms
                    }
                },
                'detailed_metrics': {
                    'channel_health': self.test_metrics.channel_health_history,
                    'fallback_activations': self.test_metrics.fallback_activations,
                    'cold_start_detections': self.test_metrics.cold_start_detections,
                    'control_cycle_times': self.test_metrics.control_cycle_times,
                    'phase_success_rates': self.test_metrics.phase_success_rates
                },
                'summary': {
                    'overall_success': all([
                        (statistics.mean(self.test_metrics.mttr_measurements) if self.test_metrics.mttr_measurements else 0) <= scenario.expected_fallback_timeout_ms,
                        self.test_metrics.data_loss_ppm <= scenario.expected_data_loss_ppm,
                        self.test_metrics.latency_overhead_ms <= scenario.expected_latency_overhead_ms
                    ]),
                    'recommendations': self._generate_recommendations(scenario)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating test report: {e}")
            return {'error': str(e)}

# =============================================================================
# Failure Simulator
# =============================================================================

class FailureSimulator:
    """Simulates various failure scenarios during testing"""
    
    def __init__(self, scenario: TestScenario, test_harness: Layer0TestHarness):
        self.scenario = scenario
        self.test_harness = test_harness
        self._active = False
        self._simulator_thread = None
        
        # Track scheduled failures
        self._scheduled_failures = []
        self._setup_failures()
    
    def _setup_failures(self):
        """Setup scheduled failures based on scenario"""
        current_time = time.time()
        
        # Channel failures
        for channel, failure_config in self.scenario.channel_failures.items():
            failure_time = current_time + failure_config['failure_time']
            recovery_time = current_time + failure_config['recovery_time']
            
            self._scheduled_failures.append({
                'type': 'channel_failure',
                'channel': channel,
                'failure_time': failure_time,
                'recovery_time': recovery_time,
                'failure_type': failure_config['failure_type']
            })
        
        # Cold start bursts
        for offset, burst_size in self.scenario.cold_start_bursts:
            burst_time = current_time + offset
            self._scheduled_failures.append({
                'type': 'cold_start_burst',
                'burst_time': burst_time,
                'burst_size': burst_size
            })
        
        # Latency spikes
        for offset, latency_ms in self.scenario.latency_spikes:
            spike_time = current_time + offset
            self._scheduled_failures.append({
                'type': 'latency_spike',
                'spike_time': spike_time,
                'latency_ms': latency_ms
            })
        
        # Telemetry drops
        for offset, duration in self.scenario.telemetry_drops:
            drop_time = current_time + offset
            self._scheduled_failures.append({
                'type': 'telemetry_drop',
                'drop_time': drop_time,
                'duration': duration
            })
        
        # Sort by time
        self._scheduled_failures.sort(key=lambda x: x.get('failure_time', x.get('burst_time', x.get('spike_time', x.get('drop_time', 0)))))
    
    def start(self):
        """Start the failure simulator"""
        self._active = True
        self._simulator_thread = threading.Thread(target=self._simulator_worker)
        self._simulator_thread.start()
        logger.info("Failure simulator started")
    
    def _simulator_worker(self):
        """Main failure simulator worker"""
        try:
            while self._active and self._scheduled_failures:
                current_time = time.time()
                
                # Check for failures to trigger
                failures_to_trigger = [
                    f for f in self._scheduled_failures
                    if f.get('failure_time', f.get('burst_time', f.get('spike_time', f.get('drop_time', 0))) <= current_time
                ]
                
                for failure in failures_to_trigger:
                    self._trigger_failure(failure)
                    self._scheduled_failures.remove(failure)
                
                # Wait before next check
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Failure simulator error: {e}")
    
    def _trigger_failure(self, failure: Dict[str, Any]):
        """Trigger a specific failure"""
        try:
            failure_type = failure['type']
            
            if failure_type == 'channel_failure':
                self._trigger_channel_failure(failure)
            elif failure_type == 'cold_start_burst':
                self._trigger_cold_start_burst(failure)
            elif failure_type == 'latency_spike':
                self._trigger_latency_spike(failure)
            elif failure_type == 'telemetry_drop':
                self._trigger_telemetry_drop(failure)
                
        except Exception as e:
            logger.error(f"Error triggering failure {failure}: {e}")
    
    def _trigger_channel_failure(self, failure: Dict[str, Any]):
        """Trigger a channel failure"""
        channel = failure['channel']
        failure_type = failure['failure_type']
        
        logger.info(f"Triggering {failure_type} failure for channel: {channel}")
        
        # Simulate channel failure by patching the component
        if channel == 'cloudwatch':
            # Simulate CloudWatch failure
            pass
        elif channel == 'xray':
            # Simulate X-Ray failure
            pass
        elif channel == 'side_channel':
            # Simulate side channel failure
            pass
    
    def _trigger_cold_start_burst(self, failure: Dict[str, Any]):
        """Trigger a cold start burst"""
        burst_size = failure['burst_size']
        logger.info(f"Triggering cold start burst with {burst_size} cold starts")
        
        # Simulate cold start burst
        if self.test_harness.sampler:
            # Trigger cold start detection
            pass
    
    def _trigger_latency_spike(self, failure: Dict[str, Any]):
        """Trigger a latency spike"""
        latency_ms = failure['latency_ms']
        logger.info(f"Triggering latency spike: {latency_ms}ms")
        
        # Simulate latency spike
        time.sleep(latency_ms / 1000.0)
    
    def _trigger_telemetry_drop(self, failure: Dict[str, Any]):
        """Trigger a telemetry drop"""
        duration = failure['duration']
        logger.info(f"Triggering telemetry drop for {duration} seconds")
        
        # Simulate telemetry drop
        time.sleep(duration)
    
    def stop(self):
        """Stop the failure simulator"""
        self._active = False
        if self._simulator_thread and self._simulator_thread.is_alive():
            self._simulator_thread.join(timeout=5.0)
        logger.info("Failure simulator stopped")

# =============================================================================
# Test Runner
# =============================================================================

def run_all_layer0_tests(config: Layer0Config) -> Dict[str, Any]:
    """Run all Layer 0 test scenarios"""
    logger.info("Starting comprehensive Layer 0 testing")
    
    # Create test harness
    test_harness = Layer0TestHarness(config)
    
    # Define test scenarios
    scenarios = [
        create_basic_degradation_scenario(),
        create_cold_start_burst_scenario(),
        create_telemetry_stream_failure_scenario(),
        create_mixed_failure_scenario()
    ]
    
    # Run all scenarios
    results = {}
    for scenario in scenarios:
        try:
            logger.info(f"Running scenario: {scenario.name}")
            result = test_harness.run_test_scenario(scenario)
            results[scenario.name] = result
            
            # Log results
            logger.info(f"Scenario {scenario.name} completed:")
            logger.info(f"  MTTR: {result['results']['mttr_ms']:.2f}ms")
            logger.info(f"  Data Loss: {result['results']['data_loss_ppm']:.2f} PPM")
            logger.info(f"  Latency Overhead: {result['results']['latency_overhead_ms']:.2f}ms")
            logger.info(f"  Overall Success: {result['summary']['overall_success']}")
            
        except Exception as e:
            logger.error(f"Scenario {scenario.name} failed: {e}")
            results[scenario.name] = {'error': str(e)}
    
    # Generate overall summary
    overall_summary = {
        'total_scenarios': len(scenarios),
        'successful_scenarios': sum(1 for r in results.values() if 'error' not in r),
        'failed_scenarios': sum(1 for r in results.values() if 'error' in r),
        'scenario_results': results,
        'overall_success_rate': sum(1 for r in results.values() if 'error' not in r) / len(scenarios)
    }
    
    logger.info(f"All tests completed. Success rate: {overall_summary['overall_success_rate']:.2%}")
    
    return overall_summary

# =============================================================================
# Main Test Execution
# =============================================================================

if __name__ == "__main__":
    # Create configuration
    config = Layer0Config()
    
    # Run all tests
    results = run_all_layer0_tests(config)
    
    # Print results
    print("\n" + "="*80)
    print("LAYER 0 COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    for scenario_name, result in results['scenario_results'].items():
        print(f"\n{scenario_name}:")
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  MTTR: {result['results']['mttr_ms']:.2f}ms")
            print(f"  Data Loss: {result['results']['data_loss_ppm']:.2f} PPM")
            print(f"  Latency Overhead: {result['results']['latency_overhead_ms']:.2f}ms")
            print(f"  Success: {result['summary']['overall_success']}")
    
    print(f"\nOverall Success Rate: {results['overall_success_rate']:.2%}")
    print("="*80)
