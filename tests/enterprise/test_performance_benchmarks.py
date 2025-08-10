#!/usr/bin/env python3
"""
Enterprise Performance Benchmarks for SOC Deployment
===================================================

Rigorous performance testing suite to ensure SCAFAD meets enterprise
SOC requirements for high-volume, low-latency security event processing.

Benchmark Categories:
- Throughput: Events per second under various loads
- Latency: Response time percentiles (P50, P90, P95, P99)
- Memory Usage: Efficient memory management under load
- CPU Utilization: Optimal CPU usage patterns
- Concurrent Processing: Multi-threaded/async performance
- Stress Testing: Behavior under extreme conditions
- Resource Limits: Maximum operational capacity

SOC Requirements:
- Process 100K+ events/hour (27+ events/second sustained)
- P95 latency < 500ms, P99 latency < 1000ms
- Memory growth < 50MB over 24 hours
- CPU utilization < 80% under normal load
- 99.9% uptime and reliability
"""

import asyncio
import time
import statistics
import concurrent.futures
import threading
import json
import uuid
import gc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import sys

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, limited system monitoring")

# SCAFAD imports
from app_main import Layer0_AdaptiveTelemetryController
from app_config import ScafadConfig
from app_telemetry import TelemetryRecord, AnomalyType, ExecutionPhase, TelemetrySource
from utils.metrics import PerformanceMetricsCollector, Stopwatch


@dataclass
class BenchmarkResult:
    """Individual benchmark test result"""
    test_name: str
    passed: bool
    execution_time: float
    throughput_eps: float  # Events per second
    latency_stats: Dict[str, float]
    memory_stats: Dict[str, float]
    cpu_stats: Dict[str, float]
    error_rate: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'execution_time': self.execution_time,
            'throughput_eps': self.throughput_eps,
            'latency_stats': self.latency_stats,
            'memory_stats': self.memory_stats,
            'cpu_stats': self.cpu_stats,
            'error_rate': self.error_rate,
            'details': self.details
        }


class SOCPerformanceBenchmarks:
    """Enterprise SOC performance benchmark suite"""
    
    def __init__(self):
        self.config = ScafadConfig()
        # Configure for high performance
        self.config.verbosity = "NORMAL"  # Reduce logging overhead
        self.config.enable_detailed_telemetry = False  # Minimize telemetry overhead
        
        self.controller = Layer0_AdaptiveTelemetryController(self.config)
        self.metrics_collector = PerformanceMetricsCollector(max_samples_per_metric=10000)
        
        # Performance requirements (enterprise SOC standards)
        self.requirements = {
            'min_throughput_eps': 50,  # 50 events/second minimum
            'target_throughput_eps': 100,  # 100 events/second target
            'max_p95_latency_ms': 500,  # P95 < 500ms
            'max_p99_latency_ms': 1000,  # P99 < 1000ms
            'max_memory_growth_mb': 50,  # < 50MB growth per hour
            'max_cpu_utilization_pct': 80,  # < 80% CPU usage
            'max_error_rate_pct': 0.1,  # < 0.1% error rate
            'min_concurrent_events': 1000  # Handle 1000 concurrent events
        }
    
    def generate_test_events(self, count: int, event_type: str = "mixed") -> List[Dict[str, Any]]:
        """Generate realistic test events for benchmarking"""
        events = []
        
        event_templates = {
            "security_alert": {
                "event_type": "security_alert",
                "alert_severity": "HIGH",
                "source_ip": "192.168.1.{}",
                "destination_ip": "10.0.0.{}",
                "attack_type": "sql_injection",
                "payload": "'; DROP TABLE users; --",
                "detection_rules": ["rule_001", "rule_002"],
                "metadata": {"analyst": "soc_team", "priority": "urgent"}
            },
            "network_anomaly": {
                "event_type": "network_anomaly", 
                "anomaly_type": "unusual_traffic_pattern",
                "bytes_transferred": 1048576,
                "connection_duration": 3600,
                "protocol": "HTTPS",
                "ports": [443, 8443, 9443],
                "geo_location": {"country": "US", "city": "Virginia"}
            },
            "file_analysis": {
                "event_type": "file_analysis",
                "file_hash": "a1b2c3d4e5f6789012345678901234567890abcd",
                "file_size": 2048576,
                "file_type": "PE32",
                "suspicious_indicators": ["packed", "anti_debug", "network_activity"],
                "sandbox_results": {"malicious_score": 0.85, "family": "trojan"}
            },
            "user_behavior": {
                "event_type": "user_behavior",
                "user_id": "user_{}",
                "login_attempts": [{"timestamp": time.time(), "success": True}],
                "privilege_escalation": False,
                "data_access": ["sensitive_db", "config_files"],
                "risk_score": 0.65
            }
        }
        
        for i in range(count):
            if event_type == "mixed":
                template = event_templates[list(event_templates.keys())[i % len(event_templates)]]
            else:
                template = event_templates.get(event_type, event_templates["security_alert"])
            
            event = template.copy()
            event.update({
                "event_id": f"bench_{event_type}_{i}",
                "timestamp": time.time(),
                "correlation_id": f"benchmark_{uuid.uuid4()}",
                "test_sequence": i
            })
            
            # Add variability to IP addresses and user IDs
            if "source_ip" in event and "{}" in event["source_ip"]:
                event["source_ip"] = event["source_ip"].format(i % 254 + 1)
            if "destination_ip" in event and "{}" in event["destination_ip"]:
                event["destination_ip"] = event["destination_ip"].format(i % 254 + 1)
            if "user_id" in event and "{}" in event["user_id"]:
                event["user_id"] = event["user_id"].format(i % 1000)
            
            events.append(event)
        
        return events
    
    def get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics"""
        stats = {}
        
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                stats.update({
                    'memory_rss_mb': process.memory_info().rss / 1024 / 1024,
                    'memory_vms_mb': process.memory_info().vms / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
                    'system_cpu_percent': psutil.cpu_percent(),
                    'system_memory_percent': psutil.virtual_memory().percent,
                })
            
            # Always available stats
            stats.update({
                'timestamp': time.time(),
                'gc_objects': len(gc.get_objects()),
            })
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    async def benchmark_sustained_throughput(self) -> BenchmarkResult:
        """Test sustained throughput over extended period"""
        print("🚀 Testing Sustained Throughput...")
        
        test_duration = 60  # 1 minute test
        events_per_batch = 100
        target_eps = self.requirements['target_throughput_eps']
        batch_interval = events_per_batch / target_eps  # seconds between batches
        
        start_time = time.time()
        initial_stats = self.get_system_stats()
        
        total_events = 0
        total_errors = 0
        latencies = []
        
        context = type('Context', (), {
            'aws_request_id': 'throughput_benchmark',
            'function_name': 'scafad_performance_test'
        })()
        
        try:
            end_time = start_time + test_duration
            
            while time.time() < end_time:
                batch_start = time.time()
                
                # Generate batch of events
                events = self.generate_test_events(events_per_batch, "mixed")
                
                # Process events concurrently
                tasks = []
                for event in events:
                    event_start = time.time()
                    task = asyncio.create_task(
                        self._timed_process_event(event, context, event_start)
                    )
                    tasks.append(task)
                
                # Wait for batch completion
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result in results:
                    total_events += 1
                    if isinstance(result, Exception):
                        total_errors += 1
                    elif isinstance(result, dict) and 'latency' in result:
                        latencies.append(result['latency'])
                    else:
                        total_errors += 1
                
                # Maintain target rate
                batch_duration = time.time() - batch_start
                if batch_duration < batch_interval:
                    await asyncio.sleep(batch_interval - batch_duration)
            
            execution_time = time.time() - start_time
            final_stats = self.get_system_stats()
            
            # Calculate metrics
            throughput = total_events / execution_time
            error_rate = (total_errors / total_events * 100) if total_events > 0 else 100
            
            latency_stats = {}
            if latencies:
                latencies.sort()
                latency_stats = {
                    'mean_ms': statistics.mean(latencies) * 1000,
                    'median_ms': statistics.median(latencies) * 1000,
                    'p90_ms': latencies[int(0.90 * len(latencies))] * 1000,
                    'p95_ms': latencies[int(0.95 * len(latencies))] * 1000,
                    'p99_ms': latencies[int(0.99 * len(latencies))] * 1000,
                    'min_ms': min(latencies) * 1000,
                    'max_ms': max(latencies) * 1000
                }
            
            memory_stats = {
                'initial_mb': initial_stats.get('memory_rss_mb', 0),
                'final_mb': final_stats.get('memory_rss_mb', 0),
                'growth_mb': final_stats.get('memory_rss_mb', 0) - initial_stats.get('memory_rss_mb', 0)
            }
            
            cpu_stats = {
                'process_cpu_pct': final_stats.get('cpu_percent', 0),
                'system_cpu_pct': final_stats.get('system_cpu_percent', 0)
            }
            
            # Determine if test passed
            passed = (
                throughput >= self.requirements['min_throughput_eps'] and
                error_rate <= self.requirements['max_error_rate_pct'] and
                latency_stats.get('p95_ms', 9999) <= self.requirements['max_p95_latency_ms'] and
                memory_stats['growth_mb'] <= self.requirements['max_memory_growth_mb']
            )
            
            print(f"   Throughput: {throughput:.1f} EPS (target: {target_eps})")
            print(f"   P95 Latency: {latency_stats.get('p95_ms', 0):.1f}ms")
            print(f"   Error Rate: {error_rate:.3f}%")
            print(f"   Memory Growth: {memory_stats['growth_mb']:.1f}MB")
            
            return BenchmarkResult(
                test_name="Sustained Throughput",
                passed=passed,
                execution_time=execution_time,
                throughput_eps=throughput,
                latency_stats=latency_stats,
                memory_stats=memory_stats,
                cpu_stats=cpu_stats,
                error_rate=error_rate,
                details={
                    'total_events': total_events,
                    'total_errors': total_errors,
                    'test_duration': test_duration,
                    'target_eps': target_eps
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Sustained Throughput",
                passed=False,
                execution_time=time.time() - start_time,
                throughput_eps=0.0,
                latency_stats={},
                memory_stats={},
                cpu_stats={},
                error_rate=100.0,
                details={'error': str(e)}
            )
    
    async def _timed_process_event(self, event: Dict[str, Any], context: Any, start_time: float) -> Dict[str, Any]:
        """Process event and measure latency"""
        try:
            result = await self.controller.process_telemetry_event(event, context)
            latency = time.time() - start_time
            return {'result': result, 'latency': latency, 'success': True}
        except Exception as e:
            latency = time.time() - start_time
            return {'error': str(e), 'latency': latency, 'success': False}
    
    async def benchmark_concurrent_processing(self) -> BenchmarkResult:
        """Test concurrent event processing capacity"""
        print("⚡ Testing Concurrent Processing...")
        
        concurrent_events = self.requirements['min_concurrent_events']
        start_time = time.time()
        initial_stats = self.get_system_stats()
        
        # Generate large batch of events
        events = self.generate_test_events(concurrent_events, "mixed")
        
        context = type('Context', (), {
            'aws_request_id': 'concurrent_benchmark',
            'function_name': 'scafad_concurrent_test'
        })()
        
        try:
            # Launch all events concurrently
            tasks = []
            event_start_times = []
            
            for event in events:
                event_start = time.time()
                event_start_times.append(event_start)
                task = asyncio.create_task(
                    self._timed_process_event(event, context, event_start)
                )
                tasks.append(task)
            
            # Wait for all to complete
            print(f"   Processing {concurrent_events} concurrent events...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            execution_time = time.time() - start_time
            final_stats = self.get_system_stats()
            
            # Analyze results
            successful_results = []
            errors = 0
            latencies = []
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                elif isinstance(result, dict) and result.get('success'):
                    successful_results.append(result)
                    latencies.append(result['latency'])
                else:
                    errors += 1
            
            throughput = len(successful_results) / execution_time
            error_rate = (errors / len(results) * 100) if results else 100
            
            # Calculate latency statistics
            latency_stats = {}
            if latencies:
                latencies.sort()
                latency_stats = {
                    'mean_ms': statistics.mean(latencies) * 1000,
                    'median_ms': statistics.median(latencies) * 1000,
                    'p90_ms': latencies[int(0.90 * len(latencies))] * 1000,
                    'p95_ms': latencies[int(0.95 * len(latencies))] * 1000,
                    'p99_ms': latencies[int(0.99 * len(latencies))] * 1000,
                    'min_ms': min(latencies) * 1000,
                    'max_ms': max(latencies) * 1000
                }
            
            memory_stats = {
                'initial_mb': initial_stats.get('memory_rss_mb', 0),
                'final_mb': final_stats.get('memory_rss_mb', 0),
                'growth_mb': final_stats.get('memory_rss_mb', 0) - initial_stats.get('memory_rss_mb', 0),
                'peak_mb': final_stats.get('memory_rss_mb', 0)  # Approximation
            }
            
            cpu_stats = {
                'process_cpu_pct': final_stats.get('cpu_percent', 0),
                'system_cpu_pct': final_stats.get('system_cpu_percent', 0)
            }
            
            # Determine success
            passed = (
                len(successful_results) >= concurrent_events * 0.95 and  # 95% success rate
                error_rate <= self.requirements['max_error_rate_pct'] and
                latency_stats.get('p99_ms', 9999) <= self.requirements['max_p99_latency_ms'] and
                memory_stats['growth_mb'] <= 100  # Allow higher growth for burst
            )
            
            print(f"   Successful Events: {len(successful_results)}/{concurrent_events}")
            print(f"   P99 Latency: {latency_stats.get('p99_ms', 0):.1f}ms")
            print(f"   Error Rate: {error_rate:.3f}%")
            print(f"   Peak Throughput: {throughput:.1f} EPS")
            
            return BenchmarkResult(
                test_name="Concurrent Processing",
                passed=passed,
                execution_time=execution_time,
                throughput_eps=throughput,
                latency_stats=latency_stats,
                memory_stats=memory_stats,
                cpu_stats=cpu_stats,
                error_rate=error_rate,
                details={
                    'concurrent_events': concurrent_events,
                    'successful_events': len(successful_results),
                    'failed_events': errors
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Concurrent Processing",
                passed=False,
                execution_time=time.time() - start_time,
                throughput_eps=0.0,
                latency_stats={},
                memory_stats={},
                cpu_stats={},
                error_rate=100.0,
                details={'error': str(e)}
            )
    
    async def benchmark_memory_efficiency(self) -> BenchmarkResult:
        """Test memory usage efficiency under sustained load"""
        print("💾 Testing Memory Efficiency...")
        
        test_duration = 120  # 2 minutes
        events_per_second = 50
        total_events = test_duration * events_per_second
        
        start_time = time.time()
        initial_stats = self.get_system_stats()
        
        memory_samples = []
        processed_events = 0
        errors = 0
        
        context = type('Context', (), {
            'aws_request_id': 'memory_benchmark',
            'function_name': 'scafad_memory_test'
        })()
        
        try:
            # Process events at steady rate while monitoring memory
            for i in range(total_events):
                event_start = time.time()
                
                # Generate single event
                events = self.generate_test_events(1, "security_alert")
                event = events[0]
                
                try:
                    await self.controller.process_telemetry_event(event, context)
                    processed_events += 1
                except Exception:
                    errors += 1
                
                # Sample memory every 100 events
                if i % 100 == 0:
                    current_stats = self.get_system_stats()
                    memory_samples.append({
                        'timestamp': current_stats['timestamp'],
                        'memory_mb': current_stats.get('memory_rss_mb', 0),
                        'events_processed': processed_events
                    })
                    
                    # Force garbage collection periodically
                    if i % 1000 == 0:
                        gc.collect()
                
                # Maintain steady rate
                target_interval = 1.0 / events_per_second
                elapsed = time.time() - event_start
                if elapsed < target_interval:
                    await asyncio.sleep(target_interval - elapsed)
            
            execution_time = time.time() - start_time
            final_stats = self.get_system_stats()
            
            # Analyze memory usage
            if memory_samples:
                initial_memory = memory_samples[0]['memory_mb']
                final_memory = memory_samples[-1]['memory_mb']
                peak_memory = max(sample['memory_mb'] for sample in memory_samples)
                memory_growth = final_memory - initial_memory
                
                # Calculate memory growth rate (MB per hour)
                growth_rate_per_hour = (memory_growth / execution_time) * 3600
            else:
                initial_memory = initial_stats.get('memory_rss_mb', 0)
                final_memory = final_stats.get('memory_rss_mb', 0)
                peak_memory = final_memory
                memory_growth = final_memory - initial_memory
                growth_rate_per_hour = (memory_growth / execution_time) * 3600
            
            throughput = processed_events / execution_time
            error_rate = (errors / total_events * 100) if total_events > 0 else 100
            
            memory_stats = {
                'initial_mb': initial_memory,
                'final_mb': final_memory,
                'peak_mb': peak_memory,
                'growth_mb': memory_growth,
                'growth_rate_mb_per_hour': growth_rate_per_hour
            }
            
            # Memory efficiency requirements
            passed = (
                growth_rate_per_hour <= self.requirements['max_memory_growth_mb'] and
                error_rate <= self.requirements['max_error_rate_pct'] and
                throughput >= events_per_second * 0.95  # 95% of target rate
            )
            
            print(f"   Memory Growth: {memory_growth:.1f}MB ({growth_rate_per_hour:.1f}MB/hour)")
            print(f"   Peak Memory: {peak_memory:.1f}MB")
            print(f"   Throughput: {throughput:.1f} EPS")
            print(f"   Error Rate: {error_rate:.3f}%")
            
            return BenchmarkResult(
                test_name="Memory Efficiency",
                passed=passed,
                execution_time=execution_time,
                throughput_eps=throughput,
                latency_stats={},
                memory_stats=memory_stats,
                cpu_stats={},
                error_rate=error_rate,
                details={
                    'total_events': total_events,
                    'processed_events': processed_events,
                    'memory_samples': len(memory_samples),
                    'test_duration': test_duration
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Memory Efficiency",
                passed=False,
                execution_time=time.time() - start_time,
                throughput_eps=0.0,
                latency_stats={},
                memory_stats={},
                cpu_stats={},
                error_rate=100.0,
                details={'error': str(e)}
            )
    
    async def benchmark_stress_test(self) -> BenchmarkResult:
        """Stress test under extreme conditions"""
        print("🔥 Running Stress Test...")
        
        # Extreme parameters
        stress_duration = 30  # 30 seconds of stress
        burst_eps = 200  # 200 events/second burst
        large_event_size = 10000  # 10KB events
        
        start_time = time.time()
        initial_stats = self.get_system_stats()
        
        processed_events = 0
        errors = 0
        latencies = []
        
        context = type('Context', (), {
            'aws_request_id': 'stress_benchmark',
            'function_name': 'scafad_stress_test'
        })()
        
        try:
            end_time = start_time + stress_duration
            
            while time.time() < end_time:
                batch_start = time.time()
                
                # Generate large events
                events = []
                for i in range(10):  # Batch of 10 large events
                    event = {
                        "event_id": f"stress_{processed_events}_{i}",
                        "event_type": "stress_test",
                        "large_data": "x" * large_event_size,
                        "nested_structure": {
                            f"level_{j}": {f"data_{k}": f"value_{k}" for k in range(50)}
                            for j in range(10)
                        },
                        "timestamp": time.time(),
                        "correlation_id": f"stress_{uuid.uuid4()}"
                    }
                    events.append(event)
                
                # Process batch concurrently
                tasks = []
                for event in events:
                    event_start = time.time()
                    task = asyncio.create_task(
                        self._timed_process_event(event, context, event_start)
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result in results:
                    processed_events += 1
                    if isinstance(result, Exception):
                        errors += 1
                    elif isinstance(result, dict) and result.get('success'):
                        latencies.append(result['latency'])
                    else:
                        errors += 1
                
                # No rate limiting - stress test
            
            execution_time = time.time() - start_time
            final_stats = self.get_system_stats()
            
            # Calculate metrics
            throughput = processed_events / execution_time
            error_rate = (errors / processed_events * 100) if processed_events > 0 else 100
            
            latency_stats = {}
            if latencies:
                latencies.sort()
                latency_stats = {
                    'mean_ms': statistics.mean(latencies) * 1000,
                    'median_ms': statistics.median(latencies) * 1000,
                    'p90_ms': latencies[int(0.90 * len(latencies))] * 1000,
                    'p95_ms': latencies[int(0.95 * len(latencies))] * 1000,
                    'p99_ms': latencies[int(0.99 * len(latencies))] * 1000,
                    'max_ms': max(latencies) * 1000
                }
            
            memory_stats = {
                'initial_mb': initial_stats.get('memory_rss_mb', 0),
                'final_mb': final_stats.get('memory_rss_mb', 0),
                'growth_mb': final_stats.get('memory_rss_mb', 0) - initial_stats.get('memory_rss_mb', 0)
            }
            
            cpu_stats = {
                'process_cpu_pct': final_stats.get('cpu_percent', 0),
                'system_cpu_pct': final_stats.get('system_cpu_percent', 0)
            }
            
            # Stress test passes if system remains stable
            passed = (
                error_rate <= 5.0 and  # Allow higher error rate under stress
                throughput > 0 and  # System continues processing
                memory_stats['growth_mb'] < 200 and  # Memory doesn't explode
                latencies  # System responds to at least some events
            )
            
            print(f"   Events Processed: {processed_events}")
            print(f"   Throughput: {throughput:.1f} EPS")
            print(f"   Error Rate: {error_rate:.1f}%")
            print(f"   Max Latency: {latency_stats.get('max_ms', 0):.1f}ms")
            print(f"   Memory Growth: {memory_stats['growth_mb']:.1f}MB")
            
            return BenchmarkResult(
                test_name="Stress Test",
                passed=passed,
                execution_time=execution_time,
                throughput_eps=throughput,
                latency_stats=latency_stats,
                memory_stats=memory_stats,
                cpu_stats=cpu_stats,
                error_rate=error_rate,
                details={
                    'stress_duration': stress_duration,
                    'target_burst_eps': burst_eps,
                    'large_event_size': large_event_size,
                    'processed_events': processed_events
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                test_name="Stress Test",
                passed=False,
                execution_time=time.time() - start_time,
                throughput_eps=0.0,
                latency_stats={},
                memory_stats={},
                cpu_stats={},
                error_rate=100.0,
                details={'error': str(e)}
            )
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite"""
        print("🏁 Enterprise SOC Performance Benchmarks")
        print("=" * 50)
        
        # System information
        print("\n📊 System Information:")
        if PSUTIL_AVAILABLE:
            print(f"   CPU Count: {psutil.cpu_count()}")
            print(f"   Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        print(f"   Python Version: {sys.version}")
        
        benchmark_start_time = time.time()
        
        # Run all benchmarks
        benchmarks = [
            ("Sustained Throughput", self.benchmark_sustained_throughput),
            ("Concurrent Processing", self.benchmark_concurrent_processing),
            ("Memory Efficiency", self.benchmark_memory_efficiency),
            ("Stress Test", self.benchmark_stress_test),
        ]
        
        results = []
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{'='*20} {benchmark_name} {'='*20}")
            
            try:
                result = await benchmark_func()
                results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"   Result: {status}")
                
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
                results.append(BenchmarkResult(
                    test_name=benchmark_name,
                    passed=False,
                    execution_time=0.0,
                    throughput_eps=0.0,
                    latency_stats={},
                    memory_stats={},
                    cpu_stats={},
                    error_rate=100.0,
                    details={'exception': str(e)}
                ))
        
        total_benchmark_time = time.time() - benchmark_start_time
        
        # Generate comprehensive report
        report = self.generate_performance_report(results, total_benchmark_time)
        
        return report
    
    def generate_performance_report(self, results: List[BenchmarkResult], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        passed_tests = sum(1 for r in results if r.passed)
        total_tests = len(results)
        
        # Calculate aggregate metrics
        throughput_results = [r.throughput_eps for r in results if r.throughput_eps > 0]
        avg_throughput = statistics.mean(throughput_results) if throughput_results else 0
        max_throughput = max(throughput_results) if throughput_results else 0
        
        # Collect all latency data
        all_p95_latencies = []
        all_p99_latencies = []
        for result in results:
            if result.latency_stats:
                if 'p95_ms' in result.latency_stats:
                    all_p95_latencies.append(result.latency_stats['p95_ms'])
                if 'p99_ms' in result.latency_stats:
                    all_p99_latencies.append(result.latency_stats['p99_ms'])
        
        worst_p95 = max(all_p95_latencies) if all_p95_latencies else 0
        worst_p99 = max(all_p99_latencies) if all_p99_latencies else 0
        
        # Memory analysis
        memory_growths = [r.memory_stats.get('growth_mb', 0) for r in results if r.memory_stats]
        max_memory_growth = max(memory_growths) if memory_growths else 0
        
        # Error rate analysis
        error_rates = [r.error_rate for r in results]
        max_error_rate = max(error_rates) if error_rates else 100
        avg_error_rate = statistics.mean(error_rates) if error_rates else 100
        
        # Determine overall performance grade
        if passed_tests == total_tests and max_throughput >= self.requirements['target_throughput_eps']:
            performance_grade = "A"  # Excellent
        elif passed_tests >= total_tests * 0.75 and max_throughput >= self.requirements['min_throughput_eps']:
            performance_grade = "B"  # Good
        elif passed_tests >= total_tests * 0.5:
            performance_grade = "C"  # Acceptable
        else:
            performance_grade = "F"  # Fail
        
        # Requirements compliance
        compliance = {
            'throughput_compliance': max_throughput >= self.requirements['min_throughput_eps'],
            'latency_compliance': worst_p95 <= self.requirements['max_p95_latency_ms'] and worst_p99 <= self.requirements['max_p99_latency_ms'],
            'memory_compliance': max_memory_growth <= self.requirements['max_memory_growth_mb'],
            'reliability_compliance': max_error_rate <= self.requirements['max_error_rate_pct']
        }
        
        overall_compliance = all(compliance.values())
        
        report = {
            'report_metadata': {
                'generated_at': time.time(),
                'total_benchmark_time': total_time,
                'scafad_version': '1.0.0',
                'test_environment': 'enterprise_performance'
            },
            'executive_summary': {
                'performance_grade': performance_grade,
                'overall_compliance': overall_compliance,
                'tests_passed': passed_tests,
                'total_tests': total_tests,
                'max_throughput_eps': max_throughput,
                'worst_p95_latency_ms': worst_p95,
                'worst_p99_latency_ms': worst_p99,
                'max_memory_growth_mb': max_memory_growth,
                'max_error_rate_pct': max_error_rate
            },
            'requirements_compliance': compliance,
            'performance_metrics': {
                'throughput': {
                    'average_eps': avg_throughput,
                    'maximum_eps': max_throughput,
                    'requirement_min_eps': self.requirements['min_throughput_eps'],
                    'requirement_target_eps': self.requirements['target_throughput_eps']
                },
                'latency': {
                    'worst_p95_ms': worst_p95,
                    'worst_p99_ms': worst_p99,
                    'requirement_p95_ms': self.requirements['max_p95_latency_ms'],
                    'requirement_p99_ms': self.requirements['max_p99_latency_ms']
                },
                'reliability': {
                    'average_error_rate_pct': avg_error_rate,
                    'maximum_error_rate_pct': max_error_rate,
                    'requirement_max_error_rate_pct': self.requirements['max_error_rate_pct']
                }
            },
            'detailed_results': [r.to_dict() for r in results],
            'recommendations': self.generate_performance_recommendations(results, compliance),
            'soc_readiness': self.assess_soc_readiness(performance_grade, overall_compliance)
        }
        
        return report
    
    def generate_performance_recommendations(self, results: List[BenchmarkResult], compliance: Dict[str, bool]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if not compliance['throughput_compliance']:
            recommendations.append("Optimize event processing pipeline to increase throughput")
            recommendations.append("Consider implementing event batching and parallel processing")
        
        if not compliance['latency_compliance']:
            recommendations.append("Reduce processing latency by optimizing detection algorithms")
            recommendations.append("Implement caching for frequently accessed data")
        
        if not compliance['memory_compliance']:
            recommendations.append("Implement better memory management and garbage collection")
            recommendations.append("Optimize data structures to reduce memory footprint")
        
        if not compliance['reliability_compliance']:
            recommendations.append("Improve error handling and retry mechanisms")
            recommendations.append("Implement circuit breakers for external dependencies")
        
        # Specific recommendations based on failed tests
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            for test in failed_tests:
                if test.test_name == "Stress Test":
                    recommendations.append("Implement backpressure mechanisms for high-load scenarios")
                elif test.test_name == "Concurrent Processing":
                    recommendations.append("Optimize async processing and resource allocation")
        
        if not recommendations:
            recommendations.append("Excellent performance! Monitor and maintain current optimization levels")
        
        return recommendations
    
    def assess_soc_readiness(self, performance_grade: str, overall_compliance: bool) -> Dict[str, Any]:
        """Assess SOC deployment readiness"""
        
        if performance_grade in ["A", "B"] and overall_compliance:
            readiness_level = "PRODUCTION_READY"
            readiness_description = "System meets all enterprise SOC performance requirements"
            deployment_recommendation = "APPROVED for production SOC deployment"
        elif performance_grade == "C" and overall_compliance:
            readiness_level = "CONDITIONALLY_READY"
            readiness_description = "System meets minimum requirements but has room for improvement"
            deployment_recommendation = "Approved with performance monitoring and optimization plan"
        elif performance_grade in ["A", "B", "C"]:
            readiness_level = "REQUIRES_OPTIMIZATION"
            readiness_description = "System shows good performance but fails compliance requirements"
            deployment_recommendation = "Address compliance issues before production deployment"
        else:
            readiness_level = "NOT_READY"
            readiness_description = "System does not meet enterprise SOC performance standards"
            deployment_recommendation = "Significant optimization required before deployment"
        
        return {
            'readiness_level': readiness_level,
            'description': readiness_description,
            'deployment_recommendation': deployment_recommendation,
            'performance_grade': performance_grade,
            'compliance_status': overall_compliance
        }


# Main execution
async def main():
    """Execute performance benchmarks"""
    benchmark_suite = SOCPerformanceBenchmarks()
    
    # Run comprehensive benchmarks
    performance_report = await benchmark_suite.run_all_benchmarks()
    
    # Print executive summary
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 50)
    
    summary = performance_report['executive_summary']
    print(f"Performance Grade: {summary['performance_grade']}")
    print(f"Overall Compliance: {'✅ PASS' if summary['overall_compliance'] else '❌ FAIL'}")
    print(f"Tests Passed: {summary['tests_passed']}/{summary['total_tests']}")
    print(f"Max Throughput: {summary['max_throughput_eps']:.1f} EPS")
    print(f"Worst P95 Latency: {summary['worst_p95_latency_ms']:.1f}ms")
    print(f"Max Memory Growth: {summary['max_memory_growth_mb']:.1f}MB")
    print(f"Max Error Rate: {summary['max_error_rate_pct']:.3f}%")
    
    # SOC Readiness Assessment
    soc_readiness = performance_report['soc_readiness']
    print(f"\n🏢 SOC Deployment Readiness: {soc_readiness['readiness_level']}")
    print(f"   {soc_readiness['description']}")
    print(f"   Recommendation: {soc_readiness['deployment_recommendation']}")
    
    # Save detailed report
    report_filename = f"scafad_performance_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(performance_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_filename}")
    
    if summary['performance_grade'] in ['A', 'B'] and summary['overall_compliance']:
        print("\n🎉 SCAFAD is performance-ready for enterprise SOC deployment!")
    else:
        print(f"\n⚠️ Performance optimization needed before SOC deployment")
    
    return performance_report


if __name__ == "__main__":
    report = asyncio.run(main())