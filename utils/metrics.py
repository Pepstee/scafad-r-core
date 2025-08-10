"""SCAFAD Performance Metrics
==========================

Comprehensive performance monitoring and metrics collection for SCAFAD Layer 0.
Provides timing, memory, throughput, and resource utilization tracking.
"""

from __future__ import annotations
import time
import statistics
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from enum import Enum

# Try to import resource module for system metrics
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

# Try to import psutil for enhanced system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics that can be collected"""
    TIMING = "timing"
    MEMORY = "memory"
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    THROUGHPUT = "throughput"


@dataclass
class MetricSample:
    """Individual metric sample"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class MetricSummary:
    """Statistical summary of metric samples"""
    count: int
    sum_value: float
    min_value: float
    max_value: float
    mean: float
    median: float
    percentile_95: float
    percentile_99: float
    
    @classmethod
    def from_samples(cls, samples: List[MetricSample]) -> 'MetricSummary':
        if not samples:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        values = [s.value for s in samples]
        return cls(
            count=len(values),
            sum_value=sum(values),
            min_value=min(values),
            max_value=max(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            percentile_95=cls._percentile(values, 95),
            percentile_99=cls._percentile(values, 99)
        )
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


class PerformanceMetricsCollector:
    """Enhanced metrics collector with statistical analysis"""
    
    def __init__(self, max_samples_per_metric: int = 1000):
        self._lock = threading.RLock()
        self._max_samples = max_samples_per_metric
        
        # Store raw samples
        self._samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples_per_metric))
        self._metric_types: Dict[str, MetricType] = {}
        
        # Aggregated metrics
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        
        # System resource tracking
        self._process_start_time = time.time()
        self._track_system_resources = PSUTIL_AVAILABLE or RESOURCE_AVAILABLE
    
    def record_timing(self, metric_name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric"""
        with self._lock:
            self._metric_types[metric_name] = MetricType.TIMING
            sample = MetricSample(time.time(), max(0.0, duration), labels or {})
            self._samples[metric_name].append(sample)
    
    def record_memory(self, metric_name: str, memory_mb: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a memory usage metric"""
        with self._lock:
            self._metric_types[metric_name] = MetricType.MEMORY
            sample = MetricSample(time.time(), max(0.0, memory_mb), labels or {})
            self._samples[metric_name].append(sample)
    
    def increment_counter(self, metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        with self._lock:
            self._metric_types[metric_name] = MetricType.COUNTER
            self._counters[metric_name] += value
            sample = MetricSample(time.time(), value, labels or {})
            self._samples[metric_name].append(sample)
    
    def set_gauge(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        with self._lock:
            self._metric_types[metric_name] = MetricType.GAUGE
            self._gauges[metric_name] = value
            sample = MetricSample(time.time(), value, labels or {})
            self._samples[metric_name].append(sample)
    
    def record_throughput(self, metric_name: str, operations: int, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record throughput metric (operations per second)"""
        if duration <= 0:
            return
        
        ops_per_second = operations / duration
        with self._lock:
            self._metric_types[metric_name] = MetricType.THROUGHPUT
            sample = MetricSample(time.time(), ops_per_second, labels or {})
            self._samples[metric_name].append(sample)
    
    def record_histogram(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric"""
        with self._lock:
            self._metric_types[metric_name] = MetricType.HISTOGRAM
            sample = MetricSample(time.time(), value, labels or {})
            self._samples[metric_name].append(sample)
    
    def get_metric_summary(self, metric_name: str) -> Optional[MetricSummary]:
        """Get statistical summary for a metric"""
        with self._lock:
            if metric_name not in self._samples:
                return None
            return MetricSummary.from_samples(list(self._samples[metric_name]))
    
    def get_current_system_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics"""
        metrics = {}
        
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                metrics.update({
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / (1024 * 1024),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'open_files': len(process.open_files()),
                })
                
                # System-wide metrics
                metrics.update({
                    'system_cpu_percent': psutil.cpu_percent(),
                    'system_memory_percent': psutil.virtual_memory().percent,
                    'system_load_avg': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
                })
            
            elif RESOURCE_AVAILABLE:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                metrics.update({
                    'user_time': usage.ru_utime,
                    'system_time': usage.ru_stime,
                    'max_memory_kb': usage.ru_maxrss,
                    'page_faults': usage.ru_majflt + usage.ru_minflt,
                })
        
        except Exception:
            pass  # Gracefully handle any system metric collection errors
        
        # Runtime metrics
        metrics['uptime_seconds'] = time.time() - self._process_start_time
        
        return metrics
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            report = {
                'timestamp': time.time(),
                'system_metrics': self.get_current_system_metrics(),
                'metrics_summary': {},
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'sample_counts': {name: len(samples) for name, samples in self._samples.items()}
            }
            
            # Add statistical summaries
            for metric_name in self._samples:
                summary = self.get_metric_summary(metric_name)
                if summary:
                    report['metrics_summary'][metric_name] = {
                        'type': self._metric_types[metric_name].value,
                        'count': summary.count,
                        'sum': summary.sum_value,
                        'min': summary.min_value,
                        'max': summary.max_value,
                        'mean': summary.mean,
                        'median': summary.median,
                        'p95': summary.percentile_95,
                        'p99': summary.percentile_99
                    }
            
            return report
    
    def clear_metrics(self, metric_pattern: Optional[str] = None) -> None:
        """Clear metrics, optionally matching a pattern"""
        with self._lock:
            if metric_pattern:
                import re
                pattern = re.compile(metric_pattern)
                to_remove = [name for name in self._samples.keys() if pattern.match(name)]
                for name in to_remove:
                    del self._samples[name]
                    self._metric_types.pop(name, None)
                    self._counters.pop(name, None)
                    self._gauges.pop(name, None)
            else:
                self._samples.clear()
                self._metric_types.clear()
                self._counters.clear()
                self._gauges.clear()


class Stopwatch:
    """Enhanced context manager for timing operations"""
    
    def __init__(self, collector: PerformanceMetricsCollector, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels or {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def __enter__(self) -> 'Stopwatch':
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter()
        if self._start_time is not None:
            duration = self._end_time - self._start_time
            # Add error label if exception occurred
            labels = self.labels.copy()
            if exc_type is not None:
                labels['error'] = exc_type.__name__
            
            self.collector.record_timing(self.metric_name, duration, labels)
    
    @property
    def duration(self) -> Optional[float]:
        """Get the measured duration (available after context exit)"""
        if self._start_time is None:
            return None
        end_time = self._end_time or time.perf_counter()
        return end_time - self._start_time


@contextmanager
def time_operation(collector: PerformanceMetricsCollector, operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for timing operations"""
    with Stopwatch(collector, operation_name, labels) as sw:
        yield sw


class ThroughputTracker:
    """Track throughput over time windows"""
    
    def __init__(self, window_size_seconds: float = 60.0):
        self.window_size = window_size_seconds
        self._operations = deque()
        self._lock = threading.Lock()
    
    def record_operation(self, timestamp: Optional[float] = None) -> None:
        """Record a single operation"""
        ts = timestamp or time.time()
        with self._lock:
            self._operations.append(ts)
            self._cleanup_old_operations(ts)
    
    def record_batch(self, count: int, timestamp: Optional[float] = None) -> None:
        """Record multiple operations at once"""
        ts = timestamp or time.time()
        with self._lock:
            for _ in range(count):
                self._operations.append(ts)
            self._cleanup_old_operations(ts)
    
    def get_throughput(self, timestamp: Optional[float] = None) -> float:
        """Get current operations per second"""
        ts = timestamp or time.time()
        with self._lock:
            self._cleanup_old_operations(ts)
            return len(self._operations) / self.window_size
    
    def _cleanup_old_operations(self, current_time: float) -> None:
        """Remove operations outside the time window"""
        cutoff = current_time - self.window_size
        while self._operations and self._operations[0] < cutoff:
            self._operations.popleft()


class MetricsAggregator:
    """Aggregate metrics from multiple collectors"""
    
    def __init__(self):
        self.collectors: List[PerformanceMetricsCollector] = []
    
    def add_collector(self, collector: PerformanceMetricsCollector) -> None:
        """Add a collector to aggregate"""
        self.collectors.append(collector)
    
    def generate_aggregated_report(self) -> Dict[str, Any]:
        """Generate report aggregating all collectors"""
        if not self.collectors:
            return {}
        
        # Start with the first collector's report
        aggregated = self.collectors[0].generate_performance_report()
        
        # Merge additional collectors
        for collector in self.collectors[1:]:
            report = collector.generate_performance_report()
            
            # Merge counters
            for key, value in report.get('counters', {}).items():
                aggregated['counters'][key] = aggregated['counters'].get(key, 0) + value
            
            # For gauges, take the latest value
            aggregated['gauges'].update(report.get('gauges', {}))
            
            # For metrics summary, we'd need more complex merging logic
            # For now, just include all summaries with collector prefixes
            for key, summary in report.get('metrics_summary', {}).items():
                aggregated['metrics_summary'][f"collector_{len(self.collectors)}_{key}"] = summary
        
        aggregated['num_collectors'] = len(self.collectors)
        return aggregated


# Global default collector instance
_default_collector = PerformanceMetricsCollector()


def get_default_collector() -> PerformanceMetricsCollector:
    """Get the default global metrics collector"""
    return _default_collector


def record_timing(metric_name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record timing using the default collector"""
    _default_collector.record_timing(metric_name, duration, labels)


def record_memory(metric_name: str, memory_mb: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record memory usage using the default collector"""
    _default_collector.record_memory(metric_name, memory_mb, labels)


def increment_counter(metric_name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Increment counter using the default collector"""
    _default_collector.increment_counter(metric_name, value, labels)


# Module exports
__all__ = [
    'MetricType', 'MetricSample', 'MetricSummary',
    'PerformanceMetricsCollector', 'Stopwatch', 'ThroughputTracker', 'MetricsAggregator',
    'time_operation', 'get_default_collector',
    'record_timing', 'record_memory', 'increment_counter'
]
