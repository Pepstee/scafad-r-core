# utils/metrics.py
from __future__ import annotations
import time
from typing import Dict

class PerformanceMetricsCollector:
    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._memory_mb: Dict[str, float] = {}

    def record_processing_time(self, component: str, duration: float) -> None:
        self._timings[component] = self._timings.get(component, 0.0) + max(0.0, duration)

    def record_memory_usage(self, component: str, memory_mb: float) -> None:
        self._memory_mb[component] = max(self._memory_mb.get(component, 0.0), memory_mb)

    def generate_performance_report(self) -> Dict[str, Dict[str, float]]:
        return {"timings": dict(self._timings), "memory_mb": dict(self._memory_mb)}

class Stopwatch:
    """
    Context manager for timing blocks.
    """
    def __init__(self, collector: PerformanceMetricsCollector, component: str):
        self.collector = collector
        self.component = component

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self._t0
        self.collector.record_processing_time(self.component, dt)
