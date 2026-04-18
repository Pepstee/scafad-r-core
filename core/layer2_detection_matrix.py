"""Basic Layer 2 multi-vector detection matrix for SCAFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.layer1_pipeline import Layer1ProcessedRecord


@dataclass
class DetectionSignal:
    """One detector's contribution to the Layer 2 score matrix."""

    detector_name: str
    score: float
    confidence: float
    rationale: str


@dataclass
class Layer2DetectionResult:
    """Aggregated output from the Layer 2 multi-vector detectors."""

    record_id: str
    signals: List[DetectionSignal] = field(default_factory=list)
    aggregate_score: float = 0.0
    anomaly_indicated: bool = False


class RuleChainEngine:
    """Simple rule-based checks over telemetry content."""

    def evaluate(self, record: Dict[str, Any]) -> DetectionSignal:
        telemetry = record.get("telemetry_data", {}) or {}
        numeric_hits = 0
        for value in telemetry.values():
            if isinstance(value, (int, float)) and value > 80:
                numeric_hits += 1
        score = min(1.0, numeric_hits * 0.25)
        return DetectionSignal(
            detector_name="rule_chain",
            score=score,
            confidence=0.7,
            rationale=f"Detected {numeric_hits} high numeric telemetry values",
        )


class DriftTracker:
    """Lightweight drift heuristic based on volatility markers."""

    def evaluate(self, record: Dict[str, Any]) -> DetectionSignal:
        telemetry = record.get("telemetry_data", {}) or {}
        drift_keys = {"latency_drift", "memory_drift", "cpu_drift", "variance"}
        hits = sum(1 for key in telemetry if key in drift_keys)
        score = min(1.0, hits * 0.3)
        return DetectionSignal(
            detector_name="drift_tracker",
            score=score,
            confidence=0.65,
            rationale=f"Observed {hits} drift-oriented telemetry markers",
        )


class GraphImmunizedDetector:
    """Stub graph-style detector using relationship density hints."""

    def evaluate(self, record: Dict[str, Any]) -> DetectionSignal:
        telemetry = record.get("telemetry_data", {}) or {}
        edge_count = telemetry.get("graph_edge_count", 0)
        node_count = max(telemetry.get("graph_node_count", 1), 1)
        density = edge_count / node_count
        score = min(1.0, density / 10.0)
        return DetectionSignal(
            detector_name="graph_immunized",
            score=score,
            confidence=0.6,
            rationale=f"Graph density heuristic evaluated to {density:.2f}",
        )


class SemanticDeviationCore:
    """Semantic deviation heuristic based on anomaly descriptors."""

    def evaluate(self, record: Dict[str, Any]) -> DetectionSignal:
        anomaly_type = str(record.get("anomaly_type", "benign")).lower()
        execution_phase = str(record.get("execution_phase", "")).lower()
        suspicious_tokens = ("spike", "drift", "timeout", "injection", "exfiltration")
        score = 0.0
        if anomaly_type != "benign":
            score += 0.45
        if any(token in anomaly_type for token in suspicious_tokens):
            score += 0.35
        if execution_phase in {"initialization", "execution"}:
            score += 0.1
        score = min(1.0, score)
        return DetectionSignal(
            detector_name="semantic_deviation",
            score=score,
            confidence=0.75,
            rationale=f"Semantic analysis based on anomaly_type={anomaly_type}",
        )


class MultiVectorDetectionMatrix:
    """Co-ordinates the basic Layer 2 detector family."""

    def __init__(self) -> None:
        self.detectors = (
            RuleChainEngine(),
            DriftTracker(),
            GraphImmunizedDetector(),
            SemanticDeviationCore(),
        )

    def analyze(self, record: Dict[str, Any] | Layer1ProcessedRecord) -> Layer2DetectionResult:
        if isinstance(record, Layer1ProcessedRecord):
            envelope = record.to_dict()
        else:
            envelope = record
        signals = [detector.evaluate(envelope) for detector in self.detectors]
        if signals:
            aggregate = sum(signal.score for signal in signals) / len(signals)
        else:
            aggregate = 0.0
        return Layer2DetectionResult(
            record_id=str(envelope.get("record_id") or envelope.get("event_id") or "unknown"),
            signals=signals,
            aggregate_score=round(aggregate, 4),
            anomaly_indicated=aggregate >= 0.3,
        )


__all__ = [
    "DetectionSignal",
    "Layer2DetectionResult",
    "RuleChainEngine",
    "DriftTracker",
    "GraphImmunizedDetector",
    "SemanticDeviationCore",
    "MultiVectorDetectionMatrix",
]
