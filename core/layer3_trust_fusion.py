"""Basic Layer 3 trust-weighted fusion for SCAFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.layer2_detection_matrix import DetectionSignal, Layer2DetectionResult


@dataclass
class Layer3FusionResult:
    """Trust-weighted fused score plus detector weights."""

    fused_score: float
    volatility_adjustment: float
    trust_weights: Dict[str, float] = field(default_factory=dict)
    leading_signals: List[DetectionSignal] = field(default_factory=list)


class TrustWeightedFusionEngine:
    """Fuses Layer 2 signals with simple confidence and trust weighting."""

    def __init__(self, base_trust: float = 0.8) -> None:
        self.base_trust = base_trust

    def fuse(self, detection: Layer2DetectionResult) -> Layer3FusionResult:
        if not detection.signals:
            return Layer3FusionResult(fused_score=0.0, volatility_adjustment=0.0)

        weights: Dict[str, float] = {}
        weighted_total = 0.0
        weight_sum = 0.0

        for signal in detection.signals:
            weight = max(0.1, min(1.0, self.base_trust * signal.confidence))
            weights[signal.detector_name] = round(weight, 4)
            weighted_total += signal.score * weight
            weight_sum += weight

        raw_fused = weighted_total / weight_sum if weight_sum else 0.0
        volatility_adjustment = min(0.15, abs(detection.aggregate_score - raw_fused))
        fused_score = max(0.0, min(1.0, raw_fused - volatility_adjustment))

        leading = sorted(detection.signals, key=lambda item: item.score, reverse=True)[:2]
        return Layer3FusionResult(
            fused_score=round(fused_score, 4),
            volatility_adjustment=round(volatility_adjustment, 4),
            trust_weights=weights,
            leading_signals=leading,
        )


__all__ = ["Layer3FusionResult", "TrustWeightedFusionEngine"]
