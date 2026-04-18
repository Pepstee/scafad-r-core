"""Basic Layer 4 explainability and decision trace for SCAFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.layer2_detection_matrix import Layer2DetectionResult
from core.layer3_trust_fusion import Layer3FusionResult


@dataclass
class Layer4DecisionTrace:
    """Budgeted explanation trace for analyst-facing decisions."""

    decision: str
    severity: str
    explanation_summary: str
    explanation_points: List[str] = field(default_factory=list)
    redacted_fields: List[str] = field(default_factory=list)
    trace_metadata: Dict[str, str] = field(default_factory=dict)


class ExplainabilityDecisionEngine:
    """Creates small but useful explanations from Layer 2 and Layer 3 results."""

    def build_trace(
        self,
        detection: Layer2DetectionResult,
        fusion: Layer3FusionResult,
        redacted_fields: List[str] | None = None,
    ) -> Layer4DecisionTrace:
        redacted = list(redacted_fields or [])
        if fusion.fused_score >= 0.75:
            decision = "escalate"
            severity = "high"
        elif fusion.fused_score >= 0.3:
            decision = "review"
            severity = "medium"
        else:
            decision = "observe"
            severity = "low"

        top_points = [
            f"{signal.detector_name} contributed score {signal.score:.2f} ({signal.rationale})"
            for signal in fusion.leading_signals
        ]
        summary = (
            f"Decision={decision} with fused_score={fusion.fused_score:.2f} "
            f"from {len(detection.signals)} Layer 2 signals."
        )
        return Layer4DecisionTrace(
            decision=decision,
            severity=severity,
            explanation_summary=summary,
            explanation_points=top_points,
            redacted_fields=redacted,
            trace_metadata={
                "aggregate_score": f"{detection.aggregate_score:.2f}",
                "volatility_adjustment": f"{fusion.volatility_adjustment:.2f}",
            },
        )


__all__ = ["Layer4DecisionTrace", "ExplainabilityDecisionEngine"]
