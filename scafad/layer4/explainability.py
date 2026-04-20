"""Layer 4 explainability and decision trace for the module-split SCAFAD architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from layer2.detection_matrix import Layer2DetectionResult
from layer3.trust_fusion import Layer3FusionResult


@dataclass
class Layer4DecisionTrace:
    record_id: str
    trace_id: str
    decision: str
    severity: str
    explanation_summary: str
    explanation_points: List[str] = field(default_factory=list)
    recommended_action: str = "monitor"
    evidence_items: List[str] = field(default_factory=list)
    redacted_fields: List[str] = field(default_factory=list)
    trace_metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "trace_id": self.trace_id,
            "decision": self.decision,
            "severity": self.severity,
            "explanation_summary": self.explanation_summary,
            "explanation_points": list(self.explanation_points),
            "recommended_action": self.recommended_action,
            "evidence_items": list(self.evidence_items),
            "redacted_fields": list(self.redacted_fields),
            "trace_metadata": dict(self.trace_metadata),
        }


class ExplainabilityDecisionEngine:
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
            recommended_action = "immediate_containment_review"
        elif fusion.fused_score >= 0.3:
            decision = "review"
            severity = "medium"
            recommended_action = "analyst_triage"
        else:
            decision = "observe"
            severity = "low"
            recommended_action = "monitor"

        top_points = [
            f"{signal.detector_name} contributed score {signal.score:.2f} ({signal.rationale})"
            for signal in fusion.leading_signals
        ]
        evidence_items = list(detection.evidence_summary)
        summary = (
            f"Decision={decision} with fused_score={fusion.fused_score:.2f} "
            f"from {len(detection.signals)} Layer 2 signals and consensus={detection.detector_consensus:.2f}."
        )
        return Layer4DecisionTrace(
            record_id=detection.record_id,
            trace_id=fusion.trace_id or detection.trace_id,
            decision=decision,
            severity=severity,
            explanation_summary=summary,
            explanation_points=top_points,
            recommended_action=recommended_action,
            evidence_items=evidence_items,
            redacted_fields=redacted,
            trace_metadata={
                "aggregate_score": f"{detection.aggregate_score:.2f}",
                "volatility_adjustment": f"{fusion.volatility_adjustment:.2f}",
                "risk_band": fusion.risk_band,
                "consensus_strength": f"{fusion.consensus_strength:.2f}",
            },
        )


__all__ = ["Layer4DecisionTrace", "ExplainabilityDecisionEngine"]

