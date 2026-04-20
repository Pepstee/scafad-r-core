"""Layer 5 threat alignment for the module-split SCAFAD architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from layer4.explainability import Layer4DecisionTrace


@dataclass
class ThreatAlignmentResult:
    record_id: str
    trace_id: str
    tactics: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    campaign_cluster: str = "baseline-observation"
    alignment_confidence: float = 0.0
    attack_story: str = "No threat pattern matched."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "trace_id": self.trace_id,
            "tactics": list(self.tactics),
            "techniques": list(self.techniques),
            "campaign_cluster": self.campaign_cluster,
            "alignment_confidence": self.alignment_confidence,
            "attack_story": self.attack_story,
        }


class ThreatAlignmentEngine:
    def align(self, anomaly_type: str, trace: Layer4DecisionTrace) -> ThreatAlignmentResult:
        label = str(anomaly_type).lower()
        tactics: List[str] = []
        techniques: List[str] = []

        if "timeout" in label or "dos" in label:
            tactics.append("impact")
            techniques.append("service-degradation")
        if "injection" in label:
            tactics.append("execution")
            techniques.append("command-and-scripting-interpreter")
        if "exfil" in label or "leak" in label:
            tactics.append("exfiltration")
            techniques.append("automated-exfiltration")
        if "spike" in label or "drift" in label:
            tactics.append("discovery")
            techniques.append("cloud-service-discovery")
        if not tactics:
            tactics.append("collection")
            techniques.append("telemetry-observation")

        cluster = f"{trace.decision}-{tactics[0]}"
        story = (
            f"Decision {trace.decision} aligned with tactic {tactics[0]} "
            f"because anomaly label '{label}' and explanation severity '{trace.severity}' "
            f"indicate a {trace.trace_metadata.get('risk_band', 'low')} risk pattern."
        )
        return ThreatAlignmentResult(
            record_id=trace.record_id,
            trace_id=trace.trace_id,
            tactics=tactics,
            techniques=techniques,
            campaign_cluster=cluster,
            alignment_confidence=round(min(1.0, 0.45 + (0.15 * len(tactics))), 4),
            attack_story=story,
        )


__all__ = ["ThreatAlignmentResult", "ThreatAlignmentEngine"]

