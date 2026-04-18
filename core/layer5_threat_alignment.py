"""Basic Layer 5 threat alignment for SCAFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.layer4_explainability import Layer4DecisionTrace


@dataclass
class ThreatAlignmentResult:
    """Threat-alignment output including MITRE-style tactics and grouping."""

    tactics: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    campaign_cluster: str = "baseline-observation"


class ThreatAlignmentEngine:
    """Maps decisions and anomaly labels to lightweight MITRE-style outputs."""

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
        return ThreatAlignmentResult(
            tactics=tactics,
            techniques=techniques,
            campaign_cluster=cluster,
        )


__all__ = ["ThreatAlignmentResult", "ThreatAlignmentEngine"]
