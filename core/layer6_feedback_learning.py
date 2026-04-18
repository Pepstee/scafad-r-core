"""Basic Layer 6 feedback and learning loop for SCAFAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.layer5_threat_alignment import ThreatAlignmentResult


@dataclass
class FeedbackEvent:
    """One analyst or system feedback observation."""

    record_id: str
    analyst_label: str
    trust_delta: float


@dataclass
class Layer6FeedbackState:
    """Rolling feedback state used for simple trust modulation."""

    adjusted_trust: float
    replay_queue_size: int
    feedback_events: List[FeedbackEvent] = field(default_factory=list)


class FeedbackLearningEngine:
    """Small trust-modulation loop to make higher layers usable now."""

    def __init__(self, initial_trust: float = 0.8) -> None:
        self.current_trust = initial_trust
        self.replay_queue: List[str] = []
        self.feedback_events: List[FeedbackEvent] = []

    def ingest_feedback(
        self,
        record_id: str,
        analyst_label: str,
        threat_alignment: ThreatAlignmentResult,
    ) -> Layer6FeedbackState:
        label = analyst_label.lower()
        delta = 0.05 if label in {"confirmed", "malicious", "true_positive"} else -0.05
        if "impact" in threat_alignment.tactics:
            delta += 0.02

        self.current_trust = max(0.1, min(1.0, self.current_trust + delta))
        event = FeedbackEvent(record_id=record_id, analyst_label=analyst_label, trust_delta=round(delta, 4))
        self.feedback_events.append(event)
        self.replay_queue.append(record_id)

        return Layer6FeedbackState(
            adjusted_trust=round(self.current_trust, 4),
            replay_queue_size=len(self.replay_queue),
            feedback_events=list(self.feedback_events),
        )


__all__ = [
    "FeedbackEvent",
    "Layer6FeedbackState",
    "FeedbackLearningEngine",
]
