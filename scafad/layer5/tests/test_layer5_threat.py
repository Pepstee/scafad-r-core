"""
scafad/layer5/tests/test_layer5_threat.py
==========================================

T-023 — Layer 5 MITRE ATT&CK threat-alignment unit tests.

Covers ThreatAlignmentEngine.align() — maps anomaly labels to MITRE
tactics/techniques and produces a human-readable attack story.

WP: Day-4 (7-day sprint)
"""

from __future__ import annotations

import json
import unittest

from layer4.explainability import Layer4DecisionTrace
from layer5.threat_alignment import ThreatAlignmentEngine, ThreatAlignmentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(
    record_id: str = "rec-001",
    trace_id: str = "trace-001",
    decision: str = "review",
    severity: str = "medium",
    explanation_summary: str = "Test summary.",
    risk_band: str = "medium",
) -> Layer4DecisionTrace:
    return Layer4DecisionTrace(
        record_id=record_id,
        trace_id=trace_id,
        decision=decision,
        severity=severity,
        explanation_summary=explanation_summary,
        trace_metadata={"risk_band": risk_band},
    )


def _align(anomaly_type: str, **trace_kwargs) -> ThreatAlignmentResult:
    engine = ThreatAlignmentEngine()
    trace = _make_trace(**trace_kwargs)
    return engine.align(anomaly_type, trace)


# ---------------------------------------------------------------------------
# Tests: tactic/technique mapping
# ---------------------------------------------------------------------------

class TestTacticMapping(unittest.TestCase):
    """Anomaly labels must map to the expected MITRE ATT&CK tactics."""

    def test_timeout_maps_to_impact(self) -> None:
        result = _align("timeout")
        self.assertIn("impact", result.tactics)

    def test_dos_maps_to_impact(self) -> None:
        result = _align("dos_attack")
        self.assertIn("impact", result.tactics)

    def test_injection_maps_to_execution(self) -> None:
        result = _align("adversarial_injection")
        self.assertIn("execution", result.tactics)

    def test_exfil_maps_to_exfiltration(self) -> None:
        result = _align("data_exfiltration")
        self.assertIn("exfiltration", result.tactics)

    def test_leak_maps_to_exfiltration(self) -> None:
        result = _align("memory_leak")
        self.assertIn("exfiltration", result.tactics)

    def test_spike_maps_to_discovery(self) -> None:
        result = _align("cpu_spike")
        self.assertIn("discovery", result.tactics)

    def test_drift_maps_to_discovery(self) -> None:
        result = _align("behavioural_drift")
        self.assertIn("discovery", result.tactics)

    def test_unknown_label_defaults_to_collection(self) -> None:
        result = _align("benign")
        self.assertIn("collection", result.tactics)
        self.assertIn("telemetry-observation", result.techniques)


class TestTechniqueMapping(unittest.TestCase):
    """Technique list must be non-empty and contain recognised ATT&CK technique names."""

    def test_injection_technique_present(self) -> None:
        result = _align("injection")
        self.assertIn("command-and-scripting-interpreter", result.techniques)

    def test_exfil_technique_present(self) -> None:
        result = _align("exfil")
        self.assertIn("automated-exfiltration", result.techniques)

    def test_spike_technique_present(self) -> None:
        result = _align("spike")
        self.assertIn("cloud-service-discovery", result.techniques)

    def test_dos_technique_present(self) -> None:
        result = _align("dos")
        self.assertIn("service-degradation", result.techniques)

    def test_techniques_non_empty(self) -> None:
        for label in ("benign", "timeout", "injection", "exfil", "spike"):
            result = _align(label)
            self.assertGreater(len(result.techniques), 0,
                               f"Empty techniques for label '{label}'")


# ---------------------------------------------------------------------------
# Tests: alignment confidence
# ---------------------------------------------------------------------------

class TestAlignmentConfidence(unittest.TestCase):
    """alignment_confidence must be in [0.0, 1.0] and scale with tactic count."""

    def test_confidence_in_range(self) -> None:
        for label in ("benign", "timeout", "injection", "exfil", "spike", "dos"):
            result = _align(label)
            self.assertGreaterEqual(result.alignment_confidence, 0.0,
                                    f"Confidence below 0 for '{label}'")
            self.assertLessEqual(result.alignment_confidence, 1.0,
                                 f"Confidence above 1 for '{label}'")

    def test_multiple_tactics_raises_confidence(self) -> None:
        # A label that matches multiple patterns will have more tactics
        # → higher confidence than a single-tactic match.
        multi = _align("injection_exfil")   # matches both injection and exfil
        single = _align("benign")           # defaults to single collection tactic
        self.assertGreater(
            multi.alignment_confidence,
            single.alignment_confidence,
        )

    def test_confidence_is_float(self) -> None:
        result = _align("benign")
        self.assertIsInstance(result.alignment_confidence, float)


# ---------------------------------------------------------------------------
# Tests: attack story
# ---------------------------------------------------------------------------

class TestAttackStory(unittest.TestCase):
    """attack_story must be a non-empty string that references the decision and tactic."""

    def test_story_is_non_empty(self) -> None:
        result = _align("timeout")
        self.assertIsInstance(result.attack_story, str)
        self.assertGreater(len(result.attack_story), 10)

    def test_story_references_decision(self) -> None:
        trace = _make_trace(decision="escalate")
        result = ThreatAlignmentEngine().align("timeout", trace)
        self.assertIn("escalate", result.attack_story)

    def test_story_references_tactic(self) -> None:
        result = _align("timeout")
        self.assertIn("impact", result.attack_story)


# ---------------------------------------------------------------------------
# Tests: campaign cluster
# ---------------------------------------------------------------------------

class TestCampaignCluster(unittest.TestCase):
    """campaign_cluster must encode decision and primary tactic."""

    def test_cluster_contains_decision(self) -> None:
        trace = _make_trace(decision="review")
        result = ThreatAlignmentEngine().align("exfil", trace)
        self.assertIn("review", result.campaign_cluster)

    def test_cluster_contains_primary_tactic(self) -> None:
        result = _align("timeout")
        self.assertIn("impact", result.campaign_cluster)

    def test_cluster_is_non_empty_string(self) -> None:
        result = _align("benign")
        self.assertIsInstance(result.campaign_cluster, str)
        self.assertGreater(len(result.campaign_cluster), 0)


# ---------------------------------------------------------------------------
# Tests: field identity and I-15 contract
# ---------------------------------------------------------------------------

class TestFieldIdentityAndContract(unittest.TestCase):
    """record_id/trace_id preserved; to_dict() JSON-serialisable with required keys."""

    def test_record_id_preserved(self) -> None:
        trace = _make_trace(record_id="my-rec")
        result = ThreatAlignmentEngine().align("benign", trace)
        self.assertEqual(result.record_id, "my-rec")

    def test_trace_id_preserved(self) -> None:
        trace = _make_trace(trace_id="t-99")
        result = ThreatAlignmentEngine().align("benign", trace)
        self.assertEqual(result.trace_id, "t-99")

    def test_to_dict_is_json_serialisable(self) -> None:
        result = _align("timeout")
        json.dumps(result.to_dict())  # must not raise

    def test_to_dict_required_keys(self) -> None:
        result = _align("benign")
        d = result.to_dict()
        required = {
            "record_id", "trace_id", "tactics", "techniques",
            "campaign_cluster", "alignment_confidence", "attack_story",
        }
        self.assertTrue(required.issubset(d.keys()),
                        f"Missing: {required - d.keys()}")

    def test_to_dict_tactics_is_list(self) -> None:
        result = _align("benign")
        self.assertIsInstance(result.to_dict()["tactics"], list)

    def test_to_dict_techniques_is_list(self) -> None:
        result = _align("benign")
        self.assertIsInstance(result.to_dict()["techniques"], list)


if __name__ == "__main__":
    unittest.main()
