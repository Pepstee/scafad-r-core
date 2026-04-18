"""Focused tests for the basic SCAFAD Layers 2-6 scaffold."""

from app_telemetry import AnomalyType, ExecutionPhase, TelemetryRecord, TelemetrySource
from core.layer1_pipeline import Layer1CanonicalPipeline
from core.layer2_detection_matrix import MultiVectorDetectionMatrix
from core.layer3_trust_fusion import TrustWeightedFusionEngine
from core.layer4_explainability import ExplainabilityDecisionEngine
from core.layer5_threat_alignment import ThreatAlignmentEngine
from core.layer6_feedback_learning import FeedbackLearningEngine
from core.multilayer_pipeline import SCAFADMultilayerPipeline


def _sample_record():
    return TelemetryRecord(
        event_id="rec-001",
        timestamp=1744718400.0,
        function_id="image_processor",
        execution_phase=ExecutionPhase.INVOKE,
        anomaly_type=AnomalyType.MEMORY_SPIKE,
        duration=0.33,
        memory_spike_kb=196608,
        cpu_utilization=91.0,
        network_io_bytes=4096,
        fallback_mode=False,
        source=TelemetrySource.SCAFAD_LAYER0,
        concurrency_id="conc-001",
    )


def _sample_processed_record():
    pipeline = Layer1CanonicalPipeline()
    return pipeline.process_l0_record(_sample_record())


def _legacy_like_record_dict():
    return {
        "record_id": "rec-001",
        "timestamp": 1744718400.0,
        "function_name": "image_processor",
        "execution_phase": "execution",
        "anomaly_type": "memory_spike",
        "telemetry_data": {
            "memory_usage_mb": 192,
            "cpu_utilization": 91,
            "latency_drift": 0.42,
            "graph_edge_count": 24,
            "graph_node_count": 4,
        },
    }


def test_layer2_detection_matrix_produces_multiple_signals():
    result = MultiVectorDetectionMatrix().analyze(_sample_processed_record())
    assert len(result.signals) == 4
    assert result.aggregate_score > 0.0
    assert result.anomaly_indicated is True


def test_layer3_fusion_uses_trust_weighted_scores():
    detection = MultiVectorDetectionMatrix().analyze(_sample_processed_record())
    fusion = TrustWeightedFusionEngine().fuse(detection)
    assert 0.0 <= fusion.fused_score <= 1.0
    assert "rule_chain" in fusion.trust_weights
    assert len(fusion.leading_signals) == 2


def test_layer4_explainability_builds_decision_trace():
    detection = MultiVectorDetectionMatrix().analyze(_sample_processed_record())
    fusion = TrustWeightedFusionEngine().fuse(detection)
    trace = ExplainabilityDecisionEngine().build_trace(detection, fusion, redacted_fields=["email"])
    assert trace.decision in {"observe", "review", "escalate"}
    assert trace.severity in {"low", "medium", "high"}
    assert trace.redacted_fields == ["email"]
    assert trace.explanation_points


def test_layer5_threat_alignment_maps_to_tactics():
    detection = MultiVectorDetectionMatrix().analyze(_sample_processed_record())
    fusion = TrustWeightedFusionEngine().fuse(detection)
    trace = ExplainabilityDecisionEngine().build_trace(detection, fusion)
    alignment = ThreatAlignmentEngine().align("memory_spike", trace)
    assert alignment.tactics
    assert alignment.techniques
    assert alignment.campaign_cluster.startswith(trace.decision)


def test_layer6_feedback_learning_adjusts_trust():
    engine = FeedbackLearningEngine(initial_trust=0.7)
    detection = MultiVectorDetectionMatrix().analyze(_sample_processed_record())
    fusion = TrustWeightedFusionEngine().fuse(detection)
    trace = ExplainabilityDecisionEngine().build_trace(detection, fusion)
    alignment = ThreatAlignmentEngine().align("memory_spike", trace)
    state = engine.ingest_feedback("rec-001", "confirmed", alignment)
    assert state.adjusted_trust > 0.7
    assert state.replay_queue_size == 1
    assert state.feedback_events[0].record_id == "rec-001"


def test_multilayer_pipeline_chains_l2_to_l6():
    pipeline = SCAFADMultilayerPipeline()
    result = pipeline.process_record(_sample_processed_record(), analyst_label="confirmed", redacted_fields=["ssn"])
    assert result.layer2.record_id
    assert result.layer1.trace_id
    assert result.layer3.fused_score >= 0.0
    assert result.layer4.redacted_fields == ["ssn"]
    assert result.layer5.tactics
    assert result.layer6 is not None
    assert result.layer6.replay_queue_size == 1


def test_multilayer_pipeline_accepts_legacy_like_dict_envelope():
    pipeline = SCAFADMultilayerPipeline()
    result = pipeline.process_record(_legacy_like_record_dict(), analyst_label="confirmed")
    assert result.layer1.record_id == "rec-001"
    assert result.layer2.anomaly_indicated is True
