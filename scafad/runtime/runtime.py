"""Canonical application-level runtime entrypoint for SCAFAD."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

from layer0.telemetry import (
    AnomalyType,
    ExecutionPhase,
    TelemetryRecord,
    TelemetrySource,
)
from layer0.adapter import RCoreToLayer1Adapter
from layer1.pipeline import Layer1CanonicalPipeline, Layer1ProcessedRecord
from .pipeline import MultilayerPipelineResult, SCAFADMultilayerPipeline


@dataclass
class CanonicalRuntimeResult:
    """Stable end-to-end runtime result for the canonical software path."""

    layer0_record: TelemetryRecord
    adapted_record: Dict[str, Any]
    layer1_record: Layer1ProcessedRecord
    multilayer_result: MultilayerPipelineResult

    def to_dict(self) -> Dict[str, Any]:
        layer0_payload = self.layer0_record.to_dict()
        return {
            "layer0_record": copy.deepcopy(layer0_payload),
            "adapted_record": copy.deepcopy(self.adapted_record),
            "layer1_record": self.layer1_record.to_dict(),
            "multilayer_result": self.multilayer_result.to_dict(),
        }


class SCAFADCanonicalRuntime:
    """Single explicit execution surface for the canonical SCAFAD pipeline."""

    def __init__(
        self,
        adapter: Optional[RCoreToLayer1Adapter] = None,
        layer1_pipeline: Optional[Layer1CanonicalPipeline] = None,
        multilayer_pipeline: Optional[SCAFADMultilayerPipeline] = None,
    ) -> None:
        self.adapter = adapter or RCoreToLayer1Adapter()
        self.layer1_pipeline = layer1_pipeline or Layer1CanonicalPipeline(self.adapter)
        self.multilayer_pipeline = multilayer_pipeline or SCAFADMultilayerPipeline()

    def process_record(
        self,
        record: TelemetryRecord,
        analyst_label: Optional[str] = None,
        redacted_fields: Optional[list[str]] = None,
    ) -> CanonicalRuntimeResult:
        adapted = self.adapter.adapt(record)
        layer1_record = self.layer1_pipeline.process_adapted_record(adapted)
        multilayer_result = self.multilayer_pipeline.process_record(
            layer1_record,
            analyst_label=analyst_label,
            redacted_fields=redacted_fields,
        )
        return CanonicalRuntimeResult(
            layer0_record=record,
            adapted_record=adapted,
            layer1_record=layer1_record,
            multilayer_result=multilayer_result,
        )

    def process_event(
        self,
        event: Dict[str, Any],
        analyst_label: Optional[str] = None,
        redacted_fields: Optional[list[str]] = None,
    ) -> CanonicalRuntimeResult:
        record = self.build_record(event)
        return self.process_record(
            record,
            analyst_label=analyst_label,
            redacted_fields=redacted_fields,
        )

    def build_record(self, event: Dict[str, Any]) -> TelemetryRecord:
        payload_size_bytes = int(event.get("payload_size_bytes", 0) or 0)
        if payload_size_bytes <= 0:
            payload_size_bytes = len(str(event).encode("utf-8"))

        anomaly_type = self._map_anomaly(event.get("anomaly") or event.get("anomaly_type"))
        execution_phase = self._map_phase(event.get("execution_phase") or event.get("phase"))
        telemetry_fields = event.get("telemetry_fields", {}) or {}
        execution_environment = event.get("execution_environment", {}) or {}

        return TelemetryRecord(
            event_id=str(event.get("event_id") or event.get("payload_id") or event.get("record_id") or "runtime-event"),
            timestamp=float(event.get("timestamp") or event.get("invocation_timestamp") or 0.0),
            function_id=str(event.get("function_id") or event.get("function_profile_id") or event.get("function_name") or "runtime_function"),
            execution_phase=execution_phase,
            anomaly_type=anomaly_type,
            duration=float(event.get("duration") or telemetry_fields.get("duration_ms", 0.0)) / (1000.0 if telemetry_fields.get("duration_ms") is not None and event.get("duration") is None else 1.0),
            memory_spike_kb=int(event.get("memory_spike_kb") or (telemetry_fields.get("max_memory_used_mb", 0) * 1024)),
            cpu_utilization=float(event.get("cpu_utilization") or event.get("cpu", 0.0) or (88.0 if anomaly_type != AnomalyType.BENIGN else 26.0)),
            network_io_bytes=int(event.get("network_io_bytes") or (self._count_network_calls(event.get("network_calls")) * 1024)),
            fallback_mode=bool(event.get("fallback_mode", False) or event.get("force_starvation", False)),
            source=TelemetrySource.SCAFAD_LAYER0,
            concurrency_id=str(event.get("concurrency_id") or "runtime-concurrency"),
            container_id=event.get("container_id"),
            region=event.get("region") or execution_environment.get("region"),
            runtime_version=event.get("runtime_version") or execution_environment.get("runtime"),
            trigger_type=event.get("trigger_type") or event.get("httpMethod"),
            payload_size_bytes=payload_size_bytes,
            adversarial_score=float(event.get("adversarial_score", 0.0) or 0.0),
            economic_risk_score=float(event.get("economic_risk_score", 0.0) or 0.0),
            silent_failure_probability=float(event.get("silent_failure_probability", 0.0) or 0.0),
            completeness_score=float(event.get("completeness_score", 1.0) or 1.0),
            confidence_level=float(event.get("confidence_level", 1.0) or 1.0),
            data_quality_score=float(event.get("data_quality_score", 1.0) or 1.0),
            custom_fields=copy.deepcopy(event.get("custom_fields") or event.get("generic_data") or {}),
            tags=copy.deepcopy(event.get("tags") or {}),
        )

    @staticmethod
    def _count_network_calls(raw_network_calls: Any) -> int:
        if isinstance(raw_network_calls, list):
            return len(raw_network_calls)
        return int(raw_network_calls or 0)

    @staticmethod
    def _map_phase(raw_phase: Any) -> ExecutionPhase:
        raw = str(raw_phase or "invoke").lower()
        mapping = {
            "init": ExecutionPhase.INIT,
            "initialization": ExecutionPhase.INIT,
            "invoke": ExecutionPhase.INVOKE,
            "execution": ExecutionPhase.INVOKE,
            "shutdown": ExecutionPhase.SHUTDOWN,
            "completion": ExecutionPhase.SHUTDOWN,
            "error": ExecutionPhase.ERROR,
            "timeout": ExecutionPhase.TIMEOUT,
        }
        return mapping.get(raw, ExecutionPhase.INVOKE)

    @staticmethod
    def _map_anomaly(raw_anomaly: Any) -> AnomalyType:
        raw = str(raw_anomaly or "benign").lower()
        for member in AnomalyType:
            if member.value == raw:
                return member
        fallback_map = {
            "none": AnomalyType.BENIGN,
            "suspicious": AnomalyType.MEMORY_SPIKE,
            "malicious": AnomalyType.ADVERSARIAL_INJECTION,
        }
        return fallback_map.get(raw, AnomalyType.BENIGN)


__all__ = ["CanonicalRuntimeResult", "SCAFADCanonicalRuntime"]
