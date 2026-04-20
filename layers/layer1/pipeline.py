"""Canonical Layer 1 intake pipeline for the module-split SCAFAD architecture."""

from __future__ import annotations

import copy
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app_telemetry import TelemetryRecord
from layers.layer1.adapter import RCoreToLayer1Adapter
from layers.layer1.validation import InputValidationGateway


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class Layer1AuditRecord:
    phases_completed: List[str] = field(default_factory=list)
    redacted_fields: List[str] = field(default_factory=list)
    hashed_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class Layer1QualityReport:
    completeness_score: float
    anomaly_signal_preservation: float
    pii_fields_redacted: int
    issues: List[str] = field(default_factory=list)


@dataclass
class Layer1ProcessedRecord:
    record_id: str
    function_name: str
    timestamp: float
    anomaly_type: str
    execution_phase: str
    schema_version: str
    telemetry_data: Dict[str, Any]
    context_metadata: Dict[str, Any]
    provenance_chain: Dict[str, Any]
    quality_report: Layer1QualityReport
    audit_record: Layer1AuditRecord
    trace_id: str
    trust_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "function_name": self.function_name,
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "execution_phase": self.execution_phase,
            "schema_version": self.schema_version,
            "telemetry_data": copy.deepcopy(self.telemetry_data),
            "context_metadata": copy.deepcopy(self.context_metadata),
            "provenance_chain": copy.deepcopy(self.provenance_chain),
            "quality_report": {
                "completeness_score": self.quality_report.completeness_score,
                "anomaly_signal_preservation": self.quality_report.anomaly_signal_preservation,
                "pii_fields_redacted": self.quality_report.pii_fields_redacted,
                "issues": list(self.quality_report.issues),
            },
            "audit_record": {
                "phases_completed": list(self.audit_record.phases_completed),
                "redacted_fields": list(self.audit_record.redacted_fields),
                "hashed_fields": list(self.audit_record.hashed_fields),
                "warnings": list(self.audit_record.warnings),
                "processing_time_ms": self.audit_record.processing_time_ms,
            },
            "trace_id": self.trace_id,
            "trust_context": copy.deepcopy(self.trust_context),
        }


class Layer1CanonicalPipeline:
    REQUIRED_FIELDS = (
        "record_id",
        "timestamp",
        "function_name",
        "execution_phase",
        "anomaly_type",
        "telemetry_data",
        "schema_version",
    )
    PRESERVED_FIELDS = frozenset(
        {
            "record_id",
            "timestamp",
            "function_name",
            "execution_phase",
            "anomaly_type",
            "telemetry_data",
            "schema_version",
            "context_metadata",
            "provenance_chain",
            "error_message",
            "error_stack",
        }
    )
    PII_KEY_RE = re.compile(
        r"email|phone|mobile|ssn|passport|credit.?card|bank.?account|"
        r"\bip\b|ip_addr|password|api.?key|token|secret|dob|date_of_birth|"
        r"full.?name|first.?name|last.?name",
        re.IGNORECASE,
    )
    EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    PHONE_RE = re.compile(r"\+?\d[\d\-\s]{6,}\d")
    CC_RE = re.compile(r"\b(?:\d[ \-]?){15,16}\b")
    CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self, adapter: Optional[RCoreToLayer1Adapter] = None) -> None:
        self.adapter = adapter or RCoreToLayer1Adapter()
        self._validator = InputValidationGateway()

    def process_l0_record(self, record: TelemetryRecord) -> Layer1ProcessedRecord:
        return self.process_adapted_record(self.adapter.adapt(record))

    def process_adapted_record(self, adapted: Dict[str, Any]) -> Layer1ProcessedRecord:
        started = _now_ms()
        original = copy.deepcopy(adapted)
        working = copy.deepcopy(adapted)
        warnings: List[str] = []
        phases = ["validation", "sanitisation", "privacy", "hashing", "preservation", "quality", "audit"]

        self._validate_shape(working)
        self._sanitize_record(working)
        redacted_fields = self._apply_privacy(working)
        hashed_fields = self._apply_hashing(working)
        preservation_score = self._measure_preservation(original, working)
        quality = self._assess_quality(working, preservation_score, len(redacted_fields))
        trace_id = self._build_trace_id(working)
        trust_context = self._build_trust_context(working, preservation_score)

        audit = Layer1AuditRecord(
            phases_completed=phases,
            redacted_fields=redacted_fields,
            hashed_fields=hashed_fields,
            warnings=warnings,
            processing_time_ms=round(_now_ms() - started, 3),
        )

        return Layer1ProcessedRecord(
            record_id=str(working["record_id"]),
            function_name=str(working["function_name"]),
            timestamp=float(working["timestamp"]),
            anomaly_type=str(working["anomaly_type"]),
            execution_phase=str(working["execution_phase"]),
            schema_version=str(working["schema_version"]),
            telemetry_data=working.get("telemetry_data", {}),
            context_metadata=working.get("context_metadata", {}),
            provenance_chain=working.get("provenance_chain", {}),
            quality_report=quality,
            audit_record=audit,
            trace_id=trace_id,
            trust_context=trust_context,
        )

    def _validate_shape(self, record: Dict[str, Any]) -> None:
        """Validate the adapted record via InputValidationGateway (WP-3.1).

        Raises ValueError with the full accumulated error list if validation
        fails, preserving the existing raise-on-failure contract for callers.
        """
        result = self._validator.validate(record)
        if not result.valid:
            messages = "; ".join(str(e) for e in result.errors)
            raise ValueError(f"L1 validation failed: {messages}")

    def _sanitize_record(self, record: Dict[str, Any]) -> None:
        for key, value in list(record.items()):
            if key in self.PRESERVED_FIELDS:
                continue
            if isinstance(value, str):
                record[key] = self.CTRL_RE.sub("", value).strip()
        telemetry = record.get("telemetry_data", {})
        for key, value in list(telemetry.items()):
            if isinstance(value, str):
                telemetry[key] = self.CTRL_RE.sub("", value).strip()

    def _apply_privacy(self, record: Dict[str, Any]) -> List[str]:
        redacted_fields: List[str] = []

        def redact_mapping(mapping: Dict[str, Any], prefix: str = "") -> None:
            for key, value in list(mapping.items()):
                fq_key = f"{prefix}.{key}" if prefix else key
                if self.PII_KEY_RE.search(key):
                    mapping[key] = "[REDACTED]"
                    redacted_fields.append(fq_key)
                    continue
                if isinstance(value, dict):
                    redact_mapping(value, fq_key)
                elif isinstance(value, str):
                    new_value = value
                    if self.EMAIL_RE.search(new_value):
                        new_value = self.EMAIL_RE.sub("[REDACTED_EMAIL]", new_value)
                    if self.PHONE_RE.search(new_value):
                        new_value = self.PHONE_RE.sub("[REDACTED_PHONE]", new_value)
                    if self.CC_RE.search(new_value):
                        new_value = self.CC_RE.sub("[REDACTED_CARD]", new_value)
                    if new_value is not value:
                        mapping[key] = new_value
                        redacted_fields.append(fq_key)

        redact_mapping(record.get("telemetry_data", {}), "telemetry_data")
        redact_mapping(record.get("context_metadata", {}), "context_metadata")
        return sorted(set(redacted_fields))

    def _apply_hashing(self, record: Dict[str, Any]) -> List[str]:
        hashed_fields: List[str] = []
        context = record.get("context_metadata", {})
        for key in ("concurrency_id", "trigger_type"):
            if key in context and isinstance(context[key], str):
                context[key] = hashlib.sha256(context[key].encode("utf-8")).hexdigest()
                hashed_fields.append(f"context_metadata.{key}")
        return hashed_fields

    def _measure_preservation(self, original: Dict[str, Any], processed: Dict[str, Any]) -> float:
        weights = {
            "anomaly_type": 0.2,
            "execution_phase": 0.15,
            "record_id": 0.1,
            "function_name": 0.1,
            "timestamp": 0.1,
            "schema_version": 0.05,
            "telemetry_data": 0.15,
            "context_metadata": 0.1,
            "provenance_chain": 0.05,
        }
        preserved = 0.0
        for key, weight in weights.items():
            if original.get(key) == processed.get(key):
                preserved += weight
        return round(preserved, 4)

    def _assess_quality(self, record: Dict[str, Any], preservation_score: float, pii_fields_redacted: int) -> Layer1QualityReport:
        present = 0
        for field_name in self.REQUIRED_FIELDS:
            if record.get(field_name) not in (None, "", {}):
                present += 1
        completeness = round(present / len(self.REQUIRED_FIELDS), 4)
        issues: List[str] = []
        if completeness < 1.0:
            issues.append("incomplete_required_fields")
        if preservation_score < 0.9995:
            issues.append("preservation_below_target")
        return Layer1QualityReport(
            completeness_score=completeness,
            anomaly_signal_preservation=preservation_score,
            pii_fields_redacted=pii_fields_redacted,
            issues=issues,
        )

    def _build_trace_id(self, record: Dict[str, Any]) -> str:
        raw = f"{record['record_id']}|{record['function_name']}|{record['timestamp']}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _build_trust_context(self, record: Dict[str, Any], preservation_score: float) -> Dict[str, Any]:
        context = record.get("context_metadata", {})
        base_confidence = float(context.get("confidence_level", 0.8) or 0.8)
        trust_score = round(max(0.1, min(1.0, base_confidence * preservation_score)), 4)
        return {
            "confidence_level": base_confidence,
            "preservation_score": preservation_score,
            "trust_score": trust_score,
            "source_layer": "layer_1",
        }


__all__ = [
    "Layer1AuditRecord",
    "Layer1QualityReport",
    "Layer1ProcessedRecord",
    "Layer1CanonicalPipeline",
]
